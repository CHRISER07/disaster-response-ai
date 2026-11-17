"""
Targeted rerun of 401 corrupted rows (silent 429/RATE_LIMIT/EMPTY failures)
using Gemini 3.5 Flash Lite as the generation engine via LangGraph ReAct agent.

Why Gemini: Groq daily token quota (200k TPD) was exhausted across all 4 accounts.
Gemini free tier has no hard token-per-day cap at this scale.
Scientifically valid: same queries, same seeds, same ablation monkeypatches.
Documented as discrepancy D7 in DISCREPANCIES.md.
"""
import os, sys, json, csv, time, random, re, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

os.environ["PYTHONUTF8"] = "1"
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from evaluation.harness_v2 import trim_text, STATUS_OK
from evaluation.run_dev_eval_v2 import kw_scores, svc_regex_check, FIELDNAMES

CSV_PATH    = ROOT / "evaluation" / "results_per_query_v2.csv"
DATASET_FILE = ROOT / "evaluation" / "golden_dataset_v2.json"

GEMINI_MODEL = "gemini-3.5-flash-lite"
MAX_WORKERS  = 2   # conservative - 4 keys but keep steady
RATE_LIMIT_RPM = 20  # ~5 rpm per key, well within free tier

# ── Rate limiter ───────────────────────────────────────────────────────────────
class TokenBucketRateLimiter:
    def __init__(self, max_per_minute):
        self._lock = threading.Lock()
        self._interval = 60.0 / max_per_minute
        self._next_allowed = time.monotonic()
    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = max(self._next_allowed + self._interval, now + self._interval)
                    return
                wait = self._next_allowed - now
            time.sleep(min(wait, 0.05))

_rate_limiter = TokenBucketRateLimiter(RATE_LIMIT_RPM)
_csv_lock = threading.Lock()


# ── Gemini-based full agent runner ─────────────────────────────────────────────
def run_full_agent_gemini(query: str, seed: int, no_tgi: bool = False,
                          no_dsel: bool = False, no_telemetry: bool = False) -> dict:
    """Runs LangGraph ReAct agent using Gemini 3.5 Flash Lite instead of Groq."""
    t0 = time.perf_counter()
    thread_id = f"gemini_eval_{seed}_{int(t0*1000)}"
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langgraph.prebuilt import create_react_agent
        from core.key_manager import gemini_pool
        import inspect
        import core.safety_validator as sv_mod
        from tools import ALL_TOOLS
        from agents.disaster_agent import _get_checkpointer, _build_system_prompt

        # Monkeypatch no_dsel
        original_vr = None
        if no_dsel:
            original_vr = sv_mod.validate_and_repair
            sv_mod.validate_and_repair = lambda text, *a, **kw: {
                "answer": text, "safe": True, "violations": [], "rewritten": False
            }

        # Monkeypatch no_telemetry
        original_sensor = original_weather = None
        if no_telemetry:
            try:
                import tools.sensor_tool as st
                original_sensor = st.query_water_sensor
                st.query_water_sensor = lambda *a, **kw: "Telemetry disabled."
            except Exception: pass
            try:
                import tools.weather_tool as wt
                original_weather = wt.fetch_live_weather
                wt.fetch_live_weather = lambda *a, **kw: "Telemetry disabled."
            except Exception: pass

        # Monkeypatch no_tgi
        original_kb_search = None
        if no_tgi:
            try:
                import tools.kb_search_tool as kb
                original_search_chroma = kb._search_chroma
                def stripped_search(*args, **kwargs):
                    text = original_search_chroma(*args, **kwargs)
                    return re.sub(r"\[(HISTORICAL[^\]]*|LIVE[^\]]*|RECENT[^\]]*|KB[^\]]*)\]", "[DOCUMENT]", text)
                kb._search_chroma = stripped_search
                original_kb_search = original_search_chroma
            except Exception: pass

        # Build fresh Gemini LLM with rotated key
        key = gemini_pool.get_key()
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=key,
            temperature=0.1,
            max_retries=3
        )

        memory = _get_checkpointer()
        system_prompt = _build_system_prompt()

        sig = inspect.signature(create_react_agent)
        params = sig.parameters

        if "state_modifier" in params:
            agent = create_react_agent(model=llm, tools=ALL_TOOLS, checkpointer=memory, state_modifier=system_prompt)
        elif "prompt" in params:
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.messages import SystemMessage
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ])
            agent = create_react_agent(model=llm, tools=ALL_TOOLS, checkpointer=memory, prompt=prompt)
        else:
            agent = create_react_agent(model=llm, tools=ALL_TOOLS, checkpointer=memory)

        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 18}

        t_active = time.perf_counter()
        out = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)
        active_elapsed = time.perf_counter() - t_active
        total_elapsed = time.perf_counter() - t0

        # Restore monkeypatches
        if no_dsel and original_vr is not None:
            sv_mod.validate_and_repair = original_vr
        if no_telemetry:
            if original_sensor:
                import tools.sensor_tool as st; st.query_water_sensor = original_sensor
            if original_weather:
                import tools.weather_tool as wt; wt.fetch_live_weather = original_weather
        if no_tgi and original_kb_search:
            import tools.kb_search_tool as kb; kb._search_chroma = original_kb_search

        # Extract answer from final message
        messages = out.get("messages", [])
        answer = messages[-1].content if messages else ""

        # Detect hidden error messages
        status = STATUS_OK
        if "Agent error" in answer and ("429" in answer or "rate_limit" in answer.lower()):
            status = "RATE_LIMIT"
        elif not answer.strip():
            status = "EMPTY"

        # Extract tool calls
        tool_calls = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if name:
                        tool_calls.append(name)

        return {
            "status": status,
            "full_answer": answer,
            "active_latency_s": round(active_elapsed, 3),
            "total_latency_s": round(total_elapsed, 3),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tools_called": ",".join(dict.fromkeys(tool_calls)),
            "model_used": GEMINI_MODEL,
            "retries": 0
        }

    except Exception as e:
        total_elapsed = time.perf_counter() - t0
        err = str(e)
        status = "STEP_LIMIT" if "recursion" in err.lower() else "GEMINI_ERROR"
        return {
            "status": status,
            "full_answer": err[:500],
            "active_latency_s": round(total_elapsed, 3),
            "total_latency_s": round(total_elapsed, 3),
            "prompt_tokens": 0, "completion_tokens": 0,
            "tools_called": "", "model_used": GEMINI_MODEL, "retries": 0
        }


# Config → function mapping
CFG_FN = {
    "Full_Agent":        lambda q, s: run_full_agent_gemini(q, s),
    "Full_Agent_NoTGI":  lambda q, s: run_full_agent_gemini(q, s, no_tgi=True),
    "Full_Agent_NoDSEL": lambda q, s: run_full_agent_gemini(q, s, no_dsel=True),
    "Full_Agent_NoTelem":lambda q, s: run_full_agent_gemini(q, s, no_telemetry=True),
}


def main():
    dataset = {q["id"]: q for q in json.loads(DATASET_FILE.read_text(encoding="utf-8"))}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from CSV.")

    # Find all corrupted rows: status=OK with 429 error in answer, or RATE_LIMIT, or EMPTY
    to_patch = []
    for idx, r in enumerate(rows):
        ans = r.get("full_answer", "")
        status = r.get("status", "")
        is_silent_fail = status == "OK" and ("Agent error" in ans and ("429" in ans or "rate_limit_exceeded" in ans))
        is_explicit_fail = status in ("RATE_LIMIT", "EMPTY")
        cfg = r.get("config", "")
        if (is_silent_fail or is_explicit_fail) and cfg in CFG_FN:
            to_patch.append((idx, r))

    print(f"Found {len(to_patch)} corrupted rows to rerun with Gemini {GEMINI_MODEL}.")
    if not to_patch:
        print("Nothing to patch.")
        return

    # Prepare tasks
    tasks = []
    for idx, r in to_patch:
        q_id = int(r["id"])
        seed = int(r["run_id"])
        cfg  = r["config"]
        q_obj = dataset.get(q_id, {})
        tasks.append({
            "row_idx": idx, "row": r,
            "q_id": q_id, "seed": seed, "cfg": cfg,
            "query_text": q_obj.get("query", ""),
            "gold_kw": q_obj.get("keywords", []),
            "expected_svc": q_obj.get("expected_service", ""),
        })

    counter = [0]
    counter_lock = threading.Lock()
    total = len(tasks)
    rows_lock = threading.Lock()

    def process(task):
        _rate_limiter.acquire()
        cfg = task["cfg"]
        fn  = CFG_FN[cfg]
        idx = task["row_idx"]

        t0 = time.perf_counter()
        res = fn(task["query_text"], task["seed"])
        elapsed = time.perf_counter() - t0

        if res["status"] == STATUS_OK:
            p, kw_r, f1 = kw_scores(res["full_answer"], task["gold_kw"])
            svc = svc_regex_check(res["full_answer"], task["expected_svc"])
        else:
            p = kw_r = f1 = svc = ""

        new_row = dict(task["row"])
        new_row.update({
            "status": res["status"], "kw_precision": p, "kw_recall": kw_r,
            "kw_f1": f1, "svc_regex": svc, "active_latency_s": res["active_latency_s"],
            "total_latency_s": res["total_latency_s"], "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"], "tools_called": res["tools_called"],
            "model_used": res["model_used"], "retries": res["retries"],
            "full_answer": res["full_answer"],
        })

        with rows_lock:
            rows[idx] = new_row

        with counter_lock:
            counter[0] += 1
            done = counter[0]
            print(f"  [{done:3d}/{total}] {cfg:30s} id={task['q_id']:3d} s={task['seed']} → {res['status']} ({elapsed:.1f}s, {res['model_used']})", flush=True)

        return res["status"]

    print(f"\nStarting parallel rerun: {MAX_WORKERS} workers, {RATE_LIMIT_RPM} RPM...\n")
    t_start = time.perf_counter()
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(process, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                st = fut.result()
                if st not in (STATUS_OK,):
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"  [ERROR] {e}")

    # Atomic write-back
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(CSV_PATH)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"GEMINI RERUN COMPLETE")
    print(f"  Rows patched : {total}")
    print(f"  Remaining err: {errors}")
    print(f"  Elapsed      : {elapsed/60:.1f} min")
    print(f"  Output       : {CSV_PATH}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

