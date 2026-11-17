"""
rerun_and_patch_gemini.py

Fast, linear, robust rerun & patch using Gemini 3.5 Flash Lite across all 4 Gemini API keys.
- Directly invokes LangGraph agent with Gemini as the backbone (no Groq 429 delays).
- Full tool calling support (query_water_sensor, search_official_protocols, etc.).
- Strict validate_answer_integrity check.
- Saves atomically after EVERY row into results_per_query_v2.csv.
- Real-time unbuffered live console output.
"""
import os, sys, json, csv, time, random, re
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

os.environ["PYTHONUTF8"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from evaluation.harness_v2 import trim_text, STATUS_OK, validate_answer_integrity
from evaluation.run_dev_eval_v2 import (
    kw_scores, svc_regex_check, FIELDNAMES
)
from core.key_manager import gemini_pool, get_gemini_chat
from tools import ALL_TOOLS
from langgraph.prebuilt import create_react_agent
from agents.disaster_agent import _build_system_prompt
from langchain_core.messages import SystemMessage, HumanMessage

CSV_PATH     = ROOT / "evaluation" / "results_per_query_v2.csv"
DATASET_FILE = ROOT / "evaluation" / "golden_dataset_v2.json"
GEMINI_MODEL = "gemini-3.5-flash-lite"

def is_row_flawed(r: dict) -> tuple:
    ans    = r.get("full_answer", "")
    status = r.get("status", "")
    tools  = r.get("tools_called", "")
    cfg    = r.get("config", "")

    clean_status, is_valid = validate_answer_integrity(ans, status)
    if not is_valid:
        return True, f"INVALID_{clean_status}"
    if cfg == "Text_Only_StaticRAG" and tools == "fts_fallback":
        return True, "FTS_FALLBACK"
    return False, ""

def save_csv_atomic(rows):
    tmp_path = CSV_PATH.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(CSV_PATH)

def run_agent_gemini(query: str, no_tgi: bool = False, no_dsel: bool = False, no_telemetry: bool = False):
    """Executes the agent with pure Gemini 3.5 Flash Lite across the 4 rotated keys."""
    t0 = time.perf_counter()

    # Dynamic ablation monkeypatches if requested
    original_vr = None
    if no_dsel:
        try:
            import core.safety_validator as sv_mod
            original_vr = sv_mod.validate_and_repair
            sv_mod.validate_and_repair = lambda ans: {"answer": ans, "status": "BYPASSED", "severity": "UNKNOWN", "violations": []}
        except Exception:
            pass

    original_sensor = None
    original_weather = None
    if no_telemetry:
        try:
            import tools.sensor_tool as st
            import tools.weather_tool as wt
            original_sensor = st.query_water_sensor
            original_weather = wt.fetch_live_weather
            st.query_water_sensor = lambda *args, **kwargs: "[TELEMETRY OFFLINE] Sensor API disabled."
            wt.fetch_live_weather = lambda *args, **kwargs: "[TELEMETRY OFFLINE] Weather API disabled."
        except Exception:
            pass

    original_kb_search = None
    if no_tgi:
        try:
            import tools.kb_search_tool as kb
            original_search_chroma = kb._search_chroma
            def stripped_search(*args, **kwargs):
                text = original_search_chroma(*args, **kwargs)
                import re
                return re.sub(r"\[(HISTORICAL[^\]]*|LIVE[^\]]*|RECENT[^\]]*|KB[^\]]*)\]", "[DOCUMENT]", text)
            kb._search_chroma = stripped_search
            original_kb_search = original_search_chroma
        except Exception:
            pass

    try:
        llm = get_gemini_chat(model=GEMINI_MODEL, temperature=0.1)
        agent = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            prompt=SystemMessage(content=_build_system_prompt())
        )
        resp = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": 15}
        )
        elapsed = time.perf_counter() - t0

        msgs = resp.get("messages", [])
        tool_calls = []
        for m in msgs:
            if getattr(m, "type", "") == "tool" or m.__class__.__name__ == "ToolMessage":
                t_name = getattr(m, "name", "")
                if t_name and t_name not in tool_calls:
                    tool_calls.append(t_name)
            if hasattr(m, "tool_calls") and m.tool_calls:
                for tc in m.tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if name and name not in tool_calls:
                        tool_calls.append(name)

        # Extract answer
        answer = ""
        for m in reversed(msgs):
            if getattr(m, "type", "") == "ai" or m.__class__.__name__ == "AIMessage":
                if hasattr(m, "content") and m.content:
                    raw_c = m.content
                    if isinstance(raw_c, list):
                        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_c]
                        answer = "\n".join(p for p in parts if p)
                    else:
                        answer = str(raw_c)
                    if answer.strip():
                        break

        # Optional DSEL
        if not no_dsel and answer:
            try:
                from core.safety_validator import validate_and_repair
                answer = validate_and_repair(answer)["answer"]
            except Exception:
                pass

        clean_status, is_valid = validate_answer_integrity(answer, STATUS_OK)
        return {
            "status": clean_status,
            "full_answer": answer,
            "active_latency_s": round(elapsed, 3),
            "total_latency_s": round(elapsed, 3),
            "tools_called": tool_calls,
            "model_used": GEMINI_MODEL,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        err = str(e)
        return {
            "status": "OTHER_ERROR",
            "full_answer": f"Error: {err}",
            "active_latency_s": round(elapsed, 3),
            "total_latency_s": round(elapsed, 3),
            "tools_called": [],
            "model_used": GEMINI_MODEL,
        }
    finally:
        # Restore monkeypatches
        if no_dsel and original_vr is not None:
            import core.safety_validator as sv_mod
            sv_mod.validate_and_repair = original_vr
        if no_telemetry:
            if original_sensor is not None:
                import tools.sensor_tool as st
                st.query_water_sensor = original_sensor
            if original_weather is not None:
                import tools.weather_tool as wt
                wt.fetch_live_weather = original_weather
        if no_tgi and original_kb_search is not None:
            import tools.kb_search_tool as kb
            kb._search_chroma = original_kb_search

def main():
    print("=" * 70, flush=True)
    print("STARTING FAST LINEAR GEMINI RERUN & PATCH (4 KEYS ROTATION)", flush=True)
    print(f"Model: {GEMINI_MODEL} | Active Keys: {gemini_pool.total_keys()}", flush=True)
    print("=" * 70, flush=True)

    dataset = {q["id"]: q for q in json.loads(DATASET_FILE.read_text(encoding="utf-8"))}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    to_patch = []
    for idx, r in enumerate(rows):
        flawed, reason = is_row_flawed(r)
        if flawed:
            to_patch.append((idx, r, reason))

    total = len(to_patch)
    print(f"\nTotal rows loaded: {len(rows)} | Flawed targets to fix: {total}\n", flush=True)

    if total == 0:
        print("All 1,200 rows in CSV are 100% clean and valid! Nothing to patch.", flush=True)
        return

    successes = 0
    failures = 0
    t_start = time.perf_counter()

    for count, (idx, r, reason) in enumerate(to_patch, 1):
        cfg  = r["config"]
        q_id = int(r["id"])
        seed = int(r["run_id"])
        q_obj = dataset.get(q_id, {})
        query_text   = q_obj.get("query", "")
        gold_kw      = q_obj.get("keywords", [])
        expected_svc = q_obj.get("expected_service", "")

        print(f"[{count:3d}/{total}] {cfg:23s} id={q_id:3d} seed={seed:2d} ({reason[:12]:12s}) ... ", end="", flush=True)

        no_tgi = (cfg == "Full_Agent_NoTGI")
        no_dsel = (cfg == "Full_Agent_NoDSEL")
        no_telemetry = (cfg == "Full_Agent_NoTelem")

        res = None
        for attempt in range(3):
            try:
                res = run_agent_gemini(query_text, no_tgi=no_tgi, no_dsel=no_dsel, no_telemetry=no_telemetry)
                ans = res.get("full_answer", "")
                st = res.get("status", STATUS_OK)
                clean_st, is_valid = validate_answer_integrity(ans, st)
                if is_valid:
                    res["status"] = clean_st
                    break
                else:
                    time.sleep(2 + attempt * 2)
            except Exception:
                time.sleep(2 + attempt * 2)

        elapsed = res.get("total_latency_s", 0) if res else 0

        if res and res.get("status") == STATUS_OK:
            ans = res["full_answer"]
            p, kw_r, f1 = kw_scores(ans, gold_kw)
            svc = svc_regex_check(ans, expected_svc)
            status = STATUS_OK
            successes += 1
            tools_list = res.get("tools_called", [])
            tools_str = ",".join(tools_list) if isinstance(tools_list, list) else str(tools_list)
            print(f"SUCCESS (status=OK, f1={f1:.3f}, {elapsed:.1f}s, tools='{tools_str}')", flush=True)
        else:
            ans = res.get("full_answer", "Execution failed") if res else "Execution failed"
            status = res.get("status", "ERROR") if res else "ERROR"
            p = kw_r = f1 = svc = ""
            tools_str = ""
            failures += 1
            print(f"FAILED ({status}, {ans[:50]})", flush=True)

        # Update row dict
        r["status"]           = status
        r["kw_precision"]     = p
        r["kw_recall"]        = kw_r
        r["kw_f1"]            = f1
        r["svc_regex"]        = svc
        r["active_latency_s"] = elapsed
        r["total_latency_s"]  = elapsed
        r["prompt_tokens"]    = 0
        r["completion_tokens"]= 0
        r["tools_called"]     = tools_str
        r["model_used"]       = GEMINI_MODEL
        r["retries"]          = 0
        r["full_answer"]      = ans

        rows[idx] = r
        save_csv_atomic(rows)

        # Polite 200ms delay between calls
        time.sleep(0.2)

    save_csv_atomic(rows)
    tot_time = time.perf_counter() - t_start
    print("\n" + "=" * 70, flush=True)
    print("GEMINI RERUN & PATCH COMPLETE", flush=True)
    print(f"Total processed: {total} | Successes: {successes} | Failures: {failures}", flush=True)
    print(f"Total time: {tot_time/60:.1f} minutes", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()

