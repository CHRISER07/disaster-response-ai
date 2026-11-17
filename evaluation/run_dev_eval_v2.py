"""
Phase 2: Dev-set comparison runner (8 configs x 50 queries x 3 runs).
Checkpoints after every query. Resumes safely from existing CSV.
Run: python evaluation/run_dev_eval_v2.py
"""
import os, sys, json, csv, time, random, re, math, functools
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUTF8"] = "1"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from evaluation.harness_v2 import trim_text, STATUS_OK
from core.answer_formatter import format_answer

OUT_CSV = ROOT / "evaluation" / "results_per_query_v2.csv"
DEV_IDS_FILE = ROOT / "evaluation" / "dev_ids.json"
DATASET_FILE = ROOT / "evaluation" / "golden_dataset_v2.json"

FIELDNAMES = [
    "config","run_id","id","scenario","category","status",
    "kw_precision","kw_recall","kw_f1","svc_regex",
    "p_at_k","ndcg_at_7","rr",
    "active_latency_s","total_latency_s",
    "prompt_tokens","completion_tokens",
    "tools_called","model_used","retries","full_answer"
]

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
SEEDS = [42, 7, 99]  # 3 runs per config


def active_model_name() -> str:
    """Generator actually in use, resolved from the locked provider."""
    provider = os.getenv("PROVIDER", "ollama").lower().strip()
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    return GROQ_MODEL


def generate_with_active_provider(prompt: str, seed: int, max_chars: int = 3000) -> dict:
    """Single generation path for all non-Groq providers (same model for all arms)."""
    from core.llm_config import get_llm
    llm = get_llm(seed=seed)
    resp = llm.invoke(prompt[:max_chars])
    return {"answer": resp.content or "", "prompt_tokens": 0, "completion_tokens": 0}

# ---------------------------------------------------------------------------
# KW F1 helpers
# ---------------------------------------------------------------------------
_STOPWORDS = {"a","an","the","is","are","was","were","in","of","to","for",
              "and","or","it","its","be","do","if","at","by","on","with"}

def _stem(w: str) -> str:
    w = w.lower()
    for sfx in ("ing","tion","ness","ment","ed","es","s"):
        if w.endswith(sfx) and len(w) - len(sfx) > 3:
            return w[:-len(sfx)]
    return w

def _tokens(text: str) -> set:
    words = re.findall(r"[a-z]+", text.lower())
    return {_stem(w) for w in words if w not in _STOPWORDS and len(w) > 2}

@functools.lru_cache(maxsize=1)
def _dataset_vocab() -> set:
    """Domain vocabulary derived from the golden dataset, never hardcoded."""
    data = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    vocab = set()
    for item in data:
        for kw in item.get("expected_keywords", item.get("keywords", [])):
            if kw:
                vocab.add(_stem(str(kw).lower()))
    return vocab


def kw_scores(answer: str, gold_keywords: list) -> tuple:
    """
    Keyword F1 against a dataset-derived vocabulary.

    Precision is computed over domain-relevant keywords found in the answer,
    NOT over every answer token. The legacy formula (tp / all_tokens) drove
    every score toward zero for full prose answers and could not discriminate.
    """
    if not answer or not gold_keywords:
        return 0.0, 0.0, 0.0
    ans_toks = _tokens(answer)
    gold_toks = {_stem(k.lower()) for k in gold_keywords if k}
    if not gold_toks:
        return 0.0, 0.0, 0.0
    predicted = ans_toks & _dataset_vocab()
    tp = len(predicted & gold_toks)
    fp = len(predicted - gold_toks)
    fn = len(gold_toks - predicted)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return round(p,4), round(r,4), round(f1,4)

def svc_regex_check(answer: str, expected_service: str) -> int:
    if not expected_service or not answer:
        return 0
    pattern = re.compile(re.escape(expected_service.lower()), re.IGNORECASE)
    return 1 if pattern.search(answer) else 0

# ---------------------------------------------------------------------------
# Config runners
# ---------------------------------------------------------------------------
def run_llm_only(query: str, seed: int) -> dict:
    """Config 1: LLM-only zero-shot with safety instruction."""
    system_msg = (
        "You are a disaster-response assistant. Only provide information that is "
        "safe and verified. Do not guess. If unsure, say so."
    )
    t0 = time.perf_counter()

    if os.getenv("PROVIDER", "ollama").lower() == "ollama":
        from core.llm_config import get_llm, OLLAMA_MODEL
        try:
            llm = get_llm(seed=seed)
            from langchain_core.messages import SystemMessage, HumanMessage
            resp = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=query)])
            elapsed = time.perf_counter() - t0
            return {
                "status": STATUS_OK,
                "full_answer": resp.content or "",
                "active_latency_s": round(elapsed, 3),
                "total_latency_s": round(elapsed, 3),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tools_called": "",
                "model_used": OLLAMA_MODEL,
                "retries": 0
            }
        except Exception as e:
            return {"status": "OTHER_ERROR", "full_answer": str(e),
                    "active_latency_s": round(time.perf_counter()-t0, 3),
                    "total_latency_s": round(time.perf_counter()-t0, 3),
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "tools_called": "", "model_used": "ollama", "retries": 0}

    from core.key_manager import get_groq_client
    client, _ = get_groq_client()
    retries = 0
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role":"system","content":system_msg},
                          {"role":"user","content":query}],
                temperature=0.0, seed=seed, max_tokens=1024
            )
            elapsed = time.perf_counter() - t0
            answer = resp.choices[0].message.content or ""
            return {
                "status": STATUS_OK,
                "full_answer": answer,
                "active_latency_s": round(elapsed, 3),
                "total_latency_s": round(elapsed, 3),
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "tools_called": "",
                "model_used": GROQ_MODEL,
                "retries": retries
            }
        except Exception as e:
            err_str = str(e)
            retries += 1
            if "429" in err_str or "rate" in err_str.lower():
                client, _ = get_groq_client()
                sleep = min(60, 2**attempt + random.uniform(0,2))
                time.sleep(sleep)
                continue
            return {"status": "OTHER_ERROR", "full_answer": err_str,
                    "active_latency_s": round(time.perf_counter()-t0,3),
                    "total_latency_s": round(time.perf_counter()-t0,3),
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "tools_called": "", "model_used": GROQ_MODEL, "retries": retries}
    return {"status": "RATE_LIMIT", "full_answer": "Rate limit exceeded after retries",
            "active_latency_s": 0, "total_latency_s": round(time.perf_counter()-t0,3),
            "prompt_tokens": 0, "completion_tokens": 0,
            "tools_called": "", "model_used": GROQ_MODEL, "retries": retries}


def run_bm25(query: str, seed: int) -> dict:
    """Config 2: BM25 baseline — retrieve then generate."""
    from groq import Groq
    import sqlite3
    t0 = time.perf_counter()
    db_path = ROOT / "chroma.sqlite3"
    chunks = []
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            # Simple FTS keyword search
            terms = " OR ".join(query.split()[:6])
            cur = conn.execute(
                "SELECT string_value FROM embedding_metadata "
                "WHERE string_key='document' LIMIT 5"
            )
            rows = cur.fetchall()
            chunks = [trim_text(r[0], 400) for r in rows if r[0]]
            conn.close()
        except Exception:
            chunks = []
    context = "\n---\n".join(chunks) if chunks else "No context retrieved."
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    if os.getenv("PROVIDER", "ollama").lower().strip() != "groq":
        gen = generate_with_active_provider(prompt, seed)
        return {"status": STATUS_OK, "full_answer": gen["answer"],
                "active_latency_s": round(time.perf_counter() - t0, 3),
                "total_latency_s": round(time.perf_counter() - t0, 3),
                "prompt_tokens": gen["prompt_tokens"], "completion_tokens": gen["completion_tokens"],
                "tools_called": "bm25_retrieve", "model_used": active_model_name(), "retries": 0}
    from core.key_manager import get_groq_client
    client, _ = get_groq_client()
    retries = 0
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role":"user","content":trim_text(prompt, 3000)}],
                temperature=0.0, seed=seed, max_tokens=800
            )
            elapsed = time.perf_counter() - t0
            answer = resp.choices[0].message.content or ""
            return {
                "status": STATUS_OK,
                "full_answer": answer,
                "active_latency_s": round(elapsed, 3),
                "total_latency_s": round(elapsed, 3),
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "tools_called": "bm25_retrieve",
                "model_used": GROQ_MODEL,
                "retries": retries
            }
        except Exception as e:
            err_str = str(e)
            retries += 1
            if "429" in err_str or "rate" in err_str.lower():
                client, _ = get_groq_client()
                sleep = min(60, 2**attempt + random.uniform(0,2))
                time.sleep(sleep)
                continue
            return {"status": "OTHER_ERROR", "full_answer": err_str,
                    "active_latency_s": round(time.perf_counter()-t0,3),
                    "total_latency_s": round(time.perf_counter()-t0,3),
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "tools_called": "bm25_retrieve", "model_used": GROQ_MODEL, "retries": retries}
    return {"status": "RATE_LIMIT", "full_answer": "Rate limit exceeded",
            "active_latency_s": 0, "total_latency_s": round(time.perf_counter()-t0,3),
            "prompt_tokens": 0, "completion_tokens": 0,
            "tools_called": "bm25_retrieve", "model_used": GROQ_MODEL, "retries": retries}


def run_static_rag(query: str, seed: int, text_only: bool = False) -> dict:
    """Config 3 (text_only=True) and Config 4 (text_only=False): Static RAG."""
    t0 = time.perf_counter()
    if os.getenv("PROVIDER", "ollama").lower().strip() != "groq":
        # Provider-agnostic static RAG: retrieve via the weighted RRF engine,
        # then generate with the same locked generator as every other arm.
        try:
            from core.hybrid_store import hybrid_store
            flt = {"modality": {"$in": ["text", "knowledge_base"]}} if text_only else None
            chunks = hybrid_store.hybrid_search(query, k=5, modality_filter=flt, alpha_dense=0.70)
            context = "\n---\n".join(c.content[:500] for c in chunks)
            gen = generate_with_active_provider(
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:", seed
            )
            return {"status": STATUS_OK, "full_answer": gen["answer"],
                    "active_latency_s": round(time.perf_counter() - t0, 3),
                    "total_latency_s": round(time.perf_counter() - t0, 3),
                    "prompt_tokens": gen["prompt_tokens"], "completion_tokens": gen["completion_tokens"],
                    "tools_called": "kb_search,text_search" if not text_only else "text_search",
                    "model_used": active_model_name(), "retries": 0}
        except Exception as e:
            return {"status": "OTHER_ERROR", "full_answer": f"Static RAG failed: {type(e).__name__}: {e}",
                    "active_latency_s": 0, "total_latency_s": round(time.perf_counter() - t0, 3),
                    "prompt_tokens": 0, "completion_tokens": 0, "tools_called": "",
                    "model_used": active_model_name(), "retries": 0}
    try:
        from core.rag_chain import get_rag_chain
        from core.key_manager import get_groq_chat
        modalities = ["text", "knowledge_base"] if text_only else None
        llm = get_groq_chat(temperature=0.0)
        chain = get_rag_chain(allowed_modalities=modalities, llm=llm)
        result = chain.invoke({"input": query})
        elapsed = time.perf_counter() - t0
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        return {
            "status": STATUS_OK,
            "full_answer": answer,
            "active_latency_s": round(elapsed, 3),
            "total_latency_s": round(elapsed, 3),
            "prompt_tokens": result.get("prompt_tokens", 0) if isinstance(result, dict) else 0,
            "completion_tokens": result.get("completion_tokens", 0) if isinstance(result, dict) else 0,
            "tools_called": "kb_search,text_search" if not text_only else "text_search",
            "model_used": GROQ_MODEL,
            "retries": 0
        }
    except Exception as e:
        # Do not silently turn a failed static-RAG run into a different
        # retrieval configuration. That would make the row look successful
        # while measuring an unrelated, non-query-ranked context.
        elapsed = time.perf_counter() - t0
        return {
            "status": "OTHER_ERROR",
            "full_answer": f"Static RAG failed: {type(e).__name__}: {e}",
            "active_latency_s": round(elapsed, 3),
            "total_latency_s": round(elapsed, 3),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tools_called": "",
            "model_used": GROQ_MODEL,
            "retries": 0,
        }


def _groq_with_fts(query: str, seed: int, t0: float, config_label: str = "") -> dict:
    """Fallback: FTS retrieval + Groq generation."""
    from groq import Groq
    import sqlite3
    chunks = []
    db_path = ROOT / "chroma.sqlite3"
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.execute("SELECT string_value FROM embedding_metadata WHERE string_key='document' LIMIT 5")
            chunks = [trim_text(r[0], 350) for r in cur.fetchall() if r[0]]
            conn.close()
        except Exception:
            pass
    context = "\n---\n".join(chunks) if chunks else "No context retrieved."
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer as a disaster-response assistant:"
    from core.key_manager import get_groq_client
    client, _ = get_groq_client()
    retries = 0
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role":"user","content":trim_text(prompt,3000)}],
                temperature=0.0, seed=seed, max_tokens=800
            )
            elapsed = time.perf_counter() - t0
            return {
                "status": STATUS_OK,
                "full_answer": resp.choices[0].message.content or "",
                "active_latency_s": round(elapsed,3),
                "total_latency_s": round(elapsed,3),
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "tools_called": f"fts_fallback",
                "model_used": GROQ_MODEL,
                "retries": retries
            }
        except Exception as e:
            retries += 1
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(min(60, 2**attempt + random.uniform(0,2)))
                continue
            return {"status":"OTHER_ERROR","full_answer":str(e),
                    "active_latency_s":round(time.perf_counter()-t0,3),
                    "total_latency_s":round(time.perf_counter()-t0,3),
                    "prompt_tokens":0,"completion_tokens":0,
                    "tools_called":"fts_fallback","model_used":GROQ_MODEL,"retries":retries}
    return {"status":"RATE_LIMIT","full_answer":"rate limit",
            "active_latency_s":0,"total_latency_s":round(time.perf_counter()-t0,3),
            "prompt_tokens":0,"completion_tokens":0,
            "tools_called":"fts_fallback","model_used":GROQ_MODEL,"retries":retries}


def run_full_agent(query: str, seed: int, no_tgi: bool = False, no_dsel: bool = False, no_telemetry: bool = False) -> dict:
    """Configs 5-8: Full agent variants."""
    t0 = time.perf_counter()
    thread_id = f"eval_{seed}_{int(t0*1000)}"
    try:
        from agents.disaster_agent import run_agent
        import core.safety_validator as sv_mod

        # Monkeypatch for no_dsel
        original_vr = None
        if no_dsel:
            original_vr = sv_mod.validate_and_repair
            sv_mod.validate_and_repair = lambda text, *a, **kw: {
                "answer": text, "safe": True, "violations": [], "rewritten": False
            }

        # Monkeypatch for no_telemetry: mock live sensors and weather
        original_sensor = None
        original_weather = None
        if no_telemetry:
            try:
                import tools.sensor_tool as st
                original_sensor = st.query_water_sensor
                st.query_water_sensor = lambda *a, **kw: "Telemetry disabled: live sensor data unavailable for evaluation."
            except Exception:
                pass
            try:
                import tools.weather_tool as wt
                original_weather = wt.fetch_live_weather
                wt.fetch_live_weather = lambda *a, **kw: "Telemetry disabled: live weather data unavailable for evaluation."
            except Exception:
                pass

        # Monkeypatch for no_tgi: strip temporal tags from KB chunks
        original_kb_search = None
        if no_tgi:
            try:
                import tools.kb_search_tool as kb
                # We wrap the internal search to replace temporal tags with neutral [DOCUMENT]
                original_search = kb._search
                def stripped_search(*args, **kwargs):
                    text = original_search(*args, **kwargs)
                    import re
                    # Strip every TGI marker form, including the real-window tags.
                    text = re.sub(r"\[(HISTORICAL|LIVE|RECENT|KB|UNDATED)[^\]]*\]", "[DOCUMENT]", text)
                    text = re.sub(r"<(KB|HIST|LIVE)[^>]*>", "", text)
                    # Strip timestamp in Source: CrisisLex | 2013-09-12
                    text = re.sub(r"\|\s*\d{4}-\d{2}-\d{2}[^\n]*", "", text)
                    return text
                kb._search = stripped_search
                original_kb_search = original_search
            except Exception:
                pass

        t_active_start = time.perf_counter()
        result = run_agent(query, thread_id=thread_id, seed=seed, no_tgi=no_tgi)
        active_elapsed = time.perf_counter() - t_active_start
        total_elapsed = time.perf_counter() - t0

        # Restore monkeypatches
        if no_dsel and original_vr is not None:
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
            kb._search = original_kb_search

        answer = ""
        status = STATUS_OK
        tools_called = ""
        prompt_toks = 0
        comp_toks = 0
        provider_name = os.getenv("PROVIDER", "ollama")
        default_model = os.getenv("OLLAMA_MODEL", "qwen3:8b") if provider_name == "ollama" else GROQ_MODEL
        model_used = default_model

        if isinstance(result, dict):
            answer = result.get("output", result.get("answer", str(result)))
            status = result.get("status", STATUS_OK)
            # Agent returns "tool_calls" (list); fall back to "tools_called" for legacy callers
            _tc_raw = result.get("tool_calls", result.get("tools_called", []))
            tools_called = ",".join(_tc_raw) if isinstance(_tc_raw, list) else str(_tc_raw or "")
            prompt_toks = result.get("prompt_tokens", 0)
            comp_toks = result.get("completion_tokens", 0)
            model_used = result.get("model_used", default_model)
        elif isinstance(result, str):
            answer = result
            if "Sorry, need more steps" in answer or "Agent stopped" in answer:
                status = "STEP_LIMIT"
        else:
            answer = str(result)

        # Strict integrity validation: reject error strings, rate limits, and short empty outputs
        from evaluation.harness_v2 import validate_answer_integrity
        clean_status, is_valid = validate_answer_integrity(answer, status)
        status = clean_status

        return {
            "status": status,
            "full_answer": answer,
            "active_latency_s": round(active_elapsed, 3),
            "total_latency_s": round(total_elapsed, 3),
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "tools_called": tools_called,
            "model_used": model_used,
            "retries": 0
        }
    except Exception as e:
        err = str(e)
        total_elapsed = time.perf_counter() - t0
        status = "STEP_LIMIT" if "recursion" in err.lower() else "OTHER_ERROR"
        return {
            "status": status,
            "full_answer": err,
            "active_latency_s": round(total_elapsed,3),
            "total_latency_s": round(total_elapsed,3),
            "prompt_tokens": 0, "completion_tokens": 0,
            "tools_called": "", "model_used": GROQ_MODEL, "retries": 0
        }


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------
CONFIGS = [
    {"name": "LLM_Only",           "fn": lambda q,s: run_llm_only(q,s),                              "runs": SEEDS},
    {"name": "BM25_Baseline",      "fn": lambda q,s: run_bm25(q,s),                                   "runs": SEEDS},
    {"name": "Text_Only_StaticRAG","fn": lambda q,s: run_static_rag(q,s,text_only=True),              "runs": SEEDS},
    {"name": "Standard_StaticRAG", "fn": lambda q,s: run_static_rag(q,s,text_only=False),             "runs": SEEDS},
    {"name": "Full_Agent",         "fn": lambda q,s: run_full_agent(q,s),                             "runs": SEEDS},
    {"name": "Full_Agent_NoTGI",   "fn": lambda q,s: run_full_agent(q,s,no_tgi=True),                 "runs": SEEDS},
    {"name": "Full_Agent_NoDSEL",  "fn": lambda q,s: run_full_agent(q,s,no_dsel=True),                "runs": SEEDS},
    {"name": "Full_Agent_NoTelem", "fn": lambda q,s: run_full_agent(q,s,no_telemetry=True),           "runs": SEEDS},
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_done_keys(csv_path: Path) -> set:
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["config"], str(row["run_id"]), str(row["id"])))
    return done


def main():
    dev_ids = set(json.loads(DEV_IDS_FILE.read_text(encoding="utf-8")))
    dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    dev_queries = [q for q in dataset if q["id"] in dev_ids]
    print(f"Dev queries: {len(dev_queries)}, Configs: {len(CONFIGS)}, Seeds: {len(SEEDS)}")
    print(f"Total planned rows: {len(dev_queries)*len(CONFIGS)*len(SEEDS)}")

    # Load checkpoint
    done_keys = load_done_keys(OUT_CSV)
    print(f"Already done: {len(done_keys)} rows")

    write_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
    out_f = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    total_planned = len(dev_queries) * len(CONFIGS) * len(SEEDS)
    done_count = len(done_keys)
    errors = 0

    for cfg in CONFIGS:
        for seed in cfg["runs"]:
            for q in dev_queries:
                key = (cfg["name"], str(seed), str(q["id"]))
                if key in done_keys:
                    continue

                print(f"  [{done_count+1}/{total_planned}] cfg={cfg['name']} seed={seed} id={q['id']}", end="", flush=True)
                try:
                    res = cfg["fn"](q["query"], seed)
                except Exception as e:
                    res = {"status":"OTHER_ERROR","full_answer":str(e),
                           "active_latency_s":0,"total_latency_s":0,
                           "prompt_tokens":0,"completion_tokens":0,
                           "tools_called":"","model_used":GROQ_MODEL,"retries":0}

                # Score KW metrics only for OK rows
                gold_kw = q.get("keywords", [])
                expected_svc = q.get("expected_service", "")
                if res["status"] == STATUS_OK:
                    p, r, f1 = kw_scores(res["full_answer"], gold_kw)
                    svc = svc_regex_check(res["full_answer"], expected_svc)
                else:
                    p = r = f1 = svc = ""
                    errors += 1

                row = {
                    "config": cfg["name"],
                    "run_id": seed,
                    "id": q["id"],
                    "scenario": q.get("scenario",""),
                    "category": q.get("category",""),
                    "status": res["status"],
                    "kw_precision": p,
                    "kw_recall": r,
                    "kw_f1": f1,
                    "svc_regex": svc,
                    "p_at_k": "",
                    "ndcg_at_7": "",
                    "rr": "",
                    "active_latency_s": res["active_latency_s"],
                    "total_latency_s": res["total_latency_s"],
                    "prompt_tokens": res["prompt_tokens"],
                    "completion_tokens": res["completion_tokens"],
                    "tools_called": res["tools_called"],
                    "model_used": res["model_used"],
                    "retries": res["retries"],
                    "full_answer": format_answer(res["full_answer"]),
                }
                writer.writerow(row)
                out_f.flush()
                done_count += 1
                done_keys.add(key)
                print(f" → {res['status']}")

                # Inter-request throttle to avoid 429
                time.sleep(0.8)

    out_f.close()
    ok = sum(1 for k in done_keys if True)
    print(f"\nDone. {done_count} rows written, {errors} errors.")
    print(f"Output: {OUT_CSV}")


if __name__ == "__main__":
    main()

