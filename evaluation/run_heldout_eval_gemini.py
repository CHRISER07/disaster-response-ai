"""
run_heldout_eval_gemini.py

Pure Gemini 3.5 Flash Lite Held-Out Evaluation Runner across 4 rotated Gemini keys.
- Linear execution with per-row atomic flush to results_heldout_v2.csv.
- Zero Groq rate-limiting or 429 stalls.
- All configurations evaluated: LLM_Only, BM25_Baseline, Standard_StaticRAG, Full_Agent, Full_Agent_NoTGI, Full_Agent_NoDSEL.
- Strict validate_answer_integrity check.
"""
import os, sys, json, csv, time, random, re
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

os.environ["PYTHONUTF8"] = "1"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from evaluation.harness_v2 import trim_text, STATUS_OK, validate_answer_integrity
from evaluation.run_dev_eval_v2 import (
    kw_scores, svc_regex_check, FIELDNAMES, load_done_keys
)
from core.key_manager import gemini_pool, get_gemini_chat
from tools import ALL_TOOLS
from langgraph.prebuilt import create_react_agent
from agents.disaster_agent import _build_system_prompt
from langchain_core.messages import SystemMessage, HumanMessage

OUT_CSV          = ROOT / "evaluation" / "results_heldout_v2.csv"
HELDOUT_IDS_FILE = ROOT / "evaluation" / "heldout_ids.json"
DATASET_FILE     = ROOT / "evaluation" / "golden_dataset_v2.json"

GEMINI_MODEL = "gemini-3.5-flash-lite"
SEEDS = [42, 7, 99]

# --- Config functions with pure Gemini ---

def run_llm_only_gemini(query: str, seed: int) -> dict:
    t0 = time.perf_counter()
    llm = get_gemini_chat(model=GEMINI_MODEL, temperature=0.1)
    prompt = f"You are a disaster-response assistant.\nQuestion: {query}\nProvide an immediate operational answer:"
    res = llm.invoke(prompt)
    elapsed = time.perf_counter() - t0
    raw_c = res.content
    ans = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_c) if isinstance(raw_c, list) else str(raw_c)
    clean_st, is_v = validate_answer_integrity(ans, STATUS_OK)
    return {
        "status": clean_st, "full_answer": ans,
        "active_latency_s": round(elapsed, 3), "total_latency_s": round(elapsed, 3),
        "prompt_tokens": 0, "completion_tokens": 0,
        "tools_called": "", "model_used": GEMINI_MODEL, "retries": 0
    }

def run_bm25_gemini(query: str, seed: int) -> dict:
    t0 = time.perf_counter()
    from rag.retriever import retrieve_bm25_from_chroma
    docs = retrieve_bm25_from_chroma(query, k=5)
    context = "\n\n".join(f"[{d.metadata.get('source','KB')}]: {d.page_content}" for d in docs)
    llm = get_gemini_chat(model=GEMINI_MODEL, temperature=0.1)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer based ONLY on context:"
    res = llm.invoke(prompt)
    elapsed = time.perf_counter() - t0
    raw_c = res.content
    ans = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_c) if isinstance(raw_c, list) else str(raw_c)
    clean_st, is_v = validate_answer_integrity(ans, STATUS_OK)
    return {
        "status": clean_st, "full_answer": ans,
        "active_latency_s": round(elapsed, 3), "total_latency_s": round(elapsed, 3),
        "prompt_tokens": 0, "completion_tokens": 0,
        "tools_called": "bm25_retrieve", "model_used": GEMINI_MODEL, "retries": 0
    }

def run_static_rag_gemini(query: str, seed: int) -> dict:
    t0 = time.perf_counter()
    from rag.retriever import get_retriever
    retriever = get_retriever(k=5, search_type="mmr")
    docs = retriever.invoke(query)
    context = "\n\n".join(f"[{d.metadata.get('source','KB')}]: {d.page_content}" for d in docs)
    llm = get_gemini_chat(model=GEMINI_MODEL, temperature=0.1)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer strictly based on evidence:"
    res = llm.invoke(prompt)
    elapsed = time.perf_counter() - t0
    raw_c = res.content
    ans = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_c) if isinstance(raw_c, list) else str(raw_c)
    clean_st, is_v = validate_answer_integrity(ans, STATUS_OK)
    return {
        "status": clean_st, "full_answer": ans,
        "active_latency_s": round(elapsed, 3), "total_latency_s": round(elapsed, 3),
        "prompt_tokens": 0, "completion_tokens": 0,
        "tools_called": "chromadb_mmr_retrieve", "model_used": GEMINI_MODEL, "retries": 0
    }

def run_agent_gemini_task(query: str, seed: int, no_tgi: bool = False, no_dsel: bool = False) -> dict:
    from evaluation.rerun_and_patch_gemini import run_agent_gemini
    res = run_agent_gemini(query, no_tgi=no_tgi, no_dsel=no_dsel, no_telemetry=False)
    tools_l = res.get("tools_called", [])
    tools_str = ",".join(tools_l) if isinstance(tools_l, list) else str(tools_l or "")
    return {
        "status": res.get("status", STATUS_OK),
        "full_answer": res.get("full_answer", ""),
        "active_latency_s": res.get("active_latency_s", 0),
        "total_latency_s": res.get("total_latency_s", 0),
        "prompt_tokens": 0, "completion_tokens": 0,
        "tools_called": tools_str,
        "model_used": GEMINI_MODEL,
        "retries": 0
    }

CONFIGS = [
    {"name": "LLM_Only",           "fn": lambda q,s: run_llm_only_gemini(q,s),          "runs": SEEDS},
    {"name": "BM25_Baseline",      "fn": lambda q,s: run_bm25_gemini(q,s),               "runs": SEEDS},
    {"name": "Standard_StaticRAG", "fn": lambda q,s: run_static_rag_gemini(q,s),        "runs": SEEDS},
    {"name": "Full_Agent",         "fn": lambda q,s: run_agent_gemini_task(q,s),         "runs": SEEDS},
    {"name": "Full_Agent_NoTGI",   "fn": lambda q,s: run_agent_gemini_task(q,s,no_tgi=True), "runs": [42]},
    {"name": "Full_Agent_NoDSEL",  "fn": lambda q,s: run_agent_gemini_task(q,s,no_dsel=True),"runs": [42]},
]

def main():
    print("=" * 70, flush=True)
    print("STARTING PURE GEMINI HELDOUT EVALUATION (4 ROTATED KEYS)", flush=True)
    print(f"Model: {GEMINI_MODEL} | Active Keys: {gemini_pool.total_keys()}", flush=True)
    print("=" * 70, flush=True)

    heldout_ids = set(json.loads(HELDOUT_IDS_FILE.read_text(encoding="utf-8")))
    dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    heldout_queries = [q for q in dataset if q["id"] in heldout_ids]
    print(f"Heldout queries: {len(heldout_queries)}", flush=True)

    done_keys = load_done_keys(OUT_CSV)
    print(f"Checkpoint: {len(done_keys)} rows already done in CSV.", flush=True)

    tasks_to_run = []
    for cfg in CONFIGS:
        for seed in cfg["runs"]:
            for q in heldout_queries:
                key = (cfg["name"], str(seed), str(q["id"]))
                if key not in done_keys:
                    tasks_to_run.append({
                        "cfg_name": cfg["name"],
                        "fn":       cfg["fn"],
                        "seed":     seed,
                        "query":    q,
                    })

    total = len(tasks_to_run)
    print(f"Heldout tasks to run: {total}\n", flush=True)

    if total == 0:
        print("All heldout tasks already complete!", flush=True)
        return

    write_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
    out_f  = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        out_f.flush()

    t_start = time.perf_counter()
    successes = 0
    errors = 0

    for count, t in enumerate(tasks_to_run, 1):
        cfg_name = t["cfg_name"]
        seed     = t["seed"]
        q        = t["query"]
        fn       = t["fn"]

        print(f"[{count:4d}/{total}] {cfg_name:23s} id={q['id']:3d} s={seed:2d} ... ", end="", flush=True)

        res = None
        for attempt in range(3):
            try:
                res = fn(q["query"], seed)
                ans = res.get("full_answer", "")
                st = res.get("status", STATUS_OK)
                clean_st, is_v = validate_answer_integrity(ans, st)
                if is_v:
                    res["status"] = clean_st
                    break
                time.sleep(1 + attempt)
            except Exception:
                time.sleep(1 + attempt)

        elapsed = res.get("total_latency_s", 0) if res else 0

        gold_kw      = q.get("keywords", [])
        expected_svc = q.get("expected_service", "")
        if res and res["status"] == STATUS_OK:
            p, r, f1 = kw_scores(res["full_answer"], gold_kw)
            svc = svc_regex_check(res["full_answer"], expected_svc)
            successes += 1
            print(f"SUCCESS (f1={f1:.3f}, {elapsed:.1f}s, tools='{res.get('tools_called','')}')", flush=True)
        else:
            p = r = f1 = svc = ""
            errors += 1
            ans_prev = (res.get("full_answer","") if res else "Failed")[:40]
            print(f"FAILED ({res.get('status','ERROR') if res else 'ERROR'}, {ans_prev})", flush=True)

        row = {
            "config":           cfg_name,
            "run_id":           seed,
            "id":               q["id"],
            "scenario":         q.get("scenario",""),
            "category":         q.get("category",""),
            "status":           res["status"] if res else "ERROR",
            "kw_precision":     p,
            "kw_recall":        r,
            "kw_f1":            f1,
            "svc_regex":        svc,
            "p_at_k": "", "ndcg_at_7": "", "rr": "",
            "active_latency_s": res["active_latency_s"] if res else elapsed,
            "total_latency_s":  res["total_latency_s"] if res else elapsed,
            "prompt_tokens":    0,
            "completion_tokens":0,
            "tools_called":     res["tools_called"] if res else "",
            "model_used":       GEMINI_MODEL,
            "retries":          0,
            "full_answer":      res["full_answer"] if res else "",
        }

        writer.writerow(row)
        out_f.flush()
        time.sleep(0.15)

    out_f.close()
    tot_time = time.perf_counter() - t_start
    print("\n" + "=" * 70, flush=True)
    print("PURE GEMINI HELDOUT EVALUATION COMPLETE", flush=True)
    print(f"Total processed: {total} | Successes: {successes} | Errors: {errors}", flush=True)
    print(f"Total time: {tot_time/60:.1f} minutes", flush=True)
    print(f"Saved: {OUT_CSV}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()

