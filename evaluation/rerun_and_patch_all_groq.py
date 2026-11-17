"""
rerun_and_patch_all_groq.py (v3 - Robust Multi-Key Failover)

Systematic, live-reporting rerun and patch for all remaining flawed rows in results_per_query_v2.csv.
- Rotates across all 7 Groq keys and 4 Gemini keys via LangGraph agent circuit breaker.
- NEVER evicts healthy keys on transient 429 rate limit pauses.
- Enforces strict validate_answer_integrity gate: zero error strings can ever pass.
- Saves progress atomically to results_per_query_v2.csv after EVERY row.
- Live unbuffered console output for real-time visibility.
"""
import os, sys, json, csv, time, random, re
from pathlib import Path

# Force unbuffered stdout so every single line is immediately visible live
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

os.environ["PYTHONUTF8"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from evaluation.harness_v2 import trim_text, STATUS_OK, validate_answer_integrity
from evaluation.run_dev_eval_v2 import (
    run_llm_only, run_bm25, run_static_rag, run_full_agent,
    kw_scores, svc_regex_check, FIELDNAMES, GROQ_MODEL
)
from core.key_manager import groq_pool, gemini_pool

CSV_PATH     = ROOT / "evaluation" / "results_per_query_v2.csv"
DATASET_FILE = ROOT / "evaluation" / "golden_dataset_v2.json"

CONFIG_DISPATCH = {
    "LLM_Only":            lambda q, s: run_llm_only(q, s),
    "BM25_Baseline":       lambda q, s: run_bm25(q, s),
    "Text_Only_StaticRAG": lambda q, s: run_static_rag(q, s, text_only=True),
    "Standard_StaticRAG":  lambda q, s: run_static_rag(q, s, text_only=False),
    "Full_Agent":          lambda q, s: run_full_agent(q, s),
    "Full_Agent_NoTGI":    lambda q, s: run_full_agent(q, s, no_tgi=True),
    "Full_Agent_NoDSEL":   lambda q, s: run_full_agent(q, s, no_dsel=True),
    "Full_Agent_NoTelem":  lambda q, s: run_full_agent(q, s, no_telemetry=True),
}

def is_row_flawed(r: dict) -> tuple:
    ans    = r.get("full_answer", "")
    status = r.get("status", "")
    tools  = r.get("tools_called", "")
    cfg    = r.get("config", "")

    # Universal integrity check
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

def main():
    print("=" * 70, flush=True)
    print("STARTING ROBUST MULTI-KEY RERUN & PATCH", flush=True)
    print(f"Active Groq Keys in Pool: {groq_pool.total_keys()} | Gemini Keys: {gemini_pool.total_keys()}", flush=True)
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

        fn = CONFIG_DISPATCH.get(cfg)
        if not fn:
            print(f"[{count:3d}/{total}] SKIP unknown config: {cfg}", flush=True)
            continue

        print(f"[{count:3d}/{total}] {cfg:23s} id={q_id:3d} seed={seed:2d} ({reason[:12]:12s}) ... ", end="", flush=True)

        res = None
        max_attempts = 4
        for attempt in range(max_attempts):
            t0 = time.perf_counter()
            try:
                res = fn(query_text, seed)
                ans = res.get("full_answer", "")
                raw_st = res.get("status", STATUS_OK)
                clean_st, is_valid = validate_answer_integrity(ans, raw_st)

                if is_valid:
                    res["status"] = clean_st
                    break
                else:
                    # Transient rate limit or error: sleep briefly with jitter and retry with next key
                    sleep_time = min(15, 2**attempt + random.uniform(1.0, 3.0))
                    time.sleep(sleep_time)
            except Exception as e:
                time.sleep(min(15, 2**attempt + random.uniform(1.0, 3.0)))

        elapsed = time.perf_counter() - t0

        if res and res.get("status") == STATUS_OK:
            ans = res["full_answer"]
            p, kw_r, f1 = kw_scores(ans, gold_kw)
            svc = svc_regex_check(ans, expected_svc)
            status = STATUS_OK
            successes += 1
            print(f"SUCCESS (status=OK, f1={f1:.3f}, {elapsed:.1f}s, tools='{res.get('tool_calls', res.get('tools_called',''))}')", flush=True)
        else:
            ans = res.get("full_answer", "Execution failed") if res else "Execution failed"
            status = res.get("status", "ERROR") if res else "ERROR"
            p = kw_r = f1 = svc = ""
            failures += 1
            print(f"FAILED ({status}, {ans[:50]})", flush=True)

        # Determine tools called cleanly
        raw_tools = res.get("tool_calls", res.get("tools_called", "")) if res else ""
        if isinstance(raw_tools, list):
            tools_str = ",".join(raw_tools)
        else:
            tools_str = str(raw_tools or "")

        # Update row dict
        r["status"]           = status
        r["kw_precision"]     = p
        r["kw_recall"]        = kw_r
        r["kw_f1"]            = f1
        r["svc_regex"]        = svc
        r["active_latency_s"] = res.get("active_latency_s", round(elapsed, 3)) if res else round(elapsed, 3)
        r["total_latency_s"]  = res.get("total_latency_s", round(elapsed, 3)) if res else round(elapsed, 3)
        r["prompt_tokens"]    = res.get("prompt_tokens", 0) if res else 0
        r["completion_tokens"]= res.get("completion_tokens", 0) if res else 0
        r["tools_called"]     = tools_str
        r["model_used"]       = res.get("model_used", GROQ_MODEL) if res else GROQ_MODEL
        r["retries"]          = res.get("retries", 0) if res else 0
        r["full_answer"]      = ans

        rows[idx] = r

        # Save progress after every single row so zero work is ever lost
        save_csv_atomic(rows)

        # Polite 200ms pause between queries to stay safely under RPM limits
        time.sleep(0.2)

    save_csv_atomic(rows)
    tot_time = time.perf_counter() - t_start
    print("\n" + "=" * 70, flush=True)
    print("RERUN & PATCH COMPLETE", flush=True)
    print(f"Total processed: {total} | Successes: {successes} | Failures: {failures}", flush=True)
    print(f"Total time: {tot_time/60:.1f} minutes", flush=True)
    print(f"Updated CSV: {CSV_PATH}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
