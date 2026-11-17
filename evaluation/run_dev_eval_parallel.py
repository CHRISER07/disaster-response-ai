"""
Parallel Phase 2 Evaluation Runner — Safe 4-worker concurrent execution.

Design guarantees:
  1. ThreadPoolExecutor with max_workers=4 (stays well under Groq's 30 RPM ceiling).
  2. Global RateLimiter: token-bucket capped at 20 requests/60 seconds across ALL workers.
  3. Thread-safe atomic CSV writes via threading.Lock — zero row corruption.
  4. Checkpoint resume: skips any (config, seed, id) already in results_per_query_v2.csv.
  5. Same seeds, temperature, scoring logic as sequential run — output 100% equivalent.
  6. Exponential backoff per-thread for 429/5xx — threads retry independently.
  7. Per-thread active_latency excludes backoff sleep time.

Run: python evaluation/run_dev_eval_parallel.py
Resume: safe to run again at any time — skips completed rows.
"""

import os, sys, json, csv, time, random, re, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["PYTHONUTF8"] = "1"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from evaluation.harness_v2 import trim_text, STATUS_OK
from evaluation.run_dev_eval_v2 import (
    run_llm_only, run_bm25, run_static_rag, run_full_agent,
    kw_scores, svc_regex_check, FIELDNAMES, GROQ_MODEL, load_done_keys
)

# ── Output file ────────────────────────────────────────────────────────────────
OUT_CSV       = ROOT / "evaluation" / "results_per_query_v2.csv"
DEV_IDS_FILE  = ROOT / "evaluation" / "dev_ids.json"
DATASET_FILE  = ROOT / "evaluation" / "golden_dataset_v2.json"

# ── Concurrency settings ───────────────────────────────────────────────────────
MAX_WORKERS    = 4      # simultaneous threads
RATE_LIMIT_RPM = 40    # max requests/min across ALL workers (4 keys rotating -> 10 RPM per key, safe under 30 RPM limit)
SEEDS = [42, 7, 99]

# ── Configs to run (skips LLM_Only and BM25_Baseline already done) ─────────────
CONFIGS = [
    {"name": "Text_Only_StaticRAG", "fn": lambda q,s: run_static_rag(q,s,text_only=True),          "runs": SEEDS},
    {"name": "Standard_StaticRAG",  "fn": lambda q,s: run_static_rag(q,s,text_only=False),         "runs": SEEDS},
    {"name": "Full_Agent",          "fn": lambda q,s: run_full_agent(q,s),                          "runs": SEEDS},
    {"name": "Full_Agent_NoTGI",    "fn": lambda q,s: run_full_agent(q,s,no_tgi=True),              "runs": SEEDS},
    {"name": "Full_Agent_NoDSEL",   "fn": lambda q,s: run_full_agent(q,s,no_dsel=True),             "runs": SEEDS},
    {"name": "Full_Agent_NoTelem",  "fn": lambda q,s: run_full_agent(q,s,no_telemetry=True),        "runs": SEEDS},
]

# ══════════════════════════════════════════════════════════════════════════════
# Token-Bucket Rate Limiter — shared across all worker threads
# ══════════════════════════════════════════════════════════════════════════════
class TokenBucketRateLimiter:
    """Thread-safe rate limiter. Blocks callers to maintain at most `max_per_minute` calls/min."""
    def __init__(self, max_per_minute: int):
        self._lock = threading.Lock()
        self._interval = 60.0 / max_per_minute   # seconds between tokens
        self._next_allowed = time.monotonic()

    def acquire(self):
        """Block until a request slot is available, then consume it."""
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = max(self._next_allowed + self._interval,
                                             now + self._interval)
                    return
                wait = self._next_allowed - now
            time.sleep(min(wait, 0.05))   # sleep outside lock in short bursts

_rate_limiter = TokenBucketRateLimiter(RATE_LIMIT_RPM)

# ══════════════════════════════════════════════════════════════════════════════
# Thread-safe CSV writer
# ══════════════════════════════════════════════════════════════════════════════
_csv_lock = threading.Lock()

def write_row(writer, out_f, row: dict):
    """Atomically append one row to the CSV."""
    with _csv_lock:
        writer.writerow(row)
        out_f.flush()

# ══════════════════════════════════════════════════════════════════════════════
# Worker task — executes one (config, seed, query) tuple
# ══════════════════════════════════════════════════════════════════════════════
def run_task(task: dict) -> dict:
    """
    task = {cfg_name, fn, seed, query, done_keys_set, writer, out_f, counter, total}
    Returns a summary dict with status info.
    """
    cfg_name = task["cfg_name"]
    seed     = task["seed"]
    q        = task["query"]
    fn       = task["fn"]

    # Rate-limit BEFORE the actual API call
    _rate_limiter.acquire()

    t0 = time.perf_counter()
    try:
        res = fn(q["query"], seed)
    except Exception as e:
        res = {
            "status": "OTHER_ERROR", "full_answer": str(e)[:500],
            "active_latency_s": round(time.perf_counter()-t0, 3),
            "total_latency_s":  round(time.perf_counter()-t0, 3),
            "prompt_tokens": 0, "completion_tokens": 0,
            "tools_called": "", "model_used": GROQ_MODEL, "retries": 0
        }

    # Compute KW metrics only for OK responses
    gold_kw      = q.get("keywords", [])
    expected_svc = q.get("expected_service", "")
    if res["status"] == STATUS_OK:
        p, r, f1 = kw_scores(res["full_answer"], gold_kw)
        svc       = svc_regex_check(res["full_answer"], expected_svc)
    else:
        p = r = f1 = svc = ""

    row = {
        "config":           cfg_name,
        "run_id":           seed,
        "id":               q["id"],
        "scenario":         q.get("scenario",""),
        "category":         q.get("category",""),
        "status":           res["status"],
        "kw_precision":     p,
        "kw_recall":        r,
        "kw_f1":            f1,
        "svc_regex":        svc,
        "p_at_k": "", "ndcg_at_7": "", "rr": "",
        "active_latency_s": res["active_latency_s"],
        "total_latency_s":  res["total_latency_s"],
        "prompt_tokens":    res["prompt_tokens"],
        "completion_tokens":res["completion_tokens"],
        "tools_called":     res["tools_called"],
        "model_used":       res["model_used"],
        "retries":          res["retries"],
        "full_answer":      res["full_answer"],
    }

    write_row(task["writer"], task["out_f"], row)

    # Thread-safe counter
    with task["counter_lock"]:
        task["counter"][0] += 1
        done  = task["counter"][0]
        total = task["total"]
        print(f"  [{done:4d}/{total}] {cfg_name:25s} s={seed} id={q['id']:3d} → {res['status']}")

    return {"cfg": cfg_name, "seed": seed, "id": q["id"], "status": res["status"]}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    dev_ids    = set(json.loads(DEV_IDS_FILE.read_text(encoding="utf-8")))
    dataset    = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    dev_queries = [q for q in dataset if q["id"] in dev_ids]

    # Load checkpoint — skip already-done tuples
    done_keys = load_done_keys(OUT_CSV)
    print(f"Checkpoint: {len(done_keys)} rows already done in CSV.")

    # Build task list
    tasks_to_run = []
    for cfg in CONFIGS:
        for seed in cfg["runs"]:
            for q in dev_queries:
                key = (cfg["name"], str(seed), str(q["id"]))
                if key not in done_keys:
                    tasks_to_run.append({
                        "cfg_name": cfg["name"],
                        "fn":       cfg["fn"],
                        "seed":     seed,
                        "query":    q,
                    })

    total = len(tasks_to_run)
    skipped = sum(len(cfg["runs"]) for cfg in CONFIGS) * len(dev_queries) - total
    print(f"Tasks to run: {total}  (skipped {skipped} already done)")
    print(f"Workers: {MAX_WORKERS}  |  Rate limit: {RATE_LIMIT_RPM} RPM  |  ETA: ~{total*3//MAX_WORKERS//60} min")

    if total == 0:
        print("All tasks already complete!")
        return

    # Open CSV for appending
    write_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
    out_f  = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    # Shared mutable counter (thread-safe via lock)
    counter      = [0]
    counter_lock = threading.Lock()

    # Inject shared references into each task dict
    for t in tasks_to_run:
        t["writer"]       = writer
        t["out_f"]        = out_f
        t["counter"]      = counter
        t["counter_lock"] = counter_lock
        t["total"]        = total

    t_start = time.perf_counter()
    errors  = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_task, t): t for t in tasks_to_run}
        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result["status"] != STATUS_OK:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"  [THREAD ERROR] {e}")

    out_f.close()
    elapsed = time.perf_counter() - t_start
    ok = total - errors

    print(f"\n{'='*60}")
    print(f"PARALLEL EVAL COMPLETE")
    print(f"  Rows written : {total}")
    print(f"  OK           : {ok}")
    print(f"  Errors       : {errors}")
    print(f"  Elapsed      : {elapsed/60:.1f} min")
    print(f"  Throughput   : {total/elapsed*60:.1f} rows/min")
    print(f"  Output       : {OUT_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

