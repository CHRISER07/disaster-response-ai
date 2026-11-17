"""
Parallel Phase 3 Held-Out Evaluation Runner — 4-worker concurrent execution.
Runs held-out evaluation across 150 heldout queries.
"""
import os, sys, json, csv, time, random, re, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["PYTHONUTF8"] = "1"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from evaluation.harness_v2 import trim_text, STATUS_OK
from evaluation.run_dev_eval_v2 import (
    run_llm_only, run_bm25, run_static_rag, run_full_agent,
    kw_scores, svc_regex_check, FIELDNAMES, GROQ_MODEL, load_done_keys
)

OUT_CSV          = ROOT / "evaluation" / "results_heldout_v2.csv"
HELDOUT_IDS_FILE = ROOT / "evaluation" / "heldout_ids.json"
DATASET_FILE     = ROOT / "evaluation" / "golden_dataset_v2.json"

MAX_WORKERS    = 4
RATE_LIMIT_RPM = 40
SEEDS = [42, 7, 99]

CONFIGS = [
    {"name": "LLM_Only",           "fn": lambda q,s: run_llm_only(q,s),                   "runs": SEEDS},
    {"name": "BM25_Baseline",      "fn": lambda q,s: run_bm25(q,s),                        "runs": SEEDS},
    {"name": "Standard_StaticRAG", "fn": lambda q,s: run_static_rag(q,s,text_only=False),  "runs": SEEDS},
    {"name": "Full_Agent",         "fn": lambda q,s: run_full_agent(q,s),                  "runs": SEEDS},
    {"name": "Full_Agent_NoTGI",   "fn": lambda q,s: run_full_agent(q,s,no_tgi=True),      "runs": [42]},
    {"name": "Full_Agent_NoDSEL",  "fn": lambda q,s: run_full_agent(q,s,no_dsel=True),     "runs": [42]},
]

class TokenBucketRateLimiter:
    def __init__(self, max_per_minute: int):
        self._lock = threading.Lock()
        self._interval = 60.0 / max_per_minute
        self._next_allowed = time.monotonic()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = max(self._next_allowed + self._interval,
                                             now + self._interval)
                    return
                wait = self._next_allowed - now
            time.sleep(min(wait, 0.05))

_rate_limiter = TokenBucketRateLimiter(RATE_LIMIT_RPM)
_csv_lock = threading.Lock()

def write_row(writer, out_f, row: dict):
    with _csv_lock:
        writer.writerow(row)
        out_f.flush()

def run_task(task: dict) -> dict:
    cfg_name = task["cfg_name"]
    seed     = task["seed"]
    q        = task["query"]
    fn       = task["fn"]

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

    with task["counter_lock"]:
        task["counter"][0] += 1
        done  = task["counter"][0]
        total = task["total"]
        print(f"  [{done:4d}/{total}] {cfg_name:25s} s={seed} id={q['id']:3d} → {res['status']}", flush=True)

    return {"cfg": cfg_name, "seed": seed, "id": q["id"], "status": res["status"]}

def main():
    heldout_ids = set(json.loads(HELDOUT_IDS_FILE.read_text(encoding="utf-8")))
    dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    heldout_queries = [q for q in dataset if q["id"] in heldout_ids]
    print(f"Heldout queries: {len(heldout_queries)}")

    done_keys = load_done_keys(OUT_CSV)
    print(f"Checkpoint: {len(done_keys)} rows already done in CSV.")

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
    print(f"Heldout tasks to run: {total} | Workers: {MAX_WORKERS} | RPM: {RATE_LIMIT_RPM}")

    if total == 0:
        print("All heldout tasks already complete!")
        return

    write_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
    out_f  = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    counter      = [0]
    counter_lock = threading.Lock()

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
                res = fut.result()
                if res["status"] != STATUS_OK:
                    errors += 1
            except Exception as e:
                errors += 1

    out_f.close()
    elapsed = time.perf_counter() - t_start
    print(f"\nPARALLEL HELDOUT EVAL COMPLETE in {elapsed/60:.1f} min! Rows: {total}, Errors: {errors}")

if __name__ == "__main__":
    main()

