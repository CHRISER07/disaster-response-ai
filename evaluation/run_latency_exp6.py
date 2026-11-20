"""EXP-6: latency under concurrent load, local locked generator.

Sweep concurrency 1 / 5 / 10 for BM25 and Full Agent over a fixed set of 10
dev queries. Latency is wall-clock measured by the caller, so it includes any
queueing introduced by concurrency. No LLM-based classification is involved:
the outcomes are timestamps and statuses only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUTF8", "1")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:3b")

from evaluation.run_dev_eval_v2 import run_bm25, run_full_agent

DATASET = ROOT / "evaluation" / "golden_dataset_200q.json"
OUT_JSON = ROOT / "evaluation" / "latency_concurrent_results.json"
OUT_CSV = ROOT / "evaluation" / "latency_concurrent_results.csv"
QUERY_IDS = list(range(1, 11))  # fixed dev subset, frozen before the run
LEVELS = [1, 5, 10]
CONFIGS = [("BM25", run_bm25), ("Full_Agent", run_full_agent)]
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(sorted_values[0], 3)
    k = (len(sorted_values) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(sorted_values[int(k)], 3)
    return round(sorted_values[f] * (c - k) + sorted_values[c] * (k - f), 3)


def stats(values):
    s = sorted(values)
    return {
        "n": len(s),
        "mean_s": round(statistics.mean(s), 3) if s else None,
        "p50_s": percentile(s, 50), "p90_s": percentile(s, 90),
        "p95_s": percentile(s, 95), "p99_s": percentile(s, 99),
        "min_s": round(s[0], 3) if s else None,
        "max_s": round(s[-1], 3) if s else None,
    }


def one_call(fn, qid, query, seed, config, level):
    t0 = time.perf_counter()
    status = "OK"
    try:
        res = fn(query, seed)
        status = str(res.get("status", "OK"))
    except Exception as e:
        status = f"ERROR:{type(e).__name__}"
    dt = time.perf_counter() - t0
    return {"config": config, "level": level, "query_id": qid,
            "seed": seed, "status": status, "latency_s": round(dt, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,5,10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test query cap")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    by_id = {int(x["id"]): x["query"] for x in dataset}
    queries = [(qid, by_id[qid]) for qid in QUERY_IDS if qid in by_id]
    if args.limit:
        queries = queries[:args.limit]
    print(f"queries={len(queries)} levels={levels} model={MODEL}", flush=True)

    rows = []
    summary = {"metadata": {"model": MODEL, "provider": os.getenv("PROVIDER"),
                            "seed": args.seed, "query_ids": QUERY_IDS,
                            "metric": "caller wall-clock seconds, includes queueing",
                            "generated_utc": datetime.now(timezone.utc).isoformat()},
               "levels": {}}
    for cfg_name, fn in CONFIGS:
        for level in levels:
            batch = []
            t_batch = time.perf_counter()
            with ThreadPoolExecutor(max_workers=level) as pool:
                futures = [pool.submit(one_call, fn, qid, q, args.seed, cfg_name, level)
                           for qid, q in queries]
                for fut in as_completed(futures):
                    batch.append(fut.result())
            wall = time.perf_counter() - t_batch
            rows.extend(batch)
            ok = [r["latency_s"] for r in batch if r["status"] == "OK"]
            summary["levels"].setdefault(str(level), {})[cfg_name] = {
                **stats(ok),
                "batch_wall_s": round(wall, 3),
                "throughput_rps": round(len(ok) / wall, 3) if wall and ok else None,
                "status_counts": dict(Counter(r["status"] for r in batch)),
            }
            print(f"{cfg_name:12s} level={level:2d} n_ok={len(ok):3d} "
                  f"P50={summary['levels'][str(level)][cfg_name]['p50_s']}s "
                  f"P95={summary['levels'][str(level)][cfg_name]['p95_s']}s "
                  f"wall={wall:.2f}s", flush=True)

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", OUT_JSON, OUT_CSV)
    print(json.dumps(summary["levels"], indent=2))


if __name__ == "__main__":
    main()
