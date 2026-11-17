"""
evaluation/run_tgi_v4.py

Corrected temporal-conflict evaluation: ALL four arms on the SAME generator
(Groq qwen/qwen3.8-27b), current pipeline, key rotation + backoff, bounded
checkpoint/resume, and a completion marker. This supersedes tgi_results_v2.csv
whose TGI/NoTGI arms used the pre-fix pipeline (fake TGI, markdown output).

Arms:
  LLM_Only          - zero-shot, no tools, no TGI
  Standard_StaticRAG - single-turn static retrieval, no TGI
  Full_Agent_NoTGI  - full agent, temporal tags stripped
  Full_Agent_TGI    - full agent, real rolling-window TGI

Run:
  python evaluation/run_tgi_v4.py --limit 2        # smoke test
  python evaluation/run_tgi_v4.py --max-new 20     # bounded chunk
  python evaluation/run_tgi_v4.py                  # full (384 rows)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUTF8", "1")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

# GENERATOR LOCK: all four arms must use the SAME generator for a valid
# comparison. A detached worker sets PROVIDER/OLLAMA_MODEL explicitly; if not
# set, default to the local 3B GPU model (fast, 100% GPU on this hardware).
os.environ.setdefault("PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:3b")

from evaluation.run_dev_eval_v2 import (
    run_llm_only, run_static_rag, run_full_agent,
)
from evaluation.run_tgi_eval_v2 import detect_evacuation
from core.answer_formatter import format_answer

CONFLICT_SET = ROOT / "evaluation" / "temporal_conflict_set.json"
OUT = ROOT / "evaluation" / "tgi_results_v4.csv"
# Stale marker from an interrupted run must never make a partial result look
# complete. Clear at startup; rewritten only when the full loop finishes.
OUT.with_suffix(".complete").unlink(missing_ok=True)
# Record the actual generator in every row.
MODEL = (os.getenv("OLLAMA_MODEL") if os.getenv("PROVIDER", "ollama") == "ollama"
         else os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"))
SEEDS = [42, 7, 99]
FIELDS = [
    "config", "run_id", "query_id", "query", "has_temporal_cue",
    "live_reading_ft", "flood_stage_ft", "expected_label", "model_used",
    "status", "actual_label", "false_evacuation", "full_answer",
]
CONFIGS = [
    ("LLM_Only", lambda q, s: run_llm_only(q, s)),
    ("Standard_StaticRAG", lambda q, s: run_static_rag(q, s, text_only=False)),
    ("Full_Agent_NoTGI", lambda q, s: run_full_agent(q, s, no_tgi=True)),
    ("Full_Agent_TGI", lambda q, s: run_full_agent(q, s)),
]


def call_with_backoff(fn, query, seed, max_attempts=9):
    """Invoke a config. For local ollama there are no rate limits: one call.
    For Groq, rotate keys with classified retry on 429/400/restricted keys."""
    import agents.disaster_agent as da

    if os.getenv("PROVIDER", "ollama") == "ollama":
        try:
            da.get_agent.cache_clear()
            res = fn(query, seed)
            if str(res.get("status", "")).upper() == "OK":
                return res
            return res  # preserve real status (do not mask local errors)
        except Exception as e:
            return {"status": "OTHER_ERROR", "full_answer": str(e),
                    "model_used": MODEL, "retries": 0}

    import core.key_manager as km
    err = "uninitialized"
    for attempt in range(max_attempts):
        try:
            os.environ["GROQ_API_KEY"] = km.groq_pool.get_key()
            da.get_agent.cache_clear()
        except Exception as e:
            err = str(e); time.sleep(2); continue
        try:
            res = fn(query, seed)
            if str(res.get("status", "")).upper() == "OK":
                return res
            err = str(res.get("full_answer", "")) or "error"
        except Exception as e:
            err = str(e)
        low = err.lower()
        if "organization has been restricted" in low or "invalid api key" in low or "401" in low:
            try:
                km.groq_pool.mark_bad(km.groq_pool.get_key(), reason="restricted/invalid")
            except Exception:
                pass
            time.sleep(0.5)
            continue
        retryable = ("429" in low or "rate" in low or "token" in low
                     or "tool call validation failed" in low)
        if retryable:
            m = re.search(r"try again in ([0-9.]+)s", err, re.IGNORECASE)
            delay = float(m.group(1)) if m else min(20.0, 2.0 * (2 ** min(attempt, 3)))
            time.sleep(max(1.0, delay))
            continue
        return {"status": "OTHER_ERROR", "full_answer": err,
                "model_used": MODEL, "retries": attempt + 1}
    return {"status": "RATE_LIMIT", "full_answer": f"retries exhausted: {err}",
            "model_used": MODEL, "retries": max_attempts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-new", type=int, default=None)
    args = ap.parse_args()

    queries = json.loads(CONFLICT_SET.read_text(encoding="utf-8"))
    if args.limit:
        queries = queries[: args.limit]
    active = [c for c in CONFIGS if args.config is None or c[0] == args.config]
    if not active:
        raise SystemExit(f"Unknown --config. Valid: {[c[0] for c in CONFIGS]}")

    done = set()
    if OUT.exists():
        for r in csv.DictReader(open(OUT, encoding="utf-8")):
            if r["status"] == "OK":
                done.add((r["config"], str(r["run_id"]), r["query_id"]))

    write_header = not OUT.exists() or OUT.stat().st_size == 0
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        total = len(active) * len(SEEDS) * len(queries)
        n = new_written = 0
        for name, fn in active:
            for seed in SEEDS:
                for item in queries:
                    key = (name, str(seed), item["query_id"])
                    if key in done:
                        continue
                    if args.max_new is not None and new_written >= args.max_new:
                        print(f"Reached --max-new={args.max_new}; stopping.", flush=True)
                        f.flush()
                        return
                    n += 1; new_written += 1
                    res = call_with_backoff(fn, item["query"], seed)
                    answer = format_answer(res.get("full_answer", ""))
                    is_evac = detect_evacuation(answer) if res["status"] == "OK" else False
                    false_evac = 1 if (is_evac and item["live_reading_ft"] < item["flood_stage_ft"]) else 0
                    w.writerow({
                        "config": name, "run_id": seed, "query_id": item["query_id"],
                        "query": item["query"], "has_temporal_cue": item["has_temporal_cue"],
                        "live_reading_ft": item["live_reading_ft"], "flood_stage_ft": item["flood_stage_ft"],
                        "expected_label": item["expected_label"], "model_used": res.get("model_used", MODEL),
                        "status": res["status"], "actual_label": "EVACUATION" if is_evac else "NO_EVACUATION",
                        "false_evacuation": false_evac, "full_answer": answer,
                    })
                    f.flush()
                    print(f"[{n}/{total}] {name} seed={seed} {item['query_id']} "
                          f"-> evac={is_evac} false={false_evac} status={res['status']}", flush=True)
                    time.sleep(0.5)
    OUT.with_suffix(".complete").write_text(
        f"rows={sum(1 for _ in csv.DictReader(open(OUT, encoding='utf-8')))}\n"
        f"finished={datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
    print("Done:", OUT)


if __name__ == "__main__":
    main()
