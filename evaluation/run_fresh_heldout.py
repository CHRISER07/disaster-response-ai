"""
evaluation/run_fresh_heldout.py

Fresh head-to-head evaluation of the RESTORED system.

Differences from run_heldout_eval_v2.py:
  - eligible IDs are exactly the 150 held-out IDs from heldout_ids.json minus dev_ids; IDs 176-200 are ordinary domain queries (hazard, triage, live telemetry, damage), not meta queries, and must not be dropped
  - provider is locked at launch and re-asserted before every call
  - every stored answer passes through format_answer (no asterisks)
  - uses the corrected dataset-derived KW-F1 scorer
  - writes to results_heldout_fresh.csv (never overwrites old artifacts)
  - every row records the actual model_used

Run:
  python evaluation/run_fresh_heldout.py --limit 5      # smoke test
  python evaluation/run_fresh_heldout.py                # full
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUTF8", "1")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

LOCKED_PROVIDER = os.getenv("PROVIDER", "ollama")
LOCKED_MODEL = (os.getenv("OLLAMA_MODEL") if LOCKED_PROVIDER == "ollama"
                else os.getenv("GROQ_MODEL"))
OUT = ROOT / "evaluation" / "results_heldout_fresh.csv"
# A stale marker from an interrupted smoke test must never let a partial run
# look complete. Remove it at startup; it is rewritten only at the real end.
OUT.with_suffix(".complete").unlink(missing_ok=True)
DATASET = ROOT / "evaluation" / "golden_dataset_v2.json"
DEV_IDS = ROOT / "evaluation" / "dev_ids.json"
HELDOUT_IDS = ROOT / "evaluation" / "heldout_ids.json"

from evaluation.run_dev_eval_v2 import (
    run_llm_only, run_bm25, run_static_rag, run_full_agent,
    kw_scores, svc_regex_check, FIELDNAMES, STATUS_OK,
)
from core.answer_formatter import format_answer

SEEDS = [42]
CONFIGS = [
    ("LLM_Only", lambda q, s: run_llm_only(q, s)),
    ("BM25_Baseline", lambda q, s: run_bm25(q, s)),
    ("Standard_StaticRAG", lambda q, s: run_static_rag(q, s, text_only=False)),
    ("Full_Agent", lambda q, s: run_full_agent(q, s)),
    ("Full_Agent_NoTGI", lambda q, s: run_full_agent(q, s, no_tgi=True)),
    ("Full_Agent_NoDSEL", lambda q, s: run_full_agent(q, s, no_dsel=True)),
    ("Full_Agent_NoTelem", lambda q, s: run_full_agent(q, s, no_telemetry=True)),
]


def reassert() -> None:
    os.environ["PROVIDER"] = LOCKED_PROVIDER
    if LOCKED_MODEL:
        key = "OLLAMA_MODEL" if LOCKED_PROVIDER == "ollama" else "GROQ_MODEL"
        os.environ[key] = LOCKED_MODEL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of queries (for smoke tests)")
    ap.add_argument("--config", default=None,
                    help="run only this config name (e.g. Full_Agent)")
    ap.add_argument("--max-new", type=int, default=None,
                    help="stop after this many NEW rows are written; supports chunked foreground runs")
    ap.add_argument("--ids", default=None,
                    help="comma-separated query IDs to target for repair re-runs")
    args = ap.parse_args()

    dev = set(json.loads(DEV_IDS.read_text(encoding="utf-8")))
    held = set(json.loads(HELDOUT_IDS.read_text(encoding="utf-8")))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    queries = [q for q in dataset if q["id"] in held and q["id"] not in dev]
    assert len(queries) == 150, f"expected 150 held-out queries, got {len(queries)}"
    if args.ids:
        wanted_ids = {int(x) for x in args.ids.split(",") if x.strip()}
        queries = [q for q in queries if q["id"] in wanted_ids]
        assert len(queries) == len(wanted_ids), "some requested IDs are not eligible held-out IDs"
    if args.limit:
        queries = queries[: args.limit]
    active_configs = [c for c in CONFIGS if args.config is None or c[0] == args.config]
    if not active_configs:
        raise SystemExit(f"Unknown --config '{args.config}'. Valid: {[c[0] for c in CONFIGS]}")
    print(f"Fresh heldout: {len(queries)} queries | provider={LOCKED_PROVIDER} | model={LOCKED_MODEL}")

    done = set()
    if OUT.exists():
        for r in csv.DictReader(open(OUT, encoding="utf-8")):
            if r["status"] == "OK":
                done.add((r["config"], str(r["run_id"]), str(r["id"])))

    write_header = not OUT.exists() or OUT.stat().st_size == 0
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        total = len(active_configs) * len(SEEDS) * len(queries)
        n = 0
        new_written = 0
        for name, fn in active_configs:
            for seed in SEEDS:
                for q in queries:
                    key = (name, str(seed), str(q["id"]))
                    if key in done:
                        continue
                    if args.max_new is not None and new_written >= args.max_new:
                        print(f"Reached --max-new={args.max_new}; stopping cleanly.", flush=True)
                        f.flush()
                        return
                    n += 1
                    new_written += 1
                    reassert()
                    try:
                        res = fn(q["query"], seed)
                    except Exception as exc:
                        res = {"status": "OTHER_ERROR", "full_answer": str(exc),
                               "active_latency_s": 0, "total_latency_s": 0,
                               "prompt_tokens": 0, "completion_tokens": 0,
                               "tools_called": "", "model_used": LOCKED_MODEL, "retries": 0}
                    answer = format_answer(res.get("full_answer", ""))
                    if res["status"] == STATUS_OK:
                        p, r, f1 = kw_scores(answer, q.get("keywords", []))
                        svc = svc_regex_check(answer, q.get("expected_service", ""))
                    else:
                        p = r = f1 = svc = ""
                    writer.writerow({
                        "config": name, "run_id": seed, "id": q["id"],
                        "scenario": q.get("scenario", ""), "category": q.get("category", ""),
                        "status": res["status"], "kw_precision": p, "kw_recall": r,
                        "kw_f1": f1, "svc_regex": svc, "p_at_k": "", "ndcg_at_7": "", "rr": "",
                        "active_latency_s": res.get("active_latency_s", 0),
                        "total_latency_s": res.get("total_latency_s", 0),
                        "prompt_tokens": res.get("prompt_tokens", 0),
                        "completion_tokens": res.get("completion_tokens", 0),
                        "tools_called": res.get("tools_called", ""),
                        "model_used": res.get("model_used", LOCKED_MODEL),
                        "retries": res.get("retries", 0),
                        "full_answer": answer,
                    })
                    f.flush()
                    print(f"[{n}/{total}] {name} id={q['id']} -> {res['status']}", flush=True)
                    time.sleep(0.3)
    # Completion marker
    OUT.with_suffix(".complete").write_text(
        f"rows={sum(1 for _ in csv.DictReader(open(OUT, encoding='utf-8')))}\n",
        encoding="utf-8",
    )
    print("Done. Output:", OUT)


if __name__ == "__main__":
    main()
