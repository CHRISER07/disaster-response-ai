"""Controlled DAB-200 rerun: one locked generator, three seeds, both splits.

All eight configurations use PROVIDER=ollama, OLLAMA_MODEL=qwen2.5:3b and
temperature 0. The leakage-free scorer is applied uniformly. Results are
written to a fresh CSV so archived cohorts are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUTF8", "1")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

LOCKED_PROVIDER = os.getenv("PROVIDER", "ollama")
LOCKED_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
if LOCKED_PROVIDER != "ollama" or not LOCKED_MODEL:
    raise SystemExit("controlled run requires a locked local generator")

OUT = ROOT / "evaluation" / os.getenv(
    "CONTROLLED_OUT_NAME", "results_dab_controlled.csv")
DATASET = ROOT / "evaluation" / "golden_dataset_v2.json"
DEV_IDS = ROOT / "evaluation" / "dev_ids.json"
HELDOUT_IDS = ROOT / "evaluation" / "heldout_ids.json"

from evaluation.run_dev_eval_v2 import (
    run_llm_only, run_bm25, run_static_rag, run_full_agent,
    kw_scores_leakfree, svc_regex_check, active_model_name,
)
from core.answer_formatter import format_answer

SEEDS = [42, 7, 99]
SCORER_VERSION = "leakfree-token-v1"

# Proposed arms first, then comparative baselines.
CONFIGS = [
    ("Full_Agent", lambda q, s: run_full_agent(q, s)),
    ("Full_Agent_NoTGI", lambda q, s: run_full_agent(q, s, no_tgi=True)),
    ("Full_Agent_NoDSEL", lambda q, s: run_full_agent(q, s, no_dsel=True)),
    ("Full_Agent_NoTelem", lambda q, s: run_full_agent(q, s, no_telemetry=True)),
    ("LLM_Only", lambda q, s: run_llm_only(q, s)),
    ("BM25_Baseline", lambda q, s: run_bm25(q, s)),
    ("Standard_StaticRAG", lambda q, s: run_static_rag(q, s, text_only=False)),
    ("Text_Only_StaticRAG", lambda q, s: run_static_rag(q, s, text_only=True)),
]

FIELDNAMES = [
    "config", "split", "run_id", "id", "scenario", "category", "status",
    "kw_precision", "kw_recall", "kw_f1", "svc_regex",
    "active_latency_s", "total_latency_s", "prompt_tokens", "completion_tokens",
    "tools_called", "model_used", "retries", "full_answer", "kw_scorer_version",
]


def reassert() -> None:
    os.environ["PROVIDER"] = LOCKED_PROVIDER
    os.environ["OLLAMA_MODEL"] = LOCKED_MODEL
    os.environ["OLLAMA_TEMPERATURE"] = "0"


def load_existing():
    """Return (rows_by_key, problems). Duplicate keys keep the latest row."""
    if not OUT.exists():
        return {}, []
    raw = list(csv.DictReader(open(OUT, encoding="utf-8")))
    problems = []
    models = {r["model_used"] for r in raw if r.get("model_used", "").strip()}
    scorers = {r.get("kw_scorer_version", "") for r in raw
               if r.get("kw_scorer_version", "")}
    if len(models) > 1 or (models and models != {LOCKED_MODEL}):
        problems.append(f"mixed generators: {models}")
    if len(scorers) > 1:
        problems.append(f"mixed scorers: {scorers}")
    by = {}
    for r in raw:
        by[(r["config"], r["split"], str(r["run_id"]), str(r["id"]))] = r
    if len(by) != len(raw):
        print(f"note: {len(raw) - len(by)} duplicate keys collapsed to latest")
    return by, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--splits", default="dev,heldout")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    dev = {int(x) for x in json.loads(DEV_IDS.read_text(encoding="utf-8"))}
    held = {int(x) for x in json.loads(HELDOUT_IDS.read_text(encoding="utf-8"))}
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    by_id = {int(q["id"]): q for q in data}
    split_ids = {"dev": sorted(dev), "heldout": sorted(held)}
    active_configs = [c for c in CONFIGS if args.config is None or c[0] == args.config]
    if not active_configs:
        raise SystemExit(f"unknown config {args.config}")

    expected = len(active_configs) * len(SEEDS) * sum(len(split_ids[s]) for s in splits)
    print(f"controlled DAB: {len(active_configs)} configs x {len(SEEDS)} seeds x "
          f"{sum(len(split_ids[s]) for s in splits)} queries = {expected} trials")
    print(f"provider={LOCKED_PROVIDER} model={LOCKED_MODEL} scorer={SCORER_VERSION}")

    existing, problems = load_existing()
    if problems:
        raise SystemExit("refusing to resume: " + "; ".join(problems))
    done = {k for k, r in existing.items() if r["status"] == "OK"}

    write_header = not OUT.exists() or OUT.stat().st_size == 0
    f = open(OUT, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    n = 0
    for name, fn in active_configs:
        for seed in SEEDS:
            for split in splits:
                for qid in split_ids[split]:
                    key = (name, split, str(seed), str(qid))
                    if key in done:
                        continue
                    if args.limit is not None and n >= args.limit:
                        f.flush()
                        f.close()
                        print(f"limit {args.limit} reached")
                        return
                    n += 1
                    reassert()
                    item = by_id[qid]
                    result = None
                    try:
                        result = fn(item["query"], seed)
                    except Exception as exc:
                        result = {"status": "OTHER_ERROR",
                                  "full_answer": f"{type(exc).__name__}: {exc}",
                                  "active_latency_s": 0, "total_latency_s": 0,
                                  "prompt_tokens": 0, "completion_tokens": 0,
                                  "tools_called": "", "model_used": "", "retries": 0}
                    status = result.get("status", "UNKNOWN")
                    raw_answer = result.get("full_answer", "")
                    if status == "OK":
                        p, r, f1 = kw_scores_leakfree(
                            raw_answer, item.get("keywords", []))
                        svc = svc_regex_check(raw_answer, item.get("expected_service", ""))
                        stored = format_answer(raw_answer)
                    else:
                        p = r = f1 = svc = ""
                        stored = raw_answer
                    writer.writerow({
                        "config": name, "split": split, "run_id": seed, "id": qid,
                        "scenario": item.get("scenario", ""),
                        "category": item.get("category", ""),
                        "status": status,
                        "kw_precision": p, "kw_recall": r, "kw_f1": f1,
                        "svc_regex": svc,
                        "active_latency_s": result.get("active_latency_s", 0),
                        "total_latency_s": result.get("total_latency_s", 0),
                        "prompt_tokens": result.get("prompt_tokens", 0),
                        "completion_tokens": result.get("completion_tokens", 0),
                        "tools_called": result.get("tools_called", ""),
                        # Keep blank on error rows: no generator was confirmed.
                        "model_used": result.get("model_used", ""),
                        "retries": result.get("retries", 0),
                        "full_answer": stored,
                        "kw_scorer_version": SCORER_VERSION,
                    })
                    f.flush()
                    print(f"[{n}/{expected}] {name} {split} seed={seed} id={qid} -> {status}",
                          flush=True)
    f.close()

    rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
    latest, dups = {}, 0
    for r in rows:
        k = (r["config"], r["split"], str(r["run_id"]), str(r["id"]))
        if k in latest:
            dups += 1
        latest[k] = r
    if dups:
        with open(OUT, "w", encoding="utf-8", newline="") as g:
            w = csv.DictWriter(g, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerows(latest.values())
        rows = list(latest.values())
        print(f"collapsed {dups} duplicate rows")

    counts = Counter(r["status"] for r in rows)
    expected_keys = {
        (name, split, str(seed), str(qid))
        for name, _ in active_configs
        for seed in SEEDS
        for split in splits
        for qid in split_ids[split]
    }
    present_keys = {
        (r["config"], r["split"], str(r["run_id"]), str(r["id"])) for r in rows
    }
    full = expected_keys <= present_keys
    marker = OUT.with_suffix(".complete")
    if full:
        marker.write_text(
            f"rows={len(rows)}\nexpected={expected}\n"
            + "".join(f"{k}={v}\n" for k, v in sorted(counts.items()))
            + f"generator={LOCKED_MODEL}\nscorer={SCORER_VERSION}\n",
            encoding="utf-8")
    else:
        marker.unlink(missing_ok=True)
    print(f"final rows={len(rows)}/{expected} status={dict(counts)} marker={full}")


if __name__ == "__main__":
    main()
