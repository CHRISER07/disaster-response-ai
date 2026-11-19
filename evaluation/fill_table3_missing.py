"""Generate comparable held-out rows for the missing Table III cells.

Uses the published generator (Qwen 3.8-27B on Groq) and the exact legacy KW-F1
scorer from evaluation/scratch/recompute_kw.py. Replaces the invalid legacy
BM25 rows, and adds Text-Only Static RAG rows (dev-only in the published run).

Run: python evaluation/fill_table3_missing.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["PROVIDER"] = "groq"
os.environ.setdefault("GROQ_MODEL", "qwen/qwen3.8-27b")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
os.environ["PROVIDER"] = "groq"
os.environ["GROQ_MODEL"] = "qwen/qwen3.8-27b"

from evaluation.run_dev_eval_v2 import (
    run_bm25, run_static_rag, FIELDNAMES, STATUS_OK,
)
from core.answer_formatter import format_answer

OUT = ROOT / "evaluation" / "results_heldout_v2.csv"
DATASET = ROOT / "evaluation" / "golden_dataset_v2.json"
HELDOUT = ROOT / "evaluation" / "heldout_ids.json"
SEEDS = [42, 7, 99]
STOP = {"a", "an", "the", "is", "are", "was", "were", "in", "of", "to",
        "for", "and", "or", "it", "its", "be", "do", "if", "at", "by", "on", "with"}


def stem(w: str) -> str:
    for sfx in ("ing", "tion", "ness", "ment", "ed", "es", "s"):
        if w.endswith(sfx) and len(w) - len(sfx) > 3:
            return w[:-len(sfx)]
    return w


def tokens(text: str) -> set:
    return {stem(w) for w in re.findall(r"[a-z]+", text.lower())
            if w not in STOP and len(w) > 2}


def legacy_kw(answer: str, gold: list) -> tuple:
    at = tokens(answer)
    gt = {stem(k.lower()) for k in gold if k}
    if not at or not gt:
        return 0.0, 0.0, 0.0
    tp = len(at & gt)
    p = round(tp / len(at), 4)
    r = round(tp / len(gt), 4)
    f1 = round(2 * p * r / (p + r), 4) if p + r else 0.0
    return p, r, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    held = set(json.loads(HELDOUT.read_text(encoding="utf-8")))
    ds = {q["id"]: q for q in json.loads(DATASET.read_text(encoding="utf-8"))}
    queries = [ds[i] for i in sorted(held) if i in ds]
    if args.limit:
        queries = queries[: args.limit]

    rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
    # Remove both target arms so every row is regenerated under the published
    # generator. Partial/error rows from an earlier attempt must not survive.
    targets = {"BM25_Baseline", "Text_Only_StaticRAG"}
    kept = [r for r in rows if r["config"] not in targets]
    removed = len(rows) - len(kept)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader(); w.writerows(kept)
    print(f"Removed {removed} old rows; kept {len(kept)}", flush=True)

    done = {(r["config"], r["run_id"], r["id"]) for r in kept}
    tasks = [
        ("BM25_Baseline", lambda q, s: run_bm25(q, s)),
        ("Text_Only_StaticRAG", lambda q, s: run_static_rag(q, s, text_only=True)),
    ]

    with open(OUT, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for name, fn in tasks:
            for seed in SEEDS:
                for q in queries:
                    key = (name, str(seed), str(q["id"]))
                    if key in done:
                        continue
                    # The Groq pool contains a few restricted keys; retry so a
                    # transient bad-key draw does not poison a table cell.
                    res = None
                    for attempt in range(6):
                        try:
                            candidate = fn(q["query"], seed)
                            if candidate.get("status") == STATUS_OK:
                                res = candidate
                                break
                            err = str(candidate.get("full_answer", ""))[:200]
                        except Exception as exc:
                            err = str(exc)[:200]
                        time.sleep(1.0 + attempt)
                    if res is None:
                        res = {"status": "OTHER_ERROR", "full_answer": err,
                               "active_latency_s": 0, "total_latency_s": 0,
                               "prompt_tokens": 0, "completion_tokens": 0,
                               "tools_called": "", "model_used": "qwen/qwen3.8-27b",
                               "retries": 6}
                    answer = format_answer(res.get("full_answer", ""))
                    p, r, f1 = legacy_kw(answer, q.get("keywords", [])) if res["status"] == STATUS_OK else ("", "", "")
                    writer.writerow({
                        "config": name, "run_id": seed, "id": q["id"],
                        "scenario": q.get("scenario", ""), "category": q.get("category", ""),
                        "status": res["status"], "kw_precision": p, "kw_recall": r,
                        "kw_f1": f1, "svc_regex": "", "p_at_k": "", "ndcg_at_7": "", "rr": "",
                        "active_latency_s": res.get("active_latency_s", 0),
                        "total_latency_s": res.get("total_latency_s", 0),
                        "prompt_tokens": res.get("prompt_tokens", 0),
                        "completion_tokens": res.get("completion_tokens", 0),
                        "tools_called": res.get("tools_called", ""),
                        "model_used": res.get("model_used", "qwen/qwen3.8-27b"),
                        "retries": res.get("retries", 0),
                        "full_answer": answer,
                    })
                    f.flush()
                    print(f"{name} seed={seed} id={q['id']} f1={f1}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
