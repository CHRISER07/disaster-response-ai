"""
evaluation/rescore_kw_f1.py

Principled, deterministic keyword-F1 re-scoring of existing evaluation answers.

Why this exists
---------------
The legacy metric computed precision as (matched keywords)/(all answer tokens),
which mechanically drives scores toward zero for any full prose answer and makes
the metric useless for comparing systems. This module replaces it with a
dataset-derived domain vocabulary:

  VOCAB = set of stems of every expected_keyword in the golden dataset
  predicted = (stems found in the answer) INTERSECT VOCAB
  gold      = stems of this query's expected_keywords

  TP = |predicted AND gold|   FP = |predicted - gold|   FN = |gold - predicted|
  precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = harmonic mean

No numbers are hardcoded. The vocabulary, the gold sets and every score are
derived at run time from the dataset and the stored full answers.

Run:
    python evaluation/rescore_kw_f1.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# golden_dataset_v2.json is the file the heldout runner actually used; it has
# all 200 IDs (including 191-200) with keywords. golden_dataset_200q.json has
# only 190 IDs and would silently drop 30 rows per config.
DATASET = ROOT / "evaluation" / "golden_dataset_v2.json"
ANSWERS = ROOT / "evaluation" / "results_heldout_v2.csv"
OUT_CSV = ROOT / "evaluation" / "kw_f1_rescored.csv"
OUT_JSON = ROOT / "evaluation" / "kw_f1_rescored_summary.json"

SUFFIXES = ("ations", "ation", "ating", "ated", "ing", "ion", "ed", "es", "s")


def stem(word: str) -> str:
    w = word.lower().strip()
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: len(w) - len(suf)]
    return w


def answer_stems(answer: str) -> set[str]:
    return {stem(t) for t in re.findall(r"\b\w+\b", answer or "")}


def score(answer: str, expected: list[str], vocab: set[str]) -> dict:
    gold = {stem(k) for k in expected}
    predicted = answer_stems(answer) & vocab
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "gold_stems": sorted(gold),
        "predicted_domain_stems": sorted(predicted),
    }


def mean_ci(values: list[float]) -> dict:
    n = len(values)
    if not n:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci95": 0.0}
    m = sum(values) / n
    if n < 2:
        return {"n": n, "mean": round(m, 4), "std": 0.0, "ci95": 0.0}
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    return {
        "n": n,
        "mean": round(m, 4),
        "std": round(sd, 4),
        "ci95": round(1.96 * sd / math.sqrt(n), 4),
    }


def main() -> None:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    by_id = {str(item["id"]): item for item in dataset}

    all_keywords = [kw for item in dataset for kw in item.get("expected_keywords", [])]
    vocab = {stem(kw) for kw in all_keywords}
    print(f"Derived domain vocabulary: {len(vocab)} stems from {len(all_keywords)} gold keywords")

    rows = []
    with open(ANSWERS, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status", "").upper() != "OK":
                continue
            item = by_id.get(str(r.get("id", "")))
            if not item:
                continue
            s = score(r.get("full_answer", ""), item.get("expected_keywords", []), vocab)
            rows.append({
                "config": r["config"],
                "id": r["id"],
                "kw_precision": s["precision"],
                "kw_recall": s["recall"],
                "kw_f1": s["f1"],
                "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            })

    if not rows:
        print("No scoreable rows.")
        return

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_config: dict[str, list[float]] = {}
    for r in rows:
        by_config.setdefault(r["config"], []).append(r["kw_f1"])
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_answers": str(ANSWERS.relative_to(ROOT)),
        "dataset": str(DATASET.relative_to(ROOT)),
        "method": "dataset-derived domain vocabulary; stemmed bag-of-keywords F1",
        "vocab_size": len(vocab),
        "scored_rows": len(rows),
        "by_config": {cfg: mean_ci(vals) for cfg, vals in sorted(by_config.items())},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nCorrected keyword-F1 by config (all OK rows):")
    for cfg, stats in summary["by_config"].items():
        print(f"  {cfg:22s} {stats['mean']:.4f} +/- {stats['ci95']:.4f}  (n={stats['n']})")
    print(f"\nWrote {OUT_CSV.name} and {OUT_JSON.name}")


if __name__ == "__main__":
    main()
