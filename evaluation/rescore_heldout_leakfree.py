"""Rescore archived DAB-200 answers with the leakage-free scorer.

Writes *_leakfree.csv artifacts with identical schema; no model calls.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.run_dev_eval_v2 import kw_scores_leakfree

BY_ID = {str(q["id"]): q for q in json.loads(
    (ROOT / "evaluation" / "golden_dataset_v2.json").read_text(encoding="utf-8"))}


def rescore(name):
    src = ROOT / "evaluation" / name
    dst = ROOT / "evaluation" / name.replace(".csv", "_leakfree.csv")
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    for row in rows:
        if row["status"] == "OK" and row["id"] in BY_ID:
            p, r, f1 = kw_scores_leakfree(
                row["full_answer"], BY_ID[row["id"]].get("keywords", []))
            row["kw_precision"], row["kw_recall"], row["kw_f1"] = p, r, f1
        else:
            row["kw_precision"] = row["kw_recall"] = row["kw_f1"] = ""
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", dst.name, len(rows))


if __name__ == "__main__":
    rescore("results_per_query_v2.csv")
    rescore("results_heldout_v2.csv")
