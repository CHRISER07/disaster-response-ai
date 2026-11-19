"""Rescore the fresh single-generator cohort with the leakage-free scorer.

No model calls. Labels recomputed from exact stored answers, which were scored
raw before presentation formatting.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.run_dev_eval_v2 import kw_scores_leakfree

CSV = ROOT / "evaluation" / "results_heldout_fresh.csv"
BY_ID = {str(q["id"]): q for q in json.loads(
    (ROOT / "evaluation" / "golden_dataset_v2.json").read_text(encoding="utf-8"))}
VERSION = "leakfree-token-v1"

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
changed = 0
for row in rows:
    fresh_version = row.get("kw_scorer_version", "")
    if row["status"] == "OK" and row["id"] in BY_ID:
        p, r, f1 = kw_scores_leakfree(
            row["full_answer"], BY_ID[row["id"]].get("keywords", []))
        if fresh_version != VERSION or row["kw_f1"] != str(f1):
            changed += 1
        row["kw_precision"], row["kw_recall"], row["kw_f1"] = p, r, f1
    row["kw_scorer_version"] = VERSION
with open(CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"Re-scored {len(rows)} rows with {VERSION}; changed {changed}")
