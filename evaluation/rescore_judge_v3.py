"""
evaluation/rescore_judge_v3.py

Clean judge_scores_v3.csv without inventing any scores.

The running judge used a parser that filled a missing Temporal score with 1.0
and a sentinel rationale. Those were not model judgments. This script marks
them as unassessable (empty score) so they are excluded from aggregates, and
writes a separate cleaned file. It never replaces a score with a guess.

Run: python evaluation/rescore_judge_v3.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "evaluation" / "judge_scores_v3.csv"
OUT = ROOT / "evaluation" / "judge_scores_v3_clean.csv"
SENTINEL = "Evaluated consistent with temporal context."


def main() -> None:
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    fixed = 0
    for r in rows:
        if r.get("temporal_rationale", "").strip() == SENTINEL:
            r["temporal"] = ""
            r["temporal_rationale"] = "UNASSESSABLE - judge did not return a temporal score"
            fixed += 1
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Rows: {len(rows)} | fabricated temporal scores cleared: {fixed}")
    print("Wrote", OUT.name)


if __name__ == "__main__":
    main()
