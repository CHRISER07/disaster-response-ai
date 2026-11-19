"""Deterministically re-score the authoritative DSEL CSV with scorer v3.

No model calls. Labels are recomputed from the exact stored answer text, so the
result is reproducible and the stored flags stay consistent with aggregation.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.safety_validator import validate_and_repair
from evaluation.eval_dsel_ablation import SCORER_VERSION

CSV = ROOT / "evaluation" / "dsel_ablation_raw_qwen25_3b.csv"


def main() -> None:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    changed = 0
    for row in rows:
        fresh = int(not validate_and_repair(row["response"], is_dangerous_query=True)["safe"])
        if fresh != int(row["is_violation"]):
            changed += 1
        row["is_violation"] = fresh
        row["scorer_version"] = SCORER_VERSION
    with open(CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Re-scored {len(rows)} rows with {SCORER_VERSION}; changed {changed}")


if __name__ == "__main__":
    main()
