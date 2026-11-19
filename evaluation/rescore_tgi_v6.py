"""Re-score stored TGI v6 answers with the corrected evacuation detector.

No model calls: labels are recomputed from the exact stored answer text.
The bug: a formatter-inserted "Recommendation: NOT AVAILABLE" placeholder made
the detector ignore a real "Mandatory evacuation" sentence in the prose.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.run_tgi_eval_v2 import detect_evacuation

CSV = ROOT / "evaluation" / "tgi_results_v6.csv"


def main() -> None:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    changed = 0
    for row in rows:
        is_evac = detect_evacuation(row["full_answer"])
        expected_pos = row["expected_label"] == "EVACUATION"
        new_fp = int(is_evac and not expected_pos)
        new_fn = int((not is_evac) and expected_pos)
        new_correct = int(is_evac == expected_pos)
        if (new_fp != int(row["false_evacuation"]) or
                new_fn != int(row["missed_evacuation"]) or
                new_correct != int(row["correct"])):
            changed += 1
        row["actual_label"] = "EVACUATION" if is_evac else "NO_EVACUATION"
        row["false_evacuation"] = new_fp
        row["missed_evacuation"] = new_fn
        row["correct"] = new_correct
    with open(CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Re-scored {len(rows)} rows; changed {changed}")


if __name__ == "__main__":
    main()
