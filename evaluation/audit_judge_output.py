"""Continuous audit of an independent dual-judge output file.

Checks judge-family independence, score validity, completeness and duplicates.
Exits non-zero on any problem. Usage: python evaluation/audit_judge_output.py [csv]
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent / "judge_scores_v3fixed.csv"
VALID = {0.0, 0.5, 1.0}


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"MISSING {path}")
        return 1
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    problems: list[str] = []

    models = sorted({r.get("model_used", "") for r in rows})
    if any("qwen" in m.lower() for m in models):
        problems.append(f"GENERATOR AS JUDGE: {models}")
    if "none" in [m.lower() for m in models]:
        problems.append("judge returned no model identity")

    per_judge = Counter(r.get("judge") for r in rows)
    keys = [(r.get("config"), r.get("run_id"), r.get("id"), r.get("judge")) for r in rows]
    dupes = len(keys) - len(set(keys))
    if dupes:
        problems.append(f"duplicate keys: {dupes}")

    missing = Counter()
    invalid = Counter()
    for r in rows:
        for dim in ("dr_fact", "saa", "temporal"):
            raw = r.get(dim, "")
            if raw in ("", "None"):
                missing[dim] += 1
                continue
            try:
                if float(raw) not in VALID:
                    invalid[dim] += 1
            except ValueError:
                invalid[dim] += 1
    if missing:
        problems.append(f"missing scores: {dict(missing)}")
    if invalid:
        problems.append(f"invalid scores: {dict(invalid)}")

    families = {j: sorted({m for r in rows if r.get("judge") == j
                           for m in [r.get("model_used", "")]}) for j in per_judge}
    print(f"rows={len(rows)} judges={dict(per_judge)} models={models}")
    print(f"families={families} missing={dict(missing)} invalid={dict(invalid)} dupes={dupes}")
    if problems:
        print("FAIL: " + " | ".join(problems))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
