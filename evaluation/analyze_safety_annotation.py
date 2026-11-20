"""Analyze the human safety annotation sheet once labeling is complete.

Reads safety_annotation_sheet.csv (blinded) plus the join key is not needed for
safety rates: the sheet deliberately hid arm labels, so this script expects the
annotator to have been given a separate arm assignment, OR it reads the
completed sheet with an arm column added by the annotator.

Because the original sheet hid arm labels, this script supports two modes:
  --with-arm  : the CSV has an 'arm' column filled from the key
  --key FILE  : join by response_id to a key CSV that contains the arm

Reports per-arm Unsafe rate with Wilson 95% CI, and Fisher exact tests between
arms. Refusals/warnings labelled Safe. Single annotator: no kappa is computed
and this is stated in the output.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
SHEET = ROOT / "evaluation" / "safety_annotation_sheet.csv"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0, c - m) * 100, 1), round(min(1, c + m) * 100, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(SHEET))
    ap.add_argument("--key", default=None, help="CSV with response_id,config columns")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.sheet, encoding="utf-8")))
    arm = {}
    if args.key:
        for r in csv.DictReader(open(args.key, encoding="utf-8")):
            arm[r["response_id"]] = r["config"]
    elif "arm" in rows[0]:
        for r in rows:
            arm[r["response_id"]] = r["arm"]
    else:
        raise SystemExit("Provide --key with response_id,config or add an 'arm' column to the sheet")

    labels = [r.get("annotator_1_label", "").strip().upper()
              for r in rows if r.get("annotator_1_label", "").strip()]
    print(f"labeled {len(labels)}/{len(rows)} responses")
    valid = {"SAFE", "UNSAFE", "AMBIGUOUS"}
    bad = [x for x in labels if x not in valid]
    if bad:
        print("unexpected labels:", set(bad))

    by = defaultdict(lambda: {"n": 0, "unsafe": 0, "ambiguous": 0})
    for r in rows:
        lab = r.get("annotator_1_label", "").strip().upper()
        if not lab:
            continue
        cfg = arm.get(r["response_id"])
        if not cfg:
            continue
        b = by[cfg]
        b["n"] += 1
        b["unsafe"] += int(lab == "UNSAFE")
        b["ambiguous"] += int(lab == "AMBIGUOUS")

    print("\nPer-arm unsafe rate (single annotator, no kappa):")
    for cfg in sorted(by):
        b = by[cfg]
        rate = b["unsafe"] / b["n"] if b["n"] else 0
        print(f"  {cfg:18s} n={b['n']:3d} unsafe={b['unsafe']:3d} "
              f"rate={rate:.3f} CI95%={wilson(b['unsafe'], b['n'])} "
              f"ambiguous={b['ambiguous']}")

    names = sorted(by)
    if len(names) >= 2:
        a, z = names[0], names[-1]
        aa, az = by[a], by[z]
        tab = [[aa["unsafe"], aa["n"] - aa["unsafe"]],
               [az["unsafe"], az["n"] - az["unsafe"]]]
        odds, p = fisher_exact(tab)
        print(f"\nFisher exact {a} vs {z}: odds={odds:.3f} p={p:.4g}")
    print("\nNOTE: one annotator -> point estimates with Wilson CI only; no inter-annotator agreement.")


if __name__ == "__main__":
    main()
