"""
evaluation/rescore_dsel_violations.py

Deterministically re-score DSEL ablation violations from stored full answers.

Why: the failure-to-warn rule counted clear refusals ("I can't provide ...")
as violations because SAFETY_WORDS did not include refusal forms. That inflated
the Raw_LLM rate. This script applies the corrected validator to every stored
answer and writes a corrected CSV plus summary — no re-run, no API calls,
no hardcoded scores.

Run (after the ablation finishes):
    python evaluation/rescore_dsel_violations.py --tag local
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from core.safety_validator import validate_and_repair


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return [round(max(0, c - m) * 100, 1), round(min(1, c + m) * 100, 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="local")
    args = ap.parse_args()

    src = ROOT / "evaluation" / f"dsel_ablation_raw{('_' + args.tag) if args.tag else ''}.csv"
    out = ROOT / "evaluation" / f"dsel_ablation_raw{('_' + args.tag) if args.tag else ''}_rescored.csv"
    summ = ROOT / "evaluation" / f"dsel_ablation_results{('_' + args.tag) if args.tag else ''}_rescored.json"

    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    # Keep the latest trial per key (retries may have earlier failed rows).
    latest = {}
    for r in rows:
        latest[(r["config"], str(r["seed"]), r["prompt_id"])] = r
    selected = list(latest.values())

    totals = {}
    for cfg in sorted({r["config"] for r in selected}):
        sub = [r for r in selected if r["config"] == cfg]
        ok = [r for r in sub if r["status"].upper() == "OK"]
        v = 0
        for r in ok:
            vr = validate_and_repair(r["response"], is_dangerous_query=True)
            r["is_violation"] = 1 if not vr["safe"] else 0
            v += int(r["is_violation"])
        totals[cfg] = {"evaluated": len(ok), "errors": len(sub) - len(ok),
                       "violations": v,
                       "rate": round(v / len(ok), 4) if ok else None,
                       "wilson_95_ci": wilson(v, len(ok))}

    fn = list(rows[0].keys())
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(selected)

    summ.write_text(json.dumps({
        "source": src.name,
        "method": "corrected validator (refusals counted as safe) applied to stored answers",
        "summary": totals,
    }, indent=2), encoding="utf-8")

    print(f"Re-scored {len(selected)} trials from {src.name}")
    for cfg, t in totals.items():
        print(f"  {cfg:20s} {t['violations']:3d}/{t['evaluated']:3d} "
              f"rate={t['rate']} errors={t['errors']} CI={t['wilson_95_ci']}")
    print("Wrote", out.name, "and", summ.name)


if __name__ == "__main__":
    main()
