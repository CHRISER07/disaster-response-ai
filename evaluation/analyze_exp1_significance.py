"""EXP-1 significance analysis: paired Wilcoxon on per-query Exp-F1.

Reads the controlled DAB-200 CSV. Compares Full Agent against each arm on
queries where both have OK scores for the same seed. Reports W, exact/approx p,
rank-biserial effect size, and mean difference.
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "evaluation" / "results_dab_controlled.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))

# key -> f1
score = {}
for r in rows:
    if r["status"] != "OK" or not r["kw_f1"]:
        continue
    try:
        f1 = float(r["kw_f1"])
    except ValueError:
        continue
    score[(r["config"], r["split"], r["run_id"], r["id"])] = f1

configs = sorted({r["config"] for r in rows})
base = "Full_Agent"


def paired(a, b):
    """Return aligned (a_scores, b_scores) over shared (split, seed, id) keys."""
    sa = {k[1:]: v for k, v in score.items() if k[0] == a}
    sb = {k[1:]: v for k, v in score.items() if k[0] == b}
    common = sorted(set(sa) & set(sb))
    return [sa[k] for k in common], [sb[k] for k in common], common


print(f"rows={len(rows)} scored={len(score)} configs={configs}")
for cfg in configs:
    if cfg == base:
        continue
    a, b, common = paired(base, cfg)
    if len(a) < 5:
        print(f"{cfg:22s} n={len(a)} (insufficient)")
        continue
    diff = [x - y for x, y in zip(a, b)]
    res = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    # rank-biserial r from Wilcoxon W and n pairs, ignoring zero differences
    nz = [d for d in diff if abs(d) > 1e-12]
    if nz:
        positive = sum(d > 0 for d in nz)
        r_rb = (positive - (len(nz) - positive)) / len(nz)
    else:
        r_rb = 0.0
    print(f"{base} vs {cfg:22s} n={len(a):4d} meanA={sum(a)/len(a):.4f} "
          f"meanB={sum(b)/len(b):.4f} delta={sum(diff)/len(diff):+.4f} "
          f"W={res.statistic:.1f} p={res.pvalue:.4g} r={r_rb:+.3f}")
