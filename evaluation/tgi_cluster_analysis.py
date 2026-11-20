"""Cluster-aware reanalysis of the three TGI suites.

The seeds are repeated measures on the same query texts, so treating every
(query, seed) row as independent inflates the sample size. This script
clusters by base query text (strips the -N/-P condition suffix), aggregates
correctness within each cluster, and runs Wilcoxon signed-rank on the
cluster-level paired differences. It also reports missed-evacuation counts,
which are the safety-relevant direction.

No metric is fitted to the outcome; this is a statistical correction only.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent

SUITES = [
    ("TGI v6 (12 scenarios)", ROOT / "evaluation" / "tgi_results_v6.csv",
     "Full_Agent_TGI", "Full_Agent_NoTGI"),
    ("EXP-2 (50 scenarios)", ROOT / "evaluation" / "tgi_exp2_results.csv",
     "Full_Agent_TGI", "Full_Agent_NoTGI"),
    ("EXP-2b (50 scenarios)", ROOT / "evaluation" / "tgi_exp2b_results.csv",
     "Full_Agent_TGI", "Full_Agent_NoTGI"),
]


def base_cluster(query_id: str) -> str:
    return re.sub(r"-(N|P)$", "", query_id)


def analyse(name, path, tgi, notgi):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("assessable") != "1":
            continue
        for key in ("correct", "false_evacuation", "missed_evacuation"):
            by[base_cluster(r["query_id"])][r["config"] + "_" + key].append(int(r[key]))

    clusters = sorted(by)
    d_correct = []
    d_missed = []
    d_fp = []
    for c in clusters:
        d = by[c]
        t = d[tgi + "_correct"]; n = d[notgi + "_correct"]
        if not t or not n:
            continue
        d_correct.append(sum(t) / len(t) - sum(n) / len(n))
        tm = d[tgi + "_missed_evacuation"]; nm = d[notgi + "_missed_evacuation"]
        d_missed.append(sum(nm) / len(nm) - sum(tm) / len(tm))  # positive = TGI fewer misses
        tf = d[tgi + "_false_evacuation"]; nf = d[notgi + "_false_evacuation"]
        d_fp.append(sum(tf) / len(tf) - sum(nf) / len(nf))

    print(f"\n=== {name} ===")
    print(f"clusters={len(clusters)}  rows={len(rows)}")
    for label, vals in (("correct(TGI-NoTGI)", d_correct),
                        ("misses avoided(NoTGI-TGI)", d_missed),
                        ("false evac(TGI-NoTGI)", d_fp)):
        nz = [v for v in vals if abs(v) > 1e-12]
        print(f"  {label:28s} mean={sum(vals)/len(vals):+.4f} "
              f"nonzero_clusters={len(nz)}/{len(vals)}")
        if nz:
            res = wilcoxon(vals, alternative="two-sided", zero_method="wilcox")
            print(f"    Wilcoxon W={res.statistic:.1f} p={res.pvalue:.4g}")
        else:
            print("    Wilcoxon undefined (no nonzero differences)")


if __name__ == "__main__":
    for s in SUITES:
        analyse(*s)
