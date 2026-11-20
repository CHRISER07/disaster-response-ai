"""EXP-7D preregistered statistical analysis.

Unit of analysis = unique scenario (n=60 primary), not execution.
Three seeds are repeated measures; aggregate within scenario per arm.
Primary comparisons: TGI vs TimestampOnly, TGI vs NoTGI, TimestampOnly vs NoTGI.
Holm correction across the three planned Wilcoxon tests.
Scenario-clustered bootstrap 95% CI for each arm's accuracy and for paired deltas.
Direction-specific results reported separately.
Structured decision/reason consistency and abstention reported separately.
"""
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "evaluation" / "exp7d_results.csv"
ARMS = ["TGI", "TimestampOnly", "NoTGI"]
COMPARISONS = [("TGI", "TimestampOnly"), ("TGI", "NoTGI"), ("TimestampOnly", "NoTGI")]
RNG = np.random.default_rng(42)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0, c - m) * 100, 1), round(min(1, c + m) * 100, 1))


def reason_consistent(answer, decision):
    """Does the reason text agree with the structured decision?"""
    m = re.search(r'"reason"\s*:\s*"([^"]*)"', answer)
    if not m:
        return None
    r = m.group(1).lower()
    if decision == "CURRENT_HAZARD":
        return "hazard" in r or "above" in r or "exceed" in r or "danger" in r
    if decision == "NO_CURRENT_HAZARD":
        return "no current" in r or "no flood" in r or "below" in r or "no hazard" in r or "safe" in r
    return None


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    primary = [r for r in rows if r["group"] == "primary"
               and r["consumption_ok"] == "1" and r["parse_ok"] == "1"]
    diagnostic = [r for r in rows if r["group"] == "diagnostic"
                  and r["consumption_ok"] == "1" and r["parse_ok"] == "1"]

    # Aggregate correctness to scenario level (mean over 3 seeds) per arm.
    per = defaultdict(dict)
    meta = {}
    for r in primary:
        per[r["scenario_id"]].setdefault(r["arm"], []).append(int(r["correct"]))
        meta[r["scenario_id"]] = (r["condition"], r["age_code"])
    scenarios = sorted(per)
    for s in scenarios:
        for arm in ARMS:
            per[s][arm] = float(np.mean(per[s][arm]))

    counts = {a: sum(1 for r in primary if r["arm"] == a) for a in ARMS}
    print(f"primary rows={len(primary)} scenarios={len(scenarios)} per-arm execs={counts}")

    # Per-arm scenario accuracy with clustered bootstrap CI (resample scenarios).
    S = np.array(scenarios)
    def arm_vec(arm):
        return np.array([per[s][arm] for s in scenarios])
    boot_stats = {}
    print("\n=== per-arm accuracy (scenario-level, bootstrap CI) ===")
    for arm in ARMS:
        v = arm_vec(arm)
        point = v.mean()
        boots = []
        for _ in range(10000):
            idx = RNG.integers(0, len(v), len(v))
            boots.append(v[idx].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        boot_stats[arm] = (point, lo, hi, v)
        k = int(round(v.sum()))  # sum of per-scenario means = correct/total
        print(f"{arm:14s} mean={point:.4f} 95%CI=[{lo:.4f},{hi:.4f}] "
              f"wilson_exec={wilson(sum(int(r['correct']) for r in primary if r['arm']==arm), 180)}")

    # Wilcoxon scenario-level paired comparisons + Holm.
    print("\n=== scenario-level paired Wilcoxon ===")
    tests = []
    for a, b in COMPARISONS:
        x, y = arm_vec(a), arm_vec(b)
        d = x - y
        if np.allclose(d, 0):
            tests.append((a, b, 0.0, 1.0, d))
            print(f"{a} vs {b}: all zero differences, p=1")
        else:
            res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            tests.append((a, b, res.statistic, res.pvalue, d))
            print(f"{a} vs {b}: W={res.statistic:.1f} p_raw={res.pvalue:.4g} "
                  f"mean_delta={d.mean():+.4f}")
    # Holm
    order = sorted(range(len(tests)), key=lambda i: tests[i][3])
    m = len(tests)
    holm = {}
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * tests[i][3])
        adj = max(adj, prev)
        prev = adj
        holm[i] = adj
    print("Holm-adjusted:")
    for i, (a, b, W, p, d) in enumerate(tests):
        # bootstrap CI for paired delta
        boots = []
        for _ in range(10000):
            idx = RNG.integers(0, len(d), len(d))
            boots.append(d[idx].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {a} vs {b}: p_holm={holm[i]:.4g} delta95=[{lo:+.4f},{hi:+.4f}]")

    # Direction-specific
    print("\n=== direction-specific (scenario-level) ===")
    for cond in ("fresh_danger_stale_safe", "fresh_safe_stale_danger"):
        ss = [s for s in scenarios if meta[s][0] == cond]
        print(f"-- {cond} (n={len(ss)})")
        for arm in ARMS:
            v = np.array([per[s][arm] for s in ss])
            print(f"   {arm:14s} mean={v.mean():.4f} correct={int(round(v.sum()))}/{len(ss)*3}")

    # Consistency across seeds and structured consistency / abstention
    print("\n=== execution-level diagnostics ===")
    for arm in ARMS:
        rs = [r for r in primary if r["arm"] == arm]
        consistent = sum(1 for s in scenarios
                         if len({r["correct"] for r in rs if r["scenario_id"] == s}) == 1)
        abst = sum(r["decision"] == "ABSTAIN" for r in rs)
        cc = [reason_consistent(r["raw_answer"], r["decision"]) for r in rs]
        cc_vals = [c for c in cc if c is not None]
        print(f"{arm:14s} scenarios_consistent_across_seeds={consistent}/{len(scenarios)} "
              f"abstain={abst}/{len(rs)} decision-reason consistent="
              f"{sum(cc_vals)}/{len(cc_vals)}")

    # Diagnostic
    print("\n=== diagnostic (separate, 20 scenarios) ===")
    for arm in ARMS:
        rs = [r for r in diagnostic if r["arm"] == arm]
        if rs:
            print(f"{arm:14s} n={len(rs)} acc={sum(int(r['correct']) for r in rs)/len(rs):.4f}")


if __name__ == "__main__":
    main()
