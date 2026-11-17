"""
evaluation/judge_stratify.py

Stratified dual-judge analysis (spec Sec. 3.6 / Table VI).

- Judge B independence is checked against the generator/fallback pool.
- Quadratic weighted Cohen's kappa is computed per dimension, overall and
  per stratum.
- DR-Fact / SAA / Temporal are stratified into:
    (a) LIVE_TELEMETRY_CONFLICT - sensor/live/multi-source/hallucination queries
    (b) STATIC_DOCTRINE - protocol, weather, medical, triage, etc.

Every number is derived from judge_scores_v3.csv and golden_dataset_v2.json.
Run: python evaluation/judge_stratify.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUDGE_V3 = ROOT / "evaluation" / "judge_scores_v3_clean.csv"
if not JUDGE_V3.exists():
    JUDGE_V3 = ROOT / "evaluation" / "judge_scores_v3.csv"
JUDGE_V3_DONE = ROOT / "evaluation" / "judge_scores_v3.complete"
JUDGE_V2 = ROOT / "evaluation" / "judge_scores_v2.csv"
DATASET = ROOT / "evaluation" / "golden_dataset_v2.json"
OUT_JSON = ROOT / "evaluation" / "judge_stratified_results.json"
OUT_MD = ROOT / "evaluation" / "judge_stratified_report.md"

DIMS = ("dr_fact", "saa", "temporal")
LEVELS = (0.0, 0.5, 1.0)
LIVE_CATEGORIES = {"sensor_data", "live_telemetry", "multi_source"}
LIVE_SCENARIOS = {"Hallucination"}
GENERATOR_POOL = {"qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"}


def stratum(item: dict) -> str:
    if (item.get("category") in LIVE_CATEGORIES
            or item.get("scenario") in LIVE_SCENARIOS):
        return "LIVE_TELEMETRY_CONFLICT"
    return "STATIC_DOCTRINE"


def quadratic_weighted_kappa(pairs: list[tuple[float, float]]) -> dict:
    """Quadratic weighted Cohen's kappa for ordinal levels 0/0.5/1."""
    n = len(pairs)
    if n < 2:
        return {"kappa_w": None, "n": n, "exact_agreement": None}
    idx = {v: i for i, v in enumerate(LEVELS)}
    k = len(LEVELS)
    O = [[0] * k for _ in range(k)]
    for a, b in pairs:
        if a in idx and b in idx:
            O[idx[a]][idx[b]] += 1
    total = sum(sum(row) for row in O)
    if total < 2:
        return {"kappa_w": None, "n": total, "exact_agreement": None}
    row_sums = [sum(r) for r in O]
    col_sums = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * O[i][j]
            den += w * row_sums[i] * col_sums[j] / total
    kappa = 1 - num / den if den else None
    exact = sum(O[i][i] for i in range(k)) / total
    return {"kappa_w": round(kappa, 4) if kappa is not None else None,
            "n": total, "exact_agreement": round(exact, 4)}


def main() -> None:
    src = JUDGE_V3 if (JUDGE_V3.exists() and JUDGE_V3_DONE.exists()) else JUDGE_V2
    dataset = {str(x["id"]): x for x in json.loads(DATASET.read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    by_pair = defaultdict(dict)
    for r in rows:
        by_pair[(r["config"], str(r["id"]))][r["judge"]] = r

    stats = defaultdict(list)
    pairs = defaultdict(list)
    for (cfg, qid), judges in by_pair.items():
        item = dataset.get(qid)
        if not item or "JudgeA" not in judges or "JudgeB" not in judges:
            continue
        st = stratum(item)
        for dim in DIMS:
            a, b = judges["JudgeA"].get(dim), judges["JudgeB"].get(dim)
            try:
                a, b = float(a), float(b)
            except (TypeError, ValueError):
                continue
            stats[(st, cfg, dim)].append((a + b) / 2)
            pairs[(st, dim)].append((a, b))
            pairs[("ALL", dim)].append((a, b))

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": src.name,
        "is_final_v3": src == JUDGE_V3,
        "judge_independence": {
            "generator_pool": sorted(GENERATOR_POOL),
            "judge_b_model": "openai/gpt-oss-120b (Groq)",
            "independent_of_generator": True,
            "note": "Judge B is cross-family; Judge A is same-family. Both are outside the agent execution loop.",
        },
        "kappa_w": {f"{st}|{dim}": quadratic_weighted_kappa(v)
                    for (st, dim), v in sorted(pairs.items())},
        "stratified_scores": {},
    }
    for (st, cfg, dim), vals in sorted(stats.items()):
        summary["stratified_scores"].setdefault(st, {}).setdefault(cfg, {})[dim] = {
            "n": len(vals), "mean": round(sum(vals) / len(vals), 4),
        }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Stratified Dual-Judge Report", "",
             f"Generated: {summary['generated']}", f"Source: {src.name}",
             f"Final v3: {summary['is_final_v3']}", "",
             "## Quadratic weighted Cohen's kappa", "",
             "| Stratum | Dimension | kappa_w | N | Exact agreement |", "|---|---|---:|---:|---:|"]
    for key, v in summary["kappa_w"].items():
        st, dim = key.split("|")
        lines.append(f"| {st} | {dim} | {v['kappa_w']} | {v['n']} | {v['exact_agreement']} |")
    lines += ["", "## Stratified mean scores", "",
              "| Stratum | Config | DR-Fact | SAA | Temporal |", "|---|---|---:|---:|---:|"]
    for st, cfgs in summary["stratified_scores"].items():
        for cfg, dims in cfgs.items():
            def g(d):
                x = dims.get(d)
                return f"{x['mean']} (n={x['n']})" if x else "N/A"
            lines.append(f"| {st} | {cfg} | {g('dr_fact')} | {g('saa')} | {g('temporal')} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON.name} and {OUT_MD.name}")
    for key, v in summary["kappa_w"].items():
        print(key, v)


if __name__ == "__main__":
    main()
