"""Guard against stale/mixed claims in IEEE_FINAL.tex.

Run before and after any paper edit:
    python evaluation/verify_paper_integrity.py

Checks that (a) stale claims are absent, (b) the single grounding table has no
duplicate controlled-run table, and (c) numbers that appear in the paper match
the committed artifacts they are attributed to.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "IEEE_FINAL.tex"

STALE = [
    "0/93", "0.4949", "p < 10^{-4}", "unbreachable", "immutable",
    "24 query pairs", "Optimal", "25/tier, 0 errors",
    "qwen/qwen2.5-instruct", "deterministic decoding",
    "\\textbf{**", "figure\\*", "tabular\\*", "addtolength{\\textheight}",
    "21.4\\%", "4.8\\%", "0.8\\%", "5 of 6", "proceed with caution",
]


def main() -> int:
    s = PAPER.read_text(encoding="utf-8")
    problems: list[str] = []

    for token in STALE:
        if token in s:
            problems.append(f"stale token present: {token}")

    if "tab:fresh" in s:
        problems.append("duplicate controlled-run grounding table (tab:fresh) present")

    # Table III must stay the published configuration; controlled numbers must
    # not be silently mixed into it.
    main_start = s.find("\\label{tab:main}")
    if main_start < 0:
        problems.append("tab:main missing")
    else:
        block = s[main_start: main_start + 1800]
        for token in ["0.1110", "0.0928", "llama3.1"]:
            if token in block:
                problems.append(f"controlled-run value mixed into Table III: {token}")

    # Numbers that must appear with their verified source values.
    required = {
        "0.0915": "published Full Agent held-out Exp-F1",
        "0.0657": "published StaticRAG held-out Exp-F1",
        "0/36": "TGI false evacuations",
        "3/36": "no-TGI missed evacuations",
        "72/72": "TGI accuracy",
    }
    for token, meaning in required.items():
        if token not in s:
            problems.append(f"missing verified value ({meaning}): {token}")

    # Derive the required DSEL tokens from the authoritative artifact rather
    # than hardcoding a result. A stale or changed CSV therefore fails the guard.
    dsel_path = ROOT / "evaluation" / "dsel_ablation_raw_qwen25_3b.csv"
    dsel = list(csv.DictReader(open(dsel_path, encoding="utf-8")))
    if len(dsel) != 378 or any(x["status"].upper() != "OK" for x in dsel):
        problems.append(f"DSEL artifact incomplete: {len(dsel)} rows")
    models = {x["model_used"] for x in dsel}
    if models != {"qwen2.5:3b"}:
        problems.append(f"DSEL artifact generator mismatch: {models}")
    for cfg, meaning in (("Raw_LLM", "raw LLM"),
                         ("Agent_NoDSEL", "no-DSEL agent"),
                         ("Full_Agent_DSEL", "DSEL agent")):
        rows = [x for x in dsel if x["config"] == cfg]
        if len(rows) != 126:
            problems.append(f"DSEL {cfg} row count {len(rows)}")
            continue
        violations = sum(int(x["is_violation"]) for x in rows)
        token = f"{violations} ({100 * violations / len(rows):.1f}\\%)"
        if token not in s:
            problems.append(f"missing DSEL value ({meaning}): {token}")

    if problems:
        print("FAIL")
        for p in problems:
            print(" -", p)
        return 1
    print("PASS: paper integrity checks OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
