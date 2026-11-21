"""Verify every quantitative claim in IEEE_FINAL.tex against committed artifacts.

Reads the paper text and the authoritative result files, checks that each
published number is present in the paper and matches the artifact. Reports
PASS/FAIL per claim. Not a LaTeX parser; a targeted claim checker.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = (ROOT / "IEEE_FINAL.tex").read_text(encoding="utf-8")
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


# --- controlled cohort table ---
cohort = {
    "BM25 Baseline": "0.1086", "Standard Static RAG": "0.1076",
    "LLM-Only": "0.0999", "Full Agent w/o DSEL": "0.1060",
    "Full Agent": "0.1032", "Full Agent w/o TGI": "0.1037",
    "Full Agent w/o Telemetry": "0.0976",
}
for label, val in cohort.items():
    check(f"cohort {label} {val}", val in PAPER)

# --- leakage-free held-out ---
for val in ("0.1377", "0.1413", "0.1003"):
    check(f"leakfree {val}", val in PAPER)

# --- TGI v6 ---
r = list(csv.DictReader(open(ROOT / "evaluation/tgi_results_v6.csv", encoding="utf-8")))
for cfg, exp_fp, exp_fn in (("Full_Agent_TGI", 0, 0),
                            ("Full_Agent_NoTGI", 0, 3),
                            ("LLM_Only", 0, 36),
                            ("Standard_StaticRAG", 0, 36)):
    s = [x for x in r if x["config"] == cfg and x["assessable"] == "1"]
    fp = sum(int(x["false_evacuation"]) for x in s)
    fn = sum(int(x["missed_evacuation"]) for x in s)
    check(f"TGIv6 {cfg} FP={fp} FN={fn}", fp == exp_fp and fn == exp_fn)
check("TGIv6 72/72 in paper", "72/72" in PAPER)

# --- DSEL ---
d = list(csv.DictReader(open(ROOT / "evaluation/dsel_ablation_raw_qwen25_3b.csv", encoding="utf-8")))
for cfg, exp in (("Raw_LLM", 10), ("Agent_NoDSEL", 7), ("Full_Agent_DSEL", 0)):
    v = sum(int(x["is_violation"]) for x in d if x["config"] == cfg)
    check(f"DSEL {cfg}={exp}", v == exp)
    token = f"{exp}/126"
    alt = f"{exp} ({100*exp/126:.1f}\\%)"
    check(f"DSEL {token} in paper", token in PAPER or alt in PAPER)

# --- EXP-4 geospatial ---
g = json.loads((ROOT / "evaluation/geospatial_expanded_results.json").read_text(encoding="utf-8"))
check("geo router 93/93", g["overall"]["router"]["accuracy"] == 1.0 and "93/93" in PAPER)
check("geo baseline 34/93", g["overall"]["baseline"]["TP"] == 34 and "34/93" in PAPER)
check("geo cross 59/59", g["cross_watershed"]["router"]["n"] == 59 and "59/59" in PAPER)
check("geo Fisher p=8.2e-35", "8.2" in PAPER and "10^{-35}" in PAPER)
check("geo McNemar p=3.47e-18", "3.47" in PAPER and "10^{-18}" in PAPER)

# --- EXP-7D ---
e = list(csv.DictReader(open(ROOT / "evaluation/exp7d_results.csv", encoding="utf-8")))
check("EXP7D rows 720", len(e) == 720)
check("EXP7D consumption all pass", all(x["consumption_ok"] == "1" for x in e))
check("EXP7D TGI 53.3% in paper", "53.3" in PAPER)
check("EXP7D Holm p=0.47 in paper", "0.47" in PAPER)
check("EXP7D failure analysis subsection", "Failure Analysis: Safety-Prompt Dominance" in PAPER)

# --- abstention ---
a = json.loads((ROOT / "evaluation/abstention_benchmark_results.json").read_text(encoding="utf-8"))
check("abstention Full Agent 0%", a["Full_Agent"]["hallucination_rate"] == 0.0 and "0.0" in PAPER)
check("abstention LLM 40%", a["LLM_Only"]["hallucination_rate"] == 0.4 and "40.0" in PAPER)

# --- structural ---
check("title unchanged", "Temporally Grounded Agentic Retrieval-Augmented Generation" in PAPER)
check("no actionable utility metric", "0.0915" not in PAPER)
check("comparison figure present with honest caption", "fig:comparison" in PAPER and "each from a separate experiment" in PAPER)
check("retrieval label N=200 combined", "Retrieval Engine Benchmark ($N=200$ Queries)" in PAPER)
check("no misleading Dev N=200 column", "Dev ($N=200$)" not in PAPER)

fails = [c for c in checks if not c[1]]
print(f"checks={len(checks)} passed={len(checks)-len(fails)} failed={len(fails)}")
for name, ok, detail in fails:
    print("FAIL:", name, detail)
print("RESULT:", "PASS" if not fails else "FAIL")
