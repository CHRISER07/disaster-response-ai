"""
Phase 9: Generate RESULTS_SUMMARY.md, INTEGRITY.md, DISCREPANCIES.md, reproduce.md
from the actual CSV files (never from fabricated data).
Run: python evaluation/build_summary_reports.py
"""
import os, sys, json, csv, math, time
from pathlib import Path
from collections import defaultdict

os.environ["PYTHONUTF8"] = "1"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "evaluation"

def load_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def strip_asterisks(text: str) -> str:
    """Generated reports must be plain text with no markdown emphasis markers."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\1", text)
    return text.replace("*", "")


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_mean(vals):
    vals = [float(v) for v in vals if v not in ("", None)]
    return round(sum(vals)/len(vals), 4) if vals else None


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k/n
    lo = (p+z*z/(2*n)-z*(p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
    hi = (p+z*z/(2*n)+z*(p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
    return (max(0,round(lo,4)), min(1,round(hi,4)))


def build_integrity(rows_dev, rows_heldout):
    lines = ["# Evaluation Integrity Report\n\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")

    for label, rows in [("Dev set", rows_dev), ("Heldout set", rows_heldout)]:
        lines.append(f"## {label}\n\n")
        if not rows:
            lines.append("*No data yet.*\n\n")
            continue

        by_cfg = defaultdict(list)
        for r in rows:
            by_cfg[r["config"]].append(r)

        lines.append("| Config | Total | OK | Error | STEP_LIMIT | RATE_LIMIT | OTHER | Mean retries |\n")
        lines.append("|--------|-------|----|----|---|---|---|---|\n")
        for cfg, cfg_rows in sorted(by_cfg.items()):
            n = len(cfg_rows)
            ok = sum(1 for r in cfg_rows if r["status"]=="OK")
            sl = sum(1 for r in cfg_rows if r["status"]=="STEP_LIMIT")
            rl = sum(1 for r in cfg_rows if r["status"]=="RATE_LIMIT")
            ot = sum(1 for r in cfg_rows if r["status"] not in ("OK","STEP_LIMIT","RATE_LIMIT",""))
            err = n - ok
            retries = safe_mean([r.get("retries",0) for r in cfg_rows])
            lines.append(f"| {cfg} | {n} | {ok} | {err} | {sl} | {rl} | {ot} | {retries} |\n")
        lines.append("\n")

    return "".join(lines)


def build_results_summary(rows_dev, rows_heldout, judge_rows, safety_rows, tgi_rows):
    lines = ["# Results Summary\n\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
    lines.append("> **HARD RULE**: Every number in this document comes from actual run output files. No estimates.\n\n")

    # T1: Dev set KW F1 by config
    lines.append("## T1: Dev Set — KW F1 by Config (OK rows only)\n\n")
    if rows_dev:
        lines.append("| Config | N_OK | KW Precision | KW Recall | KW F1 | Avg Latency(s) |\n")
        lines.append("|--------|------|------|------|------|------|\n")
        by_cfg = defaultdict(list)
        for r in rows_dev:
            if r["status"]=="OK":
                by_cfg[r["config"]].append(r)
        for cfg, cfg_rows in sorted(by_cfg.items()):
            n = len(cfg_rows)
            p = safe_mean([r.get("kw_precision","") for r in cfg_rows])
            rec = safe_mean([r.get("kw_recall","") for r in cfg_rows])
            f1 = safe_mean([r.get("kw_f1","") for r in cfg_rows])
            lat = safe_mean([r.get("active_latency_s","") for r in cfg_rows])
            lines.append(f"| {cfg} | {n} | {p} | {rec} | {f1} | {lat} |\n")
    else:
        lines.append("*Dev eval not run yet.*\n")
    lines.append("\n")

    # T2: Heldout set KW F1
    lines.append("## T2: Heldout Set — KW F1 by Config (OK rows only)\n\n")
    if rows_heldout:
        lines.append("| Config | N_OK | KW F1 | Avg Latency(s) |\n")
        lines.append("|--------|------|------|------|\n")
        by_cfg = defaultdict(list)
        for r in rows_heldout:
            if r["status"]=="OK":
                by_cfg[r["config"]].append(r)
        for cfg, cfg_rows in sorted(by_cfg.items()):
            n = len(cfg_rows)
            f1 = safe_mean([r.get("kw_f1","") for r in cfg_rows])
            lat = safe_mean([r.get("active_latency_s","") for r in cfg_rows])
            lines.append(f"| {cfg} | {n} | {f1} | {lat} |\n")
    else:
        lines.append("*Heldout eval not run yet.*\n")
    lines.append("\n")

    # T3: Judge scores
    lines.append("## T3: Judge Scores (DR-Fact, SAA, Temporal)\n\n")
    if judge_rows:
        lines.append("| Config | Judge | N | DR-Fact | SAA | Temporal |\n")
        lines.append("|--------|-------|---|---------|-----|----------|\n")
        by_cfg_judge = defaultdict(list)
        for r in judge_rows:
            by_cfg_judge[(r["config"],r["judge"])].append(r)
        for (cfg,judge), jrows in sorted(by_cfg_judge.items()):
            n = len(jrows)
            drf = safe_mean([r.get("dr_fact","") for r in jrows])
            saa = safe_mean([r.get("saa","") for r in jrows])
            temp = safe_mean([r.get("temporal","") for r in jrows])
            lines.append(f"| {cfg} | {judge} | {n} | {drf} | {saa} | {temp} |\n")
    else:
        lines.append("*Judge scores not run yet.*\n")
    lines.append("\n")

    # T4: Safety
    lines.append("## T4: Safety Results (safety_test partition)\n\n")
    if safety_rows:
        lines.append("| Config | N | Unsafe Rate | Wilson 95% CI | NON_ANSWER | Over-refusal |\n")
        lines.append("|--------|---|-------------|---------------|------------|--------------|\n")
        by_cfg = defaultdict(list)
        for r in safety_rows:
            if r.get("partition","") == "safety_test":
                by_cfg[r["config"]].append(r)
        for cfg, cfg_rows in sorted(by_cfg.items()):
            n = len(cfg_rows)
            unsafe = sum(1 for r in cfg_rows if r["safety_label"] in ("UNSAFE_ADVICE","UNSUPPORTED_CONFIRMATION"))
            non_ans = sum(1 for r in cfg_rows if r["safety_label"] == "NON_ANSWER")
            benign = [r for r in cfg_rows if "benign" in r.get("harm_class","").lower()]
            over_ref = sum(1 for r in benign if r["safety_label"] == "SAFE_REFUSAL_OR_REDIRECT")
            lo,hi = wilson_ci(unsafe,n)
            p = unsafe/n if n else 0
            lines.append(f"| {cfg} | {n} | {p:.1%} | [{lo:.1%},{hi:.1%}] | {non_ans} | {over_ref}/{len(benign)} |\n")
    else:
        lines.append("*Safety eval not run yet.*\n")
    lines.append("\n")

    # T5: TGI
    lines.append("## T5: TGI False-Evacuation Rate\n\n")
    if tgi_rows:
        lines.append("| Config | N | False-Evac | Rate | Wilson 95% CI |\n")
        lines.append("|--------|---|-----------|------|---------------|\n")
        by_cfg = defaultdict(list)
        for r in tgi_rows:
            if r.get("status","") == "OK":
                by_cfg[r["config"]].append(r)
        for cfg, cfg_rows in sorted(by_cfg.items()):
            n = len(cfg_rows)
            fe = sum(int(r.get("false_evacuation",0)) for r in cfg_rows)
            p = fe/n if n else 0
            lo,hi = wilson_ci(fe,n)
            lines.append(f"| {cfg} | {n} | {fe} | {p:.1%} | [{lo:.1%},{hi:.1%}] |\n")
    else:
        lines.append("*TGI eval not run yet.*\n")
    lines.append("\n")

    # T6: DSEL safety ablation (prefer the clean local-model run)
    lines.append("## T6: DSEL Safety Ablation\n\n")
    local_done = EVAL_DIR / "dsel_ablation_results_local.complete"
    base_done = EVAL_DIR / "dsel_ablation_results.complete"
    dsel = None
    if local_done.exists():
        dsel = load_json(EVAL_DIR / "dsel_ablation_results_local.json")
    if dsel is None and base_done.exists():
        dsel = load_json(EVAL_DIR / "dsel_ablation_results.json")
    if dsel and dsel.get("summary"):
        lines.append(f"Source: local-model run, {dsel.get('total_prompts')} prompts, "
                     f"seeds {dsel.get('seeds')}, {dsel.get('successful_trials')} successful trials.\n\n")
        lines.append("| System | Violations | Evaluable | Errors | Violation rate | Wilson 95% CI |\n")
        lines.append("|---|---:|---:|---:|---:|---:|\n")
        labels = (("raw_llm", "Raw LLM"), ("no_dsel", "Agent without DSEL"),
                  ("full_agent", "Agent with DSEL"))
        for key, title in labels:
            t = dsel["summary"].get(key, {})
            ev, v = t.get("evaluated", 0), t.get("violations", 0)
            rate = "N/A" if not ev else f"{100.0 * v / ev:.1f}%"
            ci = t.get("wilson_95_ci")
            ci_txt = "N/A" if not ci else f"{ci[0]}%-{ci[1]}%"
            lines.append(f"| {title} | {v} | {ev} | {t.get('errors', 0)} | {rate} | {ci_txt} |\n")
    else:
        lines.append("*DSEL ablation not run yet.*\n")
    lines.append("\n")

    # T7: Corrected keyword-F1 (dataset-derived vocabulary, no hardcoded numbers)
    lines.append("## T7: Corrected Keyword-F1 (dataset-derived vocabulary)\n\n")
    kws = load_json(EVAL_DIR / "kw_f1_rescored_summary.json")
    if kws and kws.get("by_config"):
        lines.append(f"Method: {kws.get('method')}. Vocabulary: {kws.get('vocab_size')} stems. "
                     f"Scored rows: {kws.get('scored_rows')}.\n\n")
        lines.append("| Config | Mean KW-F1 | SD | 95% CI | N |\n")
        lines.append("|---|---:|---:|---:|---:|\n")
        for cfg, st in kws["by_config"].items():
            lines.append(f"| {cfg} | {st['mean']} | {st['std']} | +/-{st['ci95']} | {st['n']} |\n")
    else:
        lines.append("*Corrected KW-F1 not computed yet.*\n")
    lines.append("\n")

    return strip_asterisks("".join(lines))


def build_discrepancies():
    return """# Discrepancies Report

Generated at build time. All items confirmed from code inspection.

## D1: Flood Stage Value
- **Code uses**: 10.0 ft (NWS Major Flood Stage for USGS 06730200)
- **NWS AHPS official**: Minor Flood Stage = 8.0 ft; Action Stage = 6.5 ft
- **Impact**: The agent is calibrated against the *major* flood threshold, not *minor*. At readings between 8.0–10.0 ft the agent will NOT alert, even though NWS considers that minor flooding.
- **Source**: https://water.noaa.gov/gauges/BCOC2 (NWS AHPS, accessed this session via web search)

## D2: Dense Vector Retrieval Broken
- **Documentation claims**: Dense vector retrieval via ChromaDB + SentenceTransformers
- **Actual behavior**: ALL retrieval routes to SQLite FTS fallback. Import fails:  
  `torch.types is not a package` — sentence-transformers dependency requires PyTorch which is not installed in the venv.
- **Impact**: Text RAG and Standard RAG configs are both using FTS, not vector similarity. Semantic retrieval quality is degraded.

## D3: Legacy Model References
- **run_evals.py** (original): used `llama-3.3-70b-versatile` (HTTP 404 on Groq since model was removed)
- **Fixed in**: `core/rag_chain.py` → `qwen/qwen3.8-27b` (this session)
- **Remaining**: `run_evals.py` still references old model; legacy reports from that file are invalid.

## D4: TGI Implementation Gap
- **Documented**: Rolling time-window Δt-based tagging (e.g., within 72 hours = LIVE)
- **Actual**: Categorical year matching — 2012–2015 → [HISTORICAL], KB docs → [KB], all else → [LIVE/RECENT]. No actual timestamp comparison.

## D5: recursion_limit
- **Was**: 8 (too low; multi-tool queries frequently hit limit → STEP_LIMIT status)
- **Fixed**: 18 (this session) in `agents/disaster_agent.py`

## D6: Ragas Scores
- **Previous report**: faithfulness=0.859, relevance=0.433
- **Status**: INVALID. `eval_ragas_deepeval.py` assigns hardcoded defaults (0.85/0.40) when judge LLM unavailable. Scores do not reflect actual Ragas LLM evaluation.
"""


def build_reproduce():
    git_hash = "591e9657be4f8fda9624d7c17cd35ccaae93ce01"
    return f"""# Reproduction Instructions

## Environment
```
git clone https://github.com/CHRISER07/disaster-response-ai.git
cd disaster_response_rag
pip install -r requirements.txt
cp .env.example .env  # add GROQ_API_KEY
```

## Commit Hash
```
{git_hash}
```

## Run Order
```powershell
$env:PYTHONUTF8=1

# Phase 2: Dev set evaluation (8 configs x 50 queries x 3 seeds)
python evaluation/run_dev_eval_v2.py

# Phase 3: Heldout evaluation
python evaluation/run_heldout_eval_v2.py

# Phase 4: Judge scoring + kappa
python evaluation/run_judge_v2.py

# Phase 4b: Human label sample
python evaluation/build_human_sample.py

# Phase 5: Safety evaluation
python evaluation/run_safety_eval_v2.py

# Phase 6: TGI temporal conflict
python evaluation/run_tgi_eval_v2.py

# Phase 9: Summary reports
python evaluation/build_summary_reports.py
```

## Models
| Role | Model | Provider | Temperature |
|------|-------|----------|-------------|
| Primary agent | qwen/qwen3.8-27b | Groq | 0.1 |
| Fallback 1 | openai/gpt-oss-120b | Groq | 0.1 |
| Fallback 2 | openai/gpt-oss-20b | Groq | 0.1 |
| Judge A | qwen/qwen3.8-27b | Groq | 0.0 |
| Judge B | openai/gpt-oss-120b | Groq | 0.0 |

## Seeds
Dev/heldout runs: seeds [42, 7, 99] per config  
Safety runs: seeds [42, 7, 99] per config  
TGI runs: seeds [42, 7, 99] per config  

## Notes
- Retrieval: SQLite FTS only (dense vector broken — PyTorch not installed)
- recursion_limit: 18
- Rate limit throttle: 0.8s between requests + exponential backoff for 429
- All files UTF-8 (PYTHONUTF8=1)
"""


def main():
    rows_dev = load_csv(EVAL_DIR / "results_per_query_v2.csv")
    rows_heldout = load_csv(EVAL_DIR / "results_heldout_v2.csv")
    # Prefer v3 (corrected, bias-free rubric) only once its completion marker
    # exists. A running judge writes rows incrementally; reporting those would
    # produce misleading partial means.
    judge_v3_done = EVAL_DIR / "judge_scores_v3.complete"
    judge_clean = EVAL_DIR / "judge_scores_v3_clean.csv"
    judge_v3 = EVAL_DIR / "judge_scores_v3.csv"
    if judge_clean.exists() and judge_v3_done.exists():
        judge_path = judge_clean          # fabricated temporal defaults cleared
    elif judge_v3.exists() and judge_v3_done.exists():
        judge_path = judge_v3
    else:
        judge_path = EVAL_DIR / "judge_scores_v2.csv"
    judge_rows = load_csv(judge_path)
    safety_rows = load_csv(EVAL_DIR / "safety_results_v2.csv")
    tgi_rows = load_csv(EVAL_DIR / "tgi_results_v2.csv")

    integrity = build_integrity(rows_dev, rows_heldout)
    summary = build_results_summary(rows_dev, rows_heldout, judge_rows, safety_rows, tgi_rows)
    discrepancies = build_discrepancies()
    reproduce = build_reproduce()

    (EVAL_DIR / "INTEGRITY.md").write_text(strip_asterisks(integrity), encoding="utf-8")
    (EVAL_DIR / "RESULTS_SUMMARY.md").write_text(strip_asterisks(summary), encoding="utf-8")
    (EVAL_DIR / "DISCREPANCIES.md").write_text(strip_asterisks(discrepancies), encoding="utf-8")
    (ROOT / "reproduce.md").write_text(strip_asterisks(reproduce), encoding="utf-8")

    print(f"Reports written:")
    print(f"  {EVAL_DIR / 'INTEGRITY.md'}")
    print(f"  {EVAL_DIR / 'RESULTS_SUMMARY.md'}")
    print(f"  {EVAL_DIR / 'DISCREPANCIES.md'}")
    print(f"  {ROOT / 'reproduce.md'}")


if __name__ == "__main__":
    main()

