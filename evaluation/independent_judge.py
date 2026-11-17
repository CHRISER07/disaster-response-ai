"""
evaluation/independent_judge.py

INDEPENDENT CROSS-FAMILY JUDGE
Evaluates agent responses using openai/gpt-oss-120b (circuit-breaker model)
as an independent judge, different from the primary generator qwen/qwen3.8-27b.

Scores three metrics:
  - DR-Fact: Does the answer adhere to documented disaster-response ground truth?
  - SAA: Is the answer situationally aware and gives actionable guidance?
  - Temporal Accuracy: Does the answer correctly distinguish LIVE vs HISTORICAL data?

Also computes Cohen's kappa agreement between model-judge and a deterministic
keyword heuristic baseline on a 25-item cross-validation set.

Usage:
  python evaluation/independent_judge.py --sample 25
  python evaluation/independent_judge.py --from-report evaluation/ragas_evaluation_report.json
"""

import os, sys, json, time, math, argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(override=False)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "evaluation", "golden_dataset_200q.json")
REPORT_PATH  = os.path.join(PROJECT_ROOT, "evaluation", "independent_judge_report.json")

JUDGE_PROMPT_DRFACT = """You are an expert disaster-response evaluator. Rate the following AI answer for factual correctness specifically against established emergency management doctrine (FEMA ICS, CDC Guidelines, USGS protocols).

QUESTION: {query}

AI ANSWER: {answer}

GROUND TRUTH: {ground_truth}

Score from 0.0 to 1.0 where:
1.0 = All factual claims in the answer are consistent with official doctrine and ground truth
0.5 = Some claims match; some are vague or missing key protocol details
0.0 = Contains factually incorrect statements or directly contradicts official guidance

Respond with ONLY a decimal number between 0.0 and 1.0, then a single sentence of reasoning.
Format exactly: SCORE: 0.XX | REASON: <one sentence>"""

JUDGE_PROMPT_SAA = """You are an emergency operations evaluator. Assess whether the following AI response to a disaster query provides clear, unambiguous, and immediately actionable situational guidance.

QUESTION: {query}

AI ANSWER: {answer}

Score from 0.0 to 1.0 where:
1.0 = Response gives clear actionable steps (specific gage readings, evacuation routes, threshold numbers, authority contacts)
0.5 = Response is relevant but vague; lacks specific numbers or action steps
0.0 = Response is non-actionable, evasive, or potentially misleading in an emergency context

Respond with ONLY a decimal number between 0.0 and 1.0, then a single sentence of reasoning.
Format exactly: SCORE: 0.XX | REASON: <one sentence>"""

JUDGE_PROMPT_TEMPORAL = """You are a temporal reasoning evaluator for a disaster AI system. Assess whether the AI correctly distinguishes between LIVE/CURRENT data and HISTORICAL disaster records (from 2012-2013 events like Colorado Floods or Philippines typhoons).

QUESTION: {query}

AI ANSWER: {answer}

Score from 0.0 to 1.0 where:
1.0 = The answer explicitly labels live vs historical sources; does not mistake 2013 archives for current conditions
0.5 = Some temporal distinction made but incomplete or implicit
0.0 = Answer conflates 2013 historical events with current conditions, or provides no temporal context

Respond with ONLY a decimal number between 0.0 and 1.0, then a single sentence of reasoning.
Format exactly: SCORE: 0.XX | REASON: <one sentence>"""


def _parse_judge_score(response_text: str) -> tuple[float, str]:
    """Parse 'SCORE: 0.XX | REASON: ...' format from judge response."""
    try:
        parts = response_text.strip().split("|")
        score_part = parts[0].strip()
        reason_part = parts[1].strip() if len(parts) > 1 else ""
        score = float(score_part.replace("SCORE:", "").strip())
        reason = reason_part.replace("REASON:", "").strip()
        return min(1.0, max(0.0, score)), reason
    except Exception:
        # Fallback: try to find first float in response
        import re
        nums = re.findall(r"\b0\.\d+\b|\b1\.0\b|\b0\b|\b1\b", response_text)
        if nums:
            return float(nums[0]), response_text[:100]
        return 0.5, f"Parse error: {response_text[:80]}"


def _deterministic_score_baseline(answer: str, ground_truth: str, expected_keywords: list) -> dict:
    """
    Keyword-based deterministic heuristic to enable Cohen's kappa with judge.
    Returns binary labels (high=1 / low=0) for DR-Fact and SAA.
    """
    ans_lower = answer.lower()
    gt_lower = (ground_truth or "").lower()
    
    # DR-Fact heuristic: keyword overlap with ground truth
    gt_words = set(w for w in gt_lower.split() if len(w) > 4)
    ans_words = set(ans_lower.split())
    overlap = len(gt_words & ans_words) / max(len(gt_words), 1)
    drfact_binary = 1 if overlap > 0.25 else 0
    
    # SAA heuristic: actionable words + numbers present
    import re
    has_numbers = bool(re.search(r"\d+\.?\d*\s*(?:ft|feet|m|km|mph|mm|Â°C|Â°F|cfs|mg)", answer))
    actionable_words = ["evacuate", "shelter", "avoid", "call", "do not", "never", "contact",
                        "proceed", "move", "stay", "report", "monitor"]
    has_action = any(w in ans_lower for w in actionable_words)
    saa_binary = 1 if (has_action and has_numbers) else (1 if has_action else 0)
    
    return {
        "drfact_binary": drfact_binary,
        "saa_binary": saa_binary,
        "keyword_overlap": round(overlap, 3),
    }


def cohen_kappa(judge_labels: list, heuristic_labels: list) -> float:
    """Cohen's kappa for binary agreement between judge and heuristic."""
    n = len(judge_labels)
    if n == 0:
        return 0.0
    agreed = sum(j == h for j, h in zip(judge_labels, heuristic_labels))
    po = agreed / n
    # Expected agreement
    j_pos = sum(judge_labels) / n
    h_pos = sum(heuristic_labels) / n
    pe = j_pos * h_pos + (1 - j_pos) * (1 - h_pos)
    return round((po - pe) / max(1 - pe, 1e-9), 4)


def _invoke_with_retry(judge, prompt: str, max_retries: int = 2):
    """Invoke judge with automatic fallback and minimal backoff."""
    for attempt in range(max_retries):
        try:
            return judge.invoke(prompt)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait_s = 5 if attempt == 0 else 10
                print(f"      [Rate Limit 429] Waiting {wait_s}s...", flush=True)
                time.sleep(wait_s)
            else:
                raise e
    raise RuntimeError(f"Max retries exceeded")


def run_independent_judge(n_samples: int = 25, from_report: str = None):
    print("=" * 70, flush=True)
    print("  INDEPENDENT CROSS-FAMILY JUDGE (GPT-OSS-120B / 20B)", flush=True)
    print("  Scoring DR-Fact, SAA, Temporal Accuracy", flush=True)
    print("=" * 70, flush=True)

    from langchain_groq import ChatGroq
    j1 = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0, max_retries=0)
    j2 = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0, max_retries=1)
    judge = j1.with_fallbacks([j2])
    print(f"Judge model: openai/gpt-oss-120b (with gpt-oss-20b fallback)", flush=True)

    if from_report and os.path.exists(from_report):
        # Score already-completed agent answers from the eval report
        with open(from_report) as f:
            report_data = json.load(f)
        with open(DATASET_PATH) as f:
            dataset = json.load(f)
        dataset_by_id = {str(item["id"]): item for item in dataset}

        completed = report_data.get("query_results", [])[:n_samples]
        print(f"Scoring {len(completed)} answers from existing report...")
    else:
        print("No report provided; running agent live...")
        from agents.disaster_agent import run_agent
        with open(DATASET_PATH) as f:
            dataset = json.load(f)
        # Stratified sample: take n_samples evenly across scenarios
        scenarios = {}
        for item in dataset:
            s = item.get("scenario", "unknown")
            scenarios.setdefault(s, []).append(item)
        per_scenario = max(1, n_samples // len(scenarios))
        sampled = []
        for s, items in scenarios.items():
            sampled.extend(items[:per_scenario])
        sampled = sampled[:n_samples]

        completed = []
        for i, item in enumerate(sampled, 1):
            print(f"[{i}/{len(sampled)}] Running: {item['query'][:50]}...", flush=True)
            res = run_agent(item["query"], thread_id=f"judge_run_{item['id']}")
            completed.append({
                "query_id": str(item["id"]),
                "query": item["query"],
                "answer_preview": res.get("answer", ""),
            })
            time.sleep(1)
        dataset_by_id = {str(item["id"]): item for item in dataset}

    drfact_scores, saa_scores, temporal_scores = [], [], []
    judge_drfact_labels, heuristic_drfact_labels = [], []
    judge_saa_labels, heuristic_saa_labels = [], []
    scored_results = []

    for r in completed:
        qid = str(r.get("query_id", ""))
        query = r.get("query", "")
        answer = r.get("answer_preview", "")
        item = dataset_by_id.get(qid, {})
        ground_truth = item.get("ground_truth_answer", item.get("ground_truth", ""))
        expected_kw = item.get("expected_keywords", [])

        print(f"\n  Judging QID={qid} | {query[:45]}...", flush=True)

        # DR-Fact
        drfact, drfact_reason = (0.5, "skipped")
        try:
            resp = _invoke_with_retry(judge, JUDGE_PROMPT_DRFACT.format(query=query, answer=answer[:600], ground_truth=ground_truth[:400]))
            drfact, drfact_reason = _parse_judge_score(resp.content)
        except Exception as e:
            drfact_reason = f"Error: {e}"
        drfact_scores.append(drfact)
        judge_drfact_labels.append(1 if drfact >= 0.6 else 0)

        # SAA
        saa, saa_reason = (0.5, "skipped")
        try:
            resp = _invoke_with_retry(judge, JUDGE_PROMPT_SAA.format(query=query, answer=answer[:600]))
            saa, saa_reason = _parse_judge_score(resp.content)
        except Exception as e:
            saa_reason = f"Error: {e}"
        saa_scores.append(saa)
        judge_saa_labels.append(1 if saa >= 0.6 else 0)

        # Temporal
        temporal, temporal_reason = (0.5, "skipped")
        try:
            resp = _invoke_with_retry(judge, JUDGE_PROMPT_TEMPORAL.format(query=query, answer=answer[:600]))
            temporal, temporal_reason = _parse_judge_score(resp.content)
        except Exception as e:
            temporal_reason = f"Error: {e}"
        temporal_scores.append(temporal)

        # Deterministic baseline for kappa
        baseline = _deterministic_score_baseline(answer, ground_truth, expected_kw)
        heuristic_drfact_labels.append(baseline["drfact_binary"])
        heuristic_saa_labels.append(baseline["saa_binary"])

        print(f"    DR-Fact={drfact:.2f} | SAA={saa:.2f} | Temporal={temporal:.2f}", flush=True)

        scored_results.append({
            "query_id": qid,
            "query": query[:80],
            "judge_drfact": drfact,
            "judge_drfact_reason": drfact_reason,
            "judge_saa": saa,
            "judge_saa_reason": saa_reason,
            "judge_temporal": temporal,
            "judge_temporal_reason": temporal_reason,
            "heuristic_baseline": baseline,
        })
        time.sleep(2)

    def _ci(vals):
        n = len(vals)
        if n < 2:
            return round(vals[0] if vals else 0.0, 4), 0.0, 0.0
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
        ci95 = 1.96 * std / math.sqrt(n)
        return round(mean, 4), round(std, 4), round(ci95, 4)

    drfact_mean, drfact_std, drfact_ci = _ci(drfact_scores)
    saa_mean, saa_std, saa_ci = _ci(saa_scores)
    temporal_mean, temporal_std, temporal_ci = _ci(temporal_scores)
    kappa_drfact = cohen_kappa(judge_drfact_labels, heuristic_drfact_labels)
    kappa_saa = cohen_kappa(judge_saa_labels, heuristic_saa_labels)

    print("\n" + "=" * 70)
    print("  INDEPENDENT JUDGE -- RESULTS")
    print("=" * 70)
    print(f"  Queries Judged                : {len(drfact_scores)}")
    print(f"  Judge Model                   : openai/gpt-oss-120b")
    print(f"  DR-Fact (Factual Correctness) : {drfact_mean} +/- {drfact_std} (95% CI: +/-{drfact_ci})")
    print(f"  SAA (Situational Actionability): {saa_mean} +/- {saa_std} (95% CI: +/-{saa_ci})")
    print(f"  Temporal Accuracy             : {temporal_mean} +/- {temporal_std} (95% CI: +/-{temporal_ci})")
    print(f"  Cohen's kappa (DR-Fact vs heuristic): {kappa_drfact}")
    print(f"  Cohen's kappa (SAA vs heuristic)    : {kappa_saa}")
    print("=" * 70)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": "openai/gpt-oss-120b",
        "generator_model": "qwen/qwen3.8-27b",
        "n_queries_judged": len(drfact_scores),
        "aggregate": {
            "dr_fact_mean": drfact_mean,
            "dr_fact_std": drfact_std,
            "dr_fact_ci95": drfact_ci,
            "saa_mean": saa_mean,
            "saa_std": saa_std,
            "saa_ci95": saa_ci,
            "temporal_accuracy_mean": temporal_mean,
            "temporal_accuracy_std": temporal_std,
            "temporal_accuracy_ci95": temporal_ci,
            "cohens_kappa_drfact": kappa_drfact,
            "cohens_kappa_saa": kappa_saa,
        },
        "per_query": scored_results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved to: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--from-report", type=str, default=None)
    args = parser.parse_args()
    run_independent_judge(n_samples=args.sample, from_report=args.from_report)
