"""
evaluation/keyword_f1_scorer.py

DETERMINISTIC KEYWORD F1 SCORER
Uses the `expected_keywords` field from golden_dataset_200q.json as gold standard.
This is reproducible, computation-free (no LLM), and fully auditable.

Differs from the naive approach: uses stemmed matching, not exact substring,
to avoid penalizing morphological variation (e.g., "evacuate" vs "evacuation").

Usage:
  python evaluation/keyword_f1_scorer.py
  python evaluation/keyword_f1_scorer.py --input evaluation/ragas_evaluation_report.json
"""

import os, sys, json, re, math, argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "evaluation", "golden_dataset_200q.json")
REPORT_PATH  = os.path.join(PROJECT_ROOT, "evaluation", "kw_f1_report.json")


def _stem(word: str) -> str:
    """Minimal English stemmer for disaster-domain keywords."""
    word = word.lower().strip()
    for suffix in ("ations", "ation", "ating", "ated", "ing", "ion", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)]
    return word


def compute_kw_f1(answer: str, expected_keywords: list[str]) -> dict:
    """
    Compute KW-Precision, KW-Recall, and KW-F1 between answer and gold keywords.
    Stemmed matching for morphological robustness.
    """
    if not answer or not expected_keywords:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": [], "missed": []}

    ans_tokens = set(_stem(t) for t in re.findall(r"\b\w+\b", answer))
    gold_stems = [_stem(k) for k in expected_keywords]
    # For multi-word keywords: check stem of first word OR full phrase
    answer_lower = answer.lower()

    matched = []
    missed = []
    for kw, stemmed in zip(expected_keywords, gold_stems):
        kw_lower = kw.lower()
        # Match if: phrase present, first-word stem in answer tokens, or keyword stem in answer tokens
        hit = (kw_lower in answer_lower) or (stemmed in ans_tokens) or (
            len(kw.split()) > 1 and _stem(kw.split()[0]) in ans_tokens and _stem(kw.split()[-1]) in ans_tokens
        )
        if hit:
            matched.append(kw)
        else:
            missed.append(kw)

    # Count answer words that are relevant (for precision)
    gold_stem_set = set(gold_stems)
    n_relevant_in_answer = sum(1 for t in ans_tokens if t in gold_stem_set)

    precision = n_relevant_in_answer / max(len(ans_tokens), 1)
    recall    = len(matched) / max(len(expected_keywords), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "matched":   matched,
        "missed":    missed,
        "n_gold_keywords": len(expected_keywords),
    }


def compute_confidence_interval(values: list[float]) -> dict:
    """95% CI using normal approximation: 1.96 * sigma / sqrt(n)"""
    n = len(values)
    if n < 2:
        return {"mean": values[0] if values else 0.0, "std": 0.0, "ci95": 0.0, "n": n}
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    ci95 = 1.96 * std / math.sqrt(n)
    return {"mean": round(mean, 4), "std": round(std, 4), "ci95": round(ci95, 4), "n": n}


def score_from_ragas_report(ragas_report_path: str) -> dict:
    """
    Re-score completed eval runs in ragas_evaluation_report.json
    by matching query_id back to the dataset and computing proper KW-F1.
    """
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    dataset_by_id = {str(item["id"]): item for item in dataset}

    with open(ragas_report_path) as f:
        report = json.load(f)

    results = report.get("query_results", [])
    kw_f1_scores = []
    scenario_scores = {}
    scored_results = []

    print(f"\nRe-scoring {len(results)} completed queries with verified KW-F1...")
    for r in results:
        qid = str(r.get("query_id", r.get("id", "")))
        item = dataset_by_id.get(qid)
        if not item:
            continue

        expected_keywords = item.get("expected_keywords", [])
        answer = r.get("answer_preview", "")
        # Answer preview might be truncated — score what we have
        kw = compute_kw_f1(answer, expected_keywords)

        scenario = item.get("scenario", "unknown")
        if scenario not in scenario_scores:
            scenario_scores[scenario] = []
        scenario_scores[scenario].append(kw["f1"])
        kw_f1_scores.append(kw["f1"])

        scored_results.append({
            "query_id": qid,
            "scenario": scenario,
            "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "kw_f1": kw["f1"],
            "kw_precision": kw["precision"],
            "kw_recall": kw["recall"],
            "matched_keywords": kw["matched"],
            "missed_keywords": kw["missed"],
            "n_gold_keywords": kw["n_gold_keywords"],
        })

    if not kw_f1_scores:
        print("No scored results.")
        return {}

    aggregate = compute_confidence_interval(kw_f1_scores)
    per_scenario = {}
    for sc, vals in scenario_scores.items():
        per_scenario[sc] = compute_confidence_interval(vals)

    print("\n" + "=" * 70)
    print("  VERIFIED KEYWORD-F1 SCORES (from dataset gold keywords)")
    print("=" * 70)
    print(f"  Queries scored: {aggregate['n']}")
    print(f"  KW-F1 Mean:     {aggregate['mean']} ± {aggregate['std']} (95% CI: ±{aggregate['ci95']})")
    print("\n  Per-Scenario KW-F1:")
    for sc, ci in sorted(per_scenario.items()):
        print(f"    {sc:25s}: {ci['mean']:.4f} ± {ci['ci95']:.4f}  (n={ci['n']})")
    print("=" * 70)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_report": ragas_report_path,
        "scoring_method": "stemmed keyword matching against dataset expected_keywords field",
        "aggregate_kw_f1": aggregate,
        "per_scenario_kw_f1": per_scenario,
        "per_query": scored_results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to: {REPORT_PATH}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(PROJECT_ROOT, "evaluation", "ragas_evaluation_report.json"))
    args = parser.parse_args()
    score_from_ragas_report(args.input)

