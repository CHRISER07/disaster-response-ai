# ARIA Disaster Response AI — IEEE Evaluation Results
*Generated: 2026-04-15 01:26 | Queries evaluated: 30*

---

## Table 1: Ablation Study — All Metrics

| System | n | KW-F1 ↑ | KW-Prec ↑ | KW-Rec ↑ | DR-Fact ↑ | SAA ↑ | SVC ↓ | P@K ↑ | Latency(s) |
|---|---|---|---|---|---|---|---|---|---|
| **LLM Only** | 30 | 0.231 | 0.17 | 0.554 | 0.673 | 0.673 | 0 | 0.0 | 3.041 |
| **Text RAG** | 30 | 0.19 | 0.147 | 0.309 | 0.513 | 0.5 | 0 | 0.0 | 2.225 |
| **Multimodal RAG** | 30 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0.147 |

> **KW-F1**: Objective keyword-based F1 score (not LLM-biased). **DR-Fact**: LLM-as-Judge factuality (0–1). **SAA**: Situation Awareness Accuracy (0–1). **SVC**: Safety Violation Count (lower is better). **P@K**: Retrieval Precision at K=7.

---

## Table 2: Multimodal RAG — Per-Category Breakdown

| Category | n | KW-F1 | DR-Fact | SAA | SVC |
|---|---|---|---|---|---|
| communications | 1 | 0.0 | 0.0 | 0.0 | 0 |
| evacuation | 3 | 0.0 | 0.0 | 0.0 | 0 |
| hallucination_test | 5 | 0.0 | 0.0 | 0.0 | 1 |
| infrastructure | 3 | 0.0 | 0.0 | 0.0 | 0 |
| medical | 1 | 0.0 | 0.0 | 0.0 | 0 |
| missing_data | 1 | 0.0 | 0.0 | 0.0 | 0 |
| multi_source | 1 | 0.0 | 0.0 | 0.0 | 0 |
| protocol | 4 | 0.0 | 0.0 | 0.0 | 0 |
| resource_allocation | 1 | 0.0 | 0.0 | 0.0 | 0 |
| satellite_imagery | 1 | 0.0 | 0.0 | 0.0 | 0 |
| seismic | 1 | 0.0 | 0.0 | 0.0 | 0 |
| sensor_data | 2 | 0.0 | 0.0 | 0.0 | 0 |
| shelter | 1 | 0.0 | 0.0 | 0.0 | 0 |
| situational_awareness | 2 | 0.0 | 0.0 | 0.0 | 0 |
| water_levels | 1 | 0.0 | 0.0 | 0.0 | 0 |
| water_quality | 1 | 0.0 | 0.0 | 0.0 | 0 |
| weather | 1 | 0.0 | 0.0 | 0.0 | 0 |

---

## Key Findings

- **Keyword F1 improvement**: Multimodal RAG achieves -0.231 over LLM-Only (0.0 vs 0.231)
- **SAA improvement**: -100.0% over LLM-Only (0.0 vs 0.673)
- **Safety**: Multimodal RAG produces 1 safety violations vs 0 for LLM-Only
- **Retrieval**: P@K = 0.0 for Multimodal RAG

---

## Methodology Notes

- **KW-F1** is computed from`expected_keywords` in `golden_dataset.json` — no LLM-based scoring, fully reproducible.
- **DR-Fact / SAA** use LLM-as-Judge (LLaMA 3.3 70B). Acknowledge self-referential bias in paper limitations.
- **SVC** uses rule-based pattern matching on unsafe phrases — zero LLM involvement.
- All 30 queries have human-written ground truth answers in `golden_dataset.json`.
- Full per-query scores available in `results_per_query.csv`.
