# Disaster Response AI -- IEEE Evaluation Results (50q)
*Generated: 2026-06-05 16:54 | Queries evaluated: 50*

---

## Table 1: Ablation Study -- All Metrics (mean +/- std)

| System | n | KW-F1 | std | CI95 | DR-Fact | std | SAA | std | SVC | P@K | Latency(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LLM Only** | 50 | 0.2458 | 0.1359 | 0.0377 | 0.5500 | 0.4080 | 0.5720 | 0.4010 | 1 | 0.0000 | 13.34 |
| **Text RAG** | 50 | 0.2933 | 0.1786 | 0.0495 | 0.6100 | 0.3306 | 0.6340 | 0.2754 | 3 | 0.2487 | 1668.77 |
| **Multimodal RAG** | 50 | 0.3361 | 0.2243 | 0.0622 | 0.6240 | 0.3314 | 0.6980 | 0.2956 | 2 | 0.2630 | 1.21 |
| **BM25 Baseline** | 50 | 0.0341 | 0.0482 | 0.0134 | 0.0000 | 0.0000 | 0.0040 | 0.0280 | 2 | 0.0057 | 0.28 |

Note: KW-F1 is deterministic (no LLM). DR-Fact and SAA use LLM-as-Judge (Llama-3.3-70B, temperature=0). SVC uses deterministic regex. CI95 = 1.96 * SEM. All n values reflect queries with non-empty responses.

---

## Table 2: Multimodal RAG -- Per-Scenario Breakdown

| Scenario | n | KW-F1 | DR-Fact | SAA | SVC |
|---|---|---|---|---|---|
| Building_Collapse | 5 | 0.4332 | 0.4800 | 0.4200 | 0 |
| Chemical | 5 | 0.3356 | 0.4000 | 0.5400 | 1 |
| Earthquake | 2 | 0.2905 | 0.8000 | 0.9000 | 0 |
| Flood | 23 | 0.3304 | 0.7217 | 0.8043 | 0 |
| Hallucination | 5 | 0.1228 | 0.9600 | 0.9800 | 0 |
| Typhoon | 5 | 0.4130 | 0.4400 | 0.4800 | 0 |
| Wildfire | 5 | 0.4200 | 0.3200 | 0.5000 | 1 |

---

## Table 3: Multimodal RAG -- Per-Category Breakdown

| Category | n | KW-F1 | DR-Fact | SAA | SVC |
|---|---|---|---|---|---|
| communications | 1 | 0.3080 | 0.6000 | 0.8000 | 0 |
| evacuation | 6 | 0.2657 | 0.6000 | 0.7000 | 0 |
| hallucination_test | 9 | 0.1006 | 0.9333 | 0.9667 | 2 |
| infrastructure | 5 | 0.4036 | 0.5600 | 0.5800 | 0 |
| medical | 2 | 0.3405 | 0.7000 | 0.8000 | 0 |
| missing_data | 1 | 0.2500 | 1.0000 | 1.0000 | 0 |
| multi_source | 2 | 0.2725 | 0.7000 | 0.7000 | 0 |
| protocol | 4 | 0.3032 | 0.6000 | 0.8750 | 0 |
| resource_allocation | 1 | 0.1580 | 0.6000 | 0.7000 | 0 |
| satellite_imagery | 1 | 0.8570 | 0.8000 | 0.9000 | 0 |
| search_rescue | 1 | 0.3120 | 0.0000 | 0.0000 | 0 |
| seismic | 1 | 0.4210 | 0.8000 | 0.9000 | 0 |
| sensor_data | 4 | 0.4193 | 0.4000 | 0.4750 | 0 |
| shelter | 1 | 0.1210 | 0.8000 | 0.9000 | 0 |
| situational_awareness | 6 | 0.5835 | 0.4000 | 0.4333 | 0 |
| water_levels | 1 | 0.8330 | 1.0000 | 1.0000 | 0 |
| water_quality | 1 | 0.1820 | 0.6000 | 0.4000 | 0 |
| weather | 3 | 0.4093 | 0.3333 | 0.5000 | 0 |

---

## Table 4: Multimodal RAG -- Per-Difficulty Breakdown

| Difficulty | n | KW-F1 | DR-Fact | SAA |
|---|---|---|---|---|
| medium | 50 | 0.3361 | 0.6240 | 0.6980 |

---

## Table 5: Multimodal RAG -- Per-CrisisLex-Event Breakdown

| Event | n | KW-F1 | DR-Fact | SAA | SVC |
|---|---|---|---|---|---|
| unknown | 50 | 0.3361 | 0.6240 | 0.6980 | 2 |

---

## Key Findings

- KW-F1 gain over LLM Only: +0.0903 (0.3361 +/- 0.2243 vs 0.2458 +/- 0.1359)
- SAA gain over LLM Only: +22.0% (0.6980 +/- 0.2956 vs 0.5720 +/- 0.4010)
- Safety: Multimodal RAG 2 violations vs 1 for LLM Only
- Retrieval P@K: 0.2630 +/- 0.1966 (Multimodal RAG)
- Dense vs Sparse: Multimodal RAG +0.3020 KW-F1 over BM25 baseline (0.3361 vs 0.0341)

---

## Methodology

- Dataset: 50 queries from golden_dataset.json grounded in 26 real CrisisLex T26 disaster events.
- KW-F1: verbatim keyword matching (case-insensitive substring). No LLM involvement. Fully deterministic.
- DR-Fact / SAA: LLM-as-Judge (Llama-3.3-70B, temperature=0.0). Acknowledged limitation: self-referential bias when judge model matches generation model.
- SVC: deterministic regex pattern matching. Zero LLM involvement.
- BM25 Baseline: TF-IDF keyword retrieval without LLM generation. Lower bound for demonstrating dense retrieval value.
- Statistics: mean +/- population std. CI95 computed as 1.96 * (std / sqrt(n)).
- Full per-query scores: results_per_query.csv
- Per-event scores: results_per_event.csv
