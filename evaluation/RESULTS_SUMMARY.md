> **SUPERSEDED.** Generated before the second reviewer fixes. Safety rows below
> use the pre-rescore scorer (all 0.0%), T6 says the DSEL ablation was not run,
> and the BM25 value is invalid. Use `CURRENT_RESULTS.md` instead.

# Results Summary (historical)

Generated: 2026-11-17T11:41:52Z

> HARD RULE: Every number in this document comes from actual run output files. No estimates.

## T1: Dev Set — KW F1 by Config (OK rows only)

| Config | N_OK | KW Precision | KW Recall | KW F1 | Avg Latency(s) |
|--------|------|------|------|------|------|
| BM25_Baseline | 150 | 0.0397 | 0.5604 | 0.0727 | 20.3326 |
| Full_Agent | 150 | 0.0396 | 0.5498 | 0.0727 | 44.4585 |
| Full_Agent_NoDSEL | 150 | 0.049 | 0.4943 | 0.0881 | 9.0486 |
| Full_Agent_NoTGI | 150 | 0.0502 | 0.4939 | 0.0876 | 31.7511 |
| Full_Agent_NoTelem | 150 | 0.0457 | 0.4883 | 0.083 | 8.2586 |
| LLM_Only | 150 | 0.0342 | 0.5679 | 0.0634 | 20.8508 |
| Standard_StaticRAG | 150 | 0.0351 | 0.5705 | 0.0653 | 30.5793 |
| Text_Only_StaticRAG | 150 | 0.0367 | 0.5248 | 0.0681 | 7.6199 |

## T2: Heldout Set — KW F1 by Config (OK rows only)

| Config | N_OK | KW F1 | Avg Latency(s) |
|--------|------|------|------|
| BM25_Baseline | 450 | 0.044 | 7.2411 |
| Full_Agent | 450 | 0.0915 | 6.8894 |
| Full_Agent_NoDSEL | 150 | 0.0905 | 6.426 |
| Full_Agent_NoTGI | 150 | 0.0862 | 8.6129 |
| LLM_Only | 450 | 0.0821 | 3.6569 |
| Standard_StaticRAG | 450 | 0.0657 | 11.418 |

## T3: Judge Scores (DR-Fact, SAA, Temporal)

| Config | Judge | N | DR-Fact | SAA | Temporal |
|--------|-------|---|---------|-----|----------|
| BM25_Baseline | JudgeA | 600 | 0.61 | 0.5217 | 0.8183 |
| BM25_Baseline | JudgeB | 600 | 0.6358 | 0.4987 | 0.8117 |
| Full_Agent | JudgeA | 600 | 0.6208 | 0.5842 | 0.8833 |
| Full_Agent | JudgeB | 600 | 0.645 | 0.55 | 0.8383 |
| Full_Agent_NoDSEL | JudgeA | 300 | 0.6517 | 0.5917 | 0.8783 |
| Full_Agent_NoDSEL | JudgeB | 300 | 0.6367 | 0.5567 | 0.8183 |
| Full_Agent_NoTGI | JudgeA | 300 | 0.63 | 0.585 | 0.8733 |
| Full_Agent_NoTGI | JudgeB | 300 | 0.625 | 0.5083 | 0.845 |
| Full_Agent_NoTelem | JudgeA | 150 | 0.6467 | 0.5567 | 0.93 |
| Full_Agent_NoTelem | JudgeB | 150 | 0.67 | 0.5767 | 0.8433 |
| LLM_Only | JudgeA | 600 | 0.7825 | 0.6708 | 0.9233 |
| LLM_Only | JudgeB | 600 | 0.7342 | 0.5408 | 0.88 |
| Standard_StaticRAG | JudgeA | 600 | 0.4875 | 0.445 | 0.7675 |
| Standard_StaticRAG | JudgeB | 600 | 0.4858 | 0.325 | 0.7908 |
| Text_Only_StaticRAG | JudgeA | 150 | 0.5467 | 0.4933 | 0.8033 |
| Text_Only_StaticRAG | JudgeB | 150 | 0.5433 | 0.4867 | 0.8167 |

## T4: Safety Results (safety_test partition)

| Config | N | Unsafe Rate | Wilson 95% CI | NON_ANSWER | Over-refusal |
|--------|---|-------------|---------------|------------|--------------|
| Agent_NoDSEL | 126 | 0.0% | [0.0%,3.0%] | 125 | 0/30 |
| Full_Agent_DSEL | 126 | 0.0% | [0.0%,3.0%] | 126 | 0/30 |
| Raw_LLM | 126 | 0.0% | [0.0%,3.0%] | 19 | 0/30 |

## T5: TGI False-Evacuation Rate

| Config | N | False-Evac | Rate | Wilson 95% CI |
|--------|---|-----------|------|---------------|
| Full_Agent_NoTGI | 96 | 1 | 1.0% | [0.2%,5.7%] |
| Full_Agent_TGI | 96 | 1 | 1.0% | [0.2%,5.7%] |

## T6: DSEL Safety Ablation

DSEL ablation not run yet.

## T7: Corrected Keyword-F1 (dataset-derived vocabulary)

Method: dataset-derived domain vocabulary; stemmed bag-of-keywords F1. Vocabulary: 643 stems. Scored rows: 2100.

| Config | Mean KW-F1 | SD | 95% CI | N |
|---|---:|---:|---:|---:|
| BM25_Baseline | 0.1808 | 0.0841 | +/-0.0078 | 450 |
| Full_Agent | 0.2057 | 0.1047 | +/-0.0097 | 450 |
| Full_Agent_NoDSEL | 0.2066 | 0.1082 | +/-0.0173 | 150 |
| Full_Agent_NoTGI | 0.1946 | 0.1074 | +/-0.0172 | 150 |
| LLM_Only | 0.248 | 0.1272 | +/-0.0118 | 450 |
| Standard_StaticRAG | 0.2279 | 0.1273 | +/-0.0118 | 450 |

