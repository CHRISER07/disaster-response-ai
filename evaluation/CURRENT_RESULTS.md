# Current Authoritative Results

Every number below is recomputed from the committed artifact named in the last
column. When this file and an older report disagree, **this file wins**.

## Completed

| Result | Value | Source |
|---|---|---|
| DAB-200 held-out KW-F1, Full Agent (published scorer) | 0.0915 | `results_heldout_v2.csv` |
| DAB-200 held-out KW-F1, Standard Static RAG | 0.0657 | `results_heldout_v2.csv` |
| DAB-200 held-out KW-F1, LLM-Only | 0.0821 | `results_heldout_v2.csv` |
| Corrected-vocabulary KW-F1 (different scorer — do not mix) | Full 0.2057 / BM25 0.1808 / StaticRAG 0.2279 | `kw_f1_rescored.csv` |
| Adversarial violations, raw / no-DSEL / DSEL | 27/126 / 6/126 / 1/126 | `dsel_ablation_raw_local_rescored.csv` |
| DSEL Fisher p (no-DSEL vs DSEL; raw vs DSEL) | 0.1198; 5.0e-8 | recomputed |
| Temporal conflict: TGI FP / FN | 0/36 / 3/36 | `tgi_results_v5.csv` |
| Temporal conflict: no-TGI FP / FN | 10/36 / 0/36 | `tgi_results_v5.csv` |
| Temporal conflict: LLM-Only / StaticRAG missed evacuations | 36/36 each | `tgi_results_v5.csv` |
| Geospatial routing | 20/20 (cross-watershed 4/4) | `geospatial_routing_results.json` |
| Dual judge v2 κ_w | DR 0.576 / SAA 0.519 / Temporal 0.324 | `judge_scores_v2.csv` |
| Exploratory vision (100 crops) | accuracy 0.340, macro-F1 0.292 | `vision_benchmark_results.json` |

## In progress — do not report

- `results_heldout_fresh.csv`: restored-system re-run, one generator
  (`llama3.1:8b`), still being written. BM25 rows in this file are **stale**
  (constant-context retriever) and will be deleted and re-measured.
- `judge_scores_v3fixed.csv`: independent corrected-rubric judging, incomplete;
  partial per-config means are not comparable.

## Superseded artifacts (kept for provenance only)

`results_per_query_v2.csv`, `judge_scores_v3.csv`, `judge_scores_v3_clean.csv`,
`safety_results_v2.csv`, `dsel_ablation_raw.csv`,
`dsel_ablation_raw_backup.csv`, `dsel_ablation_raw_ollama_final.csv`,
`tgi_results_v2.csv`, `tgi_results_v4.csv`. These contain pre-fix scores,
markdown-formatted answers, or invalid scorers. Do not use them for claims.

The older reports `RESULTS_SUMMARY.md`, `INTEGRITY.md` and `DISCREPANCIES.md`
were generated before these fixes and are superseded by this file and by
`EXPECTED_RESULTS_CHECK.md`.
