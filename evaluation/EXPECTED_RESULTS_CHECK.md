# Expected Results — Verification Status

Every expected result stated in `IEEE_FINAL.tex`, `README.md` or the evaluation
reports is checked against the committed artifact that actually produced it.
No number is accepted because it "looks right".

## Verified (PASS)

| Expected result | Value | Artifact | Observed |
|---|---|---|---|
| Held-out KW-F1, full agent / static RAG / LLM-only | 0.0915 / 0.0657 / 0.0821 | `results_heldout_v2.csv` | exact match |
| Adversarial violation rates, raw / no-DSEL / DSEL | 21.4% (27/126) / 4.8% (6/126) / 0.8% (1/126) | `dsel_ablation_raw_local_rescored.csv` | exact match |
| Dual-judge scores and agreement | DR-Fact 0.576, SAA 0.519, Temporal 0.324; exact 64.5% | `judge_scores_v2.csv` | recomputed from 3,300 pairs |
| Geospatial routing | 20/20, cross-watershed 4/4 | `geospatial_routing_results.json` | exact match |
| Temporal conflict, TGI vs no-TGI false evacuation | 0/36 vs 10/36; accuracy 95.8% vs 86.1% | `tgi_results_v5.csv` | 288/288 OK, one generator |
| Vision, 4-class building damage | accuracy 0.3125, macro-F1 0.268 | `vision_benchmark_results.json` | 32 balanced crops, 0 unclassified |
| Paper length | ≤ 8 pages | `IEEE_FINAL.pdf` | 8 pages, clean compile |
| Answer format | no asterisks, complete template | `output_audit.json` | 0 asterisks; template defects repaired |

## Defects found and fixed

| Issue | Status |
|---|---|
| Paper claimed TGI "reduces false evacuation to 0/93" | FIXED — unsourced; replaced with verified 0/36 vs 10/36 |
| v3 judge used the Qwen generator as Judge A | FIXED — excluded, cells blank, disclosed |
| Temporal-conflict set had zero positive cases | FIXED — rebuilt as balanced 24-query counterfactual set |
| Vision metric dropped UNCLASSIFIED rows from denominator | FIXED — errors now counted as wrong |
| Held-out runner excluded 25 real queries (IDs 176–200) | FIXED — all 150 held-out IDs used |
| `format_answer` did not enforce its four-section template | FIXED — missing sections explicitly marked unavailable; 5 rows repaired |
| BM25 baseline ignored the query and fetched a constant context | FIXED — real FTS5 retrieval; all BM25 rows require regeneration |
| README reported fabricated DR-Fact/SAA/latency figures | FIXED — rewritten from verified artifacts |

## Outstanding (do not report yet)

1. **Fresh held-out re-run** — in progress (`results_heldout_fresh.csv`). The
   `tab:fresh_heldout` cells remain blank until all 150 queries × 7 configs finish
   and pass `audit_outputs.py`.
2. **BM25 rows after retrieval fix** — the existing 150 rows were generated with
   the constant-context bug and must be deleted and regenerated before BM25 is
   reported anywhere, including Table I of the paper.
3. **KW-F1 metric divergence** — Table I uses the token-normalised scorer defined
   by Eq. (4) in the paper (full agent 0.0915). The dataset-vocabulary scorer in
   `kw_f1_rescored.csv` gives a different ordering (LLM-only 0.248, static RAG
   0.228, full agent 0.206). Both are committed; the paper must not mix them. This
   is disclosed rather than silently switched.
4. **Vision quality** — 0.3125 accuracy is only modestly above the 0.25 chance
   rate; `major-damage` F1 is 0.0. Reported as a negative result, not tuned away.
