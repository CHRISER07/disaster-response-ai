# Stratified Dual-Judge Report

Generated: 2026-11-17T12:53:44.940586+00:00
Source: judge_scores_v2.csv
Final v3: False

## Quadratic weighted Cohen's kappa

| Stratum | Dimension | kappa_w | N | Exact agreement |
|---|---|---:|---:|---:|
| ALL | dr_fact | 0.5693 | 1300 | 0.6623 |
| ALL | saa | 0.512 | 1300 | 0.73 |
| ALL | temporal | 0.3042 | 1300 | 0.7038 |
| LIVE_TELEMETRY_CONFLICT | dr_fact | 0.4529 | 264 | 0.6591 |
| LIVE_TELEMETRY_CONFLICT | saa | 0.3434 | 264 | 0.678 |
| LIVE_TELEMETRY_CONFLICT | temporal | 0.3777 | 264 | 0.75 |
| STATIC_DOCTRINE | dr_fact | 0.5802 | 1036 | 0.6631 |
| STATIC_DOCTRINE | saa | 0.5426 | 1036 | 0.7432 |
| STATIC_DOCTRINE | temporal | 0.2874 | 1036 | 0.6921 |

## Stratified mean scores

| Stratum | Config | DR-Fact | SAA | Temporal |
|---|---|---:|---:|---:|
| LIVE_TELEMETRY_CONFLICT | BM25_Baseline | 0.7812 (n=40) | 0.5375 (n=40) | 0.875 (n=40) |
| LIVE_TELEMETRY_CONFLICT | Full_Agent | 0.7625 (n=40) | 0.6438 (n=40) | 0.8562 (n=40) |
| LIVE_TELEMETRY_CONFLICT | Full_Agent_NoDSEL | 0.725 (n=40) | 0.5813 (n=40) | 0.7937 (n=40) |
| LIVE_TELEMETRY_CONFLICT | Full_Agent_NoTGI | 0.725 (n=40) | 0.6062 (n=40) | 0.85 (n=40) |
| LIVE_TELEMETRY_CONFLICT | Full_Agent_NoTelem | 0.8125 (n=12) | 0.6667 (n=12) | 0.9583 (n=12) |
| LIVE_TELEMETRY_CONFLICT | LLM_Only | 0.8375 (n=40) | 0.625 (n=40) | 0.9 (n=40) |
| LIVE_TELEMETRY_CONFLICT | Standard_StaticRAG | 0.75 (n=40) | 0.475 (n=40) | 0.85 (n=40) |
| LIVE_TELEMETRY_CONFLICT | Text_Only_StaticRAG | 0.6458 (n=12) | 0.5 (n=12) | 0.9375 (n=12) |
| STATIC_DOCTRINE | BM25_Baseline | 0.5953 (n=160) | 0.5078 (n=160) | 0.8125 (n=160) |
| STATIC_DOCTRINE | Full_Agent | 0.6531 (n=160) | 0.5953 (n=160) | 0.8828 (n=160) |
| STATIC_DOCTRINE | Full_Agent_NoDSEL | 0.6125 (n=160) | 0.5719 (n=160) | 0.85 (n=160) |
| STATIC_DOCTRINE | Full_Agent_NoTGI | 0.5859 (n=160) | 0.5312 (n=160) | 0.8562 (n=160) |
| STATIC_DOCTRINE | Full_Agent_NoTelem | 0.625 (n=38) | 0.5461 (n=38) | 0.875 (n=38) |
| STATIC_DOCTRINE | LLM_Only | 0.7625 (n=160) | 0.5891 (n=160) | 0.8984 (n=160) |
| STATIC_DOCTRINE | Standard_StaticRAG | 0.4875 (n=160) | 0.4172 (n=160) | 0.7766 (n=160) |
| STATIC_DOCTRINE | Text_Only_StaticRAG | 0.4211 (n=38) | 0.4934 (n=38) | 0.7039 (n=38) |
