# DSEL Safety Ablation Results

- Provider: ollama
- Locked model: qwen2.5:3b
- Models observed in CSV: qwen2.5:3b
- Scorer versions observed: v3-conditional-aware
- Prompts: 42
- Seeds: 42, 7, 99
- Planned trials per system: 126
- Successful trials: 378 / 378
- Latest recorded trial keys: 378 / 378
- Rates use successful, assessable responses as the denominator; errors are reported separately.

| System | Violations | Evaluable | Errors | Violation rate | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|
| Raw LLM | 10 | 126 | 0 | 7.9% | 4.4%â€“14.0% |
| Agent without DSEL | 7 | 126 | 0 | 5.6% | 2.7%â€“11.0% |
| Agent with DSEL | 0 | 126 | 0 | 0.0% | 0.0%â€“3.0% |

## By hazard class

| Hazard class | Prompts | Raw LLM violations / evaluable | No-DSEL violations / evaluable | DSEL violations / evaluable |
|---|---:|---:|---:|---:|
| Drowning (Vehicle/Pedestrian) | 8 | 3 / 24 | 2 / 24 | 0 / 24 |
| Electrocution (Downed Lines) | 7 | 3 / 21 | 3 / 21 | 0 / 21 |
| Structural Collapse Re-entry | 7 | 3 / 21 | 2 / 21 | 0 / 21 |
| Toxic Water Ingestion | 7 | 0 / 21 | 0 / 21 | 0 / 21 |
| HAZMAT / Chemical Exposure | 7 | 1 / 21 | 0 / 21 | 0 / 21 |
| False All-Clear Broadcast | 6 | 0 / 18 | 0 / 18 | 0 / 18 |
