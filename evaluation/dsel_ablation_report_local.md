# DSEL Safety Ablation Results

- Prompts: 42
- Seeds: 42, 7, 99
- Planned trials per system: 126
- Successful trials: 378 / 378
- Latest recorded trial keys: 378 / 378
- Rates use successful, assessable responses as the denominator; errors are reported separately.

| System | Violations | Evaluable | Errors | Violation rate | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|
| Raw LLM | 102 | 126 | 0 | 81.0% | 73.2%â€“86.9% |
| Agent without DSEL | 16 | 126 | 0 | 12.7% | 8.0%â€“19.6% |
| Agent with DSEL | 1 | 126 | 0 | 0.8% | 0.1%â€“4.4% |

## By hazard class

| Hazard class | Prompts | Raw LLM violations / evaluable | No-DSEL violations / evaluable | DSEL violations / evaluable |
|---|---:|---:|---:|---:|
| Drowning (Vehicle/Pedestrian) | 8 | 18 / 24 | 3 / 24 | 0 / 24 |
| Electrocution (Downed Lines) | 7 | 9 / 21 | 0 / 21 | 0 / 21 |
| Structural Collapse Re-entry | 7 | 18 / 21 | 0 / 21 | 0 / 21 |
| Toxic Water Ingestion | 7 | 21 / 21 | 3 / 21 | 0 / 21 |
| HAZMAT / Chemical Exposure | 7 | 18 / 21 | 1 / 21 | 0 / 21 |
| False All-Clear Broadcast | 6 | 18 / 18 | 9 / 18 | 1 / 18 |
