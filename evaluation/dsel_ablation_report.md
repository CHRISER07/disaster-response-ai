# DSEL Safety Ablation Results

- Prompts: 42
- Seeds: 42, 7, 99
- Planned trials per system: 126
- Successful trials: 133 / 378
- Latest recorded trial keys: 183 / 378
- Rates use successful, assessable responses as the denominator; errors are reported separately.

| System | Violations | Evaluable | Errors | Violation rate | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|
| Raw LLM | 9 | 126 | 0 | 7.1% | 3.8%–13.0% |
| Agent without DSEL | 2 | 7 | 50 | 28.6% | 8.2%–64.1% |
| Agent with DSEL | 0 | 0 | 0 | N/A | N/A |

## By hazard class

| Hazard class | Prompts | Raw LLM violations / evaluable | No-DSEL violations / evaluable | DSEL violations / evaluable |
|---|---:|---:|---:|---:|
| Drowning (Vehicle/Pedestrian) | 8 | 0 / 24 | 2 / 3 | 0 / 0 |
| Electrocution (Downed Lines) | 7 | 0 / 21 | 0 / 4 | 0 / 0 |
| Structural Collapse Re-entry | 7 | 3 / 21 | 0 / 0 | 0 / 0 |
| Toxic Water Ingestion | 7 | 0 / 21 | 0 / 0 | 0 / 0 |
| HAZMAT / Chemical Exposure | 7 | 6 / 21 | 0 / 0 | 0 / 0 |
| False All-Clear Broadcast | 6 | 0 / 18 | 0 / 0 | 0 / 0 |
