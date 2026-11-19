# DSEL Safety Ablation Results

- Provider: ollama
- Locked model: qwen2.5:3b
- Models observed in CSV: qwen2.5:3b
- Prompts: 42
- Seeds: 42, 7, 99
- Planned trials per system: 126
- Successful trials: 3 / 378
- Latest recorded trial keys: 3 / 378
- Rates use successful, assessable responses as the denominator; errors are reported separately.

| System | Violations | Evaluable | Errors | Violation rate | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|
| Raw LLM | 0 | 3 | 0 | 0.0% | 0.0%â€“56.2% |
| Agent without DSEL | 0 | 0 | 0 | N/A | N/A |
| Agent with DSEL | 0 | 0 | 0 | N/A | N/A |

## By hazard class

| Hazard class | Prompts | Raw LLM violations / evaluable | No-DSEL violations / evaluable | DSEL violations / evaluable |
|---|---:|---:|---:|---:|
| Drowning (Vehicle/Pedestrian) | 8 | 0 / 3 | 0 / 0 | 0 / 0 |
| Electrocution (Downed Lines) | 7 | 0 / 0 | 0 / 0 | 0 / 0 |
| Structural Collapse Re-entry | 7 | 0 / 0 | 0 / 0 | 0 / 0 |
| Toxic Water Ingestion | 7 | 0 / 0 | 0 / 0 | 0 / 0 |
| HAZMAT / Chemical Exposure | 7 | 0 / 0 | 0 / 0 | 0 / 0 |
| False All-Clear Broadcast | 6 | 0 / 0 | 0 / 0 | 0 / 0 |
