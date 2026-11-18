# Vision Tool Benchmark (building-level)

Generated: 2026-11-18T11:00:34.741600+00:00
Model: qwen3-vl-gpu:latest
Labeled buildings: 1017 | Sampled: 32 | Assessable: 32 | Unclassified: 0

| Metric | Value |
|---|---|
| Accuracy (errors counted) | 0.3125 |
| Accuracy (assessable only) | 0.3125 |
| Macro-F1 (errors counted) | 0.2680 |
| Mean latency (s) | 24.668 |
| p95 latency (s) | 37.318 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| no-damage | 0.3889 | 0.875 | 0.5385 | 8 |
| minor-damage | 0.1429 | 0.125 | 0.1333 | 8 |
| major-damage | 0.0 | 0.0 | 0.0 | 8 |
| destroyed | 1.0 | 0.25 | 0.4 | 8 |
