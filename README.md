# Agentic Multimodal Disaster Response AI

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-green)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)

> **IEEE Research Framework** — agentic RAG for disaster response, combining live
> hydrologic/meteorological telemetry, crisis microblogs, official response doctrine,
> and building-level damage imagery behind a deterministic safety layer. All
> infrastructure is free-tier or locally hosted; no paid APIs are used.

---

## Problem Statement

During active disasters, emergency coordinators must manually correlate data from
multiple siloed sources (social media, river gauges, satellite imagery, weather,
official manuals). This introduces delay and human error. The system produces a
single grounded situational report with an auditable evidence trail.

## System Architecture

```
Real-World Data   ETL Pipeline   Hybrid Index     ReAct Agent       Dashboard
  (free)           (real data)   (17,170 docs)    (8 tools)         (live + chat)
```

Retrieval fuses dense BGE embeddings and SQLite FTS5 BM25 with weighted reciprocal
rank fusion (`alpha_dense = 0.70`, `k = 60`). A ReAct agent coordinates eight tools;
every answer passes through a deterministic safety layer (DSEL) before delivery.

## Data Sources (all real, all free)

| Modality | Source | Description |
|---|---|---|
| Social text | CrisisLex T26 | 27,933 crisis tweets from 26 events |
| Satellite imagery | xView2 / xBD | 1,017 labeled building-damage polygons |
| River sensors | USGS NWIS | Live gage height, Boulder Creek site 06730200 |
| Weather | Open-Meteo | Live temperature, precipitation, wind |
| Earthquakes | USGS FDSNWS | Live M≥3.0 seismic events |
| Protocols | FEMA / CDC | Official response manuals, 261 doctrine chunks |

## Agent Tools

```python
ALL_TOOLS = [
    fetch_live_weather,          # Open-Meteo
    query_water_sensor,          # USGS NWIS / GloFAS
    query_recent_earthquakes,    # USGS FDSNWS
    search_official_protocols,   # FEMA / CDC only
    search_social_reports,       # CrisisLex + sensor + imagery
    search_knowledge_base,       # full hybrid store
    analyze_disaster_image,      # local Qwen3-VL 4B, task="classify"
    generate_structured_alert,   # severity-tagged alert log
]
```

## Evaluation Results

All figures come from committed artifacts in `evaluation/`. See
`evaluation/RESULTS_SUMMARY.md` and `IEEE_FINAL.tex` for full tables and intervals.

| Result | Measured value | Source |
|---|---|---|
| Held-out KW-F1, full agent | 0.0915 | `results_heldout_v2.csv` |
| Held-out KW-F1, static RAG | 0.0657 | `results_heldout_v2.csv` |
| Held-out KW-F1, LLM-only | 0.0821 | `results_heldout_v2.csv` |
| Adversarial safety violations, raw LLM / no-DSEL / DSEL | 21.4% / 4.8% / 0.8% | `dsel_ablation_raw_local_rescored.csv` |
| Geospatial routing | 20/20 scenarios | `geospatial_routing_results.json` |
| Temporal conflict, TGI vs no-TGI false evacuation | 0/36 vs 10/36 | `tgi_results_v5.csv` |
| Building damage classification | 0.3125 accuracy, 0.268 macro-F1 | `vision_benchmark_results.json` |

`BM25_Baseline` scores from the legacy run are **superseded**: the baseline retrieved
a constant context instead of querying by the question. It is being re-measured with
the corrected FTS5 retriever.

## Reproducibility notes

- The originally specified 72B Qwen model is not available on the free endpoint;
  reported runs use `qwen/qwen3.8-27b`, local `llama3.1:8b`, or local `qwen2.5:3b`
  depending on the suite. Generator identity is recorded per row.
- Free Gemini quota was exhausted, so the v3 judge run is not independent and is
  excluded. Published judge results come from the independent v2 run.
- LangSmith is disabled; `logs/incident_audit_trail.jsonl` is the audit ledger.

## Quick Start

```bash
git clone https://github.com/CHRISER07/disaster-response-ai.git
cd disaster-response-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Free Groq key plus local Ollama models
echo GROQ_API_KEY=your_key_here > .env
ollama pull qwen3-vl:4b
ollama pull llama3.1:8b

python core/populate_db.py
streamlit run app.py
```

## Citation

```bibtex
@misc{disaster-response-ai-2026,
  title={Agentic Multimodal RAG for Real-Time Disaster Response},
  author={CHRISER07},
  year={2026},
  url={https://github.com/CHRISER07/disaster-response-ai}
}
```
