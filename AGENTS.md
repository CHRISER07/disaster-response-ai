# AGENTS.md — Persistent Context for the Disaster Response AI Agent

## Project Identity

**Name:** Agentic Multimodal Disaster Response AI  
**Research Goal:** IEEE-publishable framework proving that fusing text, satellite imagery, live IoT sensors, weather, and official protocols into a single LangGraph ReAct agent produces significantly higher situational awareness accuracy and lower hallucination rates than text-only RAG baselines.  
**Code Repo:** https://github.com/CHRISER07/disaster-response-ai

## What This System Does

During an active disaster (flood, earthquake, hurricane), emergency responders face simultaneous data from multiple sources. This Agentic AI:
1. Monitors live USGS water level sensors and earthquake APIs
2. Monitors live Open-Meteo weather (temperature, rain, wind)
3. Searches 3,195 embedded documents: FEMA/CDC protocols, CrisisLex tweets, xView2 satellite imagery descriptions
4. Analyzes real disaster images locally via Ollama LLaVA 7B (no cloud cost)
5. Reasons across all these sources using LangGraph ReAct, then delivers a grounded, cited, hallucination-free situational report

## Directory Layout

```
disaster_response_rag/
├── agents/
│   └── disaster_agent.py     # LangGraph ReAct agent (the brain)
├── tools/
│   ├── weather_tool.py        # Open-Meteo API (free, no auth)
│   ├── sensor_tool.py         # USGS water + earthquake API (free, no auth)  
│   ├── kb_search_tool.py      # ChromaDB FEMA/text search tool
│   ├── vision_tool.py         # Ollama LLaVA local vision
│   └── alert_tool.py          # Structured alert logger
├── core/
│   ├── vector_store.py        # ChromaDB + BGE embeddings
│   ├── populate_db.py         # Data ingestion pipeline (run once)
│   └── knowledge_loader.py    # PDF semantic chunker
├── ingestion/
│   ├── text_loader.py         # CrisisLex tweet parser (real data)
│   ├── iot_loader.py          # USGS live + historical sensor ingestion
│   ├── drone_loader.py        # xView2 VLM description loader
│   ├── vlm_processor.py       # Ollama LLaVA image processor
│   └── extract_xview2_subset.py  # Streams images from tar without full extract
├── evaluation/
│   ├── golden_dataset.json    # 6 evaluation queries with ground truth
│   ├── run_evals.py           # LLM-as-judge evaluation harness
│   └── results.md             # Generated IEEE metrics table
├── tests/
│   ├── test_tools.py          # Unit tests (all 5 tools)
│   └── test_agent.py          # Integration test (agent uses ≥2 tools)
├── data/
│   ├── kb/                    # 3 FEMA/CDC PDFs
│   ├── iot/                   # USGS sensor CSV (generated from live API)
│   ├── imagery/               # xView2 real_drone_metadata.json
│   └── alerts.jsonl           # Structured alert log (written by alert_tool)
├── app.py                     # Streamlit 3-panel dashboard
├── AGENTS.md                  # This file
├── memory.md                  # Self-improving session memory
└── requirements.txt
```

## API Conventions

| Service | Base URL | Auth Required |
|---|---|---|
| Open-Meteo Weather | `https://api.open-meteo.com/v1/forecast` | No |
| USGS Water Sensors | `https://waterservices.usgs.gov/nwis/iv/` | No |
| USGS Earthquakes | `https://earthquake.usgs.gov/fdsnws/event/1/query` | No |
| Groq LLM | `api.groq.com` | `GROQ_API_KEY` in `.env` |
| Ollama Vision | `http://localhost:11434/api/generate` | Local only |

## Metadata Schema (ChromaDB)

Every document stored follows this contract:
```python
{
    "modality": "Text | knowledge_base | imagery_description | telemetry",
    "source": "CrisisLex | FEMA_Manual | xView2_VLM | USGS_Sensor",
    "timestamp": "ISO-8601 UTC string",
    "location": "string or null"
}
```

## How To Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Populate vector database (first time or after data changes)
python core/populate_db.py --reset

# 3. Launch the dashboard
streamlit run app.py

# 4. Run evaluation harness
python evaluation/run_evals.py

# 5. Run tests
pytest tests/ -v
```

## Critical Rules

- NEVER use mock/synthetic data in production ingestion. Real USGS, real weather, real images only.
- NEVER commit `.env`, `venv/`, or `chroma_db/` to git.
- ALWAYS cite data sources in agent answers (modality + timestamp).
- NEVER hallucinate. If data is not in context, say "Data not available."
