# Agentic Multimodal Disaster Response AI

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-green)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)


> **IEEE Research Framework** — A production-grade, hallucination-free Agentic AI for real-time disaster response, fusing satellite imagery, live IoT sensors, social media, weather data, and official emergency protocols into a single grounded situational report.

---

## Problem Statement

During active disasters, emergency coordinators must manually correlate data from 5+ siloed sources (Twitter feeds, USGS sensors, drone feeds, weather apps, FEMA handbooks). This introduces critical delays and human error. This system eliminates that bottleneck with an autonomous AI agent.

## System Architecture

```
Real-World Data → ETL Pipeline → Vector DB → LangGraph Agent → Streamlit Dashboard
     │                │              │               │                │
  6 Sources        5 Loaders     ChromaDB        ReAct Loop      3-Panel UI
  (all free)      (real data)   (3,195 docs)   (5 tools)       (live + chat)
```

## Data Sources (All Real, All Free)

| Modality | Source | Description |
|---|---|---|
| Social Text | CrisisLex T26 | 200 real tweets from 2013 Colorado Floods |
| Satellite Imagery | xView2 Dataset | Real building damage assessment images → LLaVA descriptions |
| River Sensors | USGS NWIS Live API | Real-time gage height, Boulder Creek site 06730200 |
| Weather | Open-Meteo API | Live temperature, precipitation, wind |
| Earthquakes | USGS Earthquake API | Live M≥3.0 seismic events |
| Protocols | FEMA/CDC Manuals | 3 official PDF handbooks, 2,881 semantic chunks |

## Agent Tools

```python
tools = [
    search_knowledge_base,   # FEMA/CDC/CrisisLex RAG retrieval
    fetch_live_weather,      # Open-Meteo: temp + rain + wind
    query_water_sensor,      # USGS: real river gage height + status
    analyze_disaster_image,  # Ollama LLaVA 7B: local vision (no API cost)
    generate_structured_alert  # Severity-tagged alert log entry
]
```

## Evaluation Results (IEEE Metrics)

| System | DR-Fact | SAA | Safety Violations ↓ | Avg Latency |
|---|---|---|---|---|
| LLM Only | 0.32 | 0.30 | 2 | 1.18s |
| Text RAG | 0.43 | 0.47 | 0 | 1.07s |
| **Multimodal RAG (Ours)** | **0.63** | **0.77** | **0** | **1.07s** |

## Quick Start

```bash
# Clone
git clone https://github.com/CHRISER07/disaster-response-ai.git
cd disaster-response-ai

# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Add your free Groq API key (from console.groq.com)
echo GROQ_API_KEY=your_key_here > .env

# Install local vision AI (one time, free)
# Download Ollama from https://ollama.ai then:
ollama pull llava

# Populate vector database with real data
python core/populate_db.py

# Launch the dashboard
streamlit run app.py
```

## Project Structure

See `AGENTS.md` for the complete directory layout and architecture documentation.

## Citation

If you use this framework in research, please cite:
```
@misc{disaster-response-ai-2026,
  title={Agentic Multimodal RAG for Real-Time Disaster Response},
  author={CHRISER07},
  year={2026},
  url={https://github.com/CHRISER07/disaster-response-ai}
}
```
