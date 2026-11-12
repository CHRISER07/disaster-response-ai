# memory.md - Project Master Memory (Updated 2026-07-21)

This file is the canonical reference for this project session.
Updated by the agent when it learns new facts, corrects errors, or discovers user preferences.

---

## Project Identity

- **Name**: Multimodal Agentic RAG - Disaster Response AI
- **Version**: 2.0
- **Root path**: C:\Users\Acer\OneDrive\Desktop\multiagents\disaster_response_rag
- **Purpose**: End-to-end agentic AI system for real-time disaster situational intelligence

---

## Confirmed Models & APIs

- **LLM**: Groq `llama-3.3-70b-versatile` (NOT llama3-70b-8192 - decommissioned)
- **Embeddings**: `BAAI/bge-small-en-v1.5` via `langchain_huggingface.HuggingFaceEmbeddings`
- **Vision**: Ollama LLaVA 7B at `http://localhost:11434` (optional - system degrades gracefully)
- **Vector DB**: ChromaDB persistent at `./chroma_db/`
- **LangGraph**: >= 0.2.0 - uses `state_modifier=` kwarg in `create_react_agent()`
- **Agent tracing**: LangSmith (optional, LANGCHAIN_API_KEY in .env)

---

## Confirmed Data Sources

- **USGS Boulder Creek**: Site 06730200 - Boulder Creek at North 75th St, CO. Threshold: 10.0 ft
- **CrisisLex T26**: 16,849 tweets, 26 events, 2012-2013. Path: `data/text/CrisisLexT26/`
- **FEMA/CDC PDFs**: `data/knowledge_base/` - 2,906 chunks after splitting
- **xView2 imagery**: `data/imagery/` - 50 real satellite images with VLM descriptions
- **USGS NWIS Historical**: Ingested via iot_loader.py with 7-day window
- **Open-Meteo**: Free weather + geocoding API. No key required. Global coverage.
- **USGS Earthquake API**: Free. Global coverage within any radius from geocoded lat/lon.

---

## Architecture Layers

```
[Streamlit app.py] <-- thin client
      |
      v (HTTP REST or direct import)
[FastAPI api/main.py:8000]
      |
      v
[LangGraph ReAct Agent - agents/disaster_agent.py]
      |
      +-- [8 Tools in tools/]
      |       weather_tool.py       - Open-Meteo (global)
      |       sensor_tool.py        - USGS water gauge + earthquake
      |       kb_search_tool.py     - ChromaDB (3 search functions)
      |       vision_tool.py        - Ollama LLaVA (local, optional)
      |       alert_tool.py         - JSON alert logger
      |
      +-- [ChromaDB chroma_db/]
      |       4 modalities: text, telemetry, imagery_description, knowledge_base
      |
      +-- [core/ modules]
              cost_tracker.py       - per-query USD cost logging
              policy_engine.py      - pre-execution confidence gate for write tools
              safety_validator.py   - 14-pattern regex post-processor
              rag_chain.py          - legacy SimpleRAGChain for eval harness only
              vector_store.py       - lru_cache-wrapped ChromaDB loader
              knowledge_loader.py   - FEMA/CDC PDF chunker
              populate_db.py        - 4-modality ingestion pipeline
```

---

## API Endpoints (FastAPI)

- `GET  /`                    - root info
- `GET  /health/`             - liveness probe (Docker healthcheck)
- `GET  /health/ready`        - readiness probe (checks ChromaDB doc count)
- `GET  /health/env`          - env var check (no values exposed)
- `POST /agent/query`         - run agent, returns answer + tool_calls + cost_usd + latency_s
- `GET  /agent/history/{id}`  - conversation history for thread
- `DELETE /agent/history/{id}`- clear history
- `GET  /eval/results`        - eval CSV as JSON
- `GET  /eval/summary`        - aggregated per-config stats
- `GET  /eval/alerts`         - alerts.jsonl as JSON
- `DELETE /eval/alerts`       - clear alerts

---

## Docker Setup

- `Dockerfile.api`       - python:3.11-slim, installs requirements.txt + fastapi uvicorn langsmith
- `Dockerfile.dashboard` - python:3.11-slim, installs requirements.txt, runs streamlit
- `docker-compose.yml`   - two services: api (port 8000) + dashboard (port 8501)
  - api has healthcheck on /health/
  - dashboard depends_on api with condition: service_healthy
  - Shared volumes: chroma_db, data, logs
  - dashboard gets API_BASE_URL=http://api:8000 automatically

---

## Frontend (app.py) - Current State

### What Works
- 3-column layout: Live Feed | AI Analyst Chat | Alert Log
- Location presets: Boulder CO, Houston TX, New Orleans LA, Miami FL, Nashville TN
- Custom Location: geocodes any city via Open-Meteo API to real lat/lon
- Real-time Plotly gauges for river level and weather
- Chat interface with tool call badge display
- Cost summary in sidebar (reads cost_log.jsonl)
- Dual mode: standalone (direct agent import) OR API mode (calls FastAPI)
- Alert panel reading alerts.jsonl

### What Is Missing / Needs Improvement
1. NO evaluation/metrics tab - no way to view ablation results in dashboard
2. NO multi-tab layout - everything crammed into one page
3. NO system status page - no policy violations view, no cost history chart
4. CSS is minimal - dark theme but not polished or premium
5. NO real-time refresh for sensor data (manual button only)
6. NO Streamlit page_icon (was stripped as emoji)
7. NO GitHub CI/CD pipeline (no .github/workflows/)
8. NO Nginx reverse proxy config for production
9. run.bat exists but is not properly cross-platform
10. requirements.txt has fastapi/uvicorn but they are also redundantly in Dockerfile.api's pip install line

---

## Known Bugs Fixed

- 2026-06: `{current_date}` in ChatPromptTemplate string -> InvalidPromptInput crash. Fixed by _build_system_prompt() function with string concatenation.
- 2026-06: Emoji/unicode in 49 files causing UnicodeEncodeError on Windows cp1252. Fixed by strip_unicode.py.
- 2026-06: Custom Location preset hardcoded Boulder lat/lon regardless of typed city. Fixed by geocode_location() call in sidebar.
- 2026-06: sensor_tool.py had emoji in alarm strings. Fixed to ASCII [CRITICAL]/[DANGER]/[WARNING].
- 2026-06: duplicate `import os` in disaster_agent.py. Removed.
- 2026-04: era tagging in kb_search_tool.py checked "2013" in timestamp string, mislabeling 2012 events. Fixed to use temporal_tag metadata.
- 2026-04: safety_validator.py broken negative lookahead causing false negatives. Fixed.

---

## User Rules / Preferences

- No emoji or unicode characters in any code file (strictly enforced)
- Production-grade code only - no mock/synthetic data
- Free infrastructure only - Groq free tier, Open-Meteo free, USGS free
- IEEE-publishable results - all claims backed by real data
- Minimize disk usage - stream from archives rather than extracting fully
- No placeholder images - generate them
- Dark theme UI mandatory

---

## Session Goal (2026-07-21)

Complete frontend redesign + end-to-end deployment pipeline:
1. Redesign app.py with multi-page Streamlit layout (Command Center, Evaluation, System Status)
2. Add GitHub Actions CI/CD pipeline
3. Add Nginx reverse proxy config for production
4. Fix all remaining deployment gaps
5. Update memory and walkthrough
