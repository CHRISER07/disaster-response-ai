# ARIA — Agentic Response Intelligence for Disasters
## Complete System Architecture & Engineering Reference

**Version:** 2.0 (Production)
**Stack:** Python 3.11+ · LangGraph · ChromaDB · Groq (Llama-3.3-70B) · Streamlit · Open-Meteo · USGS APIs · Ollama LLaVA

---

## 1. System Overview

ARIA is a **multimodal, agentic AI system** for real-time disaster situational awareness.
It is NOT a chatbot. It is an autonomous AI analyst that:

1. Continuously monitors live data streams (weather, water sensors, seismic events)
2. Maintains a searchable memory of historical disaster reports, official protocols, and imagery
3. Reasons across ALL data sources simultaneously using a LangGraph ReAct loop
4. Generates structured, severity-tagged alerts when thresholds are crossed
5. Answers responder queries with source-cited, hallucination-free assessments

### What Makes This Different From a Standard RAG

| Standard RAG | ARIA Agentic System |
|---|---|
| Answers when asked | Can proactively detect critical conditions |
| Searches one database | Calls 7+ live tools autonomously |
| Returns retrieved text | Synthesizes cross-modal evidence |
| Static knowledge | Live USGS + Open-Meteo + USGS Earthquake APIs |
| No memory | Full conversation memory via LangGraph MemorySaver |
| No alerts | Generates formal structured alerts (logged to disk) |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD (app.py)                  │
│  ┌──────────────┐  ┌──────────────────────┐  ┌──────────────┐  │
│  │ Live Feed    │  │   ARIA Chat Panel    │  │ Alert Log    │  │
│  │ - Gauge: River│  │   (LangGraph Agent)  │  │ (alerts.jsonl│  │
│  │ - Weather card│  │   - Thought trace    │  │  CRITICAL/   │  │
│  │ - Quake list │  │   - Tool badges      │  │  HIGH/MEDIUM │  │
│  └──────┬───────┘  └──────────┬───────────┘  └──────────────┘  │
└─────────┼────────────────────┼─────────────────────────────────┘
          │                    │
          ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│              ARIA LANGGRAPH ReAct AGENT                          │
│                  (agents/disaster_agent.py)                      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  SYSTEM PROMPT: ARIA                                       │  │
│  │  - Role: Expert disaster analyst                           │  │
│  │  - Temporal Bridge: Distinguish 2013 historical vs live    │  │
│  │  - Output Format: Status / Evidence / Recommendation       │  │
│  │  - Rules: Always cite sources. Call alert if CRITICAL.     │  │
│  └───────────────────────┬────────────────────────────────────┘  │
│                          │                                        │
│  ┌───────────────────────▼────────────────────────────────────┐  │
│  │           QUERY ROUTER (tools/routing.py)                  │  │
│  │  Classifies query intent BEFORE entering ReAct loop:       │  │
│  │  - "live_weather"  → skip ChromaDB, go to weather_tool     │  │
│  │  - "live_sensor"   → skip ChromaDB, go to sensor_tool      │  │
│  │  - "knowledge_base" → search_official_protocols only       │  │
│  │  - "social"        → search_social_reports only            │  │
│  │  - "full_search"   → all tools available                   │  │
│  └───────────────────────┬────────────────────────────────────┘  │
│                          │                                        │
│         ┌────────────────┼────────────────┐                      │
│         │   ReAct Loop   │                │                      │
│         │  Reason → Act → Observe → Loop  │                      │
│         └────────────────┬────────────────┘                      │
│                          │                                        │
│         ┌────────────────▼────────────────┐                      │
│         │      TOOL REGISTRY (7 Tools)    │                      │
│         └────────────────┬────────────────┘                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────────────────────┐
        │                  │                                   │
        ▼                  ▼                                   ▼
┌───────────────┐ ┌────────────────────┐ ┌───────────────────────┐
│  LIVE APIs    │ │  CHROMADB (Local)  │ │   LOCAL AI (Ollama)   │
│               │ │                    │ │                        │
│ Open-Meteo    │ │  Collection:       │ │  LLaVA 7B             │
│  Weather API  │ │  "disaster_response│ │  - Analyzes real       │
│  (free, no key│ │                    │ │    xView2 satellite    │
│               │ │  ┌──────────────┐  │ │    disaster images     │
│ USGS NWIS     │ │  │modality=     │  │ │  - Returns structured  │
│  River gauges │ │  │"knowledge_   │  │ │    SUMMARY/DAMAGE/     │
│  (7 US sites) │ │  │ base" (FEMA) │  │ │    ACCESS assessment   │
│               │ │  ├──────────────┤  │ │                        │
│ USGS EQ API   │ │  │modality=     │  │ │  Endpoint:             │
│  Earthquake   │ │  │"text"        │  │ │  localhost:11434/      │
│  events M≥3.0 │ │  │(CrisisLex)   │  │ │  api/generate          │
│  (7 day window│ │  ├──────────────┤  │ └───────────────────────┘
└───────────────┘ │  │modality=     │  │
                  │  │"telemetry"   │  │
                  │  │(USGS sensor  │  │
                  │  │ snapshots)   │  │
                  │  ├──────────────┤  │
                  │  │modality=     │  │
                  │  │"imagery_desc"│  │
                  │  │(VLM outputs) │  │
                  │  └──────────────┘  │
                  │                    │
                  │  Embedding Model:  │
                  │  BAAI/bge-small-   │
                  │  en-v1.5 (local)   │
                  └────────────────────┘
```

---

## 3. Data Modalities & Sources

### 3.1 Social Media / Crisis Text (Historical)
- **Source:** CrisisLex T26 Dataset (2013 Colorado Floods)
- **Format:** CSV → LangChain `Document` objects
- **Key Feature:** Twitter Snowflake ID → precise UTC timestamps
- **Filtering:** Only "Related and informative" tweets
- **Volume:** Up to 1,500 documents
- **Modality tag:** `"text"`

### 3.2 Official Knowledge Base (Static)
- **Source:** 3 PDFs — CDC Preparedness Manual, FEMA Field Operations Guide, Disaster Training Manual
- **Format:** PDF → PyPDF → RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Volume:** ~3,000–5,000 chunks across all PDFs
- **Modality tag:** `"knowledge_base"`

### 3.3 IoT Sensor Data (Live + Cached)
- **Source:** USGS National Water Information System (NWIS) REST API
- **Endpoint:** `https://waterservices.usgs.gov/nwis/iv/`
- **Parameter:** `00065` = Gage Height in feet
- **Strategy:**
  - Fetches last 7 days of real USGS data on every `populate_db.py` run
  - Saves to `data/iot/water_levels.csv` as local cache
  - Only WARNING + CRITICAL readings are embedded (NORMAL readings = noise)
  - Also embeds last 24 readings regardless of status (current conditions)
- **Modality tag:** `"telemetry"`

### 3.4 Drone / Satellite Imagery (Vision AI)
- **Source:** xView2 Dataset (real satellite disaster imagery, ~2.7GB tar)
- **Processing:**
  1. `extract_xview2_subset.py` → extracts 50 real images from tar
  2. `vlm_processor.py` → sends each image to Ollama LLaVA → structured description
  3. Results saved to `data/imagery/real_drone_metadata.json`
- **Requires:** Ollama installed + `ollama pull llava` (4.7GB, one-time)
- **Modality tag:** `"imagery_description"`

### 3.5 Live Weather (Real-time)
- **Source:** Open-Meteo API (free, no API key required)
- **Geocoding:** Open-Meteo Geocoding API resolves any city name to lat/lon
- **Parameters:** temperature_2m, rain, wind_speed_10m, weather_code, humidity, feels_like
- **NOT embedded in ChromaDB** — always fetched live by the agent tool

### 3.6 Seismic Data (Real-time)
- **Source:** USGS Earthquake Hazards Program API
- **Endpoint:** `https://earthquake.usgs.gov/fdsnws/event/1/query`
- **Window:** Last 7 days, M≥3.0, within 300km of location
- **NOT embedded in ChromaDB** — always fetched live by the agent tool

---

## 4. The 7 Agent Tools (Complete Reference)

### Tool 1: `fetch_live_weather`
```
File: tools/weather_tool.py
Input: location (str) — any city name worldwide
Output: Temperature, rainfall, wind, conditions, severity flag
API: Open-Meteo (free, no key)
Routing: Direct — never goes through ChromaDB
```

### Tool 2: `query_water_sensor`
```
File: tools/sensor_tool.py
Input: location (str) — any city with a known USGS site
Output: Real-time gage height, flood status, % above/below threshold
API: USGS NWIS IV (free, no key)
Critical Behavior: If gage ≥ 130% threshold → appends
  "⚠️ CRITICAL: IMMEDIATE EVACUATION PROTOCOL REQUIRED."
Routing: Direct — never goes through ChromaDB
```

### Tool 3: `query_recent_earthquakes`
```
File: tools/sensor_tool.py
Input: location (str), min_magnitude (float, default 3.0)
Output: List of M≥threshold events in last 7 days within 300km
API: USGS Earthquake API (free, no key)
Routing: Direct — never goes through ChromaDB
```

### Tool 4: `search_official_protocols`
```
File: tools/kb_search_tool.py
Input: query (str)
Output: Top 5 chunks from FEMA/CDC manuals only
Filter: modality = "knowledge_base" (NEVER returns tweets)
Use: For official procedures, evacuation protocols, medical triage
```

### Tool 5: `search_social_reports`
```
File: tools/kb_search_tool.py
Input: query (str)
Output: Top 5 CrisisLex tweets + sensor snapshots
Filter: modality IN ["text", "telemetry", "imagery_description"]
Use: For ground-truth field reports, survivor accounts, damage sightings
```

### Tool 6: `analyze_disaster_image`
```
File: tools/vision_tool.py
Input: image_filename (str) — file from data/imagery/samples/
Output: Structured analysis:
  SUMMARY: One-line severity summary
  DAMAGE: Infrastructure damage description
  ACCESS: Road/access status for emergency vehicles
  FULL_DESC: Complete LLaVA description
API: Ollama LLaVA (localhost:11434, free, requires setup)
Fallback: Returns instructive error if Ollama not running
```

### Tool 7: `generate_structured_alert`
```
File: tools/alert_tool.py
Input: severity (LOW/MEDIUM/HIGH/CRITICAL), message (str), zone (str)
Output: Alert logged to data/alerts.jsonl + confirmation string
Behavior: Dashboard reads this file in real-time — alert appears immediately
```

---

## 5. LangGraph Agent Architecture

### ReAct Loop (Reason → Act → Observe)

```
User Query
    │
    ▼
┌───────────────────────────────────────────┐
│  Query Intent Router                      │
│  Classifies: live_sensor / live_weather / │
│  knowledge_base / social / full_search    │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│  REASON NODE (LLM: Llama-3.3-70B)        │
│  "What information do I need?"            │
│  "Which tools should I call?"             │
│  "Do I have enough to answer?"            │
└───────────────────┬───────────────────────┘
                    │  (tool calls)
                    ▼
┌───────────────────────────────────────────┐
│  ACT NODE (Tool Executor)                 │
│  Calls selected tools in parallel or seq  │
│  Captures results as ToolMessages         │
└───────────────────┬───────────────────────┘
                    │  (observations)
                    ▼
┌───────────────────────────────────────────┐
│  OBSERVE → REASON (loop)                  │
│  LLM reads tool outputs                   │
│  Decides: enough info? → Answer           │
│           not enough?  → Call more tools  │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│  FINAL ANSWER                             │
│  Format:                                  │
│  - Status: NORMAL / WARNING / DANGER /    │
│            CRITICAL                       │
│  - Evidence: [cited data points]          │
│  - Recommendation: [action items]         │
│  - Alert Generated: [yes/no + severity]   │
└───────────────────────────────────────────┘
```

### Memory Architecture
- **Type:** `MemorySaver` (in-process, RAM)
- **Scope:** Per `thread_id` — each session maintains its own conversation history
- **Persistence:** Lives for the duration of the Streamlit session
- **Thread ID:** Dashboard uses `"dash"`, CLI uses `"cli_session"`

### Temporal Context Bridge
The system prompt instructs ARIA to distinguish data eras:
- **Historical data** (CrisisLex tweets, xView2 images): Labeled as `[HISTORICAL - 2013]` context
- **Live API data** (USGS, Open-Meteo, USGS EQ): Labeled as `[LIVE - current]`
- **Rule:** If historical and live data conflict, ALWAYS prioritize live for current safety assessments
- **Use of historical:** Pattern recognition only ("In 2013, Boulder Creek rose 5ft in 2 hours under similar conditions")

---

## 6. Vector Database Schema

### ChromaDB Collection: `disaster_response`
```
Embedding Model: BAAI/bge-small-en-v1.5
Dimensions: 384
Distance: Cosine (normalized L2)
Search Type: MMR (Maximal Marginal Relevance) — reduces redundancy
Parameters: fetch_k=20, lambda_mult=0.7, k=7

Document Schema:
{
  "page_content": "<the actual text for embedding>",
  "metadata": {
    "modality": "text" | "knowledge_base" | "telemetry" | "imagery_description",
    "source": "<source identifier>",
    "timestamp": "<ISO 8601 timestamp>",
    "source_type": "<optional sub-classification>",
    ... (modality-specific fields)
  }
}
```

### Modality Filter Examples (ChromaDB $in operator)
```python
# Protocols only
{"modality": {"$in": ["knowledge_base"]}}

# Field reports only
{"modality": {"$in": ["text", "telemetry", "imagery_description"]}}

# Full search (no filter)
# (pass no filter parameter)
```

---

## 7. Data Flow: From Raw Data to Agent Answer

### Step 1: One-Time Setup (Run Once)
```
[xView2 tar] ─► extract_xview2_subset.py ─► data/imagery/samples/*.png (50 images)
                                                      │
                                                      ▼
                                            vlm_processor.py
                                            (Ollama LLaVA)
                                                      │
                                                      ▼
                                        data/imagery/real_drone_metadata.json
```

### Step 2: Database Population (Run Once or --reset)
```
python core/populate_db.py
         │
         ├─[1/4]─► text_loader.py ──────────► CrisisLex CSV ──────────┐
         │                                                              │
         ├─[2/4]─► iot_loader.py ──────────► USGS API (live) ────────►├──► ChromaDB
         │                                    │ (cache to CSV)         │    "disaster_response"
         ├─[3/4]─► drone_loader.py ─────────► real_drone_metadata.json►│    collection
         │                                                              │
         └─[4/4]─► knowledge_loader.py ─────► data/kb/*.pdf ──────────┘
```

### Step 3: Runtime (Every Query)
```
User: "Should we evacuate near Boulder River?"
         │
         ▼
  Query Router: intent = "full_search"
         │
         ▼
  ARIA ReAct Agent begins reasoning:
  ┌─────────────────────────────────────────────────────┐
  │ Reason: "I need water level + weather + protocols"  │
  └───────────┬──────────────────────────────┬──────────┘
              │                              │
              ▼                              ▼
  query_water_sensor("Boulder")    fetch_live_weather("Boulder, CO")
  → USGS API call                  → Open-Meteo API call
  → "8.2 ft — WARNING"             → "Temp: 12°C, Rain: 3.2mm/hr"
              │                              │
              └──────────────┬───────────────┘
                             ▼
              search_official_protocols("flood evacuation")
              → ChromaDB query (knowledge_base only)
              → Returns FEMA Protocol 7.3
                             │
                             ▼
              LLM synthesizes all 3 observations:
              ┌──────────────────────────────────────────┐
              │ Status: WARNING                           │
              │ Evidence:                                 │
              │  - River: 8.2ft (82% of 10ft threshold)  │
              │  - Weather: 3.2mm/hr rain continuing     │
              │  - FEMA Protocol 7.3: Pre-position evac  │
              │ Recommendation: Issue WARNING alert,      │
              │  pre-position evacuation resources        │
              │ Alert Generated: YES — MEDIUM severity    │
              └──────────────────────────────────────────┘
                             │
                             ▼
              generate_structured_alert(severity="MEDIUM", ...)
              → Writes to data/alerts.jsonl
              → Dashboard alert panel updates instantly
```

---

## 8. API Connectivity Reference

| API | URL | Auth | Rate Limit | Used By |
|---|---|---|---|---|
| Open-Meteo Weather | `api.open-meteo.com/v1/forecast` | None | Unlimited | `weather_tool.py` |
| Open-Meteo Geocoding | `geocoding-api.open-meteo.com/v1/search` | None | Unlimited | `weather_tool.py` |
| USGS NWIS Real-time | `waterservices.usgs.gov/nwis/iv/` | None | Unlimited | `sensor_tool.py`, `iot_loader.py` |
| USGS Earthquake | `earthquake.usgs.gov/fdsnws/event/1/query` | None | Unlimited | `sensor_tool.py` |
| Groq API | `api.groq.com` | API Key | 30 RPM (free) | `disaster_agent.py`, `rag_chain.py` |
| Ollama LLaVA | `localhost:11434/api/generate` | None | Local GPU/CPU | `vision_tool.py`, `vlm_processor.py` |

---

## 9. File Structure (Canonical)

```
disaster_response_rag/
│
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt                # All Python dependencies
├── .env                            # GROQ_API_KEY (never commit to git)
├── .env.example                    # Template for new users
├── .gitignore                      # Must include: .env, chroma_db/, venv/
├── run.bat                         # Windows one-click launcher
│
├── agents/
│   ├── __init__.py                 # Exports run_agent
│   └── disaster_agent.py          # LangGraph ReAct agent (ARIA core)
│
├── tools/
│   ├── __init__.py                 # Exports ALL_TOOLS list (8 tools)
│   ├── weather_tool.py            # Open-Meteo live weather
│   ├── sensor_tool.py             # USGS water + earthquake sensors
│   ├── kb_search_tool.py          # ChromaDB search (filtered tools)
│   ├── vision_tool.py             # Ollama LLaVA image analysis
│   └── alert_tool.py              # Structured alert logger
│
├── core/
│   ├── vector_store.py            # ChromaDB connection + helpers
│   ├── knowledge_loader.py        # PDF → chunks for ChromaDB
│   ├── rag_chain.py               # Legacy SimpleRAGChain (for evaluation)
│   └── populate_db.py             # Master ingestion pipeline
│
├── ingestion/
│   ├── text_loader.py             # CrisisLex tweets → Documents
│   ├── iot_loader.py              # USGS API → sensor Documents
│   ├── drone_loader.py            # VLM JSON → imagery Documents
│   ├── vlm_processor.py           # Ollama LLaVA batch image processor
│   └── extract_xview2_subset.py   # xView2 tar → sample images
│
├── data/
│   ├── text/CrisisLexT26/...      # CrisisLex dataset (2013 Colorado Floods)
│   ├── iot/water_levels.csv       # Cached USGS sensor readings
│   ├── imagery/
│   │   ├── samples/               # Extracted xView2 disaster images
│   │   ├── drone_metadata.json    # (legacy - mock data, ignored)
│   │   └── real_drone_metadata.json # VLM-processed descriptions
│   └── kb/                        # FEMA/CDC PDF manuals
│       ├── CDC_Preparedness...pdf
│       ├── disaster-preparedness-training-manual.pdf
│       └── field_operations_guide.pdf
│
├── evaluation/
│   ├── golden_dataset.json        # 30 human-written Q&A pairs
│   ├── run_evals.py               # IEEE evaluation harness
│   ├── results.md                 # Generated evaluation report
│   └── results_per_query.csv      # Per-query scores
│
├── chroma_db/                     # ChromaDB persistent storage (auto-created)
└── tests/                         # Unit tests
```

---

## 10. Quick Start Guide

### Prerequisites
```bash
# 1. Python 3.11+
python --version

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key (free at console.groq.com)
# Edit .env:
GROQ_API_KEY=gsk_your_key_here

# 4. (Optional, for real vision AI) Install Ollama
# Download: https://ollama.ai
ollama pull llava   # 4.7GB, one-time download
```

### First-Time Database Population
```bash
# This fetches live USGS data, processes PDFs, loads tweets
python core/populate_db.py --reset

# Expected output:
#   [OK] text              : 847 documents
#   [OK] telemetry         : 34 documents
#   [OK] imagery_description: 50 documents   (only if Ollama ran)
#   [OK] knowledge_base    : 4,231 documents
#   TOTAL                  : 5,162 documents
```

### Running the Dashboard
```bash
streamlit run app.py
# Opens at: http://localhost:8501
```

### Running Evaluation
```bash
# Quick test (10 queries)
python evaluation/run_evals.py --quick

# Full evaluation (all queries, ~30 min)
python evaluation/run_evals.py
```

---

## 11. Known Constraints & Design Decisions

### Why Groq (not OpenAI)?
Groq provides free inference for Llama-3.3-70B at 30 RPM — sufficient for a research prototype. The system is LLM-agnostic; swapping to OpenAI requires changing one line in `disaster_agent.py`.

### Why ChromaDB (not Pinecone/Weaviate)?
Local ChromaDB requires zero cloud credentials, zero cost, and works offline. For production deployment, the `get_vector_store()` function is the only place that needs to change.

### Why BAAI/bge-small-en-v1.5?
Best-in-class performance at 384 dimensions — runs entirely on CPU in ~100ms/batch. No GPU required. Normalized embeddings enable cosine similarity with no post-processing.

### Why MMR over simple similarity search?
Disaster queries often return highly redundant chunks (multiple tweets about the same event). MMR's diversity penalty ensures the 7 returned chunks cover different aspects, improving synthesis quality.

### Temporal Data Gap (Design Decision)
CrisisLex tweets are from 2013. USGS APIs return live 2026 data. This is **intentional** — the system demonstrates that an AI can correctly handle temporal heterogeneity. The agent's system prompt explicitly instructs it to label data eras and prioritize live data for safety decisions while using historical data for pattern matching.

### Ollama Optional
The system runs fully without Ollama. `analyze_disaster_image` returns a helpful error message if Ollama is unavailable. All other 6 tools work regardless. Vision capability is a graceful enhancement.

---

## 12. Security & Production Notes

- **`.env` must never be committed.** The `.gitignore` includes `.env` — verify before any push.
- **Groq API key rotation:** If you see 401/403 errors, generate a new key at `console.groq.com` and update `.env`.
- **USGS API:** No auth required but be respectful — do not hammer with <1s intervals. All tools use 8–15s timeouts.
- **ChromaDB is local-only.** In production, replace with a cloud vector DB. The `get_vector_store()` function is the only change point.
- **LangGraph MemorySaver** stores conversation history in RAM. It is NOT persistent across app restarts. For persistent memory, replace with `SqliteSaver`.

---

## 13. Evaluation Methodology (IEEE Paper Reference)

### Three Ablation Configurations
| Config | What It Tests |
|---|---|
| **LLM Only** | Baseline — Llama-3.3-70B with no retrieval at all |
| **Text RAG** | Retrieval from text + knowledge_base modalities only |
| **Multimodal RAG** | Full retrieval across all 4 modalities |

### Metrics
| Metric | Type | What It Measures |
|---|---|---|
| **KW-F1** | Objective (no LLM) | Keyword match against golden answers |
| **DR-Fact** | LLM-as-Judge | Disaster-specific factual accuracy (0–1) |
| **SAA** | LLM-as-Judge | Situational awareness accuracy (0–1) |
| **SVC** | Rule-based | Safety violation count (unsafe advice given) |
| **P@K** | Retrieval | Precision of retrieved docs at K=7 |

### Golden Dataset
- 30 human-written query/answer pairs
- Covers: evacuation, water safety, shelter, medical, infrastructure, protocols
- Each item has `expected_keywords`, `ground_truth_answer`, `safe_response` flag

---

*This document reflects the production architecture as of ARIA v2.0.*
*All API calls use public, free endpoints. No billing credentials required except Groq API key (free tier).*
