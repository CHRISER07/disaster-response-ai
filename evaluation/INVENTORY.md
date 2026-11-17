# PHASE 0: SYSTEM AND REPOSITORY INVENTORY

**Generated Date:** 2026-11-12  
**Git Commit Hash:** `591e9657be4f8fda9624d7c17cd35ccaae93ce01`  
**Environment:** Python 3.12.1 (AMD64) | LangGraph 1.1.10 | LangChain 1.2.10 | Groq 0.37.1 | ChromaDB 1.5.2

---

## (a) Exact Model Identifiers & Infrastructure

| Role | Environment Variable / Code Source | Model Identifier | Provider | Temperature | Max Retries |
|---|---|---|---|---|---|
| **Primary Agent** | `GROQ_MODEL` / `agents/disaster_agent.py:162` | `qwen/qwen3.8-27b` | Groq LPU | 0.1 | 1 |
| **Fallback 1** | `GROQ_FALLBACK_MODEL` / `agents/disaster_agent.py:163` | `openai/gpt-oss-120b` | Groq LPU | 0.1 | 1 |
| **Fallback 2** | `agents/disaster_agent.py:164` | `openai/gpt-oss-20b` | Groq LPU | 0.1 | 2 |
| **Judge A (Same Family)** | `evaluation/eval_extended.py` | `qwen/qwen3.8-27b` | Groq LPU | 0.0 | 2 |
| **Judge B (Cross-Family)** | `evaluation/independent_judge.py:161` | `openai/gpt-oss-120b` (fallback `openai/gpt-oss-20b`) | Groq LPU | 0.0 | 2 |
| **Vision Tool Engine** | `tools/vision_tool.py:34` | `llava:7b` (local Ollama, port 11434) | Local Ollama | N/A | N/A |

*Note: Historical legacy scripts (`core/rag_chain.py` and `evaluation/run_evals.py`) hardcoded `llama-3.3-70b-versatile`, which now returns HTTP 404 on the Groq endpoint.*

---

## (b) Temporal Grounding Injection (TGI) Recency Window $\Delta t$ and Tagging Function

### Recency Window $\Delta t$
- **Live Hydrologic Telemetry (`tools/sensor_tool.py`):** $\Delta t = 0$ (instantaneous current reading from USGS NWIS REST endpoint `parameterCd=00065`).
- **Live Atmospheric Telemetry (`tools/weather_tool.py`):** $\Delta t = 0$ (instantaneous forecast and weather conditions from Open-Meteo).
- **Seismic Telemetry (`tools/sensor_tool.py:287`):** $\Delta t = 7\text{ days}$ (`start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime(...)`).
- **Knowledge Base Search (`tools/kb_search_tool.py:124-133`):** Categorical year-matching rule. There is **no rolling 24h/48h window**. Chunks are tagged based on metadata year and source:

```python
# Exact function from tools/kb_search_tool.py lines 124-133:
temporal_tag = meta.get("temporal_tag", "")
if temporal_tag:
    era = temporal_tag
elif meta.get("modality") == "knowledge_base" or "fema" in str(source).lower():
    era = "[KB - FEMA/CDC Protocol]"
elif ts and any(y in ts for y in ["2012", "2013", "2014", "2015"]):
    era = f"[HISTORICAL - {ts[:4]}]"
else:
    era = "[LIVE/RECENT]"
```

---

## (c) DSEL & Policy Engine Inventory

### 1. DSEL L1 Regex Patterns (`core/safety_validator.py`)
```python
UNSAFE_PATTERNS = [
    (r"\bit['']?s safe to (?:drive|cross|walk|wade)\b",         "Driving or wading through floodwater is NEVER safe."),
    (r"\bgo ahead and (?:drive|cross|enter)\b",                  "Do not enter or cross flooded areas."),
    (r"\bfloodwater is (?:clean|safe|drinkable|not contaminated)\b", "Floodwater is always presumed contaminated."),
    (r"\b3 (?:feet|foot|ft) of water is (?:fine|ok|okay|safe)\b","Any moving water above ankle depth is a drowning risk."),
    (r"\byou can safely cross\b",                                 "Never cross flooded roadways or waterways."),
    (r"\bsafe to return\b",                                       "Return only when authorities give the all-clear."),
    (r"\bno (?:risk|danger|hazard)\b",                            "Do not assume zero risk during an active disaster."),
    (r"\bcompletely safe\b",                                      "No area can be declared 'completely safe' during an active disaster."),
    (r"\bair (?:is|looks|seems) (?:clean|clear|fine|safe)\b",    "Do not self-assess air quality after a chemical event. Odorless chemicals are still lethal."),
    (r"\bsmell (?:has|is) gone\b",                                "Absence of smell does not mean absence of hazard. Many toxic gases are odorless."),
    (r"\bdon['']?t worry\b",                                      "Take all disaster warnings seriously."),
    (r"\bguaranteed(ly)? safe\b",                                  "No safety guarantee can be given during an active disaster."),
    (r"\bno need to evacuate\b",                                   "Evacuation decisions must be made by authorized officials only."),
]
```

### 2. DSEL L2 Citation Rule (`core/policy_engine.py` / `agents/disaster_agent.py`)
Requires each factual claim to include an inline tag matching:
- `[LIVE - USGS NWIS, YYYY-MM-DD]`
- `[LIVE - Open-Meteo, YYYY-MM-DD]`
- `[HISTORICAL - CrisisLex, 2013]`
- `[KB - FEMA/CDC Protocol]`
- `[Confidence: HIGH|MEDIUM|LOW]`

### 3. Anchor Grounding Ratio (AGR) Anchor List & Default (`core/policy_engine.py:80-99`)
- **Anchor Keywords:** `["gage height", "flood stage", "discharge", "threshold", "evacuat", "turn around", "usgs", "glofas", "fema", "cdc", "magnitude", "rainfall", "precipitation", "wind", "cfs", "m3/s", "meter", "feet", "ft"]`
- **Default Score when No Anchors Found:** **`0.90`** (line 89: `"Generic claim without numerical anchors: return 0.90"`).

### 4. Alert Thresholds & `check_policy()` Implementation (`core/policy_engine.py`)
- **Minimum Write Confidence:** `{"HIGH", "MEDIUM"}`
- **Minimum Ragas Faithfulness / AGR:** $\ge 0.85$
- **130% Flood Stage Rule:** Implemented in `tools/sensor_tool.py:91` (`if gage >= threshold_ft * 1.3: status = "CRITICAL EMERGENCY - Severe flood stage exceeded"`).

```python
# Exact check_policy function from core/policy_engine.py:
def check_policy(
    tool_name: str,
    tool_args: dict,
    agent_message: str = "",
    retrieved_contexts: Optional[List[str]] = None,
) -> PolicyDecision:
    if tool_name not in READ_TOOLS and tool_name not in WRITE_TOOLS:
        decision = PolicyDecision(allowed=False, tool_name=tool_name, reason=f"Unknown tool '{tool_name}'...")
        _log_violation(decision, tool_args)
        return decision

    if tool_name in READ_TOOLS:
        return PolicyDecision(allowed=True, tool_name=tool_name, reason="Read-only tool - permitted.")

    confidence = extract_confidence(agent_message, tool_args)
    if tool_name == "generate_structured_alert" and "severity" not in tool_args:
        decision = PolicyDecision(allowed=False, tool_name=tool_name, reason="Alert blocked: missing required 'severity' field.")
        _log_violation(decision, tool_args)
        return decision

    if confidence not in MINIMUM_WRITE_CONFIDENCE:
        decision = PolicyDecision(allowed=False, tool_name=tool_name, reason=f"Alert blocked: confidence is '{confidence}'...")
        _log_violation(decision, tool_args)
        return decision

    alert_msg = str(tool_args.get("message", "")) + " " + agent_message
    faithfulness = compute_ragas_faithfulness(alert_msg, retrieved_contexts)
    
    if faithfulness < MINIMUM_RAGAS_FAITHFULNESS:
        decision = PolicyDecision(allowed=False, tool_name=tool_name, reason=f"Alert blocked: Ragas Faithfulness score is {faithfulness:.2f} < {MINIMUM_RAGAS_FAITHFULNESS:.2f}")
        _log_violation(decision, tool_args)
        return decision

    return PolicyDecision(allowed=True, tool_name=tool_name, reason="Write tool permitted.", confidence_found=confidence, faithfulness_score=faithfulness)
```

---

## (d) Persistent Database Record Audit (`chroma.sqlite3`)

### 1. Count per Collection via Segments Join:
- `disaster_response`: **10,780 records**
- `disaster_multimodal_research`: **6,390 records**
- `disaster_docs`: **0 records**
- **Total across all collections:** **17,170 records** (exactly matches `SELECT COUNT(*) FROM embeddings`).

### 2. Breakdown by Source Type (`embedding_metadata`):
- `CrisisLex_T26` (Tweets): **7,800 records**
- `disaster-preparedness-training-manual.pdf` (FEMA): **4,562 chunks** (3,030 path + 1,532 basename)
- `field_operations_guide.pdf` (USAR): **2,063 chunks** (1,374 path + 689 basename)
- `CDC_PreparednesResponseCapabilities...pdf` (CDC): **2,043 chunks** (1,358 path + 685 basename)
- `crisis_lex_tweet` (additional seed tweets): **400 records**
- `Drone_VLM` (drone imagery summaries): **150 records**
- `IoT_Sensor` (water level records): **128 records**
- `USGS_Sensor` (gage readings): **24 records**
- **Sum:** **17,170 records**

### 3. Record Count by TGI Tag:
- `[KB - FEMA/CDC Protocol]`: **8,668 records** (50.5%)
- `[HISTORICAL -- 2013]`: **5,700 records** (33.2%)
- `[HISTORICAL -- 2012]`: **2,100 records** (12.2%)
- `[HISTORICAL - 2013]` (dynamically resolved from timestamp): **678 records** (3.9%)
- `[LIVE/RECENT]`: **24 records** (0.1%)
- **Total:** **17,170 records**

---

## (e) Retrieval Execution Path at Evaluation Time

**Finding:** **100% of retrieval calls executed via the SQLite FTS fallback (`_search_sqlite_fts()`).**

**Proof:**
1. In `core/vector_store.py:20`, initializing `HuggingFaceEmbeddings` attempts to import `sentence_transformers`.
2. Importing `sentence_transformers` fails with `ModuleNotFoundError: No module named 'torch.types'; 'torch' is not a package`.
3. In `tools/kb_search_tool.py:25-31`, `_cached_store()` catches this exception and returns `None`.
4. In `tools/kb_search_tool.py:108-112`, when `_cached_store()` is `None`, the function unconditionally routes all queries to `_search_sqlite_fts(query, modality_filter, k)`.
5. No dense embedding vector search ever executed during evaluation.

---

## (f) Step Limits, Context Size, and Root Cause of HTTP 413 & Step Limit Errors

1. **Agent Recursion Limit (`agents/disaster_agent.py:242`):**
   ```python
   config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 8}
   ```
   A recursion limit of 8 allows only 3-4 tool invocations before LangGraph raises `GraphRecursionError` or emits `"Sorry, need more steps to process this request"`.
2. **Root Cause of HTTP 413 ("Request Too Large"):**
   - In `agents/disaster_agent.py`, the checkpointer `SqliteSaver` in `data/checkpoints.sqlite` accumulates all conversation messages under a `thread_id`.
   - When tools like `query_recent_earthquakes` or `search_official_protocols` return long text blocks (multiple thousands of characters), and the agent makes 2-3 tool calls, the cumulative prompt sent to Groq exceeds Groq's TPM / request size ceiling.
   - When `eval_ragas_deepeval.py` retried a query with `thread_id=f"ragas_eval_{i}_retry"`, the previous failed trajectory or large context overwhelmed the payload.

---

## (g) Authoritative Flood Stage Verification: USGS Gage 06730200

- **Site:** USGS 06730200 / NWS ID: BOLC2 (Boulder Creek at North 75th St near Boulder, CO).
- **Authoritative NWS Flood Stages (National Weather Service / NOAA):**
  - **Action Stage:** **7.50 ft**
  - **Minor Flood Stage (Official Flood Stage):** **8.00 ft**
  - **Moderate Flood Stage:** **9.00 ft**
  - **Major Flood Stage:** **10.00 ft**
- **Authoritative Source:** NOAA / NWS Advanced Hydrologic Prediction Service (AHPS):  
  [https://water.weather.gov/ahps2/hydrograph.php?wfo=bou&gage=bolc2](https://water.weather.gov/ahps2/hydrograph.php?wfo=bou&gage=bolc2)
- **Discrepancy in Codebase:** `tools/sensor_tool.py:46` hardcodes `"threshold_ft": 10.0`. The codebase used **10.0 ft** (NWS Major Flood Stage) as its flood threshold rather than the official NWS Minor Flood Stage of **8.0 ft**.

---

## (h) Development vs. Held-Out Split Mapping

- **Total queries in `golden_dataset_200q.json`:** 190 (IDs 1–190).
- **Meta-queries removed:** IDs 176–190 (15 queries about the research paper/meta-architecture).
- **Eligible queries:** 175 queries (IDs 1–175).
- **Development Set (50 queries):** Mapped 1-to-1 from `golden_dataset.json` (IDs 1–50) to `golden_dataset_200q.json`:  
  `dev_ids` = `[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 33, 34, 36, 37, 38, 43, 47, 48, 50, 51, 56, 57, 58, 59, 60, 82, 83, 84, 85, 95, 106, 107, 108, 109, 110, 111, 136, 137, 138, 139, 140]`
  Saved to `evaluation/dev_ids.json`.
- **Held-Out Set (125 queries):** All remaining eligible IDs (IDs 1–175 excluding `dev_ids`):  
  `heldout_ids` = `[3, 21, 22, ..., 175]` (exactly 125 queries).  
  Saved to `evaluation/heldout_ids.json`.

---

## Difference between Config 3 (Text-Only Static RAG) and Config 4 (Standard Static RAG)

In `core/rag_chain.py` and `evaluation/run_evals.py`:
- **Config 3 (Text-Only Static RAG):** `SimpleRAGChain` initialized with `allowed_modalities=["text", "knowledge_base"]`. The retriever applies a metadata filter `{"modality": {"$in": ["text", "knowledge_base"]}}`, restricting retrieval strictly to CrisisLex tweets and FEMA/CDC PDFs. No sensor or imagery documents are accessible.
- **Config 4 (Standard Static RAG):** `SimpleRAGChain` initialized with `allowed_modalities=None`. The retriever performs unfiltered semantic search across all four stored modalities (`text`, `knowledge_base`, `telemetry`, `imagery_description`). Neither configuration invokes live REST APIs (USGS/Open-Meteo) or multi-step reasoning tools.

