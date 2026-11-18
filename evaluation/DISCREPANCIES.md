> **PARTLY SUPERSEDED.** D2 (dense retrieval) and D4 (TGI) were subsequently
> addressed in code; D5 was updated. See `AUDIT_FINDINGS.md` and
> `CURRENT_RESULTS.md` for current status.

# Discrepancies Report (historical)

Generated at build time. All items confirmed from code inspection.

## D1: Flood Stage Value
- Code uses: 10.0 ft (NWS Major Flood Stage for USGS 06730200)
- NWS AHPS official: Minor Flood Stage = 8.0 ft; Action Stage = 6.5 ft
- Impact: The agent is calibrated against the major flood threshold, not minor. At readings between 8.0–10.0 ft the agent will NOT alert, even though NWS considers that minor flooding.
- Source: https://water.noaa.gov/gauges/BCOC2 (NWS AHPS, accessed this session via web search)

## D2: Dense Vector Retrieval Broken
- Documentation claims: Dense vector retrieval via ChromaDB + SentenceTransformers
- Actual behavior: ALL retrieval routes to SQLite FTS fallback. Import fails:  
  `torch.types is not a package` — sentence-transformers dependency requires PyTorch which is not installed in the venv.
- Impact: Text RAG and Standard RAG configs are both using FTS, not vector similarity. Semantic retrieval quality is degraded.

## D3: Legacy Model References
- run_evals.py (original): used `llama-3.3-70b-versatile` (HTTP 404 on Groq since model was removed)
- Fixed in: `core/rag_chain.py` → `qwen/qwen3.8-27b` (this session)
- Remaining: `run_evals.py` still references old model; legacy reports from that file are invalid.

## D4: TGI Implementation Gap
- Documented: Rolling time-window Δt-based tagging (e.g., within 72 hours = LIVE)
- Actual: Categorical year matching — 2012–2015 → [HISTORICAL], KB docs → [KB], all else → [LIVE/RECENT]. No actual timestamp comparison.

## D5: recursion_limit
- Was: 8 (too low; multi-tool queries frequently hit limit → STEP_LIMIT status)
- Fixed: 18 (this session) in `agents/disaster_agent.py`

## D6: Ragas Scores
- Previous report: faithfulness=0.859, relevance=0.433
- Status: INVALID. `eval_ragas_deepeval.py` assigns hardcoded defaults (0.85/0.40) when judge LLM unavailable. Scores do not reflect actual Ragas LLM evaluation.
