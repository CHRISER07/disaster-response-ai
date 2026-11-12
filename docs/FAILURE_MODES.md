# Failure Modes - Disaster Response AI
*Production failure analysis, observed behaviours, and mitigation strategies*

> This document is written from real observed system behavior during development and load testing.
> It is the basis for the "tell me about a production incident" interview answer.

---

## Failure Mode 1: Groq Rate Limit - Exponential Backoff in Action

### What Happens
Groq's free tier enforces rate limits of ~30 requests/minute for `llama-3.3-70b-versatile`.
When the evaluation harness runs 4 configs  58 queries back-to-back, it reliably hits this limit
around query 12-15 of each configuration pass.

### Observed Behavior (from run_evals.py logs)
```
[Q012] [STATIC] What are the FEMA protocols for flash floods?
  Attempt 1: RateLimitError - 429 Too Many Requests
  Sleeping 60s before retry...
  Attempt 2: RateLimitError - 429 Too Many Requests
  Sleeping 120s before retry...
  Attempt 3: SUCCESS (latency: 4.2s)
  KW-F1=0.712  DR-Fact=0.800  SAA=0.750  SVC=0  4.2s
```

### Root Cause
The LLM-as-Judge makes a *second* Groq call per query (judge call after generation).
With `sleep_before_judge=2s` and `sleep_after_query=5s`, the effective rate was
still ~8 req/min - too fast for the burst limit.

### Fix Applied
```python
# evaluation/run_evals.py - exponential backoff
RETRY_DELAYS = [60, 120, 240, 480, 960]
for attempt, delay in enumerate(RETRY_DELAYS):
    try:
        result = llm.invoke(query)
        break
    except RateLimitError:
        if attempt < len(RETRY_DELAYS) - 1:
            time.sleep(delay)
        else:
            raise
```

### What I'd Change for Real Production
- **Circuit breaker pattern**: After 3 consecutive failures, stop calling the same endpoint
  and immediately fall back to a smaller local model (e.g., `llama-3.1-8b-instant` on Groq)
  instead of waiting 960 seconds
- **Request queue + worker pool**: Decouple query submission from LLM calls. Queue jobs and
  process at a controlled rate (e.g., 20 req/min) using `asyncio.Semaphore`
- **Paid tier or secondary provider**: Budget $10/month on Groq Pro, or add Anthropic/OpenAI
  as a fallback provider using LangChain's `ChatOpenAI` with the same interface

---

## Failure Mode 2: ChromaDB Embedding Model Cold Start

### What Happens
On first startup, `get_embeddings()` downloads BAAI/bge-small-en-v1.5 from HuggingFace Hub
(~34MB). This takes 15-45 seconds depending on network speed. During this window, any
ChromaDB query fails with a timeout.

### Observed Behavior
```
[Streamlit] User opens dashboard  clicks "Ask AI"
[Error] ChromaDB: embedding function returned None for 1 documents
[Root cause] HuggingFace model download in progress (33% complete)
```

### Fix Applied
```python
# core/vector_store.py
@functools.lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Model is only loaded once per process - subsequent calls return cached instance."""
    ...
```
The `lru_cache` prevents re-downloads on subsequent calls. But the first cold start still
blocks the first query.

### What I'd Change for Real Production
- **Dockerfile pre-download**: Add `RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"` to the Dockerfile so the model is baked into the image layer - no download at runtime
- **Readiness probe**: The `/health/ready` endpoint checks `store._collection.count() > 0`. Kubernetes would not route traffic until this returns 200
- **Cache to persistent volume**: Mount `/root/.cache/huggingface` as a Docker volume so the model survives container restarts

---

## Failure Mode 3: USGS API Silent Fallback

### What Happens
`sensor_tool.py` has 7 hardcoded gauge sites. When a query mentions a city not in the
list (e.g., "Sacramento, CA"), the tool falls back to Boulder Creek (gauge `06730200`)
**without telling the user**. The agent then confidently cites "Boulder Creek, CO" data
for a Sacramento flood query.

### Observed Behavior (evaluation Q38 - Typhoon coastal zones)
```
Agent called: query_water_sensor("Sacramento, CA")
Tool returned: [Boulder Creek, CO - USGS 06730200 - 3.2 ft - NORMAL]
Agent answered: "Current gage height at Boulder Creek is 3.2 ft - NORMAL."
Expected: Agent should say "USGS data unavailable for Sacramento - using nearest known gauge"
```

### Fix Status
The silent fallback was replaced in `sensor_tool.py` with an explicit error message:
```python
if site_id is None:
    return f"[SENSOR ERROR] No USGS gauge configured for '{city}'. " \
           f"Available cities: {', '.join(KNOWN_SITES.keys())}. " \
           f"The agent should report this data gap explicitly."
```

### What I'd Change for Real Production
- Implement dynamic USGS nearest-gauge lookup using the USGS Site Service API:
  ```
  GET https://waterservices.usgs.gov/nwis/site/
      ?format=json&stateCd={state}&siteType=ST&siteStatus=active
  ```
  Geocode the city, find the nearest active gauge by geodesic distance, and use it dynamically

---

## Failure Mode 4: Ollama LLaVA Not Running (Vision Tool Degradation)

### What Happens
`analyze_disaster_image` requires a local Ollama server with LLaVA 7B loaded.
When Ollama is not running, the tool raises a `ConnectionRefusedError` that
propagates up to the agent, which then crashes instead of degrading gracefully.

### Observed Behavior
```
User: "Show me the satellite imagery analysis"
Agent called: analyze_disaster_image(...)
Tool raised: ConnectionRefusedError: [Errno 111] Connection refused (localhost:11434)
Agent error: ConnectionRefusedError - agent stopped responding
```

### Fix Applied
```python
# tools/vision_tool.py
try:
    response = requests.post("http://localhost:11434/api/generate", ...)
except requests.exceptions.ConnectionError:
    return (
        "[VISION UNAVAILABLE] Ollama is not running. "
        "Start with: ollama serve && ollama pull llava:7b\n"
        "Proceeding with text-based analysis only."
    )
```
The tool now returns a descriptive string instead of raising, so the agent
can self-correct and answer without the vision modality.

### What I'd Change for Real Production
- Add Ollama as a service in `docker-compose.yml`:
  ```yaml
  ollama:
    image: ollama/ollama:latest
    volumes: ["ollama_models:/root/.ollama"]
    ports: ["11434:11434"]
  ```
- Pre-pull LLaVA in the Dockerfile entrypoint: `ollama pull llava:7b`
- Add `/health/vision` endpoint that pings Ollama and returns its status

---

## Failure Mode 5: MemorySaver Memory Leak (Long Sessions)

### What Happens
LangGraph's `MemorySaver` stores the full message history for every thread_id
**in memory** (Python dict). In a multi-user Streamlit deployment, each user
session creates a new thread_id whose history grows unboundedly.

After 100 turns in a single session, the serialized message history can reach
50-100KB per thread. At 1,000 concurrent sessions, that's 50-100MB of RAM
consumed by conversation history alone.

### What I'd Change for Real Production
- Replace `MemorySaver` with `PostgresSaver` or `RedisSaver` from `langgraph-checkpoint`:
  ```python
  from langgraph.checkpoint.postgres import PostgresSaver
  memory = PostgresSaver.from_conn_string(os.environ["POSTGRES_URL"])
  ```
- Add a session TTL: purge threads inactive for >24 hours
- Cap message history to last N turns using a sliding window trim in the agent graph

---

## Summary Table

| Failure Mode | Severity | Status | Production Fix |
|---|---|---|---|
| Groq rate limit (429) | HIGH | [OK] Mitigated (backoff) | Circuit breaker + fallback model |
| Embedding cold start | MEDIUM | [OK] Mitigated (lru_cache) | Bake into Docker image layer |
| USGS silent fallback | HIGH | [OK] Fixed (explicit error) | Dynamic nearest-gauge lookup |
| Ollama not running | MEDIUM | [OK] Fixed (graceful degrade) | Add Ollama to docker-compose |
| MemorySaver leak | LOW | [WARN] Known issue | PostgresSaver + TTL |
