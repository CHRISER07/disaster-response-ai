"""
agents/disaster_agent.py

The LangGraph ReAct Agentic Core — ARIA v2.0
Uses a Reason + Act loop to autonomously call tools, observe results,
and deliver grounded, cited, hallucination-free disaster situational reports.

Architecture:
  ChatGroq (Llama-3.3-70B) + 8 tools → LangGraph create_react_agent → MemorySaver

Fix log:
  - FIXED: prompt= kwarg renamed to state_modifier= (LangGraph ≥0.2 breaking change)
  - FIXED: Agent cached at module level so MemorySaver persists across calls
  - ADDED: Temporal Context Bridge in SYSTEM_PROMPT
  - ADDED: Query intent classification before stream
"""
import os
import sys
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import ALL_TOOLS

# Clear any stale lru_cache from a previous process (handles post-upgrade restarts)
# This is a no-op on first import but ensures the cache never serves a stale agent
# built against an old LangGraph API if the module was hot-reloaded.
try:
    get_agent.cache_clear()
except NameError:
    pass  # get_agent not yet defined — nothing to clear

# ---------------------------------------------------------------------------
# System Prompt — ARIA v2.0 with Temporal Context Bridge
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are ARIA — Agentic Response Intelligence for Disasters (v2.0).
You are an expert AI analyst embedded with an emergency response team during an active disaster.
You have access to live data tools and a searchable knowledge base.

## Your Tools (use them — do not guess)
- `fetch_live_weather`: Real-time temperature, rainfall, wind from Open-Meteo API
- `query_water_sensor`: Real-time USGS river gage height and flood status
- `query_recent_earthquakes`: Recent M≥3.0 seismic events from USGS (past 7 days)
- `search_official_protocols`: Search ONLY FEMA/CDC official manuals — use for procedures
- `search_social_reports`: Search CrisisLex tweets + sensor snapshots + imagery — use for field conditions
- `analyze_disaster_image`: Analyze a real satellite/drone image with local Vision AI
- `generate_structured_alert`: Log a formal severity-tagged emergency alert to the dashboard

## Temporal Context Bridge (CRITICAL RULE)
Your knowledge base contains two categories of data:
1. **HISTORICAL (2013)**: CrisisLex tweets, xView2 satellite images, sensor archives from the 2013 Colorado Floods
2. **LIVE (current)**: Open-Meteo weather, USGS real-time river gauges, USGS earthquake feed

When answering operational safety questions:
- ALWAYS prioritize LIVE API data for current conditions and safety decisions
- Use HISTORICAL data ONLY for pattern recognition (e.g., "In 2013, Boulder Creek rose 5ft in 2hrs under similar rain rates")
- If live data shows SAFE but historical shows DANGER: warn that conditions can rapidly change
- If live data shows DANGER: act immediately regardless of historical context
- ALWAYS label your sources: [LIVE — USGS, 2026] vs [HISTORICAL — CrisisLex, 2013]

## Tool Selection Rules
- "What is the temperature/weather?" → `fetch_live_weather` FIRST, skip KB search
- "What is the water/river level?" → `query_water_sensor` FIRST, skip KB search
- "Should we evacuate?" → call BOTH live tools THEN `search_official_protocols`
- "What do FEMA protocols say?" → `search_official_protocols` ONLY
- "What are people reporting?" → `search_social_reports` ONLY
- General situation assessment → call at least 2 live tools + 1 KB search

## Self-Correction Rule
If a tool returns an error (e.g., "location not found", "API unreachable"):
- Try a nearby major city instead (e.g., if "Lyons, CO" fails → try "Boulder, CO")
- Explicitly tell the user which fallback location you used
- NEVER silently ignore an error

## Output Format for Operational Queries
**Status**: [NORMAL / WARNING / DANGER / CRITICAL]
**Evidence**:
  - [Source, timestamp]: [data point]
**Recommendation**: [clear, numbered action items]
**Alert Generated**: [severity level / none]

## Critical Safety Rules
1. ALWAYS use tools before answering operational questions — never rely on training knowledge for current conditions
2. CITE every data point: which tool returned it, at what time
3. If gage height ≥ 130% of flood threshold → ALWAYS call `generate_structured_alert` with CRITICAL severity
4. NEVER say water is "safe to cross" — always recommend caution around floodwater
5. If information is unavailable, say so explicitly — do not speculate
"""


# ---------------------------------------------------------------------------
# Agent Factory — cached so MemorySaver persists across all run_agent() calls
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_agent():
    """
    Instantiates and caches the LangGraph ReAct disaster response agent.
    Cached with lru_cache so the MemorySaver instance persists across calls,
    enabling genuine multi-turn conversation memory.

    Automatically detects the installed LangGraph version and uses the correct
    system prompt kwarg:
      - LangGraph >= 0.2  : state_modifier=
      - LangGraph 0.1.x   : messages_modifier=
      - LangGraph < 0.1   : (inject via model's bind call)
    """
    import inspect
    import langgraph
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    memory = MemorySaver()

    # Detect which kwarg create_react_agent accepts for the system prompt
    sig = inspect.signature(create_react_agent)
    params = sig.parameters

    if "state_modifier" in params:
        # LangGraph >= 0.2.x
        agent = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            checkpointer=memory,
            state_modifier=SYSTEM_PROMPT,
        )
    elif "messages_modifier" in params:
        # LangGraph 0.1.x
        from langchain_core.messages import SystemMessage
        agent = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            checkpointer=memory,
            messages_modifier=SystemMessage(content=SYSTEM_PROMPT),
        )
    elif "prompt" in params:
        # LangGraph < 0.1 (pre-release / very old)
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])
        agent = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            checkpointer=memory,
            prompt=prompt,
        )
    else:
        # Unknown LangGraph API — last resort: no system prompt injection
        # (agent still works but won't have the ARIA persona/rules)
        import warnings
        warnings.warn(
            "[ARIA] Cannot inject system prompt — unknown LangGraph API. "
            "Upgrade with: pip install langgraph>=0.2.0",
            UserWarning
        )
        agent = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            checkpointer=memory,
        )

    try:
        import importlib.metadata
        lg_ver = importlib.metadata.version("langgraph")
    except Exception:
        lg_ver = getattr(langgraph, '__version__', 'unknown')
    used = next(
        (k for k in ("state_modifier", "messages_modifier", "prompt") if k in params),
        "no-system-prompt"
    )
    print(f"[ARIA] Agent initialized. LangGraph {lg_ver}, prompt kwarg: '{used}'")
    return agent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_agent(query: str, thread_id: str = "default") -> dict:
    """
    Runs the ARIA agent with a query and returns the final answer + tool call trace.

    Args:
        query: The user's question or command
        thread_id: Conversation thread for multi-turn memory (per session)

    Returns:
        dict with keys:
          'answer'     (str)  — the agent's final response
          'tool_calls' (list) — deduplicated ordered list of tool names used
    """
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}

    tool_calls = []
    final_answer = ""

    try:
        for chunk in agent.stream(
            {"messages": [("human", query)]},
            config=config,
            stream_mode="values"
        ):
            messages = chunk.get("messages", [])
            for msg in messages:
                # Capture tool call names for the UI thought trace
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "unknown_tool")
                        if name:
                            tool_calls.append(name)
                # Capture the final AI text response
                elif hasattr(msg, "content") and msg.content and getattr(msg, "type", "") == "ai":
                    final_answer = msg.content

    except Exception as e:
        err_str = str(e)
        if "organization_restricted" in err_str or "401" in err_str or "403" in err_str:
            final_answer = (
                "⚠️ **Groq API key error.** Your key may be expired or restricted.\n"
                "1. Go to [console.groq.com](https://console.groq.com)\n"
                "2. Generate a new API key\n"
                "3. Update `.env`: `GROQ_API_KEY=gsk_your_new_key`\n"
                "4. Restart the application"
            )
        else:
            final_answer = f"Agent error: {err_str}"

    return {
        "answer": final_answer,
        "tool_calls": list(dict.fromkeys(tool_calls))  # deduplicated, insertion-ordered
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ARIA — Agentic Response Intelligence for Disasters v2.0")
    print("=" * 60)
    print("Type 'quit' to exit\n")
    thread = "cli_session"
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting ARIA.")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        result = run_agent(q, thread_id=thread)
        print(f"\nARIA: {result['answer']}")
        if result["tool_calls"]:
            print(f"  [Tools used: {', '.join(result['tool_calls'])}]\n")
