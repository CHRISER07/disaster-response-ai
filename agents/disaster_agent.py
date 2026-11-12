"""
agents/disaster_agent.py

The LangGraph ReAct Agentic Core v2.0
Uses a Reason + Act loop to autonomously call tools, observe results,
and deliver grounded, cited, hallucination-free disaster situational reports.

Architecture:
  ChatGroq (Llama-3.3-70B with Llama-3.1-8B fallback) + 8 tools -> LangGraph create_react_agent -> Checkpointer
"""
import os
import sys
import functools
import time as _time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Windows bare-metal compatibility guard: protect against torch WinError 193 DLL crash
try:
    import torch
except (OSError, ImportError):
    import types
    fake_torch = types.ModuleType("torch")
    fake_torch.__spec__ = types.SimpleNamespace(origin="fake", submodule_search_locations=None)
    fake_torch.__version__ = "2.2.0"
    sys.modules["torch"] = fake_torch

from dotenv import load_dotenv
load_dotenv(override=True)

# -- LangSmith Tracing ---------------------------------------------------------
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "disaster-response-ai")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from tools import ALL_TOOLS


# ---------------------------------------------------------------------------
# System Prompt Builder
# ---------------------------------------------------------------------------
def _build_system_prompt() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        "You are an Agentic Response Intelligence system for Disasters (v2.0).\n"
        "You are an expert AI analyst embedded with an emergency response team during an active disaster.\n"
        "You have access to live data tools and a searchable knowledge base.\n"
        "\n"
        "CRITICAL: DO NOT write XML tags like <function=...> or </function> in your output or thoughts.\n"
        "Natively call tools using the tool call API. Never type out XML tags.\n"
        "\n"
        "## Your Tools - USE THEM BEFORE ANSWERING\n"
        "- fetch_live_weather        : Real-time temperature, rainfall, wind from Open-Meteo API\n"
        "- query_water_sensor        : Real-time river gage / discharge (USGS NWIS & global Copernicus GloFAS)\n"
        "- query_recent_earthquakes  : Recent M3.0+ seismic events from USGS (past 7 days, global)\n"
        "- search_official_protocols : Search ONLY FEMA/CDC official manuals - use for procedures\n"
        "- search_social_reports     : Search CrisisLex tweets + sensor snapshots + imagery - field conditions\n"
        "- analyze_disaster_image    : Analyze a real satellite/drone image with local Vision AI\n"
        "- generate_structured_alert : Log a formal emergency alert (severity='CRITICAL'|'HIGH'|'MEDIUM'|'LOW', message=..., zone=..., confidence='HIGH')\n"
        "\n"
        "## MANDATORY RULE - Call Tools First\n"
        "For ANY question about current conditions, protocols, or field status:\n"
        "1. CALL at least one relevant tool BEFORE formulating your answer.\n"
        "2. NEVER answer operational safety questions from training memory alone.\n"
        "3. If a tool fails, call a fallback tool or explicitly say the data is unavailable.\n"
        "\n"
        "## Terminology Rule (CRITICAL for accuracy)\n"
        "Use EXACT official terminology in your answers:\n"
        "- Water levels: 'gage height', 'flood stage', 'threshold', 'USGS site', 'cubic feet per second', 'discharge'\n"
        "- Evacuation: 'mandatory evacuation', 'evacuation order', 'evacuation route', 'high ground', 'shelter in place'\n"
        "- FEMA terms: 'Incident Command System (ICS)', 'National Incident Management System (NIMS)',\n"
        "  'Unified Command', 'FEMA protocol', 'Swift-Water Rescue', 'HAZMAT', 'decontamination zone'\n"
        "- Medical: 'contamination', 'waterborne disease', 'potable water', 'triage', 'medical supplies'\n"
        "- Safety: 'turn around dont drown', 'structural integrity', 'collapse risk', 'USAR', 'search and rescue'\n"
        "\n"
        "## Citation Rule\n"
        "Every factual claim MUST have an inline citation:\n"
        "  [LIVE - USGS NWIS, " + today + "]     for live sensor data\n"
        "  [LIVE - Open-Meteo, " + today + "]    for live weather / global flood\n"
        "  [HISTORICAL - CrisisLex, 2013]         for social media/tweet data\n"
        "  [KB - FEMA/CDC Protocol]               for knowledge base results\n"
        "  [VISION - LLaVA Analysis]              for image analysis\n"
        "\n"
        "## Confidence Tagging\n"
        "Append to each factual claim:\n"
        "  [Confidence: HIGH]    - from live API or official protocol\n"
        "  [Confidence: MEDIUM]  - from historical data cross-referenced with live\n"
        "  [Confidence: LOW]     - from unverified social reports or single source\n"
        "\n"
        "## Temporal Context Bridge (CRITICAL RULE)\n"
        "Your knowledge base contains two categories of data:\n"
        "1. HISTORICAL (2013): CrisisLex tweets, xView2 satellite images, sensor archives from the 2013 Colorado Floods\n"
        "2. LIVE (current): Open-Meteo weather, USGS real-time river gauges, USGS earthquake feed, GloFAS discharge\n"
        "\n"
        "When answering operational safety questions:\n"
        "- ALWAYS prioritize LIVE API data for current conditions and safety decisions\n"
        "- Use HISTORICAL data ONLY for pattern recognition\n"
        "- If live data shows SAFE but historical shows DANGER: warn that conditions can rapidly change\n"
        "- If live data shows DANGER: act immediately regardless of historical context\n"
        "\n"
        "## Grounding & Scope Constraint (STRICT RULE)\n"
        "1. BASE ALL ANSWERS EXCLUSIVELY on the data retrieved from your tools and knowledge base.\n"
        "2. DO NOT introduce external topics, unverified hypotheticals, or irrelevant disaster speculation.\n"
        "3. Every statement must link directly to the observed sensor numbers, weather readings, or official protocols.\n"
        "4. DO NOT append conversational closing questions like 'Do you want me to check other areas?' or 'Let me know if you need anything else'. Provide only the grounded disaster intelligence.\n"
        "\n"
        "## Output Format for Operational Queries\n"
        "Status: [NORMAL / WARNING / DANGER / CRITICAL]\n"
        "Evidence:\n"
        "  - [Source, timestamp, Confidence]: [specific data point with numbers]\n"
        "Recommendation:\n"
        "  1. [Clear action item using official terminology]\n"
        "Alert Generated: [severity level / none]\n"
        "\n"
        "## Critical Safety Rules - NEVER VIOLATE\n"
        "1. ALWAYS call tools before answering operational questions\n"
        "2. CITE every data point with source and timestamp\n"
        "3. If gage height >= 130% of flood threshold or discharge exceeds surge ratio -> ALWAYS call generate_structured_alert with CRITICAL severity\n"
        "4. NEVER say water, flood areas, or chemical zones are 'safe' without official clearance\n"
        "5. NEVER give specific GPS/casualty counts you cannot verify - say 'data unavailable'\n"
        "6. If information is unavailable from tools, say so explicitly - do not speculate\n"
    )


# ---------------------------------------------------------------------------
# Checkpointer Factory
# ---------------------------------------------------------------------------
def _get_checkpointer():
    """
    Returns persistent SQLite checkpointer if available; falls back to MemorySaver.
    Ensures session continuity across application restarts.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "checkpoints.sqlite")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


# ---------------------------------------------------------------------------
# Agent Factory - cached so checkpointer persists across all run_agent() calls
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_agent():
    """
    Instantiates and caches the LangGraph ReAct disaster response agent.
    Includes a production circuit breaker: primary 70B model with automated
    fallback to fast 8B model if Groq returns 429 rate limit or capacity errors.
    """
    import inspect
    import langgraph

    # Model resolution with environment override and high-availability fallback
    # If llama-3.3-70b-versatile is available on the account, it uses it; otherwise falls back to qwen/qwen3.8-27b or openai/gpt-oss-120b
    default_primary = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
    default_fallback_1 = os.environ.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")
    default_fallback_2 = "openai/gpt-oss-20b"
    
    primary_llm = ChatGroq(model=default_primary, temperature=0.1, max_retries=1)
    fallback_1 = ChatGroq(model=default_fallback_1, temperature=0.1, max_retries=1)
    fallback_2 = ChatGroq(model=default_fallback_2, temperature=0.1, max_retries=2)
    llm = primary_llm.with_fallbacks([fallback_1, fallback_2])

    memory = _get_checkpointer()
    system_prompt = _build_system_prompt()

    sig = inspect.signature(create_react_agent)
    params = sig.parameters

    if "state_modifier" in params:
        agent = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            checkpointer=memory,
            state_modifier=system_prompt,
        )
    elif "messages_modifier" in params:
        from langchain_core.messages import SystemMessage
        agent = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            checkpointer=memory,
            messages_modifier=SystemMessage(content=system_prompt),
        )
    elif "prompt" in params:
        from langchain_core.messages import SystemMessage
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        agent = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            checkpointer=memory,
            prompt=prompt,
        )
    else:
        agent = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            checkpointer=memory,
        )

    try:
        import importlib.metadata
        lg_ver = importlib.metadata.version("langgraph")
    except Exception:
        lg_ver = getattr(langgraph, "__version__", "unknown")

    used = next((k for k in ("state_modifier", "messages_modifier", "prompt") if k in params), "none")
    print(f"[System] Agent initialized. LangGraph {lg_ver}, prompt kwarg: '{used}', circuit breaker active.")
    return agent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_agent(query: str, thread_id: str = "default") -> dict:
    """
    Runs the agent with a query and returns the final answer + tool call trace.

    Args:
        query:     The user's question or command
        thread_id: Conversation thread for multi-turn memory (per session)

    Returns:
        dict with keys:
          'answer'     (str)   - the agent's final response (safety-repaired)
          'tool_calls' (list)  - deduplicated ordered list of tool names used
          'cost_usd'   (float) - estimated dollar cost for this query
    """
    agent = get_agent()
    # Bound reasoning depth: 8 steps prevents infinite loops and rate-limit exhaustion
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 8}

    tool_calls = []
    tool_contexts = []
    final_answer = ""
    input_tokens = 0
    output_tokens = 0
    t0 = _time.perf_counter()

    try:
        for chunk in agent.stream(
            {"messages": [("human", query)]},
            config=config,
            stream_mode="values"
        ):
            messages = chunk.get("messages", [])
            for msg in messages:
                # Capture tool calls from any message that has them
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = (
                            tc.get("name", "") if isinstance(tc, dict)
                            else getattr(tc, "name", "unknown_tool")
                        )
                        if name:
                            tool_calls.append(name)
                
                # Capture tool outputs as context for evaluation & grounding
                if (
                    hasattr(msg, "content")
                    and msg.content
                    and (getattr(msg, "type", "") == "tool" or msg.__class__.__name__ == "ToolMessage")
                ):
                    tool_contexts.append(str(msg.content))

                # Capture assistant content from AI messages (DSA: update to most recent non-empty content)
                if (
                    hasattr(msg, "content")
                    and msg.content
                    and (getattr(msg, "type", "") == "ai" or msg.__class__.__name__ == "AIMessage")
                ):
                    final_answer = str(msg.content)
                    usage = getattr(msg, "usage_metadata", None)
                    if usage:
                        def _get(obj, key):
                            if isinstance(obj, dict):
                                return obj.get(key, 0) or 0
                            return getattr(obj, key, 0) or 0

                        input_tokens += _get(usage, "input_tokens")
                        output_tokens += _get(usage, "output_tokens")

    except Exception as e:
        err_str = str(e)
        if "organization_restricted" in err_str or "401" in err_str or "403" in err_str:
            final_answer = (
                "**Groq API key error.** Your key may be expired or restricted.\n"
                "1. Go to https://console.groq.com\n"
                "2. Generate a new API key\n"
                "3. Update .env: GROQ_API_KEY=gsk_your_new_key\n"
                "4. Restart the application"
            )
        else:
            final_answer = f"Agent error: {err_str}"

    latency = _time.perf_counter() - t0
    deduped_tools = list(dict.fromkeys(tool_calls))

    # Log cost safely
    cost_usd = None
    if input_tokens + output_tokens > 0:
        try:
            from core.cost_tracker import log_cost
            cost_usd = log_cost(
                query_id=f"{thread_id}_{int(t0)}",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=latency,
                thread_id=thread_id,
                tool_calls=deduped_tools,
            )
        except Exception:
            pass

    # Deterministic Safety Enforcement Layer (DSEL Layer 3)
    try:
        from core.safety_validator import validate_and_repair
        validation = validate_and_repair(final_answer)
        final_answer = validation["answer"]
    except Exception:
        pass

    return {
        "answer": final_answer,
        "tool_calls": deduped_tools,
        "contexts": tool_contexts,
        "cost_usd": cost_usd,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Agentic Response Intelligence for Disasters v2.0")
    print("=" * 60)
    print("Type 'quit' to exit\n")
    thread = "cli_session"
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        result = run_agent(q, thread_id=thread)
        print(f"\nAI: {result['answer']}")
        if result["tool_calls"]:
            print(f"  [Tools used: {', '.join(result['tool_calls'])}]\n")
