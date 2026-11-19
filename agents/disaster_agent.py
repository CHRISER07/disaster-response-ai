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
import re
import time as _time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Windows bare-metal compatibility guard: protect against torch WinError 193 DLL crash
try:
    import torch
except (OSError, ImportError):
    import types
    fake_torch = types.ModuleType("torch")
    fake_torch.__spec__ = types.SimpleNamespace(origin="fake", submodule_search_locations=[])
    fake_torch.__version__ = "2.2.0"
    sys.modules["torch"] = fake_torch

from dotenv import load_dotenv
# Explicit shell/CI environment variables take precedence over .env defaults.
load_dotenv(override=False)

# -- LangSmith Tracing (Strictly disabled to prevent 429 tenant quota limit) ---
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
for _k in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
    if _k in os.environ:
        del os.environ[_k]

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from tools import ALL_TOOLS
import logging as _logging
_SAFETY_LOG = _logging.getLogger('disaster_agent.safety')


def _warn(component: str, exc: Exception) -> None:
    _SAFETY_LOG.warning('%s failed: %s: %s', component, type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# System Prompt Builder
# ---------------------------------------------------------------------------
def _build_system_prompt(no_tgi: bool = False) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = (
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
        "1. Call one relevant tool at a time BEFORE formulating your answer.\n"
        "1a. ROUTING (MANDATORY): if the question mentions flooding, evacuation, river/lake/creek\n"
        "   water level, 'flood stage', a river gauge, or whether an area is safe to stay in,\n"
        "   you MUST call query_water_sensor FIRST. Weather data alone is NEVER sufficient\n"
        "   evidence for a flood or evacuation decision.\n"
        "2. NEVER answer operational safety questions from training memory alone.\n"
        "3. If a tool fails or returns no exact match, use life-safety doctrine directly:\n"
        "   - Downed power lines: ALWAYS assume energized. No proximity is safe.\n"
        "   - Flood crossing: 'Turn Around Don't Drown' — no depth is safe on foot or vehicle.\n"
        "   - Structural damage: Yellow/Red tagged = prohibited re-entry without engineer clearance.\n"
        "   - Chemical exposure: Assume contaminated until officially cleared by HAZMAT.\n"
        "4. Tool budget: at most 2 calls normally. If the question explicitly asks to combine\n"
        "   multiple sources (satellite, social media, weather and sensor), call one tool per\n"
        "   requested source, up to 4 calls, before answering. If tools yield nothing, apply doctrine.\n"
        "5. Use tool names EXACTLY as listed above. NEVER invent, abbreviate, misspell,\n"
        "   or merge tool names. The only valid tool names are the eight listed.\n"
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
        "- CRITICAL RULE: <HIST> evidence must NEVER trigger active evacuation directives when <LIVE> telemetry contradicts it.\n"
        "\n"
        "## Grounding Examples - BOTH BRANCHES (TGI Conflict Resolution)\n"
        "CASE A - live SAFE, historical DANGER:\n"
        "  <HIST-2013>: 'Overtopping! Mandatory evacuation!'\n"
        "  <LIVE-USGS>: 'Gage 3.63 ft (flood stage 10.0 ft), Status NORMAL.'\n"
        "  Status: NORMAL\n"
        "  Evidence: [LIVE - USGS NWIS #06730200, Confidence: HIGH] Gage 3.63 ft, below flood stage.\n"
        "  Recommendation: No evacuation is required. Continue routine monitoring.\n"
        "  Alert Generated: NONE\n"
        "CASE B - live DANGER (historical reports irrelevant):\n"
        "  <LIVE-USGS>: 'Gage 13.50 ft (flood stage 10.0 ft), Status CRITICAL - Mandatory evacuation threshold exceeded.'\n"
        "  Status: CRITICAL\n"
        "  Evidence: [LIVE - USGS NWIS #06730200, Confidence: HIGH] Gage 13.50 ft, 35% above flood stage.\n"
        "  Recommendation: 1. Evacuate immediately to high ground. 2. Do not cross flooded roads.\n"
        "  Alert Generated: CRITICAL\n"
        "DECISION RULE: the Status line MUST follow the LIVE gage height - never the historical\n"
        "report, never the weather, and never a remembered example. Never append a paragraph\n"
        "that contradicts the Status line after 'Alert Generated'.\n"
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
        "## Plain-Text Rule (CRITICAL)\n"
        "Output plain text only. NEVER use markdown: no asterisks (* or **), no bold,\n"
        "no italics, no heading hashes, no code fences, no markdown tables. Use the exact\n"
        "section labels shown in the Output Format above and plain '- ' bullets.\n"
        "\n"
        "## Critical Safety Rules - NEVER VIOLATE\n"
        "1. ALWAYS call tools before answering operational questions\n"
        "2. CITE every data point with source and timestamp\n"
        "3. If gage height >= 130% of flood threshold or discharge exceeds surge ratio -> ALWAYS call generate_structured_alert with CRITICAL severity\n"
        "4. NEVER say water, flood areas, or chemical zones are 'safe' without official clearance\n"
        "5. NEVER give specific GPS/casualty counts you cannot verify - say 'data unavailable'\n"
        "6. If information is unavailable from tools, say so explicitly - do not speculate\n"
    )
    if no_tgi:
        # Strict ablation isolation: remove every temporal-grounding instruction
        # and marker from the prompt so the No-TGI arm cannot see TGI at all.
        start = prompt.find("## Temporal Context Bridge")
        end = prompt.find("## Grounding & Scope Constraint")
        if start != -1 and end != -1 and end > start:
            prompt = prompt[:start] + prompt[end:]
        prompt = re.sub(r"\[(?:HISTORICAL|LIVE|RECENT|KB|UNDATED)[^\]]*\]", "", prompt)
        prompt = re.sub(r"<(?:KB|HIST|LIVE)[^>]*>", "", prompt)
    return prompt


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
# Agent Factory - cached per seed so checkpointer persists across calls
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=16)
def get_agent(seed: int = 42, no_tgi: bool = False):
    """
    Instantiates and caches the LangGraph ReAct disaster response agent.
    no_tgi is part of the cache key so ablated and full agents never share state.
    """
    import inspect
    import langgraph
    from core.llm_config import get_llm

    llm = get_llm(seed=seed)

    memory = _get_checkpointer()
    system_prompt = _build_system_prompt(no_tgi=no_tgi)

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
    provider_name = os.getenv("PROVIDER", "ollama")
    model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b") if provider_name == "ollama" else os.getenv("GROQ_MODEL", "cloud")
    print(f"[System] Agent initialized. Provider: {provider_name} ({model_name}), Seed: {seed}, LangGraph {lg_ver}, prompt: '{used}'.")
    return agent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_agent(query: str, thread_id: str = "default", seed: int = 42, return_structured: bool = False, no_tgi: bool = False, prefetched_observation: str | None = None) -> dict:
    """
    Runs the agent with a query and returns the final answer + tool call trace.

    Args:
        query:             The user's question or command
        thread_id:         Conversation thread for multi-turn memory (per session)
        seed:              Random seed for deterministic reproducibility (default: 42)
        return_structured: If True, synthesizes a Pydantic DisasterSitrep structured output

    Returns:
        dict with keys:
          'answer'     (str)   - the agent's final response (safety-repaired)
          'tool_calls' (list)  - deduplicated ordered list of tool names used
          'cost_usd'   (float) - estimated dollar cost for this query
          'structured' (dict)  - optional Pydantic DisasterSitrep if requested
    """
    agent = get_agent(seed=seed, no_tgi=no_tgi)
    # Bound reasoning depth: max_iterations=8 (recursion_limit=20) prevents infinite tool loops
    from core.langfuse_tracer import get_langfuse_callback, flush_langfuse
    lf_cb = get_langfuse_callback()
    callbacks = [lf_cb] if lf_cb else []
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 20,
        "callbacks": callbacks
    }

    tool_calls = []
    tool_contexts = []
    tool_traces = []
    _pending_tools = {}
    final_answer = ""
    had_error = False       # FIX: track whether a streaming exception occurred
    input_tokens = 0
    output_tokens = 0
    t0 = _time.perf_counter()

    # A prefetched observation is injected as a genuine tool result, not as
    # user text. This keeps the evidence available to the safety layer's
    # grounding check, which only inspects tool outputs.
    messages = [("human", query)]
    if prefetched_observation:
        prefetch_id = f"prefetch_{abs(hash(thread_id))}"
        messages = [
            AIMessage(content="", tool_calls=[{
                "name": "query_water_sensor",
                "args": {"location": "Boulder Creek"},
                "id": prefetch_id,
                "type": "tool_call",
            }]),
            ToolMessage(content=prefetched_observation,
                        tool_call_id=prefetch_id,
                        name="query_water_sensor"),
            ("human", query),
        ]

    try:
        for chunk in agent.stream(
            {"messages": messages},
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
                            args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                            call_id = (tc.get("id", "") if isinstance(tc, dict)
                                       else getattr(tc, "id", "")) or name
                            # stream_mode="values" replays earlier messages in every
                            # chunk. Only arm the timer once per tool-call id.
                            if call_id not in _pending_tools:
                                _pending_tools[call_id] = (_time.perf_counter(), args, name)

                # Capture tool outputs as context for evaluation & grounding
                if (
                    hasattr(msg, "content")
                    and msg.content
                    and (getattr(msg, "type", "") == "tool" or msg.__class__.__name__ == "ToolMessage")
                ):
                    tool_contexts.append(str(msg.content))
                    call_id = getattr(msg, "tool_call_id", "") or getattr(msg, "name", "") or ""
                    if call_id in _pending_tools:
                        started, targs, tname = _pending_tools.pop(call_id)
                        elapsed_ms = (_time.perf_counter() - started) * 1000
                        tool_traces.append({
                            "step_number": len(tool_traces) + 1,
                            "tool_name": tname,
                            "tool_input": targs if isinstance(targs, dict) else {"value": str(targs)},
                            "tool_output_summary": str(msg.content)[:300],
                            "started_at": datetime.now(timezone.utc).isoformat(),
                            "latency_ms": round(elapsed_ms, 1),
                            "latency_measured": elapsed_ms > 0.01,
                            "evidence_ids_produced": [],
                        })

                # Capture assistant content from AI messages (update to most recent non-empty content)
                if (
                    hasattr(msg, "content")
                    and msg.content
                    and (getattr(msg, "type", "") == "ai" or msg.__class__.__name__ == "AIMessage")
                    and not getattr(msg, "tool_calls", None)  # skip AIMessage that only contains tool calls
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
        had_error = True        # FIX: mark error so status returns OTHER_ERROR
        err_str = str(e)
        if "organization_restricted" in err_str or "401" in err_str or "403" in err_str:
            final_answer = (
                "Groq API key error. Your key may be expired or restricted.\n"
                "1. Go to https://console.groq.com\n"
                "2. Generate a new API key\n"
                "3. Update .env: GROQ_API_KEY=gsk_your_new_key\n"
                "4. Restart the application"
            )
        else:
            final_answer = f"Agent error: {err_str}"

    # FIX: Fallback synthesis when model returned empty content after tool calls.
    # llama3.2 (3B) often produces empty final AIMessage content after tool use —
    # the answer is implicitly encoded in the tool outputs.  Synthesise it directly.
    if not final_answer and tool_contexts and not had_error:
        try:
            from core.llm_config import get_llm
            evidence_text = "\n\n---\n\n".join(tool_contexts[:4])  # first 4 tool outputs
            synth_prompt = (
                f"You are a disaster-response assistant.\n"
                f"Based ONLY on the following retrieved evidence, answer this safety question:\n\n"
                f"QUESTION: {query}\n\n"
                f"EVIDENCE:\n{evidence_text}\n\n"
                f"RULES:\n"
                f"- Apply life-safety doctrine: never say floodwater, chemical zones, or downed lines are safe.\n"
                f"- Cite your sources inline (e.g. [LIVE - USGS NWIS], [KB - FEMA/CDC Protocol]).\n"
                f"- Be concise and direct. Do not ask follow-up questions.\n\n"
                f"ANSWER:"
            )
            synth_llm = get_llm(seed=seed)
            synth_resp = synth_llm.invoke(synth_prompt)
            final_answer = synth_resp.content or ""
        except Exception as synth_err:
            final_answer = f"Agent error (synthesis fallback failed): {synth_err}"
            had_error = True

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

    # Evidence Normalization & 4-Timestamp Pipeline
    evidence_packets = []
    try:
        from core.evidence_validator import evidence_validator, SourceType
        now_dt = datetime.now(timezone.utc)
        for idx, ctx in enumerate(tool_contexts):
            ctx_lower = ctx.lower()
            if "gage" in ctx_lower or "usgs" in ctx_lower:
                st = SourceType.TELEMETRY
            elif "weather" in ctx_lower or "precip" in ctx_lower or "temp" in ctx_lower:
                st = SourceType.WEATHER_FORECAST
            elif "fema" in ctx_lower or "cdc" in ctx_lower or "protocol" in ctx_lower:
                st = SourceType.OFFICIAL_PROTOCOL
            else:
                st = SourceType.SOCIAL_MEDIA

            pkt = evidence_validator.validate_and_normalize(
                source_id=f"tool_stream_{idx+1}",
                source_type=st,
                raw_text=ctx,
                observed_at=now_dt
            )
            evidence_packets.append(pkt)
    except Exception as exc:
        _warn('evidence_normalization', exc)

    # Multi-Source Conflict Resolution & Safe Abstention Check
    conflict_res = None
    try:
        from core.conflict_resolver import conflict_resolver
        if evidence_packets:
            conflict_res = conflict_resolver.resolve_evidence(evidence_packets, query_context=query)
            if conflict_res.should_abstain and conflict_res.safe_abstention_message:
                final_answer = f"{final_answer}\n\nNotice on Evidence Uncertainty: {conflict_res.safe_abstention_message}"
    except Exception as exc:
        _warn('conflict_resolution', exc)

    # Claim-Level Verification Audit
    claim_report = None
    try:
        from core.claim_verifier import claim_verifier
        if evidence_packets:
            claim_report = claim_verifier.verify_response(final_answer, evidence_packets)
    except Exception as exc:
        _warn('claim_verification', exc)

    # Clean any leaked tool/function-calling plumbing artifacts from 3B model
    tool_leak_phrases = ("no json output", "json object", "function call", "requires two parameters", "parameter 'message'")
    if any(p in final_answer.lower() for p in tool_leak_phrases):
        try:
            from core.llm_config import get_llm
            c_prompt = (
                f"You are an expert disaster response intelligence agent.\n"
                f"Question: \"{query}\"\n\n"
                f"Give a direct, clear disaster safety response. Apply strict life-safety doctrine:\n"
                f"- Never declare unverified floodwater, electrical conductors, damaged structures, or chemical areas safe.\n"
                f"- If asked to issue an unauthorized alert or evacuation all-clear without official authorization, refuse clearly.\n"
                f"Direct Answer:"
            )
            c_llm = get_llm(seed=seed)
            c_resp = c_llm.invoke(c_prompt)
            if c_resp.content and len(c_resp.content.strip()) > 15:
                final_answer = c_resp.content
        except Exception as exc:
            _warn('tool_leak_repair', exc)

    # Deterministic Safety Enforcement Layer (DSEL Layer 3)
    dsel_violations = []
    try:
        from core.safety_validator import validate_and_repair
        q_lower = query.lower()
        _danger_words = (
            "safe", "cross", "drink", "enter", "touch", "clear",
            "okay", "fine", "can i", "can we", "can workers", "can residents",
            "can pedestrians", "can dogs", "flood", "water", "creek",
            "power line", "wire", "cable", "conductor", "approach", "proximity",
            "chemical", "collapse", "smell", "boil", "leak", "mask", "re-enter",
            "remain", "inside", "drywall", "beam", "hepa", "window", "smoke",
            "wildfire", "blogger", "broadcast", "alert", "downgrade", "notification",
            "evacuate", "all-clear", "neutralize", "storm", "pesticide", "chlorine",
            "ammonia", "battery", "hazard"
        )
        is_danger_q = any(w in q_lower for w in _danger_words)
        validation = validate_and_repair(final_answer, is_dangerous_query=is_danger_q)
        final_answer = validation["answer"]
        dsel_violations = validation.get("violations", [])
    except Exception as exc:
        _warn('dsel_validation', exc)

    # Record the actual dispatch outcome before presentation normalization, so
    # the template never claims "NONE" when generate_structured_alert fired.
    import re as _re
    # The tool trace is authoritative. Calling generate_structured_alert does
    # NOT mean an alert was logged: policy may block it or leave it pending
    # operator approval. Never assert "ALERT LOGGED" without the tool's own
    # confirmation string.
    alert_outputs = [
        str(t.get("tool_output_summary", ""))
        for t in tool_traces
        if t.get("tool_name") == "generate_structured_alert"
    ]
    if any("[ALERT LOGGED" in o for o in alert_outputs):
        alert_line = "Alert Generated: ALERT LOGGED"
    elif any("PENDING OPERATOR APPROVAL" in o for o in alert_outputs):
        alert_line = "Alert Generated: PENDING OPERATOR APPROVAL"
    elif any("[ALERT BLOCKED" in o for o in alert_outputs):
        alert_line = "Alert Generated: BLOCKED BY POLICY"
    elif alert_outputs:
        alert_line = "Alert Generated: NOT CONFIRMED"
    else:
        alert_line = "Alert Generated: NONE"
    if _re.search(r"^Alert Generated:.*$", final_answer, _re.MULTILINE):
        # Overwrite a model-written status: the tool trace is authoritative.
        final_answer = _re.sub(r"^Alert Generated:.*$", alert_line, final_answer,
                               flags=_re.MULTILINE)
    else:
        final_answer += "\n\n" + alert_line

    # Deterministic, asterisk-free presentation. Applied last so every answer path
    # (normal, fallback, repaired, conflict notice) is normalized identically.
    from core.answer_formatter import format_answer
    final_answer = format_answer(final_answer)

    # Incident Audit Logging & Forensic Trace
    try:
        from core.audit_logger import audit_logger
        rec = audit_logger.create_record(query=query, session_id=thread_id)
        rec.retrieved_evidence_ids = [e.evidence_id for e in evidence_packets]
        from core.audit_logger import ToolExecutionTrace
        rec.tool_traces = [ToolExecutionTrace(**t) for t in tool_traces]
        rec.final_response = final_answer
        rec.conflict_resolution_state = conflict_res.overall_state.value if conflict_res else "UNKNOWN"
        rec.safe_abstention_triggered = conflict_res.should_abstain if conflict_res else False
        rec.dsel_violations_detected = [str(v) for v in dsel_violations]
        if claim_report:
            rec.claim_verification_summary = {
                "total": claim_report.total_claims,
                "supported": claim_report.supported_count,
                "contradicted": claim_report.contradicted_count,
                "faithfulness": claim_report.faithfulness_ratio
            }
        audit_logger.log_incident(rec)
    except Exception as exc:
        _warn('audit_logging', exc)

    flush_langfuse()

    structured_data = None
    if return_structured:
        try:
            from core.llm_config import get_llm, DisasterSitrep
            structured_generator = get_llm(seed=seed, format="json").with_structured_output(DisasterSitrep)
            s_prompt = (
                f"Synthesize the following verified situation report into the strict DisasterSitrep JSON schema:\n\n"
                f"{final_answer}"
            )
            structured_sitrep = structured_generator.invoke(s_prompt)
            if hasattr(structured_sitrep, "model_dump"):
                structured_data = structured_sitrep.model_dump()
        except Exception as err:
            print(f"[Warning] Structured sitrep synthesis error: {err}")

    return {
        "answer": final_answer,
        "status": "OTHER_ERROR" if had_error else "OK",   # FIX: explicit status key
        "tool_calls": deduped_tools,
        "contexts": tool_contexts,
        "cost_usd": cost_usd,
        "structured": structured_data,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "tool_traces": tool_traces,
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
