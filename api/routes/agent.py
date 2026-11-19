"""
api/routes/agent.py

Wraps agents/disaster_agent.py behind a FastAPI REST interface.
Includes file-backed persistent conversation history across restarts and streaming response support.

Endpoints:
  POST /agent/query       - run the LangGraph ReAct agent, return answer + tool trace
  POST /agent/stream      - stream tokens and intermediate tool events (SSE)
  GET  /agent/history     - retrieve full chat history for a thread_id
  DELETE /agent/history   - clear chat history for a thread_id
"""
import os
import sys
import time
import json
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HISTORY_FILE = os.path.join(PROJECT_ROOT, "data", "chat_history.jsonl")

# In-memory index of conversation history, synced with on-disk storage
_history: dict[str, list[dict]] = {}


def _load_history_from_disk():
    global _history
    if not os.path.exists(HISTORY_FILE):
        return
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        thread = record.get("thread_id", "default")
                        if thread not in _history:
                            _history[thread] = []
                        _history[thread].append(record.get("message", {}))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass


def _append_history_to_disk(thread_id: str, message: dict):
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"thread_id": thread_id, "message": message}) + "\n")
    except Exception:
        pass


_load_history_from_disk()


# ── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="The user's question or command")
    thread_id: str = Field(default="default", description="Conversation thread ID for multi-turn memory")
    location_context: Optional[str] = Field(default=None, description="Active monitoring location to prepend to query")


class QueryResponse(BaseModel):
    answer: str
    tool_calls: list[str]
    thread_id: str
    latency_s: float
    cost_usd: Optional[float] = None


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[dict]
    count: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    """
    Run the LangGraph ReAct disaster response agent.
    Autonomously invokes sensor APIs, weather feeds, and ChromaDB retrieval.
    """
    try:
        from agents.disaster_agent import run_agent
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Agent unavailable: {e}")

    full_query = req.query
    if req.location_context:
        full_query = f"[Active monitoring location: {req.location_context}]\n\n{req.query}"

    thread = req.thread_id
    if thread not in _history:
        _history[thread] = []

    user_msg = {"role": "user", "content": req.query, "timestamp": time.time()}
    _history[thread].append(user_msg)
    _append_history_to_disk(thread, user_msg)

    t0 = time.perf_counter()
    try:
        result = run_agent(full_query, thread_id=thread)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
    latency = round(time.perf_counter() - t0, 3)

    assistant_msg = {
        "role": "assistant",
        "content": result["answer"],
        "tool_calls": result.get("tool_calls", []),
        "timestamp": time.time(),
    }
    _history[thread].append(assistant_msg)
    _append_history_to_disk(thread, assistant_msg)

    cost = result.get("cost_usd", None)

    return QueryResponse(
        answer=result["answer"],
        tool_calls=result.get("tool_calls", []),
        thread_id=thread,
        latency_s=latency,
        cost_usd=cost,
    )


@router.post("/stream")
async def stream_query(req: QueryRequest):
    """
    Server-Sent Events (SSE) streaming endpoint for the disaster agent.
    Streams tool execution events and text chunks as they generate.
    """
    try:
        from agents.disaster_agent import get_agent
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Agent unavailable: {e}")

    full_query = req.query
    if req.location_context:
        full_query = f"[Active monitoring location: {req.location_context}]\n\n{req.query}"

    thread = req.thread_id
    if thread not in _history:
        _history[thread] = []
    user_msg = {"role": "user", "content": req.query, "timestamp": time.time()}
    _history[thread].append(user_msg)
    _append_history_to_disk(thread, user_msg)

    def event_generator():
        agent = get_agent()
        config = {"configurable": {"thread_id": thread}}
        yield f"data: {json.dumps({'type': 'start', 'thread_id': thread})}\n\n"

        final_content = ""
        tools_called = []

        try:
            for chunk in agent.stream({"messages": [("human", full_query)]}, config=config, stream_mode="values"):
                messages = chunk.get("messages", [])
                for msg in messages:
                    # Capture tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "tool")
                            if name and name not in tools_called:
                                tools_called.append(name)
                                yield f"data: {json.dumps({'type': 'tool_call', 'tool': name})}\n\n"
                    
                    # Capture AI messages
                    mtype = getattr(msg, "type", "")
                    clsname = msg.__class__.__name__
                    if (mtype == "ai" or clsname == "AIMessage") and getattr(msg, "content", ""):
                        # Buffer only. Never stream raw model text to an operator:
                        # it may contain unsafe directives before DSEL runs.
                        content_str = str(msg.content)
                        if content_str != final_content:
                            final_content = content_str

            # DSEL validation MUST complete before any answer text is delivered.
            # Fail closed: if repair itself fails, do not emit the raw answer.
            if final_content:
                try:
                    from core.safety_validator import validate_and_repair
                    danger_words = (
                        "safe", "cross", "drink", "enter", "touch", "clear",
                        "okay", "fine", "can i", "can we", "can residents",
                        "flood", "water", "creek", "power line", "chemical",
                        "collapse", "evacuate", "all-clear", "wildfire", "hazard"
                    )
                    is_danger = any(w in full_query.lower() for w in danger_words)
                    validated = validate_and_repair(final_content, is_dangerous_query=is_danger)
                    final_content = validated["answer"]
                    if validated.get("violations"):
                        yield f"data: {json.dumps({'type': 'safety_notice', 'violations': validated['violations']})}\n\n"
                except Exception as exc:
                    final_content = (
                        "Response withheld: safety validation unavailable. "
                        f"Reason: {type(exc).__name__}"
                    )
                try:
                    from core.answer_formatter import format_answer
                    final_content = format_answer(final_content)
                except Exception:
                    pass
                yield f"data: {json.dumps({'type': 'chunk', 'content': final_content})}\n\n"
            else:
                # Fallback: if stream produced no AI content, run blocking agent
                try:
                    from agents.disaster_agent import run_agent
                    result = run_agent(full_query, thread_id=thread)
                    final_content = result.get("answer", "")
                    for t in result.get("tool_calls", []):
                        if t not in tools_called:
                            tools_called.append(t)
                            yield f"data: {json.dumps({'type': 'tool_call', 'tool': t})}\n\n"
                except Exception as fallback_err:
                    final_content = f"Agent error: {fallback_err}"

            assistant_msg = {
                "role": "assistant",
                "content": final_content,
                "tool_calls": tools_called,
                "timestamp": time.time(),
            }
            _history[thread].append(assistant_msg)
            _append_history_to_disk(thread, assistant_msg)

            yield f"data: {json.dumps({'type': 'done', 'final_answer': final_content})}\n\n"
        except Exception as err:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(err)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/history/{thread_id}", response_model=HistoryResponse)
async def get_history(thread_id: str):
    """Retrieve full conversation history for a given thread."""
    msgs = _history.get(thread_id, [])
    return HistoryResponse(thread_id=thread_id, messages=msgs, count=len(msgs))


@router.delete("/history/{thread_id}")
async def clear_history(thread_id: str):
    """Clear conversation history for a given thread both in-memory and on-disk."""
    _history.pop(thread_id, None)
    # Rewrite disk history omitting the cleared thread
    if os.path.exists(HISTORY_FILE):
        try:
            surviving = []
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("thread_id") != thread_id:
                        surviving.append(line)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                f.writelines(surviving)
        except Exception:
            pass
    return {"status": "cleared", "thread_id": thread_id}
