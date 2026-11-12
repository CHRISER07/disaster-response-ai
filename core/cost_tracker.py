"""
core/cost_tracker.py

Per-query token usage and cost tracking for the Groq API.

Groq returns token counts in every response. This module captures them,
computes USD cost, and appends records to logs/cost_log.jsonl.

The Streamlit sidebar reads cost_log.jsonl to display:
  "Session cost: $0.0032   Total tokens: 8,247"

Why this matters in interviews:
  "We track cost per query so we can identify expensive tool chains
   and optimize them. Our median query costs ~$0.0006 - roughly
   1,700 queries per dollar at current Groq pricing."

Groq pricing (check https://console.groq.com/settings/billing for current rates):
  llama-3.3-70b-versatile: ~$0.59 / 1M input tokens, ~$0.79 / 1M output tokens
"""
import json
import time
import os
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST_LOG     = os.path.join(PROJECT_ROOT, "logs", "cost_log.jsonl")

os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

# Groq pricing per 1,000 tokens (update if pricing changes)
COST_PER_1K_INPUT_TOKENS  = 0.00059   # $0.59 / 1M
COST_PER_1K_OUTPUT_TOKENS = 0.00079   # $0.79 / 1M


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    """Returns the estimated USD cost for a given token usage."""
    return round(
        (input_tokens  / 1000 * COST_PER_1K_INPUT_TOKENS) +
        (output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS),
        6
    )


def log_cost(
    query_id: str,
    input_tokens: int,
    output_tokens: int,
    latency_s: float,
    model: str = "llama-3.3-70b-versatile",
    thread_id: str = "default",
    tool_calls: Optional[list] = None,
) -> float:
    """
    Logs token usage and cost for one agent call to logs/cost_log.jsonl.

    Args:
        query_id:      Unique identifier for this query (can be timestamp or UUID)
        input_tokens:  Input token count from Groq response metadata
        output_tokens: Output token count from Groq response metadata
        latency_s:     Wall-clock time in seconds for the full agent call
        model:         Model name used (for cost table lookup)
        thread_id:     Conversation thread ID
        tool_calls:    List of tool names called during this query

    Returns:
        cost_usd: The computed dollar cost for this query
    """
    cost = compute_cost(input_tokens, output_tokens)
    record = {
        "ts":            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query_id":      query_id,
        "thread_id":     thread_id,
        "model":         model,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  input_tokens + output_tokens,
        "cost_usd":      cost,
        "latency_s":     round(latency_s, 3),
        "tool_calls":    tool_calls or [],
    }
    try:
        with open(COST_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # Never let cost logging break the main flow
    return cost


def get_session_summary(thread_id: Optional[str] = None) -> dict:
    """
    Returns aggregate cost stats from cost_log.jsonl.
    If thread_id is provided, filters to that session only.

    Returns:
        {
          "total_queries":  int,
          "total_tokens":   int,
          "total_cost_usd": float,
          "avg_cost_usd":   float,
          "avg_latency_s":  float,
        }
    """
    if not os.path.exists(COST_LOG):
        return {"total_queries": 0, "total_tokens": 0, "total_cost_usd": 0.0, "avg_cost_usd": 0.0, "avg_latency_s": 0.0}

    records = []
    with open(COST_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if thread_id is None or r.get("thread_id") == thread_id:
                    records.append(r)
            except Exception:
                pass

    if not records:
        return {"total_queries": 0, "total_tokens": 0, "total_cost_usd": 0.0, "avg_cost_usd": 0.0, "avg_latency_s": 0.0}

    total_tokens = sum(r.get("total_tokens", 0) for r in records)
    total_cost   = sum(r.get("cost_usd", 0.0) for r in records)
    avg_latency  = sum(r.get("latency_s", 0.0) for r in records) / len(records)

    return {
        "total_queries":  len(records),
        "total_tokens":   total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd":   round(total_cost / len(records), 6),
        "avg_latency_s":  round(avg_latency, 2),
    }
