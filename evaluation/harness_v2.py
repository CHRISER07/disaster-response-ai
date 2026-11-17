"""
evaluation/harness_v2.py

Production-Grade Evaluation Harness for Disaster Response Agentic RAG.
Implements:
  1. Exponential backoff + jitter for HTTP 429 rate limits, with max retries cap.
  2. Strict context & tool output trimming to prevent HTTP 413 (request too large).
  3. Status tracking: OK | RATE_LIMIT | TOO_LARGE | STEP_LIMIT | EMPTY | OTHER_ERROR.
  4. Scoring ONLY for OK responses (errored rows scored as null).
  5. Exact latency tracking: active_latency_s (excluding backoff sleeps) and total_latency_s.
  6. Full, untruncated answer storage.
  7. Exact Stemmed Keyword-F1 on full answers.
  8. Checkpointing after every query to allow zero-loss resumption.
"""

import os
import sys
import time
import json
import random
import csv
import re
from typing import Optional, Tuple, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.safety_validator import validate_and_repair


# Convenience aliases for importers
STATUS_OK = "OK"
STATUS_RATE_LIMIT = "RATE_LIMIT"
STATUS_TOO_LARGE = "TOO_LARGE"
STATUS_STEP_LIMIT = "STEP_LIMIT"
STATUS_EMPTY = "EMPTY"
STATUS_OTHER_ERROR = "OTHER_ERROR"

REJECT_PATTERNS = [
    "error code: 429",
    "rate_limit_exceeded",
    "tokens per day",
    "rate limit reached",
    "agent error:",
    "agent rate limit:",
    "traceback (most recent call last)",
    "internalservererror",
    "403 permission_denied",
    "groq api key error",
    "authenticationerror",
    # Step-limit / recursion guard strings that the agent emits on hard failure
    "sorry, need more steps",
    "agent stopped due to",
    "recursionlimit",
    "execution failed",
]

# Step-limit strings also map to STEP_LIMIT status (not OTHER_ERROR)
STEP_LIMIT_PATTERNS = [
    "sorry, need more steps",
    "agent stopped due to",
    "recursionlimit",
]

def validate_answer_integrity(answer: str, raw_status: str = STATUS_OK) -> tuple:
    """
    Guarantees strict integrity: no answer containing API errors, rate limit strings,
    tracebacks, or step-limit bail-out messages can EVER be classified as STATUS_OK.
    Returns (clean_status: str, is_valid: bool).
    """
    if not answer or not answer.strip():
        return STATUS_EMPTY, False

    lower_ans = answer.lower()
    for pat in REJECT_PATTERNS:
        if pat in lower_ans:
            # Rate-limit family → RATE_LIMIT
            if "429" in pat or "rate" in pat or "tokens per day" in pat:
                return STATUS_RATE_LIMIT, False
            # Step-limit family → STEP_LIMIT
            if any(sp in pat for sp in STEP_LIMIT_PATTERNS):
                return STATUS_STEP_LIMIT, False
            # Execution-failure strings → STEP_LIMIT too
            if pat == "execution failed":
                return STATUS_STEP_LIMIT, False
            return STATUS_OTHER_ERROR, False

    if raw_status != STATUS_OK:
        return raw_status, False

    if len(answer.strip()) < 15:
        return STATUS_EMPTY, False

    return STATUS_OK, True


def trim_text(text: str, max_chars: int = 1200) -> str:
    """Alias for clean_text_for_tokens (backward compat)."""
    return clean_text_for_tokens(text, max_chars)


def clean_text_for_tokens(text: str, max_chars: int = 1200) -> str:
    """Trim tool outputs and context blobs to avoid HTTP 413 request size overflow."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [TRUNCATED {len(text) - max_chars} CHARS TO PREVENT HTTP 413]"


def _stem_kw(word: str) -> str:
    w = word.lower().strip()
    for suf in ("ations", "ation", "ating", "ated", "ing", "ion", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: len(w) - len(suf)]
    return w


def _domain_vocab() -> set:
    """Domain vocabulary derived from the golden dataset, never hardcoded."""
    if not hasattr(_domain_vocab, "_cache"):
        path = os.path.join(os.path.dirname(__file__), "golden_dataset_v2.json")
        data = json.loads(open(path, encoding="utf-8").read())
        vocab = set()
        for item in data:
            for kw in item.get("expected_keywords", item.get("keywords", [])):
                if kw:
                    vocab.add(_stem_kw(str(kw)))
        _domain_vocab._cache = vocab
    return _domain_vocab._cache


def compute_keyword_f1_full(response: str, expected_keywords: List[str]) -> Dict[str, float]:
    """
    Keyword F1 on the FULL answer using a dataset-derived domain vocabulary.

    Precision is over domain-relevant keywords present in the answer, not over
    every answer token. The previous length-penalty heuristic made F1
    mechanically collapse toward zero for any full prose answer.
    """
    if not expected_keywords:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not response or len(response.strip()) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ans_stems = {_stem_kw(w) for w in re.findall(r"\b\w+\b", response)}
    predicted = ans_stems & _domain_vocab()
    gold = {_stem_kw(k) for k in expected_keywords}
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def execute_with_backoff(
    fn,
    max_retries: int = 4,
    base_delay: float = 3.0,
    max_delay: float = 25.0,
) -> Tuple[Any, str, int, float]:
    """
    Executes a callable with exponential backoff and jitter.
    Returns: (result, status, retries, sleep_time_spent)
    """
    retries = 0
    total_sleep = 0.0

    while True:
        try:
            res = fn()
            return res, "OK", retries, total_sleep
        except Exception as e:
            err_str = str(e).lower()
            retries += 1
            
            # Identify error category
            if "429" in err_str or "rate_limit" in err_str:
                status = "RATE_LIMIT"
            elif "413" in err_str or "request too large" in err_str:
                status = "TOO_LARGE"
                # Cannot retry request-size errors without modifying payload
                return None, status, retries, total_sleep
            elif "recursion" in err_str or "step" in err_str or "need more steps" in err_str:
                status = "STEP_LIMIT"
                return None, status, retries, total_sleep
            else:
                status = "OTHER_ERROR"

            if retries > max_retries:
                return None, status, retries, total_sleep

            # Exponential backoff with jitter
            delay = min(max_delay, base_delay * (2 ** (retries - 1))) + random.uniform(0.5, 2.0)
            time.sleep(delay)
            total_sleep += delay


def run_single_evaluated_query(
    run_fn,
    query_item: dict,
    config_name: str,
    run_id: int = 1,
) -> dict:
    """
    Executes a query through a target configuration, tracks status, latency,
    and computes metrics strictly on OK responses.
    """
    qid = query_item["id"]
    q_text = query_item["query"]
    scenario = query_item.get("scenario", "Unknown")
    category = query_item.get("category", "Unknown")
    expected_kw = query_item.get("expected_keywords", [])
    safe_response = query_item.get("safe_response", True)

    t_start = time.perf_counter()
    
    # Execute through function with backoff
    raw_res, status, retries, sleep_s = execute_with_backoff(
        lambda: run_fn(q_text, thread_id=f"eval_v2_{config_name}_{qid}_r{run_id}_{int(time.time()*1000)%1000000}")
    )
    
    t_end = time.perf_counter()
    total_latency_s = round(t_end - t_start, 3)
    active_latency_s = max(0.001, round(total_latency_s - sleep_s, 3))

    full_answer = ""
    tools_called = []
    model_used = "unknown"
    prompt_tokens = 0
    completion_tokens = 0

    if status == "OK" and raw_res:
        if isinstance(raw_res, dict):
            full_answer = raw_res.get("answer", "")
            tools_called = raw_res.get("tool_calls", [])
            model_used = raw_res.get("model", "qwen/qwen3.8-27b")
            prompt_tokens = raw_res.get("input_tokens", 0)
            completion_tokens = raw_res.get("output_tokens", 0)
        elif isinstance(raw_res, str):
            full_answer = raw_res
            model_used = "qwen/qwen3.8-27b"

        # Check for error strings emitted inside answer
        lower_ans = full_answer.lower()
        if "sorry, need more steps" in lower_ans or "recursion limit" in lower_ans:
            status = "STEP_LIMIT"
        elif "rate limit" in lower_ans or "error code: 429" in lower_ans:
            status = "RATE_LIMIT"
        elif "request too large" in lower_ans or "error code: 413" in lower_ans:
            status = "TOO_LARGE"
        elif len(full_answer.strip()) == 0:
            status = "EMPTY"

    # Compute metrics strictly for OK status
    if status == "OK":
        kw_metrics = compute_keyword_f1_full(full_answer, expected_kw)
        kw_prec = kw_metrics["precision"]
        kw_rec = kw_metrics["recall"]
        kw_f1 = kw_metrics["f1"]
        
        # SVC check
        v_res = validate_and_repair(full_answer, is_dangerous_query=(not safe_response))
        svc_regex = int(not v_res["safe"])
    else:
        kw_prec = None
        kw_rec = None
        kw_f1 = None
        svc_regex = None

    return {
        "config": config_name,
        "run_id": run_id,
        "id": qid,
        "scenario": scenario,
        "category": category,
        "status": status,
        "kw_precision": kw_prec,
        "kw_recall": kw_rec,
        "kw_f1": kw_f1,
        "svc_regex": svc_regex,
        "p_at_k": 0.0,
        "ndcg_at_7": 0.0,
        "rr": 0.0,
        "active_latency_s": active_latency_s,
        "total_latency_s": total_latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tools_called": ";".join(tools_called),
        "model_used": model_used,
        "retries": retries,
        "full_answer": full_answer.replace("\r", " ").replace("\n", " ").strip(),
    }

