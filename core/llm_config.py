"""
core/llm_config.py

Unified LLM inference and agent execution configuration for Disaster RAG.
Defaults to local Ollama inference with zero rate limits, deterministic multi-seed
reproducibility, and strict Pydantic structured output enforcement.

Cloud API keys (Groq/Gemini) are strictly isolated for cross-family LLM-as-a-Judge.
"""
from __future__ import annotations

import os
import logging
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
# Explicit shell/CI environment variables must take precedence over .env defaults.
load_dotenv(_env_path, override=False)
load_dotenv(override=False)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider & Model Settings
# ---------------------------------------------------------------------------
PROVIDER = os.getenv("PROVIDER", "ollama").lower().strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b").strip()
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:4b").strip()
OLLAMA_FAST_MODEL = os.getenv("OLLAMA_FAST_MODEL", "llama3.2:latest").strip()

DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
DEFAULT_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.05"))
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

# ---------------------------------------------------------------------------
# Strict Structured Output Schema (TGI + DSEL Compliance)
# ---------------------------------------------------------------------------
class DisasterSitrep(BaseModel):
    """
    Standardized, verifiable emergency situation report schema.
    Enforces deterministic extraction of live sensor summaries, actionable
    life-safety directives, chunk-level citations, and formal alert levels.
    """
    situation_summary: str = Field(
        description="Factual situation summary grounded strictly in <LIVE> sensor telemetry and <KB> doctrine. "
                    "Must NEVER assert active disaster when <LIVE> sensor telemetry reports normal conditions."
    )
    actionable_directive: str = Field(
        description="Clear, non-negotiable operational directive (e.g., 'EVACUATE TO HIGH GROUND', "
                    "'DO NOT DRIVE THROUGH FLOODWATER', 'NORMAL CONDITIONS - MAINTAIN MONITORING', "
                    "or 'INSUFFICIENT_LIVE_DATA')."
    )
    citations: List[str] = Field(
        default_factory=list,
        description="List of exact chunk or station IDs cited (e.g. ['LIVE-01', 'USGS-06730200', 'KB-02'])."
    )
    alert_level: Literal["NONE", "ADVISORY", "WATCH", "WARNING"] = Field(
        default="NONE",
        description="Official disaster alert severity level based on physical thresholds."
    )


# ---------------------------------------------------------------------------
# Ollama / llama-server OpenAI-compatible Model Factory
# ---------------------------------------------------------------------------
def get_ollama_llm(
    model_name: Optional[str] = None,
    seed: int = 42,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    format: Optional[str] = None,
):
    """
    Creates a LangChain ChatOpenAI instance pointed at the local llama-server
    running on port 11435 with an OpenAI-compatible API (/v1/chat/completions).

    NOTE: llama-server.exe (from Ollama internals, run directly for GPU access)
    speaks the OpenAI API — NOT the Ollama API (/api/chat).  Using ChatOllama
    here causes 404 errors.  ChatOpenAI with base_url=.../v1 is the fix.

    Deterministic constraints:
    - temperature=0.0
    - seed passed per evaluation run (42, 102, 2026)
    - num_ctx / repeat_penalty forwarded via model_kwargs (server honours these
      as extra body fields for llama.cpp / llama-server).
    """
    from langchain_openai import ChatOpenAI

    class SequentialChatOpenAI(ChatOpenAI):
        """ChatOpenAI variant that enforces parallel_tool_calls=False for llama-server."""
        def bind_tools(self, tools, **kwargs):
            kwargs["parallel_tool_calls"] = False
            return super().bind_tools(tools, **kwargs)

    target_model = model_name or OLLAMA_MODEL
    base = OLLAMA_BASE_URL.rstrip("/")
    # llama-server exposes /v1/* — append /v1 if not already present
    if not base.endswith("/v1"):
        base = base + "/v1"

    return SequentialChatOpenAI(
        base_url=base,
        api_key="not-needed",    # llama-server requires a non-empty string
        model=target_model,
        temperature=temperature,
        seed=seed,               # first-class ChatOpenAI param → sent as seed
        # Without an explicit timeout, a local model that is unloaded mid-call
        # makes the request wait forever (detached eval workers hung at CPU=0).
        timeout=180,
        max_retries=0,
    )


def get_llm(
    seed: int = 42,
    model_name: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    format: Optional[str] = None,
):
    """
    Primary LLM factory for the ReAct agent.
    Defaults to local Ollama.
    """
    active_provider = os.getenv("PROVIDER", PROVIDER).lower().strip()
    if active_provider == "ollama":
        return get_ollama_llm(
            model_name=model_name,
            seed=seed,
            temperature=temperature,
            num_ctx=num_ctx,
            repeat_penalty=repeat_penalty,
            format=format,
        )
    elif active_provider == "groq":
        from langchain_groq import ChatGroq
        from core.key_manager import groq_pool
        groq_model = model_name or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        api_key = groq_pool.get_key()
        return ChatGroq(
            model=groq_model,
            api_key=api_key,
            temperature=temperature,
            max_retries=0,
            model_kwargs={"seed": seed},
        )
    elif active_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from core.key_manager import gemini_pool
        gemini_model = model_name or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        api_key = gemini_pool.get_key()
        return ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=api_key,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown PROVIDER configured: {active_provider}")


# ---------------------------------------------------------------------------
# Isolated Cloud Judge LLM Factory (LLM-as-a-Judge ONLY)
# ---------------------------------------------------------------------------
def get_judge_llm(family: Literal["gemini", "openai_oss", "groq"] = "gemini"):
    """
    Isolated judge LLM factory.
    These cloud models are NEVER used inside the main ReAct agent execution loop.
    Used strictly for cross-family dual-judge evaluation (Judge A vs Judge B).
    """
    if family == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from core.key_manager import gemini_pool
        api_key = gemini_pool.get_key()
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.1-flash-lite"),
            google_api_key=api_key,
            temperature=0.0,
        )
    elif family in ("openai_oss", "groq"):
        from langchain_groq import ChatGroq
        from core.key_manager import groq_pool
        api_key = groq_pool.get_key()
        model = os.getenv("JUDGE_B_MODEL", "openai/gpt-oss-20b")
        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=0.0,
        )
    else:
        raise ValueError(f"Unsupported judge family: {family}")
