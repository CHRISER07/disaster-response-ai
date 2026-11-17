"""
core/langfuse_tracer.py

Production Langfuse telemetry integration for the disaster response agent.
Provides automatic tracing, tool inspection, and observation callbacks.
Falls back gracefully if Langfuse credentials are not provided.
"""
import os
import logging
from typing import Optional, List
from dotenv import load_dotenv

# Explicit shell/CI environment variables take precedence over .env defaults.
# override=False here used to reset PROVIDER back to "groq" mid-run.
load_dotenv(override=False)
logger = logging.getLogger(__name__)

_HANDLER = None
_INIT_ATTEMPTED = False

def get_langfuse_callback():
    """
    Returns an active Langfuse CallbackHandler for LangChain/LangGraph,
    or None if credentials are missing or disabled.
    """
    global _HANDLER, _INIT_ATTEMPTED
    if _INIT_ATTEMPTED:
        return _HANDLER

    _INIT_ATTEMPTED = True

    enabled = os.getenv("LANGFUSE_ENABLED", "true").lower() in ("true", "1", "yes")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000").strip()

    if not enabled or not public_key or not secret_key:
        return None

    try:
        from langfuse.langchain import CallbackHandler
        _HANDLER = CallbackHandler(public_key=public_key)
        logger.info(f"[Langfuse] Initialized Langfuse callback tracer on host: {host}")
        return _HANDLER
    except Exception as e:
        logger.warning(f"[Langfuse] Failed to initialize CallbackHandler: {e}")
        return None

def flush_langfuse():
    """Flushes buffered traces to Langfuse backend."""
    global _HANDLER
    if _HANDLER is not None:
        try:
            _HANDLER.flush()
        except Exception:
            pass
