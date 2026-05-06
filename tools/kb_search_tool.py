"""
tools/kb_search_tool.py

Wraps the ChromaDB vector store as LangChain tools for the ARIA agent.
Provides THREE search tools with strict modality filtering:

  search_official_protocols()  → FEMA/CDC docs only (knowledge_base)
  search_social_reports()      → tweets + sensor snapshots + imagery (field data)
  search_knowledge_base()      → full search across all modalities (legacy / eval)

Fix log:
  - FIXED: get_vector_store() now cached — embedding model loaded once, not per-call
  - ADDED: search_official_protocols() — prevents agent from citing tweets as FEMA policy
  - ADDED: search_social_reports() — dedicated field-report search tool
"""
import sys
import os
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.tools import tool
from core.vector_store import get_vector_store as _get_vector_store

# ---------------------------------------------------------------------------
# Cached vector store — loads embedding model ONCE per process lifetime
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _cached_store():
    """Returns the ChromaDB vector store, loaded once and cached."""
    return _get_vector_store()


def _search(query: str, modality_filter: dict | None, k: int = 5) -> str:
    """
    Internal helper: runs a similarity search with optional modality filter.
    Returns a formatted multi-document string for the agent.
    """
    try:
        store = _cached_store()

        search_kwargs = {"k": k}
        if modality_filter:
            search_kwargs["filter"] = modality_filter

        docs = store.similarity_search(query, **search_kwargs)

        if not docs:
            return "[KB SEARCH] No relevant documents found for this query."

        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            modality   = meta.get("modality", "unknown").upper()
            source     = meta.get("source", "Unknown")
            ts         = meta.get("timestamp", "")
            ts_str     = f" | {ts}" if ts else ""
            era        = "[HISTORICAL — 2013]" if "2013" in ts else "[LIVE/RECENT]"
            results.append(
                f"[{i}] [{modality}] {era} Source: {source}{ts_str}\n"
                f"    {doc.page_content[:350]}"
            )

        return "\n\n".join(results)

    except Exception as e:
        return f"[KB SEARCH ERROR] {e}"


# ---------------------------------------------------------------------------
# Tool 1: Official Protocols (FEMA/CDC only)
# ---------------------------------------------------------------------------
@tool
def search_official_protocols(query: str) -> str:
    """
    Searches ONLY official FEMA/CDC emergency manuals and protocols.
    Use this for: evacuation procedures, medical triage, shelter protocols,
    command structures, official response standards.
    NEVER returns social media or sensor data — purely authoritative sources.
    """
    result = _search(
        query,
        modality_filter={"modality": {"$in": ["knowledge_base"]}},
        k=5
    )
    return f"[OFFICIAL PROTOCOLS — FEMA/CDC]\n{result}"


# ---------------------------------------------------------------------------
# Tool 2: Social Reports & Field Data
# ---------------------------------------------------------------------------
@tool
def search_social_reports(query: str) -> str:
    """
    Searches CrisisLex social media reports, USGS sensor snapshots, and
    satellite/drone imagery descriptions from the disaster knowledge base.
    Use this for: ground-truth field conditions, eyewitness accounts,
    damage reports, access route status from the field.
    NOTE: This data is from 2013 Colorado Floods — label as HISTORICAL in your answer.
    """
    result = _search(
        query,
        modality_filter={"modality": {"$in": ["text", "telemetry", "imagery_description"]}},
        k=5
    )
    return f"[FIELD REPORTS — CrisisLex / Sensors / Imagery]\n{result}"


# ---------------------------------------------------------------------------
# Tool 3: Full Knowledge Base (all modalities — kept for legacy/eval use)
# ---------------------------------------------------------------------------
@tool
def search_knowledge_base(query: str) -> str:
    """
    Searches the entire multimodal knowledge base: FEMA/CDC protocols,
    historical CrisisLex disaster tweets, sensor snapshots, and xView2
    satellite imagery descriptions.
    Returns the top 5 most relevant documents across all sources.
    Prefer search_official_protocols() or search_social_reports() for
    more targeted and accurate results.
    """
    result = _search(query, modality_filter=None, k=5)
    return f"[FULL KNOWLEDGE BASE RESULTS]\n{result}"
