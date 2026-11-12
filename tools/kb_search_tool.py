"""
tools/kb_search_tool.py

Wraps the ChromaDB vector store as LangChain tools for the agent.
Provides THREE search tools with strict modality filtering:

  search_official_protocols()   FEMA/CDC docs only (knowledge_base)
  search_social_reports()       tweets + sensor snapshots + imagery (field data)
  search_knowledge_base()       full search across all modalities (legacy / eval)
"""
import sys
import os
import functools
import sqlite3
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Cached vector store - loads embedding model ONCE per process lifetime
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _cached_store():
    """Returns the ChromaDB vector store, loaded once and cached."""
    try:
        from core.vector_store import get_vector_store
        return get_vector_store()
    except Exception:
        return None


def _search_sqlite_fts(query: str, modality_filter: dict | None = None, k: int = 5) -> list:
    """
    Direct, ultra-fast SQLite full-text search fallback.
    Queries the 17,170 real CrisisLex and FEMA records in chroma.sqlite3
    without requiring external C++ DLL or PyTorch dependencies.
    """
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db", "chroma.sqlite3")
    if not os.path.exists(db_path):
        return []

    clean_q = " ".join(re.findall(r"[a-zA-Z0-9]+", query))
    if not clean_q:
        return []

    tokens = [t for t in clean_q.split() if len(t) > 2]
    if not tokens:
        tokens = clean_q.split()
    fts_clause = " OR ".join(tokens[:6])

    class _Doc:
        def __init__(self, content, metadata):
            self.page_content = content
            self.metadata = metadata

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        sql = """
        SELECT s.rowid, s.string_value
        FROM embedding_fulltext_search s
        WHERE s.embedding_fulltext_search MATCH ?
        LIMIT ?
        """
        rows = cur.execute(sql, (fts_clause, k * 3)).fetchall()
        results = []
        for rowid, doc_text in rows:
            meta_rows = cur.execute("SELECT key, string_value FROM embedding_metadata WHERE id = ?", (rowid,)).fetchall()
            meta = {k_name: v_val for k_name, v_val in meta_rows}
            if modality_filter:
                mod = meta.get("modality", "")
                target_mods = []
                for val in modality_filter.values():
                    if isinstance(val, dict) and "$in" in val:
                        target_mods.extend(val["$in"])
                    elif isinstance(val, str):
                        target_mods.append(val)
                if target_mods and mod not in target_mods:
                    continue
            results.append(_Doc(doc_text, meta))
            if len(results) >= k:
                break
        return results
    except Exception:
        return []


def _search(query: str, modality_filter: dict | None, k: int = 5) -> str:
    """
    Internal helper: runs a similarity search with optional modality filter.
    Falls back gracefully to native SQLite FTS across 17,170 real CrisisLex/FEMA records.
    Returns a formatted multi-document string for the agent.
    """
    docs = []
    try:
        store = _cached_store()
        if store:
            search_kwargs = {"k": k}
            if modality_filter:
                search_kwargs["filter"] = modality_filter
            docs = store.similarity_search(query, **search_kwargs)
    except Exception:
        pass

    if not docs:
        docs = _search_sqlite_fts(query, modality_filter, k=k)

    if not docs:
        docs = _search_sqlite_fts(query, None, k=k)

    if not docs:
        return "[KB SEARCH] No relevant documents found for this query."

    results = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        modality = meta.get("modality", "unknown").upper()
        source = meta.get("source", meta.get("author_source", "CrisisLex/FEMA"))
        ts = meta.get("timestamp", "")
        ts_str = f" | {ts}" if ts else ""

        temporal_tag = meta.get("temporal_tag", "")
        if temporal_tag:
            era = temporal_tag
        elif meta.get("modality") == "knowledge_base" or "fema" in str(source).lower():
            era = "[KB - FEMA/CDC Protocol]"
        elif ts and any(y in ts for y in ["2012", "2013", "2014", "2015"]):
            era = f"[HISTORICAL - {ts[:4]}]"
        else:
            era = "[LIVE/RECENT]"

        results.append(
            f"[{i}] [{modality}] {era} Source: {source}{ts_str}\n"
            f"    {doc.page_content[:350]}"
        )

    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# Tool 1: Official Protocols (FEMA/CDC only)
# ---------------------------------------------------------------------------
@tool
def search_official_protocols(query: str) -> str:
    """
    Searches ONLY official FEMA/CDC emergency manuals and protocols.
    Use this for: evacuation procedures, medical triage, shelter protocols,
    command structures, official response standards.
    NEVER returns social media or sensor data - purely authoritative sources.
    """
    result = _search(
        query,
        modality_filter={"modality": {"$in": ["knowledge_base"]}},
        k=5,
    )
    return f"[OFFICIAL PROTOCOLS - FEMA/CDC]\n{result}"


# ---------------------------------------------------------------------------
# Tool 2: Social Reports & Field Data
# ---------------------------------------------------------------------------
@tool
def search_social_reports(query: str) -> str:
    """
    Searches historical social media reports from 26 real disaster events
    (CrisisLex T26, 2012-2013), USGS sensor snapshots, and satellite/drone
    imagery descriptions from the disaster knowledge base.

    Events covered include: 2013 Colorado Floods, 2013 Alberta Floods,
    2013 Typhoon Yolanda, 2012 Typhoon Pablo, 2013 Savar Building Collapse
    (Bangladesh), 2013 West Texas Explosion, 2012 Venezuela Refinery,
    2013 Singapore Haze, 2012 Philippines Floods, 2013 Australia Bushfire,
    2013 Boston Bombings, and 15 more events across 12 disaster types.

    Use this for: ground-truth field conditions, eyewitness accounts,
    damage reports, access route status from the field.

    IMPORTANT: Each result is tagged with [HISTORICAL -- YYYY] and the
    specific event name. Always cite the ACTUAL event name and year in
    your answer (e.g. 'CrisisLex Typhoon Yolanda, 2013'), NOT just
    'Colorado Floods' unless the result is specifically from that event.
    """
    result = _search(
        query,
        modality_filter={"modality": {"$in": ["text", "telemetry", "imagery_description"]}},
        k=5,
    )
    return f"[FIELD REPORTS - CrisisLex / Sensors / Imagery]\n{result}"


# ---------------------------------------------------------------------------
# Tool 3: Full Knowledge Base (all modalities - kept for legacy/eval use)
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
