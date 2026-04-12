"""
tools/kb_search_tool.py

Wraps the ChromaDB vector store as a LangChain tool for the agent.
Searches across all modalities: FEMA protocols, CrisisLex tweets, satellite imagery descriptions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.tools import tool
from core.vector_store import get_vector_store

@tool
def search_knowledge_base(query: str) -> str:
    """
    Searches the multimodal knowledge base: FEMA/CDC emergency protocols,
    historical CrisisLex disaster tweets, and xView2 satellite imagery descriptions.
    Use this to find official procedures, historical reports, or visual damage assessments.
    Returns the top 5 most relevant documents with their source and timestamp.
    """
    try:
        store = get_vector_store()
        docs = store.similarity_search(query, k=5)
        if not docs:
            return "[KB SEARCH] No relevant documents found for this query."

        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            modality = meta.get("modality", "unknown")
            source = meta.get("source", "Unknown")
            ts = meta.get("timestamp", "")
            ts_str = f" | Time: {ts}" if ts else ""
            results.append(
                f"[{i}] [{modality.upper()}] Source: {source}{ts_str}\n"
                f"    {doc.page_content[:300]}"
            )

        return "[KNOWLEDGE BASE RESULTS]\n" + "\n\n".join(results)
    except Exception as e:
        return f"[KB SEARCH ERROR] {e}"
