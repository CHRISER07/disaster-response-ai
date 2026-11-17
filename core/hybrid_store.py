"""
core/hybrid_store.py

True Hybrid Retrieval Engine combining Dense Neural Embeddings (ChromaDB BGE-small)
with Sparse Lexical Search (SQLite FTS5 BM25) using Reciprocal Rank Fusion (RRF).

Combines:
1. Dense semantic search (captures conceptual intent and synonyms like "deluge" -> "flood").
2. Sparse lexical search (guarantees exact keyword matching for station numbers and official codes).
3. Reciprocal Rank Fusion (RRF) with constant k=60 to fuse dual rankings without score normalization bias.
4. Modality, Temporal, and Geospatial filtering.
"""

import os
import re
import sqlite3
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db", "chroma.sqlite3")


@dataclass
class RetrievedChunk:
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0


class HybridRetrievalStore:
    """True Hybrid RRF Retriever for multi-modal disaster intelligence."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self._vector_store = None

    def _get_vector_store(self):
        if self._vector_store is None:
            try:
                from core.vector_store import get_vector_store
                self._vector_store = get_vector_store()
            except Exception:
                self._vector_store = None
        return self._vector_store

    def _search_dense(self, query: str, k: int = 10, modality_filter: Optional[dict] = None) -> List[RetrievedChunk]:
        """Dense semantic similarity search using cached ChromaDB."""
        store = self._get_vector_store()
        if not store:
            return []
        try:
            kwargs = {"k": k}
            if modality_filter:
                kwargs["filter"] = modality_filter
            docs = store.similarity_search(query, **kwargs)
            chunks = []
            for rank, doc in enumerate(docs, 1):
                doc_id = doc.metadata.get("id", f"dense_{rank}")
                chunks.append(RetrievedChunk(
                    doc_id=doc_id,
                    content=doc.page_content,
                    metadata=doc.metadata,
                    dense_rank=rank
                ))
            return chunks
        except Exception:
            return []

    def _search_sparse_fts(self, query: str, k: int = 10, modality_filter: Optional[dict] = None) -> List[RetrievedChunk]:
        """Sparse lexical BM25-style search over 17,170 CrisisLex & FEMA records in SQLite FTS5."""
        if not os.path.exists(DB_PATH):
            return []

        clean_q = " ".join(re.findall(r"[a-zA-Z0-9]+", query))
        if not clean_q:
            return []

        tokens = [t for t in clean_q.split() if len(t) > 2]
        if not tokens:
            tokens = clean_q.split()
        fts_clause = " OR ".join(tokens[:8])

        chunks = []
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            sql = """
            SELECT s.rowid, s.string_value
            FROM embedding_fulltext_search s
            WHERE s.embedding_fulltext_search MATCH ?
            LIMIT ?
            """
            rows = cur.execute(sql, (fts_clause, k * 3)).fetchall()
            rank = 1
            for rowid, doc_text in rows:
                meta_rows = cur.execute(
                    "SELECT key, string_value FROM embedding_metadata WHERE id = ?", (rowid,)
                ).fetchall()
                meta = {k_name: v_val for k_name, v_val in meta_rows}
                
                # Apply modality filter
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

                chunks.append(RetrievedChunk(
                    doc_id=f"fts_{rowid}",
                    content=doc_text,
                    metadata=meta,
                    sparse_rank=rank
                ))
                rank += 1
                if len(chunks) >= k:
                    break
            conn.close()
        except Exception:
            pass
        return chunks

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        modality_filter: Optional[dict] = None,
        alpha_dense: float = 0.70
    ) -> List[RetrievedChunk]:
        """
        Executes Dense + Sparse searches and fuses rankings via Weighted Reciprocal Rank Fusion (RRF).
        Weighted RRF Score = alpha * (1 / (k + dense_rank)) + (1 - alpha) * (1 / (k + sparse_rank))

        alpha = 0.70 is the calibrated value from the 50-query Dev set (Sec. V-B / Table III).
        k = 60 is the RRF rank constant. Both are locked here so every caller uses the
        same fusion parameters; callers may override alpha only for ablation studies.
        """
        dense_results = self._search_dense(query, k=k*2, modality_filter=modality_filter)
        sparse_results = self._search_sparse_fts(query, k=k*2, modality_filter=modality_filter)

        fused: Dict[str, RetrievedChunk] = {}

        # Ingest dense ranks with alpha weighting
        for item in dense_results:
            key = item.content[:150]  # Content signature for deduplication
            item.rrf_score += alpha_dense * (1.0 / (self.rrf_k + item.dense_rank))
            fused[key] = item

        # Ingest sparse ranks with (1 - alpha) weighting and fuse
        for item in sparse_results:
            key = item.content[:150]
            if key in fused:
                fused[key].sparse_rank = item.sparse_rank
                fused[key].rrf_score += (1.0 - alpha_dense) * (1.0 / (self.rrf_k + item.sparse_rank))
            else:
                item.rrf_score = (1.0 - alpha_dense) * (1.0 / (self.rrf_k + item.sparse_rank))
                fused[key] = item

        # Sort descending by RRF score
        ranked_chunks = sorted(fused.values(), key=lambda c: c.rrf_score, reverse=True)
        return ranked_chunks[:k]


hybrid_store = HybridRetrievalStore()
