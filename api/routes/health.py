"""
api/routes/health.py

Liveness and readiness endpoints.
Docker uses GET /health/ as its container healthcheck.
Kubernetes uses /health/ready as readiness probe (checks ChromaDB is loaded).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def liveness():
    """
    Liveness probe - answers 'is the process alive?'
    Returns 200 immediately with no side effects.
    Used by Docker HEALTHCHECK and load balancers.
    """
    return {"status": "ok"}


@router.get("/ready")
async def readiness():
    """
    Readiness probe - answers 'is the service ready to serve requests?'
    Checks that ChromaDB vector store is accessible and has documents.
    Returns 503 if the store is empty or unreachable.
    """
    try:
        from core.vector_store import get_vector_store
        store = get_vector_store()
        doc_count = store._collection.count()
        if doc_count == 0:
            return {"status": "not_ready", "reason": "Vector store is empty - run populate_db.py first", "doc_count": 0}
        return {"status": "ready", "doc_count": doc_count}
    except Exception as e:
        # Return 200 with not_ready status rather than 5xx so Docker healthcheck
        # distinguishes between "service crashed" and "service starting up"
        return {"status": "not_ready", "reason": str(e), "doc_count": 0}


@router.get("/env")
async def env_check():
    """
    Checks which environment variables are configured (does NOT expose values).
    Useful for debugging deployments.
    """
    keys_to_check = ["GROQ_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT", "LANGCHAIN_TRACING_V2"]
    return {
        k: ("set" if os.environ.get(k) else "missing")
        for k in keys_to_check
    }
