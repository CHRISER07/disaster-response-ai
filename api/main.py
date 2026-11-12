"""
api/main.py - FastAPI application entry point

The backend brain. Streamlit (app.py) is a thin client that calls this API.
Run with:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET  /health/           liveness check (used by Docker healthcheck)
  POST /agent/query       run the LangGraph ReAct agent
  GET  /agent/history     fetch chat history for a thread
  GET  /eval/results      serve latest eval CSV as JSON
  GET  /eval/alerts       serve latest alerts.jsonl as JSON
"""
import sys
import os

# Ensure project root is on the path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, agent, eval_routes

app = FastAPI(
    title="Disaster Response AI",
    description=(
        "Agentic Multimodal RAG system for real-time disaster situational intelligence. "
        "Integrates live USGS sensor data, Open-Meteo weather, seismic feeds, "
        "CrisisLex social reports, FEMA/CDC protocols, and satellite imagery analysis."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit dashboard running on a different port to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://dashboard:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,      prefix="/health",  tags=["Health"])
app.include_router(agent.router,       prefix="/agent",   tags=["Agent"])
app.include_router(eval_routes.router, prefix="/eval",    tags=["Evaluation"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Disaster Response AI",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health/",
    }
