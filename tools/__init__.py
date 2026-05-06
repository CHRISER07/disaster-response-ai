# tools/__init__.py
"""
ARIA Tool Registry — all 8 tools available to the LangGraph ReAct agent.

Tools:
  1. fetch_live_weather          — Open-Meteo live weather (global)
  2. query_water_sensor          — USGS real-time river gage (US)
  3. query_recent_earthquakes    — USGS earthquake feed (global)
  4. search_official_protocols   — FEMA/CDC manuals only (ChromaDB filtered)
  5. search_social_reports       — Field reports: tweets + sensors + imagery (ChromaDB filtered)
  6. search_knowledge_base       — Full ChromaDB search (all modalities, for eval)
  7. analyze_disaster_image      — Ollama LLaVA local vision AI
  8. generate_structured_alert   — Structured alert logger to dashboard
"""

from tools.weather_tool import fetch_live_weather
from tools.sensor_tool import query_water_sensor, query_recent_earthquakes
from tools.kb_search_tool import (
    search_official_protocols,
    search_social_reports,
    search_knowledge_base,
)
from tools.vision_tool import analyze_disaster_image
from tools.alert_tool import generate_structured_alert

ALL_TOOLS = [
    fetch_live_weather,
    query_water_sensor,
    query_recent_earthquakes,
    search_official_protocols,    # NEW: FEMA/CDC only
    search_social_reports,        # NEW: field reports only
    search_knowledge_base,        # Legacy: full search
    analyze_disaster_image,
    generate_structured_alert,
]
