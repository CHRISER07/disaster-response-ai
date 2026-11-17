"""
app.py - Disaster Command Center (v2.0)
Streamlit dashboard - thin UI client that calls the FastAPI backend.

Modes:
  API mode (Docker / production): API_BASE_URL env var is set
    -> all agent calls go to FastAPI at that URL
  Standalone mode (local dev, no Docker): API_BASE_URL is not set
    -> falls back to importing the agent directly (original behavior)

Run standalone:
  streamlit run app.py

Run against FastAPI backend:
  set API_BASE_URL=http://localhost:8000
  streamlit run app.py
"""
import streamlit as st
import plotly.graph_objects as go
import requests
import json
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(override=False)

st.set_page_config(
    page_title="Disaster Command Center",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- CSS -----------------------------------------------------------------------
st.markdown("""
<style>
  .block-container { padding-top: 1rem; }
  .metric-card {
    background: #1c2333; border-radius: 10px; padding: 14px;
    border-left: 4px solid #3d8ef0; margin-bottom: 10px;
  }
  .alert-CRITICAL { border-left-color: #ff4b4b !important; }
  .alert-HIGH     { border-left-color: #ffa500 !important; }
  .alert-MEDIUM   { border-left-color: #ffd700 !important; }
  .alert-LOW      { border-left-color: #00cc88 !important; }
  .tool-badge {
    display: inline-block; background: #2d3748; color: #90cdf4;
    padding: 2px 8px; border-radius: 12px; font-size: 11px;
    margin: 2px; font-family: monospace;
  }
  .status-bar { font-size: 12px; color: #718096; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# -- Constants -----------------------------------------------------------------
from core.audit_logger import ALERT_LOG_FILE as ALERTS_FILE, LEGACY_ALERTS_FILE as LEGACY_ALERTS_FILE

# Prefer the canonical log; fall back to the legacy file for old alerts.
if not os.path.exists(ALERTS_FILE) and os.path.exists(LEGACY_ALERTS_FILE):
    ALERTS_FILE = LEGACY_ALERTS_FILE
DANGER_FT   = 10.0

LOCATION_PRESETS = {
    "Boulder, CO (2013 Flood)":   {"city": "Boulder, CO",    "usgs": "06730200", "lat": 40.0150, "lon": -105.2705, "threshold": 10.0},
    "Houston, TX":                {"city": "Houston, TX",     "usgs": "08074000", "lat": 29.7604, "lon": -95.3698,  "threshold": 20.0},
    "New Orleans, LA":            {"city": "New Orleans, LA", "usgs": "07374000", "lat": 29.9511, "lon": -90.0715,  "threshold": 17.0},
    "Miami, FL":                  {"city": "Miami, FL",       "usgs": "02288990", "lat": 25.7617, "lon": -80.1918,  "threshold": 5.0},
    "Nashville, TN":              {"city": "Nashville, TN",   "usgs": "03431500", "lat": 36.1627, "lon": -86.7816,  "threshold": 40.0},
    "Custom Location":            None,
}

# -- API mode detection --------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")

# -- Session state init --------------------------------------------------------
if "run_agent" not in st.session_state:
    st.session_state.run_agent = None
    st.session_state.agent_error = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -- Load agent in standalone mode --------------------------------------------
groq_key = os.environ.get("GROQ_API_KEY", "")

if not API_BASE_URL and st.session_state.run_agent is None and groq_key:
    try:
        from agents.disaster_agent import run_agent
        st.session_state.run_agent = run_agent
    except Exception as e:
        st.session_state.agent_error = str(e)


# -- Unified agent call -------------------------------------------------------
def call_agent(query: str, location: str, thread_id: str = "dash") -> dict:
    """
    Calls FastAPI backend if API_BASE_URL is set, otherwise calls agent directly.
    Always returns dict with 'answer' and 'tool_calls'.
    """
    if API_BASE_URL:
        try:
            resp = requests.post(
                f"{API_BASE_URL}/agent/query",
                json={"query": query, "thread_id": thread_id, "location_context": location},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return {"answer": data["answer"], "tool_calls": data.get("tool_calls", [])}
        except requests.exceptions.ConnectionError:
            return {
                "answer": f"Cannot reach API at {API_BASE_URL}. Is the FastAPI server running? (uvicorn api.main:app)",
                "tool_calls": []
            }
        except Exception as e:
            return {"answer": f"API error: {e}", "tool_calls": []}
    else:
        if not st.session_state.run_agent:
            return {"answer": "Agent offline - check GROQ_API_KEY.", "tool_calls": []}
        full_prompt = f"[Active monitoring location: {location}]\n\n{query}"
        try:
            return st.session_state.run_agent(full_prompt, thread_id=thread_id)
        except Exception as e:
            return {"answer": f"Agent error: {e}", "tool_calls": []}


# -- Alert loader -------------------------------------------------------------
def load_alerts() -> list:
    """Load alerts from API (API mode) or file (standalone mode)."""
    if API_BASE_URL:
        try:
            resp = requests.get(f"{API_BASE_URL}/eval/alerts", timeout=5)
            return resp.json().get("alerts", [])
        except Exception:
            return []
    if not os.path.exists(ALERTS_FILE):
        return []
    alerts = []
    with open(ALERTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    alerts.append(json.loads(line))
                except Exception:
                    pass
    return list(reversed(alerts[-20:]))


# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.title("Disaster Command Center")

    # Mode indicator
    if API_BASE_URL:
        st.caption(f"Mode: API ({API_BASE_URL})")
        try:
            hresp = requests.get(f"{API_BASE_URL}/health/", timeout=2)
            if hresp.status_code == 200:
                st.success("FastAPI backend online")
            else:
                st.warning(f"API returned {hresp.status_code}")
        except Exception:
            st.error(f"FastAPI offline at {API_BASE_URL}")
    else:
        st.caption("Mode: Standalone (direct)")

    st.subheader("Disaster Location")
    preset = st.selectbox("Select location", list(LOCATION_PRESETS.keys()))

    if preset == "Custom Location":
        custom_city = st.text_input("City name", placeholder="e.g. Caracas, Venezuela")
        if custom_city:
            # Dynamically resolve lat/lon for the custom city
            from tools.weather_tool import geocode_location
            geo = geocode_location(custom_city)
            if geo:
                loc_cfg = {
                    "city": geo["name"],
                    "usgs": "06730200",  # Default USGS gauge, tool handles fallback
                    "lat": geo["lat"],
                    "lon": geo["lon"],
                    "threshold": 10.0
                }
            else:
                st.warning(f"Could not locate '{custom_city}', using defaults.")
                loc_cfg = {
                    "city": custom_city,
                    "usgs": "06730200",
                    "lat": 40.0150, "lon": -105.2705,
                    "threshold": 10.0
                }
        else:
            loc_cfg = {
                "city": "Boulder, CO",
                "usgs": "06730200",
                "lat": 40.0150, "lon": -105.2705,
                "threshold": 10.0
            }
    else:
        loc_cfg = LOCATION_PRESETS[preset]

    st.divider()

    # API Key check
    if groq_key:
        st.success("GROQ_API_KEY set")
    else:
        st.error("GROQ_API_KEY missing")
        st.caption("Get a free key at [console.groq.com](https://console.groq.com)")

    st.divider()

    # DB Status & Populate
    doc_count = 0
    st.subheader("Vector Database")
    try:
        from core.vector_store import get_vector_store
        store = get_vector_store()
        doc_count = store._collection.count()
        if doc_count > 0:
            st.success(f"{doc_count:,} documents indexed")
        else:
            st.warning("Database is empty!")
            if st.button("Populate Database Now", type="primary"):
                with st.spinner("Ingesting real data - this takes ~2 minutes..."):
                    import subprocess, sys
                    result = subprocess.run(
                        [sys.executable or "python", "core/populate_db.py", "--reset"],
                        capture_output=True, text=True, timeout=300,
                        cwd=os.path.dirname(__file__)
                    )
                    st.text(result.stdout[-2000:] if result.stdout else "")
                    if result.returncode == 0:
                        st.success("Database populated!")
                        st.rerun()
                    else:
                        st.error(result.stderr[-1000:])
    except Exception as e:
        st.error(f"Vector store error: {e}")
        doc_count = 0

    st.caption("BGE-small-en-v1.5 + ChromaDB")
    st.divider()

    if st.button("Refresh Live Data"):
        st.cache_data.clear()
        st.rerun()

    st.caption("**How to run correctly:**")
    st.code("venv\\Scripts\\activate\nstreamlit run app.py", language="bash")

    # Cost summary (if cost tracking is running)
    try:
        from core.cost_tracker import get_session_summary
        stats = get_session_summary()
        if stats["total_queries"] > 0:
            st.divider()
            st.caption("**Session Cost**")
            st.caption(f"Queries: {stats['total_queries']}")
            st.caption(f"Total tokens: {stats['total_tokens']:,}")
            st.caption(f"Cost: ${stats['total_cost_usd']:.4f}")
            st.caption(f"Avg latency: {stats['avg_latency_s']:.1f}s")
    except Exception:
        pass


# -- Data fetchers -------------------------------------------------------------
@st.cache_data(ttl=60)
def get_weather(lat: float, lon: float):
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,rain,wind_speed_10m,weather_code",
                "timezone": "auto"
            }, timeout=8
        )
        d = r.json()["current"]
        return {"temp": d["temperature_2m"], "rain": d["rain"],
                "wind": d["wind_speed_10m"], "ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@st.cache_data(ttl=60)
def get_sensor(site_id: str, threshold: float):
    try:
        r = requests.get(
            "https://waterservices.usgs.gov/nwis/iv/",
            params={"sites": site_id, "parameterCd": "00065", "format": "json"},
            timeout=8
        )
        ts = r.json().get("value", {}).get("timeSeries", [])
        if not ts or not ts[0]["values"][0]["value"]:
            return {"ok": False, "error": "No gauge data returned for this site"}
        val  = ts[0]["values"][0]["value"][-1]
        gage = float(val["value"])
        return {
            "gage": gage,
            "time": val["dateTime"],
            "site": ts[0]["sourceInfo"]["siteName"],
            "ok":   True
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def make_gauge(value, max_val, danger, title, unit, key):
    color = "#ff4b4b" if value >= danger else "#ffa500" if value >= danger*0.8 else "#00cc88"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#e2e8f0", "size": 13}},
        number={"suffix": f" {unit}", "font": {"color": "#e2e8f0", "size": 20}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#718096"},
            "bar": {"color": color},
            "bgcolor": "#1c2333",
            "steps": [
                {"range": [0, danger*0.8], "color": "#1a2a1a"},
                {"range": [danger*0.8, danger], "color": "#2a2a12"},
                {"range": [danger, max_val], "color": "#2a1212"},
            ],
            "threshold": {"value": danger, "line": {"color": "#ff4b4b", "width": 2}}
        }
    ))
    fig.update_layout(margin={"t":50,"b":0,"l":20,"r":20}, height=190,
                      paper_bgcolor="#0e1117", font={"color": "#e2e8f0"})
    return st.plotly_chart(fig, use_container_width=True, key=key)


# -- Header -------------------------------------------------------------------
st.markdown("# Disaster Command Center")

agent_online = (API_BASE_URL or st.session_state.run_agent)
agent_status = "Agent Online" if agent_online else "Agent Offline"
st.markdown(
    f'<p class="status-bar">{agent_status} &nbsp;|&nbsp; '
    f'{loc_cfg["city"]} &nbsp;|&nbsp; '
    f'{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} &nbsp;|&nbsp; '
    f'{doc_count:,} docs indexed</p>',
    unsafe_allow_html=True
)

# Show errors
if not groq_key and not API_BASE_URL:
    st.error(
        "**GROQ_API_KEY is not set.** The AI agent cannot run without it.\n\n"
        "1. Go to [console.groq.com](https://console.groq.com) -> create a free account -> API Keys -> Create key\n"
        "2. Copy the key\n"
        "3. Open the `.env` file in the project folder and set:\n"
        "   ```\n   GROQ_API_KEY=gsk_your_key_here\n   ```\n"
        "4. Restart the app"
    )
elif st.session_state.agent_error:
    err = st.session_state.agent_error
    if "organization_restricted" in err or "401" in err or "403" in err:
        st.error(
            "**Your Groq API key has been restricted or expired.**\n\n"
            "1. Go to [console.groq.com](https://console.groq.com)\n"
            "2. Generate a **new** API key\n"
            "3. Update `.env`: `GROQ_API_KEY=gsk_your_new_key`\n"
            "4. Restart the app"
        )
    else:
        st.error(f"Agent error: {err}")


# -- 3-Column Layout ----------------------------------------------------------
col_l, col_m, col_r = st.columns([1.2, 2.5, 1.3])

# LEFT PANEL - Live Sensor Feed
with col_l:
    st.subheader("Live Feed")

    weather = get_weather(loc_cfg["lat"], loc_cfg["lon"])
    sensor  = get_sensor(loc_cfg["usgs"], loc_cfg["threshold"])

    # Water level gauge
    if sensor and sensor.get("ok"):
        gage   = sensor["gage"]
        thresh = loc_cfg["threshold"]
        make_gauge(gage, thresh*2, thresh, "River Gage Height", "ft", "wg")
        if gage >= thresh * 1.3:    badge = "[CRITICAL]"
        elif gage >= thresh:        badge = "[DANGER]"
        elif gage >= thresh * 0.8:  badge = "[WARNING]"
        else:                       badge = "[NORMAL]"
        st.markdown(f"**{sensor.get('site', 'USGS')}**")
        st.markdown(f"`{gage:.2f} ft` - {badge}")
        st.caption(sensor["time"])
    else:
        st.warning(f"USGS unavailable: {sensor.get('error', '') if sensor else 'No response'}")

    # Weather card
    if weather and weather.get("ok"):
        rain_level = "Heavy rain" if weather["rain"] > 5 else "Light rain" if weather["rain"] > 0.5 else "Clear"
        st.markdown(f"""
<div class="metric-card">
<b>{loc_cfg['city']}</b><br>
Temp: <b>{weather['temp']:.1f} C</b><br>
Rain: <b>{weather['rain']:.1f} mm/hr</b> ({rain_level})<br>
Wind: <b>{weather['wind']:.1f} km/h</b>
</div>
""", unsafe_allow_html=True)
    else:
        st.warning(f"Weather unavailable: {weather.get('error', '') if weather else ''}")

# CENTER PANEL - Agent Chat
with col_m:
    st.subheader("AI Analyst")

    with st.container(height=480):
        if not st.session_state.chat_history:
            st.info(
                f"**System ready** - monitoring **{loc_cfg['city']}**\n\n"
                "Example questions:\n"
                "- *Should we evacuate near the river?*\n"
                "- *What are the FEMA protocols for flash floods?*\n"
                "- *What did people report during the Savar building collapse?*\n"
                "- *Are there any recent earthquakes nearby?*"
            )
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tool_calls"):
                    badges = "".join(
                        f'<span class="tool-badge">[tool] {t}</span>'
                        for t in msg["tool_calls"]
                    )
                    st.markdown(f"Tools used: {badges}", unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about the disaster situation..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("Reasoning across live data sources..."):
            result = call_agent(prompt, location=loc_cfg["city"], thread_id="dash")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "tool_calls": result.get("tool_calls", []),
            })
        st.rerun()

    if st.session_state.chat_history and st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

# RIGHT PANEL - Alert Log
with col_r:
    st.subheader("Alerts")
    alerts = load_alerts()
    if not alerts:
        st.info("No alerts yet.\nAlerts are logged when critical conditions are detected.")
    else:
        colors = {"CRITICAL": "#ff4b4b", "HIGH": "#ffa500", "MEDIUM": "#ffd700", "LOW": "#00cc88"}
        for a in alerts:
            sev   = a.get("severity", "LOW")
            ts    = a.get("timestamp", "")[:16].replace("T", " ")
            zone  = a.get("zone", "")
            msg   = a.get("message", "")
            color = colors.get(sev, "#718096")
            truncated = msg[:140] + ("..." if len(msg) > 140 else "")
            st.markdown(f"""
<div class="metric-card alert-{sev}" style="border-left-color:{color}">
<small><b style="color:{color}">{sev}</b> | {ts}</small><br>
<small>{zone}</small><br>
{truncated}
</div>""", unsafe_allow_html=True)

    if alerts and st.button("Clear alerts"):
        if API_BASE_URL:
            try:
                requests.delete(f"{API_BASE_URL}/eval/alerts", timeout=5)
            except Exception:
                pass
        else:
            if os.path.exists(ALERTS_FILE):
                os.remove(ALERTS_FILE)
        st.rerun()
