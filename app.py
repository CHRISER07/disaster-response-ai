"""
app.py — Production Streamlit Dashboard: ARIA Disaster Command Center

3-panel layout:
  Left:   Live sensor gauges (USGS water level + Open-Meteo weather)
  Center: ARIA agent chat with tool-use trace
  Right:  Structured alert log
"""
import streamlit as st
import plotly.graph_objects as go
import requests
import json
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(
    page_title="ARIA — Disaster Command Center",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 8px; margin-bottom: 8px; }
    .metric-card {
        background: #1c2333; border-radius: 10px; padding: 16px;
        border-left: 4px solid #3d8ef0; margin-bottom: 12px;
    }
    .alert-CRITICAL { border-left: 4px solid #ff4b4b !important; }
    .alert-HIGH     { border-left: 4px solid #ffa500 !important; }
    .alert-MEDIUM   { border-left: 4px solid #ffd700 !important; }
    .alert-LOW      { border-left: 4px solid #00cc88 !important; }
    .tool-badge {
        display: inline-block; background: #2d3748; color: #90cdf4;
        padding: 2px 8px; border-radius: 12px; font-size: 12px;
        margin: 2px; font-family: monospace;
    }
    h1 { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
USGS_SITE = "06730200"
DANGER_FT  = 10.0
ALERTS_FILE = os.path.join("data", "alerts.jsonl")

@st.cache_data(ttl=60)
def get_live_weather():
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 40.0150, "longitude": -105.2705,
                "current": "temperature_2m,rain,wind_speed_10m,weather_code",
                "timezone": "auto"
            }, timeout=8
        )
        d = r.json()["current"]
        return {"temp": d["temperature_2m"], "rain": d["rain"],
                "wind": d["wind_speed_10m"], "code": d["weather_code"]}
    except Exception:
        return None

@st.cache_data(ttl=60)
def get_live_sensor():
    try:
        r = requests.get(
            "https://waterservices.usgs.gov/nwis/iv/",
            params={"sites": USGS_SITE, "parameterCd": "00065", "format": "json"},
            timeout=8
        )
        ts = r.json()["value"]["timeSeries"]
        if ts:
            val = ts[0]["values"][0]["value"][-1]
            return {"gage": float(val["value"]), "time": val["dateTime"]}
    except Exception:
        pass
    return None

def load_alerts():
    if not os.path.exists(ALERTS_FILE):
        return []
    alerts = []
    with open(ALERTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                alerts.append(json.loads(line.strip()))
            except Exception:
                pass
    return list(reversed(alerts[-20:]))

def make_gauge(value, max_val, danger, title, unit):
    color = "#ff4b4b" if value >= danger else "#ffa500" if value >= danger * 0.8 else "#00cc88"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#e2e8f0", "size": 14}},
        number={"suffix": f" {unit}", "font": {"color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#718096"},
            "bar": {"color": color},
            "bgcolor": "#1c2333",
            "steps": [
                {"range": [0, danger * 0.8], "color": "#1a2a1a"},
                {"range": [danger * 0.8, danger], "color": "#2a2a1a"},
                {"range": [danger, max_val], "color": "#2a1a1a"},
            ],
            "threshold": {"value": danger, "line": {"color": "#ff4b4b", "width": 2}}
        }
    ))
    fig.update_layout(
        height=200, margin={"t": 40, "b": 0, "l": 20, "r": 20},
        paper_bgcolor="#0e1117", font={"color": "#e2e8f0"}
    )
    return fig

# ── Session State ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False
if "init_error" not in st.session_state:
    st.session_state.init_error = None

# Lazy agent import to avoid blocking page load
if not st.session_state.agent_initialized:
    try:
        from agents.disaster_agent import run_agent
        st.session_state.run_agent = run_agent
        st.session_state.agent_initialized = True
    except Exception as e:
        st.session_state.init_error = str(e)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🚨 ARIA — Agentic Response Intelligence for Disasters")
st.caption(f"Live feed | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
           f"{'🟢 Agent Online' if st.session_state.agent_initialized else '🔴 Agent Offline'}")

if st.session_state.init_error:
    st.error(f"⚠️ Agent initialization failed: {st.session_state.init_error}\n\n"
             "Ensure `GROQ_API_KEY` is set in `.env`")
    st.stop()

# ── 3-Column Layout ──────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1.2, 2.5, 1.3])

# ── LEFT: Live Sensor Feed ────────────────────────────────────────────────────
with col_left:
    st.subheader("📡 Live Data Feed")

    weather = get_live_weather()
    sensor  = get_live_sensor()

    # Water Level Gauge
    if sensor:
        st.plotly_chart(
            make_gauge(sensor["gage"], 20, DANGER_FT, "River Gage Height", "ft"),
            use_container_width=True, key="gauge_water"
        )
        gage = sensor["gage"]
        status = "CRITICAL 🔴" if gage >= DANGER_FT * 1.2 else "DANGER 🟠" if gage >= DANGER_FT else "WARNING 🟡" if gage >= DANGER_FT * 0.8 else "NORMAL 🟢"
        st.markdown(f"**USGS Boulder Creek:** `{gage:.2f} ft` — {status}")
        st.caption(f"Reading: {sensor['time']}")
    else:
        st.warning("USGS sensor unavailable")

    # Weather Card
    if weather:
        rain_status = "🌧️ HEAVY" if weather["rain"] > 5 else "🌦️ MODERATE" if weather["rain"] > 0.5 else "☁️ DRY"
        st.markdown(f"""
<div class="metric-card">
<b>🌡️ Boulder, CO — Live Weather</b><br>
Temperature: <b>{weather['temp']:.1f}°C</b><br>
Rainfall: <b>{weather['rain']:.1f} mm/hr</b> {rain_status}<br>
Wind: <b>{weather['wind']:.1f} km/h</b>
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("Weather service unavailable")

    # Vector DB Stats
    st.divider()
    st.markdown("**🗄️ Knowledge Base**")
    try:
        from core.vector_store import get_vector_store
        store = get_vector_store()
        count = store._collection.count()
        st.success(f"✓ {count:,} documents indexed")
        st.caption("BGE-small-en-v1.5 + ChromaDB")
    except Exception:
        st.warning("Vector store not connected")

    # Auto-refresh
    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()
        st.rerun()

# ── CENTER: Agent Chat ────────────────────────────────────────────────────────
with col_mid:
    st.subheader("💬 ARIA — AI Disaster Analyst")

    # Chat history display
    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.chat_history:
            st.info(
                "**ARIA is ready.** Ask operational questions like:\n\n"
                "- *\"Should we evacuate near Boulder Creek?\"*\n"
                "- *\"What does the latest satellite image show?\"*\n"
                "- *\"What is the standard protocol for flash flood response?\"*\n"
                "- *\"Are there any recent earthquakes near the disaster area?\"*"
            )
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tool_calls"):
                    tools_html = "".join(
                        f'<span class="tool-badge">⚙️ {t}</span>'
                        for t in msg["tool_calls"]
                    )
                    st.markdown(f"**Tools used:** {tools_html}", unsafe_allow_html=True)

    # Input
    if prompt := st.chat_input("Ask ARIA about the disaster situation..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.spinner("ARIA is analyzing — querying live data sources..."):
            try:
                result = st.session_state.run_agent(
                    prompt, thread_id="dashboard_session"
                )
                answer = result["answer"]
                tool_calls = result.get("tool_calls", [])
            except Exception as e:
                answer = f"Agent error: {e}"
                tool_calls = []

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "tool_calls": tool_calls
        })
        st.rerun()

# ── RIGHT: Alert Log ─────────────────────────────────────────────────────────
with col_right:
    st.subheader("🔔 Alert Log")

    alerts = load_alerts()
    if not alerts:
        st.info("No alerts generated yet.\nARIA will log alerts when critical conditions are detected.")
    else:
        for alert in alerts:
            sev = alert.get("severity", "LOW")
            ts  = alert.get("timestamp", "")[:16].replace("T", " ")
            zone = alert.get("zone", "General")
            msg  = alert.get("message", "")

            color_map = {"CRITICAL": "#ff4b4b", "HIGH": "#ffa500", "MEDIUM": "#ffd700", "LOW": "#00cc88"}
            color = color_map.get(sev, "#718096")

            st.markdown(f"""
<div class="metric-card alert-{sev}" style="border-left-color: {color};">
<small><b style="color:{color}">{sev}</b> | {ts}</small><br>
<small>Zone: {zone}</small><br>
{msg[:150]}{"..." if len(msg) > 150 else ""}
</div>
""", unsafe_allow_html=True)

    # Clear alerts button
    if alerts and st.button("🗑️ Clear Alerts"):
        if os.path.exists(ALERTS_FILE):
            os.remove(ALERTS_FILE)
        st.rerun()
