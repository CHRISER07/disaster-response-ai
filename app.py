"""
app.py — ARIA Disaster Command Center
Production Streamlit dashboard: Live sensors | Agentic AI chat | Alert log
Supports any global location via dynamic geocoding.
"""
import streamlit as st
import plotly.graph_objects as go
import requests
import json
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(
    page_title="ARIA — Disaster Command Center",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
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

# ── Constants ─────────────────────────────────────────────────────────────────
ALERTS_FILE = os.path.join(os.path.dirname(__file__), "data", "alerts.jsonl")
DANGER_FT   = 10.0

LOCATION_PRESETS = {
    "Boulder, CO (2013 Flood)":   {"city": "Boulder, CO",    "usgs": "06730200", "lat": 40.0150, "lon": -105.2705, "threshold": 10.0},
    "Houston, TX":                {"city": "Houston, TX",     "usgs": "08074000", "lat": 29.7604, "lon": -95.3698,  "threshold": 20.0},
    "New Orleans, LA":            {"city": "New Orleans, LA", "usgs": "07374000", "lat": 29.9511, "lon": -90.0715,  "threshold": 17.0},
    "Miami, FL":                  {"city": "Miami, FL",       "usgs": "02288990", "lat": 25.7617, "lon": -80.1918,  "threshold": 5.0},
    "Nashville, TN":              {"city": "Nashville, TN",   "usgs": "03431500", "lat": 36.1627, "lon": -86.7816,  "threshold": 40.0},
    "Custom Location":            None,  # handled separately
}

# ── Sidebar: Location & Setup ─────────────────────────────────────────────────
with st.sidebar:
    st.title("🚨 ARIA Controls")

    st.subheader("📍 Disaster Location")
    preset = st.selectbox("Select location", list(LOCATION_PRESETS.keys()))

    if preset == "Custom Location":
        custom_city = st.text_input("City name", placeholder="e.g. Sacramento, CA")
        loc_cfg = {
            "city": custom_city or "Boulder, CO",
            "usgs": "06730200",
            "lat": 40.0150, "lon": -105.2705,
            "threshold": 10.0
        }
    else:
        loc_cfg = LOCATION_PRESETS[preset]

    st.divider()

    # API Key check
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        st.success("✓ GROQ_API_KEY set")
    else:
        st.error("✗ GROQ_API_KEY missing")
        st.caption("Get a free key at [console.groq.com](https://console.groq.com)")

    st.divider()

    # DB Status & Populate
    doc_count = 0  # FIX: initialize before try block to prevent NameError in status bar
    st.subheader("🗄️ Vector Database")
    try:
        from core.vector_store import get_vector_store
        store = get_vector_store()
        doc_count = store._collection.count()
        if doc_count > 0:
            st.success(f"✓ {doc_count:,} documents indexed")
        else:
            st.warning("⚠ Database is empty!")
            if st.button("▶ Populate Database Now", type="primary"):
                with st.spinner("Ingesting real data — this takes ~2 minutes..."):
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

    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()
        st.rerun()

    st.caption("**How to run correctly:**")
    st.code("venv\\Scripts\\activate\nstreamlit run app.py", language="bash")

# ── Data fetchers ─────────────────────────────────────────────────────────────
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
    # FIX: Restructured with explicit early returns — no implicit None return path
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

def load_alerts():
    if not os.path.exists(ALERTS_FILE):
        return []
    alerts = []
    with open(ALERTS_FILE, encoding="utf-8") as f:
        for line in f:
            try: alerts.append(json.loads(line.strip()))
            except: pass
    return list(reversed(alerts[-20:]))

# ── Agent init (lazy) ─────────────────────────────────────────────────────────
if "run_agent" not in st.session_state:
    st.session_state.run_agent = None
    st.session_state.agent_error = None

if st.session_state.run_agent is None and groq_key:
    try:
        from agents.disaster_agent import run_agent
        st.session_state.run_agent = run_agent
    except Exception as e:
        st.session_state.agent_error = str(e)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"# 🚨 ARIA — Disaster Command Center")
agent_status = "🟢 **Agent Online**" if st.session_state.run_agent else "🔴 **Agent Offline**"
st.markdown(
    f'<p class="status-bar">{agent_status} &nbsp;|&nbsp; '
    f'📍 {loc_cfg["city"]} &nbsp;|&nbsp; '
    f'{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} &nbsp;|&nbsp; '
    f'🗄 {doc_count:,} docs indexed</p>',
    unsafe_allow_html=True
)

# Show clear error if key is missing or organization restricted
if not groq_key:
    st.error(
        "**GROQ_API_KEY is not set.** The AI agent cannot run without it.\n\n"
        "1. Go to [console.groq.com](https://console.groq.com) → create a free account → API Keys → Create key\n"
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

# ── 3-Column Layout ───────────────────────────────────────────────────────────
col_l, col_m, col_r = st.columns([1.2, 2.5, 1.3])

# LEFT PANEL — Live Sensor Feed
with col_l:
    st.subheader("📡 Live Feed")

    weather = get_weather(loc_cfg["lat"], loc_cfg["lon"])
    sensor  = get_sensor(loc_cfg["usgs"], loc_cfg["threshold"])

    # Water level gauge
    if sensor and sensor.get("ok"):
        gage = sensor["gage"]
        thresh = loc_cfg["threshold"]
        make_gauge(gage, thresh*2, thresh, "River Gage Height", "ft", "wg")
        pct = abs(((gage - thresh) / thresh) * 100)
        if gage >= thresh * 1.3:   badge = "🔴 CRITICAL"
        elif gage >= thresh:        badge = "🟠 DANGER"
        elif gage >= thresh * 0.8:  badge = "🟡 WARNING"
        else:                       badge = "🟢 NORMAL"
        st.markdown(f"**{sensor.get('site','USGS')}**")
        st.markdown(f"`{gage:.2f} ft` — {badge}")
        st.caption(sensor["time"])
    else:
        st.warning(f"USGS unavailable: {sensor.get('error','') if sensor else 'No response'}")

    # Weather card
    if weather and weather.get("ok"):
        rain_icon = "🌧️" if weather["rain"] > 5 else "🌦️" if weather["rain"] > 0.5 else "☀️"
        st.markdown(f"""
<div class="metric-card">
<b>🌡️ {loc_cfg['city']}</b><br>
Temp: <b>{weather['temp']:.1f}°C</b><br>
Rain: <b>{weather['rain']:.1f} mm/hr</b> {rain_icon}<br>
Wind: <b>{weather['wind']:.1f} km/h</b>
</div>
""", unsafe_allow_html=True)
    else:
        st.warning(f"Weather unavailable: {weather.get('error','') if weather else ''}")

# CENTER PANEL — Agent Chat
with col_m:
    st.subheader("💬 ARIA — AI Analyst")

    with st.container(height=480):
        if not st.session_state.chat_history:
            st.info(
                f"**ARIA ready** — monitoring **{loc_cfg['city']}**\n\n"
                "Example questions:\n"
                "- *Should we evacuate near the river?*\n"
                "- *What's the latest satellite imagery showing?*\n"
                "- *What are the FEMA protocols for flash floods?*\n"
                "- *Are there any recent earthquakes nearby?*"
            )
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tool_calls"):
                    badges = "".join(
                        f'<span class="tool-badge">⚙ {t}</span>'
                        for t in msg["tool_calls"]
                    )
                    st.markdown(f"Tools used: {badges}", unsafe_allow_html=True)

    if prompt := st.chat_input("Ask ARIA about the disaster situation..."):
        if not st.session_state.run_agent:
            st.error("Agent offline — check GROQ_API_KEY in the sidebar.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("ARIA is reasoning across live data sources..."):
                try:
                    # Inject location context so agent knows where we are
                    full_prompt = f"[Active monitoring location: {loc_cfg['city']}]\n\n{prompt}"
                    result = st.session_state.run_agent(full_prompt, thread_id="dash")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "tool_calls": result.get("tool_calls", [])
                    })
                except Exception as e:
                    err_str = str(e)
                    if "organization_restricted" in err_str:
                        msg = "Your Groq API key is restricted. Get a new one at [console.groq.com](https://console.groq.com)."
                    elif "401" in err_str or "403" in err_str:
                        msg = "Invalid Groq API key. Check `.env` and restart."
                    else:
                        msg = f"Agent error: {err_str}"
                    st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.rerun()

    if st.session_state.chat_history and st.button("🗑 Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

# RIGHT PANEL — Alert Log
with col_r:
    st.subheader("🔔 Alerts")
    alerts = load_alerts()
    if not alerts:
        st.info("No alerts yet.\nARIA logs alerts when critical conditions are detected.")
    else:
        colors = {"CRITICAL": "#ff4b4b", "HIGH": "#ffa500", "MEDIUM": "#ffd700", "LOW": "#00cc88"}
        for a in alerts:
            sev   = a.get("severity", "LOW")
            ts    = a.get("timestamp", "")[:16].replace("T", " ")
            zone  = a.get("zone", "")
            msg   = a.get("message", "")
            color = colors.get(sev, "#718096")
            st.markdown(f"""
<div class="metric-card alert-{sev}" style="border-left-color:{color}">
<small><b style="color:{color}">{sev}</b> | {ts}</small><br>
<small>{zone}</small><br>
{msg[:140]}{"…" if len(msg)>140 else ""}
</div>""", unsafe_allow_html=True)

    if alerts and st.button("🗑 Clear alerts"):
        os.remove(ALERTS_FILE)
        st.rerun()
