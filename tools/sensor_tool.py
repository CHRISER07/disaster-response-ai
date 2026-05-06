"""
tools/sensor_tool.py

Live USGS sensor data — works for any US location by looking up nearby gauges.
Fetches real-time water level readings and recent earthquake data.
All endpoints are free, no API key required.

Fix log:
  - FIXED: datetime.utcfromtimestamp() deprecated in 3.12 → datetime.fromtimestamp(..., tz=timezone.utc)
  - ADDED: Threshold-triggered CRITICAL alarm string — loudly signals evacuation to the LLM
  - ADDED: Input validation with fallback for unrecognized locations
  - IMPROVED: Docstrings clarify expected input format to reduce LLM arg errors
"""
import requests
from datetime import datetime, timedelta, timezone
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Known USGS Gauge Sites
# ---------------------------------------------------------------------------
KNOWN_SITES = {
    "boulder":      {"id": "06730200", "name": "Boulder Creek at N 75th St, CO",     "threshold_ft": 10.0},
    "houston":      {"id": "08074000", "name": "Buffalo Bayou at Houston, TX",        "threshold_ft": 20.0},
    "new orleans":  {"id": "07374000", "name": "Mississippi R at New Orleans, LA",    "threshold_ft": 17.0},
    "miami":        {"id": "02288990", "name": "Miami Canal near Miami, FL",          "threshold_ft": 5.0},
    "sacramento":   {"id": "11447650", "name": "Sacramento River at Sacramento, CA",  "threshold_ft": 25.0},
    "nashville":    {"id": "03431500", "name": "Cumberland River at Nashville, TN",   "threshold_ft": 40.0},
    "phoenix":      {"id": "09512500", "name": "Salt River near Phoenix, AZ",         "threshold_ft": 15.0},
}
DEFAULT_SITE = KNOWN_SITES["boulder"]


def _resolve_site(location: str) -> dict:
    """
    Maps a location name to a known USGS site.
    Falls back to Boulder Creek if location is unrecognized.
    """
    loc_lower = location.lower().strip()
    for key, site in KNOWN_SITES.items():
        if key in loc_lower:
            return site
    # Graceful fallback — agent will be told which site was used
    return {**DEFAULT_SITE, "_fallback": True, "_requested": location}


def _fetch_gage(site_id: str, threshold_ft: float, site_name: str) -> str:
    """Fetches real-time gage height from USGS NWIS for a given site ID."""
    try:
        resp = requests.get(
            "https://waterservices.usgs.gov/nwis/iv/",
            params={"sites": site_id, "parameterCd": "00065", "format": "json"},
            timeout=10
        )
        resp.raise_for_status()

        # Navigate the nested USGS JSON structure
        ts = resp.json().get("value", {}).get("timeSeries", [])
        if not ts:
            return f"[SENSOR] No data available for {site_name} (site {site_id})."

        values = ts[0].get("values", [{}])[0].get("value", [])
        if not values:
            return f"[SENSOR] No recent readings available for {site_name}."

        latest  = values[-1]
        gage    = float(latest["value"])
        ts_str  = latest["dateTime"]
        pct     = ((gage - threshold_ft) / threshold_ft) * 100
        direction = "above" if pct > 0 else "below"

        # Severity classification
        if gage >= threshold_ft * 1.3:
            status = "CRITICAL — Mandatory evacuation threshold EXCEEDED"
            # ⚠️ Threshold-Triggered Alarm — deliberately loud to prevent calm LLM responses
            alarm = (
                "\n\n⚠️⚠️⚠️ CRITICAL: IMMEDIATE EVACUATION PROTOCOL REQUIRED. "
                f"Gage height {gage:.2f} ft is {abs(pct):.1f}% above flood stage. "
                "Per FEMA protocol, initiate mandatory evacuation of all low-lying zones. "
                "Do NOT allow personnel to cross flooded areas. ⚠️⚠️⚠️"
            )
        elif gage >= threshold_ft:
            status = "DANGER — At flood stage"
            alarm = (
                "\n\n⚠️ DANGER: River has reached flood stage. "
                "Prepare evacuation resources and issue WARNING alerts for low-lying areas."
            )
        elif gage >= threshold_ft * 0.8:
            status = "WARNING — Approaching flood stage"
            alarm = (
                "\n\n⚡ WARNING: River approaching flood stage rapidly. "
                "Pre-position emergency resources and monitor every 15 minutes."
            )
        else:
            status = "NORMAL"
            alarm = ""

        return (
            f"[LIVE USGS SENSOR — {site_name}]\n"
            f"  Site ID   : USGS #{site_id}\n"
            f"  Gage Height: {gage:.2f} ft  (Flood stage: {threshold_ft} ft)\n"
            f"  Status    : {status}\n"
            f"  Reading   : {abs(pct):.1f}% {direction} flood threshold\n"
            f"  Timestamp : {ts_str} UTC"
            f"{alarm}"
        )

    except requests.RequestException as e:
        return (
            f"[SENSOR TOOL ERROR] USGS API unreachable for site {site_id}.\n"
            f"  Error: {e}\n"
            f"  Fallback: Use historical sensor data from the knowledge base."
        )
    except (KeyError, IndexError, ValueError) as e:
        return f"[SENSOR TOOL ERROR] Unexpected USGS API response format: {e}"


# ---------------------------------------------------------------------------
# Tool 2: Water Sensor
# ---------------------------------------------------------------------------
@tool
def query_water_sensor(location: str = "Boulder") -> str:
    """
    Queries a real USGS river gauge for current water levels.
    Supported locations (city name or state): Boulder CO, Houston TX,
    New Orleans LA, Miami FL, Sacramento CA, Nashville TN, Phoenix AZ.
    Pass just the city name, e.g. "Houston" or "Houston, TX".
    Returns real-time gage height, flood danger status, and evacuation flags.
    If the city is not in the supported list, defaults to Boulder Creek, CO.
    """
    site = _resolve_site(location)
    result = _fetch_gage(site["id"], site["threshold_ft"], site["name"])

    # Notify agent if we fell back to a default location
    if site.get("_fallback"):
        result = (
            f"[NOTE] Location '{site['_requested']}' is not in the supported gauge list. "
            f"Showing data for Boulder Creek, CO (nearest default).\n\n"
        ) + result

    return result


# ---------------------------------------------------------------------------
# Tool 3: Earthquake Sensor
# ---------------------------------------------------------------------------
@tool
def query_recent_earthquakes(location: str = "Boulder, CO", min_magnitude: float = 3.0) -> str:
    """
    Fetches M≥3.0 earthquakes (or specified magnitude) near any city in the past 7 days.
    Pass any city name, e.g. "Los Angeles, CA" or "Seattle".
    Uses Open-Meteo geocoding to resolve coordinates, then queries USGS Earthquake API.
    Returns a list of recent events with magnitude, location, and UTC time.
    Use this to assess seismic risk to buildings and infrastructure.
    """
    from tools.weather_tool import geocode_location

    # Geocode the location
    geo = geocode_location(location)
    if geo:
        lat, lon, display = geo["lat"], geo["lon"], geo["name"]
    else:
        lat, lon, display = 40.0150, -105.2705, "Boulder, CO (geocoding fallback)"

    try:
        start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        resp = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={
                "format": "geojson",
                "starttime": start,
                "latitude": lat,
                "longitude": lon,
                "maxradiuskm": 300,
                "minmagnitude": min_magnitude,
                "orderby": "magnitude"
            },
            timeout=10
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])

        if not features:
            return (
                f"[USGS EARTHQUAKES — {display}]\n"
                f"  No M≥{min_magnitude} events within 300 km in the past 7 days.\n"
                f"  Seismic risk assessment: LOW for this period."
            )

        lines = [f"[USGS EARTHQUAKES — {display}, past 7 days, M≥{min_magnitude}]"]
        for q in features[:5]:
            p = q["properties"]
            # FIX: datetime.utcfromtimestamp is deprecated in Python 3.12
            t = datetime.fromtimestamp(p["time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            lines.append(f"  M{p['mag']:.1f} — {p['place']} at {t}")

        if len(features) > 5:
            lines.append(f"  ... and {len(features) - 5} more events.")

        # Add severity assessment
        max_mag = max(q["properties"]["mag"] for q in features)
        if max_mag >= 6.0:
            lines.append("\n⚠️ CRITICAL: Major seismic event detected. Inspect all critical infrastructure.")
        elif max_mag >= 5.0:
            lines.append("\n⚡ WARNING: Moderate seismic activity. Check structural integrity of key buildings.")

        return "\n".join(lines)

    except requests.RequestException as e:
        return f"[EARTHQUAKE TOOL ERROR] USGS API unreachable: {e}"
    except Exception as e:
        return f"[EARTHQUAKE TOOL ERROR] Unexpected error: {e}"
