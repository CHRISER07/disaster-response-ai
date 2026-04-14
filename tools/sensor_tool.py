"""
tools/sensor_tool.py

Live USGS sensor data — works for any US location by searching nearby gauges.
Fetches water level readings and recent earthquake data.
All free, no API key required.
"""
import requests
from datetime import datetime, timedelta, timezone
from langchain_core.tools import tool

# Known high-quality flood gauge sites (expandable)
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
    """Maps a location name to a known USGS site, or returns the default."""
    loc_lower = location.lower()
    for key, site in KNOWN_SITES.items():
        if key in loc_lower:
            return site
    return DEFAULT_SITE


def _fetch_gage(site_id: str, threshold_ft: float, site_name: str) -> str:
    """Fetches real-time gage height from USGS NWIS for a given site ID."""
    try:
        resp = requests.get(
            "https://waterservices.usgs.gov/nwis/iv/",
            params={"sites": site_id, "parameterCd": "00065", "format": "json"},
            timeout=10
        )
        resp.raise_for_status()
        ts = resp.json()["value"]["timeSeries"]
        if not ts:
            return f"[SENSOR] No data available for {site_name}."

        values = ts[0]["values"][0]["value"]
        if not values:
            return f"[SENSOR] No recent readings for {site_name}."

        latest   = values[-1]
        gage     = float(latest["value"])
        ts_str   = latest["dateTime"]
        pct      = ((gage - threshold_ft) / threshold_ft) * 100

        if gage >= threshold_ft * 1.3:
            status = "CRITICAL — MANDATORY EVACUATION THRESHOLD EXCEEDED"
        elif gage >= threshold_ft:
            status = "DANGER — At flood stage"
        elif gage >= threshold_ft * 0.8:
            status = "WARNING — Approaching flood stage"
        else:
            status = "NORMAL"

        direction = "above" if pct > 0 else "below"
        return (
            f"[LIVE USGS SENSOR — {site_name}]\n"
            f"  Gage Height: {gage:.2f} ft  (Flood stage: {threshold_ft} ft)\n"
            f"  Status: {status}\n"
            f"  {abs(pct):.1f}% {direction} flood threshold\n"
            f"  Reading time: {ts_str}"
        )
    except requests.RequestException as e:
        return f"[SENSOR TOOL ERROR] USGS API unreachable: {e}"
    except (KeyError, IndexError, ValueError) as e:
        return f"[SENSOR TOOL ERROR] Unexpected API response: {e}"


@tool
def query_water_sensor(location: str = "Boulder, CO") -> str:
    """
    Queries a real USGS river gauge for current water levels at ANY supported US location.
    Supported cities: Boulder CO, Houston TX, New Orleans LA, Miami FL,
    Sacramento CA, Nashville TN, Phoenix AZ (more can be added).
    Returns real-time gage height and flood danger status.
    """
    site = _resolve_site(location)
    return _fetch_gage(site["id"], site["threshold_ft"], site["name"])


@tool
def query_recent_earthquakes(location: str = "Boulder, CO", min_magnitude: float = 3.0) -> str:
    """
    Fetches M≥3.0 earthquakes (or given magnitude) near any city in the past 7 days.
    Uses Open-Meteo geocoding to resolve the city name, then queries USGS Earthquake API.
    Use this to assess seismic risk to infrastructure.
    """
    # Geocode the location
    from tools.weather_tool import geocode_location
    geo = geocode_location(location)
    if geo:
        lat, lon, display = geo["lat"], geo["lon"], geo["name"]
    else:
        lat, lon, display = 40.0150, -105.2705, "Boulder, CO (default)"

    try:
        start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        resp = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={
                "format": "geojson",
                "starttime": start,
                "latitude": lat, "longitude": lon,
                "maxradiuskm": 300,
                "minmagnitude": min_magnitude,
                "orderby": "magnitude"
            },
            timeout=10
        )
        resp.raise_for_status()
        features = resp.json()["features"]

        if not features:
            return (
                f"[USGS EARTHQUAKES — {display}] "
                f"No M≥{min_magnitude} events within 300 km in the past 7 days. "
                f"Seismic risk: LOW."
            )

        lines = [f"[USGS EARTHQUAKES — {display}, past 7 days, M≥{min_magnitude}]"]
        for q in features[:5]:
            p = q["properties"]
            t = datetime.utcfromtimestamp(p["time"] / 1000).strftime("%Y-%m-%d %H:%M UTC")
            lines.append(f"  M{p['mag']:.1f} — {p['place']} at {t}")
        if len(features) > 5:
            lines.append(f"  ... and {len(features)-5} more events.")
        return "\n".join(lines)

    except requests.RequestException as e:
        return f"[EARTHQUAKE TOOL ERROR] USGS API unreachable: {e}"
