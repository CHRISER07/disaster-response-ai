"""
tools/sensor_tool.py

Live USGS sensor data (100% free, no API key required).
Covers river gage heights (water levels) and recent earthquake activity.
"""
import requests
from datetime import datetime, timedelta, timezone
from langchain_core.tools import tool

# USGS Site 06730200 = Boulder Creek at N 75th St — the epicenter of 2013 CO Flood
USGS_SITE_ID = "06730200"
USGS_SITE_NAME = "Boulder Creek at North 75th Street"
DANGER_THRESHOLD_FT = 10.0  # Historical flood stage for this gauge

# Boulder, CO earthquake monitoring area
BOULDER_LAT, BOULDER_LON = 40.0150, -105.2705
EARTHQUAKE_RADIUS_KM = 200

@tool
def query_water_sensor(site_id: str = USGS_SITE_ID) -> str:
    """
    Queries the USGS National Water Information System for current river gage height.
    Returns real-time water level in feet with flood danger status.
    Use this to determine if flood conditions are critical.
    """
    try:
        url = "https://waterservices.usgs.gov/nwis/iv/"
        params = {
            "sites": site_id,
            "parameterCd": "00065",  # Gage height in feet
            "format": "json"
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        time_series = data["value"]["timeSeries"]
        if not time_series:
            return f"[SENSOR] No data available for site {site_id}."

        # Get most recent reading
        values = time_series[0]["values"][0]["value"]
        if not values:
            return f"[SENSOR] No recent readings for site {site_id}."

        latest = values[-1]
        gage_height = float(latest["value"])
        timestamp = latest["dateTime"]
        site_name = time_series[0]["sourceInfo"]["siteName"]

        pct_above = ((gage_height - DANGER_THRESHOLD_FT) / DANGER_THRESHOLD_FT) * 100

        if gage_height >= DANGER_THRESHOLD_FT * 1.2:
            status = "CRITICAL — MANDATORY EVACUATION THRESHOLD EXCEEDED"
        elif gage_height >= DANGER_THRESHOLD_FT:
            status = "DANGER — At or above flood stage"
        elif gage_height >= DANGER_THRESHOLD_FT * 0.8:
            status = "WARNING — Approaching flood stage"
        else:
            status = "NORMAL"

        return (
            f"[LIVE USGS SENSOR — {site_name}]\n"
            f"  Gage Height: {gage_height:.2f} ft (Danger threshold: {DANGER_THRESHOLD_FT} ft)\n"
            f"  Status: {status}\n"
            f"  {'Above' if pct_above > 0 else 'Below'} threshold by: {abs(pct_above):.1f}%\n"
            f"  Last Reading: {timestamp}"
        )
    except requests.RequestException as e:
        return f"[SENSOR TOOL ERROR] Could not reach USGS API: {e}"
    except (KeyError, IndexError, ValueError) as e:
        return f"[SENSOR TOOL ERROR] Unexpected API response format: {e}"


@tool
def query_recent_earthquakes(min_magnitude: float = 3.0) -> str:
    """
    Fetches recent earthquakes (M≥3.0 by default) near the disaster area from USGS.
    Use this to assess seismic risk to infrastructure.
    """
    try:
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        params = {
            "format": "geojson",
            "starttime": start,
            "latitude": BOULDER_LAT,
            "longitude": BOULDER_LON,
            "maxradiuskm": EARTHQUAKE_RADIUS_KM,
            "minmagnitude": min_magnitude,
            "orderby": "magnitude"
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        features = resp.json()["features"]

        if not features:
            return (
                f"[USGS EARTHQUAKES] No M≥{min_magnitude} earthquakes detected within "
                f"{EARTHQUAKE_RADIUS_KM}km in the past 7 days. Seismic risk: LOW."
            )

        lines = [f"[USGS EARTHQUAKES — Past 7 days, M≥{min_magnitude}]"]
        for quake in features[:5]:  # report top 5 by magnitude
            props = quake["properties"]
            lines.append(
                f"  M{props['mag']:.1f} — {props['place']} at {props['time'] and datetime.utcfromtimestamp(props['time']/1000).strftime('%Y-%m-%d %H:%M UTC')}"
            )
        if len(features) > 5:
            lines.append(f"  ... and {len(features)-5} more events.")
        return "\n".join(lines)

    except requests.RequestException as e:
        return f"[EARTHQUAKE TOOL ERROR] Could not reach USGS API: {e}"
