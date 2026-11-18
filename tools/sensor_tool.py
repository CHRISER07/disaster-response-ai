"""
tools/sensor_tool.py

Live USGS sensor data & global hydrological / seismic intelligence.
Supports both US USGS river gauge networks and global Open-Meteo GloFAS river discharge.
All endpoints are free, no API key required.
Optimized with O(1) in-memory TTL caching and efficient string pooling.
"""
import time
import requests
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# High Performance In-Memory TTL Cache
# ---------------------------------------------------------------------------
class TTLCache:
    """O(1) Dictionary-backed TTL cache for API sensor data."""
    def __init__(self, ttl_seconds: float = 60.0):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        if key in self._cache:
            ts, val = self._cache[key]
            if now - ts <= self.ttl:
                return val
            else:
                del self._cache[key]
        return None

    def set(self, key: str, val: Any) -> None:
        self._cache[key] = (time.time(), val)


_SENSOR_CACHE = TTLCache(ttl_seconds=60.0)      # Cache river gauge/discharge for 60 seconds
_EARTHQUAKE_CACHE = TTLCache(ttl_seconds=120.0) # Cache earthquakes for 2 minutes


# ---------------------------------------------------------------------------
# Known USGS Gauge Sites (US Rivers)
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


def _match_usgs_site(location: str) -> dict | None:
    """Returns matching known USGS gauge if location matches a supported US site."""
    loc_lower = location.lower().strip()
    for key, site in KNOWN_SITES.items():
        if key in loc_lower:
            return site
    return None


def format_gage_report(site_id: str, threshold_ft: float, site_name: str,
                       gage: float, ts_str: str) -> str:
    """Render a gage reading in the canonical sensor-report format.

    Extracted from ``_fetch_gage`` so that counterfactual evaluation can supply
    a gage height through the exact same severity logic the live tool uses.
    """
    pct = ((gage - threshold_ft) / threshold_ft) * 100
    direction = "above" if pct > 0 else "below"

    # Severity classification
    if gage >= threshold_ft * 1.3:
        status = "CRITICAL - Mandatory evacuation threshold EXCEEDED"
        alarm = (
            "\n\n[CRITICAL] IMMEDIATE EVACUATION PROTOCOL REQUIRED. "
            f"Gage height {gage:.2f} ft is {abs(pct):.1f}% above flood stage. "
            "Per FEMA protocol, initiate mandatory evacuation of all low-lying zones. "
            "Do NOT allow personnel to cross flooded areas."
        )
    elif gage >= threshold_ft:
        status = "DANGER - At flood stage"
        alarm = (
            "\n\n[DANGER] River has reached flood stage. "
            "Prepare evacuation resources and issue WARNING alerts for low-lying areas."
        )
    elif gage >= threshold_ft * 0.8:
        status = "WARNING - Approaching flood stage"
        alarm = (
            "\n\n[WARNING] River approaching flood stage rapidly. "
            "Pre-position emergency resources and monitor every 15 minutes."
        )
    else:
        status = "NORMAL"
        alarm = ""

    return (
        f"[LIVE USGS SENSOR - {site_name}]\n"
        f"  Site ID    : USGS #{site_id}\n"
        f"  Gage Height: {gage:.2f} ft  (Flood stage: {threshold_ft} ft)\n"
        f"  Status     : {status}\n"
        f"  Reading    : {abs(pct):.1f}% {direction} flood threshold\n"
        f"  Timestamp  : {ts_str} UTC"
        f"{alarm}"
    )


def _fetch_gage(site_id: str, threshold_ft: float, site_name: str) -> str:
    """Fetches real-time gage height from USGS NWIS for a given site ID."""
    try:
        resp = requests.get(
            "https://waterservices.usgs.gov/nwis/iv/",
            params={"sites": site_id, "parameterCd": "00065", "format": "json"},
            timeout=10
        )
        resp.raise_for_status()

        ts = resp.json().get("value", {}).get("timeSeries", [])
        if not ts:
            return f"[SENSOR] No data available for {site_name} (site {site_id})."

        values = ts[0].get("values", [{}])[0].get("value", [])
        if not values:
            return f"[SENSOR] No recent readings available for {site_name}."

        latest  = values[-1]
        gage    = float(latest["value"])
        ts_str  = latest["dateTime"]

        return format_gage_report(site_id, threshold_ft, site_name, gage, ts_str)

    except requests.RequestException as e:
        return (
            f"[SENSOR TOOL ERROR] USGS API unreachable for site {site_id}.\n"
            f"  Error: {e}\n"
            f"  Fallback: Use historical sensor data from the knowledge base."
        )
    except (KeyError, IndexError, ValueError) as e:
        return f"[SENSOR TOOL ERROR] Unexpected USGS API response format: {e}"


def _fetch_global_flood(lat: float, lon: float, display_name: str) -> str:
    """
    Fetches real-time river discharge and hydrological flood data worldwide via Open-Meteo GloFAS.
    Works for any coordinates on Earth (e.g. Venezuela, Europe, Asia, Americas).
    """
    try:
        resp = requests.get(
            "https://flood-api.open-meteo.com/v1/flood",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": ["river_discharge", "river_discharge_median", "river_discharge_max"],
                "forecast_days": 1,
            },
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json().get("daily", {})
        discharges = data.get("river_discharge", [])
        medians    = data.get("river_discharge_median", [])
        maxes      = data.get("river_discharge_max", [])

        if not discharges or discharges[0] is None:
            return (
                f"[GLOBAL HYDROLOGY - {display_name}]\n"
                f"  Coordinates: {lat:.3f}, {lon:.3f}\n"
                f"  Status: River discharge data unavailable for exact coordinates. "
                f"Check regional weather and precipitation rates."
            )

        cur_flow = float(discharges[0])
        med_flow = float(medians[0]) if (medians and medians[0] is not None) else cur_flow
        max_flow = float(maxes[0]) if (maxes and maxes[0] is not None) else (med_flow * 2.0)

        ratio = (cur_flow / med_flow) if med_flow > 0 else 1.0

        if ratio >= 2.0 or (max_flow > 0 and cur_flow >= 0.9 * max_flow):
            status = "CRITICAL - Severe Flood Threat (Discharge >200% seasonal median)"
            alarm = (
                "\n\n[CRITICAL] Severe river surge detected by global hydrological sensors. "
                "Immediate low-lying area evacuation recommended. High risk of riverbank breach."
            )
        elif ratio >= 1.4:
            status = "WARNING - Elevated river flow approaching flood threshold"
            alarm = (
                "\n\n[WARNING] River discharge is elevated above seasonal median. "
                "Alert emergency teams and monitor water levels every 15 minutes."
            )
        else:
            status = "NORMAL - River discharge within typical seasonal range"
            alarm = ""

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return (
            f"[LIVE GLOBAL HYDROLOGICAL SENSOR - {display_name}]\n"
            f"  Coordinates    : {lat:.3f}, {lon:.3f} (Copernicus GloFAS Model via Open-Meteo)\n"
            f"  River Discharge: {cur_flow:.2f} m³/s\n"
            f"  Seasonal Median: {med_flow:.2f} m³/s  (Peak threshold: {max_flow:.2f} m³/s)\n"
            f"  Status         : {status}\n"
            f"  Flow Ratio     : {ratio:.1f}x seasonal baseline\n"
            f"  Timestamp      : {today} UTC"
            f"{alarm}"
        )
    except Exception as e:
        return f"[GLOBAL HYDROLOGY ERROR] Unable to retrieve flood data for {display_name}: {e}"


# ---------------------------------------------------------------------------
# Tool 2: Water Sensor (US USGS Gauges + Global Hydrological Monitoring)
# ---------------------------------------------------------------------------
@tool
def query_water_sensor(location: str = "Boulder") -> str:
    """
    Queries real-time water levels and flood status worldwide.
    For US cities with USGS gauges (Boulder CO, Houston TX, New Orleans LA, Miami FL,
    Sacramento CA, Nashville TN, Phoenix AZ), queries live USGS river gauges.
    For all other locations worldwide (including Venezuela, Europe, Asia, Latin America),
    queries global Copernicus GloFAS river discharge and flood forecasting.
    Returns real-time gage height/flow, flood status, and evacuation directives.
    """
    cache_key = location.strip().lower()
    cached = _SENSOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from tools.weather_tool import geocode_location

    # Check for direct US USGS gauge match first
    site = _match_usgs_site(location)
    if site:
        res = _fetch_gage(site["id"], site["threshold_ft"], site["name"])
        _SENSOR_CACHE.set(cache_key, res)
        return res

    # For non-USGS locations, geocode and query global hydrology
    geo = geocode_location(location)
    if geo:
        res = _fetch_global_flood(geo["lat"], geo["lon"], geo["name"])
        _SENSOR_CACHE.set(cache_key, res)
        return res

    # No wrong-location fallback. Returning Boulder data for an unresolvable
    # location is a correctness hazard: it silently answers a different
    # question. Fail closed with an explicit data-unavailable notice.
    out = (
        f"[LIVE - USGS NWIS, Data Unavailable]: No live gauge could be resolved "
        f"for the requested location '{location}'. No reading is reported. "
        f"Do not interpret any other basin's gauge as applying to this location; "
        f"verify with local emergency management before issuing any directive."
    )
    _SENSOR_CACHE.set(cache_key, out)
    return out


# ---------------------------------------------------------------------------
# Tool 3: Earthquake Sensor (Adaptive City/Regional/Country Coverage)
# ---------------------------------------------------------------------------
@tool
def query_recent_earthquakes(location: str = "Boulder, CO", min_magnitude: float = 3.0) -> str:
    """
    Fetches M>=3.0 earthquakes (or specified magnitude) near any city, region, or country in the past 7 days.
    Pass any location like 'Venezuela', 'Caracas', 'Los Angeles, CA', 'Japan', or 'Chile'.
    Uses geocoding to resolve coordinates, dynamically expands search radius for countries/regions,
    and queries the USGS Earthquake API.
    Returns list of recent events with magnitude, exact epicenter, distance, and UTC time.
    """
    cache_key = f"{location.strip().lower()}_{min_magnitude}"
    cached = _EARTHQUAKE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from tools.weather_tool import geocode_location

    # Geocode the location. Fail closed if it cannot be resolved: using Boulder
    # coordinates would report unrelated seismic events for the wrong region.
    geo = geocode_location(location)
    if not geo:
        out = (
            f"[USGS FDSNWS, Data Unavailable]: No coordinates could be resolved "
            f"for the requested location '{location}'. No seismic events are reported. "
            f"Verify the location before issuing any seismic directive."
        )
        _EARTHQUAKE_CACHE.set(cache_key, out)
        return out
    lat, lon, display = geo["lat"], geo["lon"], geo["name"]

    # Detect broad country / regional queries to set adequate radius
    loc_clean = location.strip().lower()
    country_indicators = {
        "venezuela", "colombia", "chile", "peru", "ecuador", "mexico", "japan",
        "turkey", "italy", "indonesia", "philippines", "california", "alaska", "usa", "taiwan"
    }
    is_broad_region = (
        loc_clean in country_indicators
        or "," not in location  # single-word country or territory (e.g. "Venezuela")
        or (geo and geo.get("country_code") and loc_clean == geo.get("name", "").lower())
    )
    search_radius_km = 800 if is_broad_region else 350

    try:
        start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        resp = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={
                "format": "geojson",
                "starttime": start,
                "latitude": lat,
                "longitude": lon,
                "maxradiuskm": search_radius_km,
                "minmagnitude": min_magnitude,
                "orderby": "magnitude",
                "limit": 10,
            },
            timeout=10
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])

        if not features:
            res = (
                f"[USGS EARTHQUAKES - {display}]\n"
                f"  No M>={min_magnitude} events detected within {search_radius_km} km in the past 7 days.\n"
                f"  Seismic risk assessment: LOW for this period."
            )
            _EARTHQUAKE_CACHE.set(cache_key, res)
            return res

        lines = [f"[USGS EARTHQUAKES - {display}, past 7 days, M>={min_magnitude}, radius: {search_radius_km}km]"]
        for q in features[:6]:
            p = q["properties"]
            t = datetime.fromtimestamp(p["time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            lines.append(f"  M{p['mag']:.1f} - {p['place']} at {t}")

        if len(features) > 6:
            lines.append(f"  ... and {len(features) - 6} more events.")

        # Severity assessment
        max_mag = max(q["properties"]["mag"] for q in features)
        if max_mag >= 6.0:
            lines.append("\n[CRITICAL] Major seismic event detected (M>=6.0). Inspect bridges, pipelines, and structural integrity immediately.")
        elif max_mag >= 5.0:
            lines.append("\n[WARNING] Moderate seismic activity (M>=5.0). Potential structural damage; inspect older masonry and critical utilities.")

        out = "\n".join(lines)
        _EARTHQUAKE_CACHE.set(cache_key, out)
        return out

    except requests.RequestException as e:
        return f"[EARTHQUAKE TOOL ERROR] USGS API unreachable: {e}"
    except Exception as e:
        return f"[EARTHQUAKE TOOL ERROR] Unexpected error: {e}"
