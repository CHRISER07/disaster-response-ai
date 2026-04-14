"""
tools/weather_tool.py

Live weather using Open-Meteo (free, no API key).
Accepts any city name — uses Open-Meteo geocoding to resolve coordinates.
"""
import requests
from langchain_core.tools import tool

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Moderate drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Rain showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Thunderstorm + heavy hail"
}

# Default: Boulder, CO (2013 Colorado Flood reference point)
DEFAULT_LOCATION = {"name": "Boulder, CO", "lat": 40.0150, "lon": -105.2705}


def geocode_location(city_name: str) -> dict | None:
    """
    Uses Open-Meteo's free geocoding API to resolve a city name to lat/lon.
    Returns dict with lat, lon, name, country — or None if not found.
    """
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 1, "language": "en", "format": "json"},
            timeout=8
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            r = results[0]
            return {
                "name": f"{r.get('name', city_name)}, {r.get('country_code', '')}",
                "lat": r["latitude"],
                "lon": r["longitude"]
            }
    except Exception:
        pass
    return None


@tool
def fetch_live_weather(location: str = "Boulder, CO") -> str:
    """
    Fetches live weather conditions for ANY city worldwide.
    Returns temperature (°C), precipitation (mm/hr), wind speed (km/h), and conditions.
    Accepts any city name like 'Miami, FL', 'Houston, TX', 'New Orleans' etc.
    Use this to assess whether disaster weather conditions are worsening or improving.
    """
    # Try to geocode the location
    geo = geocode_location(location)
    if geo:
        lat, lon, display_name = geo["lat"], geo["lon"], geo["name"]
    else:
        # Fall back to default coordinates
        lat, lon = DEFAULT_LOCATION["lat"], DEFAULT_LOCATION["lon"]
        display_name = DEFAULT_LOCATION["name"]

    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,rain,wind_speed_10m,weather_code,relative_humidity_2m,apparent_temperature",
                "timezone": "auto"
            },
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()["current"]

        temp        = data["temperature_2m"]
        feels_like  = data["apparent_temperature"]
        rain        = data["rain"]
        wind        = data["wind_speed_10m"]
        humidity    = data["relative_humidity_2m"]
        condition   = WMO_CODES.get(data["weather_code"], f"Code {data['weather_code']}")

        if rain > 10 or wind > 80:
            severity = "CRITICAL"
        elif rain > 5 or wind > 60:
            severity = "HIGH"
        elif rain > 1 or wind > 40:
            severity = "MODERATE"
        else:
            severity = "NORMAL"

        return (
            f"[LIVE WEATHER — {display_name}] Severity: {severity}\n"
            f"  Temperature: {temp:.1f}°C (feels like {feels_like:.1f}°C)\n"
            f"  Rainfall: {rain:.1f} mm/hr | Wind: {wind:.1f} km/h | Humidity: {humidity}%\n"
            f"  Conditions: {condition}"
        )
    except requests.RequestException as e:
        return f"[WEATHER TOOL ERROR] Could not reach Open-Meteo API: {e}"
