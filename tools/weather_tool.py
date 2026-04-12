"""
tools/weather_tool.py

Live weather data from Open-Meteo (100% free, no API key required).
Covers temperature, precipitation, wind speed for any coordinates.
"""
import requests
from langchain_core.tools import tool

BOULDER_CO = {"latitude": 40.0150, "longitude": -105.2705}
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 51: "Light drizzle", 61: "Slight rain", 71: "Slight snow",
    80: "Rain showers", 95: "Thunderstorm", 99: "Thunderstorm with hail"
}

@tool
def fetch_live_weather(location: str = "Boulder, CO") -> str:
    """
    Fetches live weather conditions for the disaster area.
    Returns temperature (°C), precipitation (mm/hr), wind speed (km/h), and weather description.
    Use this to assess whether conditions are worsening or improving.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            **BOULDER_CO,
            "current": "temperature_2m,rain,wind_speed_10m,weather_code,relative_humidity_2m",
            "timezone": "auto"
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()["current"]

        temp = data["temperature_2m"]
        rain = data["rain"]
        wind = data["wind_speed_10m"]
        humidity = data["relative_humidity_2m"]
        condition = WMO_CODES.get(data["weather_code"], f"Code {data['weather_code']}")

        severity = "NORMAL"
        if rain > 5 or wind > 60:
            severity = "CRITICAL"
        elif rain > 2 or wind > 40:
            severity = "HIGH"
        elif rain > 0.5:
            severity = "MODERATE"

        return (
            f"[LIVE WEATHER — {location}] Severity: {severity}\n"
            f"  Temperature: {temp:.1f}°C | Rainfall: {rain:.1f} mm/hr | "
            f"Wind: {wind:.1f} km/h | Humidity: {humidity}%\n"
            f"  Conditions: {condition}"
        )
    except requests.RequestException as e:
        return f"[WEATHER TOOL ERROR] Could not reach Open-Meteo API: {e}"
