# tools/__init__.py
from tools.weather_tool import fetch_live_weather
from tools.sensor_tool import query_water_sensor, query_recent_earthquakes
from tools.kb_search_tool import search_knowledge_base
from tools.vision_tool import analyze_disaster_image
from tools.alert_tool import generate_structured_alert

ALL_TOOLS = [
    fetch_live_weather,
    query_water_sensor,
    query_recent_earthquakes,
    search_knowledge_base,
    analyze_disaster_image,
    generate_structured_alert,
]
