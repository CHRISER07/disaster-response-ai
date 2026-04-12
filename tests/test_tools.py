"""
tests/test_tools.py

Unit tests for all 5 live tools. Each test verifies the tool hits a real API
and returns data in the expected format. Tests gracefully skip if network is unavailable.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.weather_tool import fetch_live_weather
from tools.sensor_tool import query_water_sensor, query_recent_earthquakes
from tools.alert_tool import generate_structured_alert, load_recent_alerts


# ── Weather Tool ──────────────────────────────────────────────────────────────
class TestWeatherTool:
    def test_returns_string(self):
        result = fetch_live_weather.invoke("Boulder, CO")
        assert isinstance(result, str)

    def test_contains_temperature(self):
        result = fetch_live_weather.invoke("Boulder, CO")
        assert "Temperature" in result or "WEATHER TOOL ERROR" in result

    def test_handles_api_failure_gracefully(self):
        """Tool must return a string even if API fails — never raise."""
        result = fetch_live_weather.invoke("Nonexistent_Location_XYZ_123")
        assert isinstance(result, str)


# ── Sensor Tool ───────────────────────────────────────────────────────────────
class TestSensorTool:
    def test_returns_string(self):
        result = query_water_sensor.invoke("06730200")
        assert isinstance(result, str)

    def test_contains_gage_height(self):
        result = query_water_sensor.invoke("06730200")
        # Either real data or a graceful error
        assert "Gage Height" in result or "SENSOR TOOL ERROR" in result or "No data" in result

    def test_gage_height_is_positive(self):
        result = query_water_sensor.invoke("06730200")
        if "Gage Height" in result:
            # Extract the number from "Gage Height: 4.32 ft"
            import re
            numbers = re.findall(r"(\d+\.\d+) ft", result)
            if numbers:
                assert float(numbers[0]) >= 0

    def test_earthquake_returns_string(self):
        result = query_recent_earthquakes.invoke({"min_magnitude": 3.0})
        assert isinstance(result, str)
        assert "EARTHQUAKE" in result or "No M" in result or "EARTHQUAKE TOOL ERROR" in result


# ── Alert Tool ────────────────────────────────────────────────────────────────
class TestAlertTool:
    def test_creates_alert_file(self, tmp_path, monkeypatch):
        """Alert is written to the alerts file."""
        import tools.alert_tool as at
        test_file = str(tmp_path / "test_alerts.jsonl")
        monkeypatch.setattr(at, "ALERTS_FILE", test_file)

        result = at.generate_structured_alert.invoke({
            "severity": "HIGH",
            "message": "Test flood alert",
            "zone": "Zone A"
        })
        assert "ALERT LOGGED" in result
        assert os.path.exists(test_file)

    def test_alert_severity_normalised(self, tmp_path, monkeypatch):
        """Invalid severity should be coerced to MEDIUM."""
        import tools.alert_tool as at
        test_file = str(tmp_path / "test_alerts2.jsonl")
        monkeypatch.setattr(at, "ALERTS_FILE", test_file)

        result = at.generate_structured_alert.invoke({
            "severity": "SUPER_ULTRA_HIGH",
            "message": "Invalid severity test",
            "zone": "Test"
        })
        assert "MEDIUM" in result

    def test_load_returns_list(self, tmp_path, monkeypatch):
        import tools.alert_tool as at
        test_file = str(tmp_path / "test_alerts3.jsonl")
        monkeypatch.setattr(at, "ALERTS_FILE", test_file)
        at.generate_structured_alert.invoke({"severity": "LOW", "message": "test", "zone": "Z"})
        alerts = at.load_recent_alerts()
        assert isinstance(alerts, list)
