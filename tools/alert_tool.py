"""
tools/alert_tool.py

Generates and persists structured emergency alerts.
Alerts are written to data/alerts.jsonl — one JSON object per line.
The Streamlit dashboard reads this file to display the alert log.
"""
import json
import os
from datetime import datetime, timezone
from langchain_core.tools import tool

ALERTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "alerts.jsonl")
SEVERITY_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

@tool
def generate_structured_alert(severity: str, message: str, zone: str = "General") -> str:
    """
    Creates a formally structured emergency alert and logs it to the alert file.
    Severity must be one of: LOW, MEDIUM, HIGH, CRITICAL.
    Use this when you identify an active threat that requires responder attention.
    Example: severity='CRITICAL', message='Water level 12.4ft exceeds 10ft threshold. Mandatory evacuation Zone A.', zone='Zone A'
    """
    severity = severity.upper()
    if severity not in SEVERITY_LEVELS:
        severity = "MEDIUM"

    alert = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "zone": zone,
        "message": message,
        "auto_generated": True
    }

    os.makedirs(os.path.dirname(ALERTS_FILE), exist_ok=True)
    with open(ALERTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert) + "\n")

    return (
        f"[ALERT LOGGED — {severity}]\n"
        f"  Zone: {zone} | Time: {alert['timestamp']}\n"
        f"  Message: {message}"
    )


def load_recent_alerts(n: int = 20) -> list[dict]:
    """Loads the most recent N alerts from the alert log. Used by the dashboard."""
    if not os.path.exists(ALERTS_FILE):
        return []
    alerts = []
    with open(ALERTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    alerts.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return alerts[-n:]
