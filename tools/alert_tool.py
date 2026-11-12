"""
tools/alert_tool.py

Generates and persists structured emergency alerts.
Alerts are written to data/alerts.jsonl - one JSON object per line.
The Streamlit dashboard reads this file to display the alert log.

Every alert passes through core/policy_engine.py BEFORE being written.
Low-confidence or malformed alerts are blocked and logged for audit.
"""
import json
import os
from datetime import datetime, timezone
from langchain_core.tools import tool

from typing import Annotated, Any

try:
    from langgraph.prebuilt import InjectedState
except Exception:
    class InjectedState:
        pass

ALERTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "alerts.jsonl")
SEVERITY_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

@tool
def generate_structured_alert(
    severity: str,
    message: str,
    zone: str = "General",
    confidence: str = "HIGH",
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """
    Creates a formally structured emergency alert and logs it to the alert file.
    Severity must be one of: LOW, MEDIUM, HIGH, CRITICAL.
    Confidence should be HIGH (from live sensors/protocols) or MEDIUM (from cross-referenced reports).
    Use this when you identify an active threat that requires responder attention.
    Example: severity='CRITICAL', message='Water level 12.4ft exceeds 10ft threshold. Mandatory evacuation Zone A.', zone='Zone A', confidence='HIGH'

    NOTE: This tool requires HIGH or MEDIUM confidence. Low-confidence alerts
    are blocked by the policy engine and must not be retried without additional data.
    """
    # Policy Gate 
    agent_msg = message
    if state and "messages" in state:
        # Search backwards for the most recent AI message (reasoning thought)
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", "") == "ai" or msg.__class__.__name__ == "AIMessage":
                content = getattr(msg, "content", "")
                if content:
                    if isinstance(content, list):
                        content = " ".join(
                            item.get("text", "") if isinstance(item, dict) else str(item)
                            for item in content
                        )
                    agent_msg = content
                    break

    try:
        from core.policy_engine import check_policy
        decision = check_policy(
            tool_name="generate_structured_alert",
            tool_args={
                "severity": severity,
                "zone": zone,
                "message": message[:200],
                "confidence": confidence,
            },
            agent_message=agent_msg,
        )
        if not decision.allowed:
            return (
                f"[ALERT BLOCKED by policy engine]\n"
                f"  Reason: {decision.reason}\n"
                f"  Confidence detected: {decision.confidence_found}\n"
                f"  Action: Gather additional verified data before generating this alert."
            )
    except ImportError:
        pass  # Policy engine not available - fail open (dev mode)

    severity = severity.upper()
    if severity not in SEVERITY_LEVELS:
        severity = "MEDIUM"

    confidence_clean = (confidence or "HIGH").strip().upper()
    if confidence_clean not in {"HIGH", "MEDIUM", "LOW"}:
        confidence_clean = "HIGH"

    alert = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "zone": zone,
        "message": message,
        "confidence": confidence_clean,
        "auto_generated": True
    }

    os.makedirs(os.path.dirname(ALERTS_FILE), exist_ok=True)
    with open(ALERTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert) + "\n")

    return (
        f"[ALERT LOGGED - {severity}]\n"
        f"  Zone: {zone} | Confidence: {confidence_clean} | Time: {alert['timestamp']}\n"
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
