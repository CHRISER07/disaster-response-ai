"""
core/temporal_grounding.py

Real Temporal Grounding Injection (TGI).

Replaces the legacy categorical year-matching rule ("2012-2015 -> HISTORICAL")
with an actual rolling time-window comparison against the current UTC time.

Windows are configuration-driven, never hardcoded in the decision logic:

    TGI_LIVE_WINDOW_HOURS   default 72    (within -> LIVE)
    TGI_RECENT_WINDOW_DAYS  default 7     (within -> RECENT, else HISTORICAL)

A timestamp is parsed from ISO-8601 (with Z or numeric UTC offset). Documents
without a timestamp are explicitly marked UNDATED; they are never silently
labelled LIVE.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

LIVE_WINDOW_HOURS = float(os.getenv("TGI_LIVE_WINDOW_HOURS", "72"))
RECENT_WINDOW_DAYS = float(os.getenv("TGI_RECENT_WINDOW_DAYS", "7"))


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into an aware UTC datetime, or None."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def temporal_tag(metadata: dict, now: Optional[datetime] = None) -> str:
    """
    Return an honest TGI tag for a document.

    Live telemetry (fresh timestamps) -> LIVE
    Recent telemetry                  -> RECENT
    Old archives                      -> HISTORICAL - YYYY
    Official protocols (undated)      -> KB - FEMA/CDC Protocol
    Undated social data               -> UNDATED
    """
    meta = metadata or {}
    source = str(meta.get("source", ""))
    modality = str(meta.get("modality", ""))

    dt = parse_timestamp(meta.get("timestamp"))
    if dt is None:
        if modality == "knowledge_base" or "fema" in source.lower() or "cdc" in source.lower():
            return "[KB - FEMA/CDC Protocol]"
        return "[UNDATED]"

    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    delta = reference - dt
    hours = delta.total_seconds() / 3600.0
    # Future-dated records are treated as current with an explicit marker.
    if hours < 0:
        return f"[LIVE - FUTURE_DATED, {dt.date().isoformat()}]"
    if hours <= LIVE_WINDOW_HOURS:
        return f"[LIVE - {hours:.0f}h ago]"
    if hours <= RECENT_WINDOW_DAYS * 24:
        return f"[RECENT - {hours / 24:.0f}d ago]"
    return f"[HISTORICAL - {dt.year}]"
