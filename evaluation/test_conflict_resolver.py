"""Verify the conflict resolver never suppresses warnings on non-comparable evidence.

Run: python evaluation/test_conflict_resolver.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.conflict_resolver import ConflictResolver, EvidenceState
from core.evidence_validator import (
    EvidencePacket, FreshnessStatus, SourceType, TemporalTier,
)

NOW = datetime.now(timezone.utc)


def packet(evid, source_type, tier, text, location, event_time, metric=None, value=None, meta=None):
    return EvidencePacket(
        evidence_id=evid, source_id=evid, source_type=source_type, raw_text=text,
        event_time=event_time, observed_at=NOW, ingested_at=NOW,
        valid_until=NOW + timedelta(hours=2), temporal_tier=tier,
        freshness=FreshnessStatus.FRESH, location_name=location,
        lat=None, lon=None, metric_name=metric, metric_value=value,
        metadata=meta or {},
    )


tel = packet(
    "tel", SourceType.TELEMETRY, TemporalTier.LIVE,
    "Gage height 4.2 ft, flood stage 10.0 ft, status NORMAL",
    "Boulder Creek", NOW, "gage_height_ft", 4.2, {"flood_stage_ft": 10.0},
)

comparable_hist = packet(
    "hist", SourceType.SOCIAL_MEDIA, TemporalTier.HISTORICAL,
    "Boulder Creek flooding, underwater, evacuate now",
    "Boulder Creek", NOW - timedelta(minutes=30),
)

noncomparable_social = packet(
    "social", SourceType.SOCIAL_MEDIA, TemporalTier.LIVE,
    "Homes flooded and residents stranded downtown",
    "Different Creek", NOW - timedelta(minutes=10),
)

r1 = ConflictResolver.resolve_evidence([tel, comparable_hist], "Boulder Creek flood?" )
r2 = ConflictResolver.resolve_evidence([tel, noncomparable_social], "street flooding?")

print("comparable    ->", r1.overall_state.value, "abstain", r1.should_abstain)
print("noncomparable ->", r2.overall_state.value, "abstain", r2.should_abstain)

ok = (
    r1.overall_state == EvidenceState.CONTRADICTED
    and r2.overall_state == EvidenceState.UNVERIFIED
    and r2.should_abstain
    and not r2.can_issue_evacuation
)
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
