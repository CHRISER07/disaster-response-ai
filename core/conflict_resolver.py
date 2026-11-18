"""
core/conflict_resolver.py

Evidence Conflict Detection, Multi-Source Reconciliation,
and Safe Abstention Engine for Disaster Intelligence.

Implements the formal 4-State Evidence Framework:
  1. SUPPORTED: Multiple corroborated streams agree on physical conditions.
  2. CONTRADICTED: Discrepant reports detected across temporal, sensor, or social sources.
  3. UNVERIFIED: Single-source eyewitness/social claims lacking sensor confirmation.
  4. INSUFFICIENT: Essential telemetry or doctrine absent; triggers safe abstention.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from core.evidence_validator import EvidencePacket, SourceType, TemporalTier, FreshnessStatus
from core.geo_router import SpatialRelevance


# A gauge reading can only contradict a field report when both observations
# describe the same incident, within a compatible time window and geographic
# area, and measured through a comparable phenomenon. Otherwise the safe state
# is UNVERIFIED, never CONTRADICTED (a below-flood-stage river gauge does not
# prove that streets are not flooded by drainage failure or flash runoff).
MAX_COMPARABLE_TIME_SECONDS = 2 * 3600
MAX_COMPARABLE_DISTANCE_KM = 10.0


def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _comparable(tel: EvidencePacket, report: EvidencePacket) -> bool:
    """True only when incident/time/region/phenomenon/measurement are compatible."""
    if tel.metric_name not in ("gage_height_ft", "water_level_ft", "river_stage_ft"):
        return False
    # Time comparability: use event time when both carry it.
    if tel.event_time and report.event_time:
        if abs((tel.event_time - report.event_time).total_seconds()) > MAX_COMPARABLE_TIME_SECONDS:
            return False
    else:
        return False
    # Geographic comparability: require named location or coordinates on both.
    if tel.location_name and report.location_name:
        if tel.location_name.strip().lower() != report.location_name.strip().lower():
            return False
    elif tel.lat is not None and tel.lon is not None and report.lat is not None and report.lon is not None:
        if _haversine_km(tel.lat, tel.lon, report.lat, report.lon) > MAX_COMPARABLE_DISTANCE_KM:
            return False
    else:
        return False
    return True


class EvidenceState(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass
class ConflictResolutionResult:
    overall_state: EvidenceState
    primary_status: str                     # "NORMAL" | "WARNING" | "DANGER" | "CRITICAL" | "INSUFFICIENT_EVIDENCE"
    resolution_summary: str
    can_issue_evacuation: bool
    should_abstain: bool
    safe_abstention_message: Optional[str] = None
    supporting_evidence: List[EvidencePacket] = field(default_factory=list)
    conflicting_evidence: List[EvidencePacket] = field(default_factory=list)
    unverified_reports: List[EvidencePacket] = field(default_factory=list)
    detailed_discrepancies: List[str] = field(default_factory=list)


class ConflictResolver:
    """Reconciles conflicting multi-source observations and enforces safe abstention."""

    @classmethod
    def resolve_evidence(
        cls,
        evidence_list: List[EvidencePacket],
        query_context: str = ""
    ) -> ConflictResolutionResult:
        """
        Reconciles a collection of validated evidence packets for an incident query.
        """
        if not evidence_list:
            return ConflictResolutionResult(
                overall_state=EvidenceState.INSUFFICIENT,
                primary_status="INSUFFICIENT_EVIDENCE",
                resolution_summary="No validated evidence packets retrieved for this operational sector.",
                can_issue_evacuation=False,
                should_abstain=True,
                safe_abstention_message=(
                    "Available evidence is insufficient to determine the current danger level. "
                    "Do not assume the area is safe. Rely on official local emergency broadcasts."
                ),
            )

        # Categorize packets by source and temporal tier
        live_telemetry = [
            e for e in evidence_list
            if e.source_type in (SourceType.TELEMETRY, SourceType.WEATHER_FORECAST)
            and e.temporal_tier == TemporalTier.LIVE
            and e.freshness != FreshnessStatus.EXPIRED
            and e.is_valid
        ]

        historical_reports = [
            e for e in evidence_list
            if e.temporal_tier == TemporalTier.HISTORICAL
            or e.freshness == FreshnessStatus.EXPIRED
        ]

        social_reports = [
            e for e in evidence_list
            if e.source_type == SourceType.SOCIAL_MEDIA
            and e.temporal_tier in (TemporalTier.LIVE, TemporalTier.RECENT)
        ]

        official_doctrine = [
            e for e in evidence_list
            if e.source_type == SourceType.OFFICIAL_PROTOCOL
        ]

        supporting = []
        conflicting = []
        unverified = []
        discrepancies = []

        # -------------------------------------------------------------------
        # Rule 1: Live Telemetry vs Historical Panic (Temporal Conflict)
        # -------------------------------------------------------------------
        # Check if live telemetry reports safe/normal, but historical reports describe flooding
        telemetry_normal = False
        telemetry_danger = False

        for tel in live_telemetry:
            if tel.metric_name == "gage_height_ft" and tel.metric_value is not None:
                flood_stage = tel.metadata.get("flood_stage_ft", 10.0)
                if tel.metric_value < flood_stage:
                    telemetry_normal = True
                    supporting.append(tel)
                else:
                    telemetry_danger = True
                    supporting.append(tel)
            elif "normal" in tel.raw_text.lower():
                telemetry_normal = True
                supporting.append(tel)

        has_historical_flood_claims = any(
            any(w in h.raw_text.lower() for w in ("flood", "evacuat", "underwater", "submerged", "overflow"))
            for h in historical_reports
        )

        has_social_flood_claims = any(
            any(w in s.raw_text.lower() for w in ("flood", "water rising", "stranded", "trapped"))
            for s in social_reports
        )

        non_comparable_conflict = False

        # Case A: Live is Normal, but Historical claims Flooding
        if telemetry_normal and has_historical_flood_claims:
            for h in historical_reports:
                if any(_comparable(t, h) for t in live_telemetry):
                    conflicting.append(h)
                else:
                    unverified.append(h)
                    non_comparable_conflict = True
            discrepancies.append(
                "Temporal Conflict Detected: disaster records describe severe flooding while live "
                "telemetry is below flood threshold; only temporally and spatially compatible "
                "reports are treated as contradictory."
            )

        # Case B: Live is Normal, but Social Media reports water entering homes
        if telemetry_normal and has_social_flood_claims:
            for s in social_reports:
                if any(_comparable(t, s) for t in live_telemetry):
                    conflicting.append(s)
                else:
                    unverified.append(s)
                    non_comparable_conflict = True
            discrepancies.append(
                "Spatial/Hydrological Discrepancy: river gauge reports normal stage while localized "
                "reports claim surface flooding; without compatible location/time/measurement the "
                "claim remains unverified rather than contradicted."
            )

        # -------------------------------------------------------------------
        # Synthesis and State Resolution
        # -------------------------------------------------------------------
        if not live_telemetry and not official_doctrine and social_reports:
            # Only unverified social media exists, no sensors or doctrine
            return ConflictResolutionResult(
                overall_state=EvidenceState.UNVERIFIED,
                primary_status="WARNING",
                resolution_summary=(
                    "Situation unverified: Field reports indicate hazard, but no live telemetry "
                    "or official sensor confirmation is currently available."
                ),
                can_issue_evacuation=False,
                should_abstain=True,
                safe_abstention_message=(
                    "Available evidence is insufficient to verify field conditions. "
                    "Unconfirmed eyewitness reports suggest rising water, but official sensor "
                    "confirmation is pending. Proceed with caution and contact emergency dispatch."
                ),
                unverified_reports=social_reports,
                detailed_discrepancies=["Lack of official live sensor validation for social media reports."],
            )

        if non_comparable_conflict:
            # Non-comparable evidence cannot be used to suppress a warning.
            return ConflictResolutionResult(
                overall_state=EvidenceState.UNVERIFIED,
                primary_status="WARNING",
                resolution_summary=(
                    "Live telemetry and field reports are not comparable in incident, time, "
                    "region or measurement; the warning is preserved pending verification."
                ),
                can_issue_evacuation=False,
                should_abstain=True,
                safe_abstention_message=(
                    "Evidence is insufficient to rule out a local hazard. Do not assume the area "
                    "is safe on the basis of a distant or non-contemporaneous gauge reading. "
                    "Follow official local emergency broadcasts."
                ),
                conflicting_evidence=conflicting,
                unverified_reports=unverified,
                detailed_discrepancies=discrepancies,
            )

        if discrepancies:
            # We have a comparable conflict, and live telemetry resolves it
            if telemetry_normal and not telemetry_danger:
                return ConflictResolutionResult(
                    overall_state=EvidenceState.CONTRADICTED,
                    primary_status="NORMAL",
                    resolution_summary=(
                        "Evidence conflict successfully reconciled: Live hydrological telemetry confirms "
                        "water stage is within normal limits. Past historical flood records are distinguished "
                        "and do not apply to current conditions."
                    ),
                    can_issue_evacuation=False,
                    should_abstain=False,
                    supporting_evidence=supporting,
                    conflicting_evidence=conflicting,
                    unverified_reports=unverified,
                    detailed_discrepancies=discrepancies,
                )
            elif telemetry_danger:
                return ConflictResolutionResult(
                    overall_state=EvidenceState.SUPPORTED,
                    primary_status="CRITICAL",
                    resolution_summary="Live telemetry confirms critical flood stage exceeded. Immediate action authorized.",
                    can_issue_evacuation=True,
                    should_abstain=False,
                    supporting_evidence=supporting,
                    detailed_discrepancies=discrepancies,
                )

        # Default clean state
        status = "NORMAL"
        if telemetry_danger:
            status = "CRITICAL"

        return ConflictResolutionResult(
            overall_state=EvidenceState.SUPPORTED,
            primary_status=status,
            resolution_summary="All available evidence streams are congruent and temporally aligned.",
            can_issue_evacuation=telemetry_danger,
            should_abstain=False,
            supporting_evidence=supporting + official_doctrine,
        )


conflict_resolver = ConflictResolver()
