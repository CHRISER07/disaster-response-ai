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

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from core.evidence_validator import EvidencePacket, SourceType, TemporalTier, FreshnessStatus
from core.geo_router import SpatialRelevance


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

        # Case A: Live is Normal, but Historical claims Flooding
        if telemetry_normal and has_historical_flood_claims:
            conflicting.extend(historical_reports)
            discrepancies.append(
                "Temporal Conflict Detected: Historical disaster records describe severe flooding, "
                "but live real-time sensors confirm river stage is below flood threshold."
            )

        # Case B: Live is Normal, but Social Media reports water entering homes
        if telemetry_normal and has_social_flood_claims:
            unverified.extend(social_reports)
            discrepancies.append(
                "Spatial/Hydrological Discrepancy: Mainstem river gauge reports normal stage, "
                "while localized social reports claim surface flooding (possible urban drainage ponding)."
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

        if discrepancies:
            # We have an active conflict, but live telemetry resolves it
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
