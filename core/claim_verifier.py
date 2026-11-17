"""
core/claim_verifier.py

Claim-Level Verification, Citation Auditing, and Evidence-to-Claim Mapping.

Addresses the critical failure mode where an LLM generates a valid-looking citation
(e.g., [USGS NWIS]) but uses it to support a fabricated or contradictory assertion
(e.g., "The river has exceeded flood stage and evacuation is necessary. [USGS]").

Features:
1. Atomic Factual Claim Extraction.
2. Direct Evidence-to-Claim Mapping using EvidencePacket IDs.
3. Numeric & Polarity Verification (verifies that cited numbers exist in the underlying observation).
4. Citation Hallucination Detection (flags physical sensors cited for evacuation policy).
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from core.evidence_validator import EvidencePacket, TemporalTier, SourceType


class ClaimVerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"               # Numbers, polarity, and facts match evidence
    CONTRADICTED = "CONTRADICTED"         # Claim directly contradicts the cited observation
    UNANCHORED_CITATION = "UNANCHORED"    # Citation attached, but evidence lacks supporting content
    UNSUPPORTED = "UNSUPPORTED"           # Factual assertion made without any retrieved evidence


@dataclass
class VerifiedClaim:
    claim_text: str
    citation_tag: Optional[str] = None
    linked_evidence_id: Optional[str] = None
    verification_status: ClaimVerificationStatus = ClaimVerificationStatus.UNSUPPORTED
    source_timestamp: Optional[str] = None
    discrepancy_note: Optional[str] = None
    extracted_numbers: List[float] = field(default_factory=list)


@dataclass
class VerificationReport:
    total_claims: int
    supported_count: int
    contradicted_count: int
    unsupported_count: int
    faithfulness_ratio: float
    is_safe: bool
    verified_claims: List[VerifiedClaim] = field(default_factory=list)


class ClaimVerifier:
    """Verifies generated response statements against underlying evidence packets."""

    @staticmethod
    def extract_claims(text: str) -> List[Tuple[str, Optional[str]]]:
        """
        Splits response into atomic declarative sentences and extracts attached citations.
        Returns list of (claim_text, citation_string).
        """
        if not text:
            return []

        # Split into lines/sentences
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\-\*])|\n+", text)
        claims = []

        for s in raw_sentences:
            s_clean = s.strip()
            # Skip empty, headers, markdown fences, or purely conversational filler
            if len(s_clean) < 15 or s_clean.startswith("#") or s_clean.startswith("```"):
                continue
            if s_clean.lower().startswith("do you want me to") or s_clean.lower().startswith("let me know"):
                continue

            # Extract bracketed citation e.g. [LIVE - USGS NWIS, 2026-11-14] or [EV-a1b2c3]
            cit_match = re.search(r"(\[(?:LIVE|HISTORICAL|KB|EV-|FIELD|VISION)[^\]]+\])", s_clean)
            citation = cit_match.group(1) if cit_match else None
            
            # Remove citation from claim text for clean lexical inspection
            claim_body = re.sub(r"\[(?:LIVE|HISTORICAL|KB|EV-|FIELD|VISION)[^\]]+\]", "", s_clean).strip()
            claims.append((claim_body, citation))

        return claims

    @staticmethod
    def extract_numbers(s: str) -> List[float]:
        """Extracts numerical quantities from a claim string."""
        matches = re.findall(r"\b\d+(?:\.\d+)?\b", s)
        nums = []
        for m in matches:
            try:
                nums.append(float(m))
            except ValueError:
                pass
        return nums

    @classmethod
    def verify_response(
        cls,
        response_text: str,
        evidence_packets: List[EvidencePacket]
    ) -> VerificationReport:
        """
        Performs end-to-end claim extraction and verification against retrieved evidence packets.
        """
        claims_raw = cls.extract_claims(response_text)
        evidence_by_id = {e.evidence_id: e for e in evidence_packets}
        evidence_by_source = {e.source_id.lower(): e for e in evidence_packets}

        verified_list: List[VerifiedClaim] = []
        supported_cnt = 0
        contradicted_cnt = 0
        unsupported_cnt = 0

        for claim_text, citation in claims_raw:
            nums_in_claim = cls.extract_numbers(claim_text)
            claim_lower = claim_text.lower()
            
            # Find matching evidence packet
            matched_evidence: Optional[EvidencePacket] = None
            if citation:
                # Check for direct EV-ID
                ev_id_match = re.search(r"EV-[a-f0-9]{12}", citation)
                if ev_id_match and ev_id_match.group(0) in evidence_by_id:
                    matched_evidence = evidence_by_id[ev_id_match.group(0)]
                else:
                    # Match by source name in citation
                    for src_key, ep in evidence_by_source.items():
                        if src_key in citation.lower() or ep.source_id.lower() in citation.lower():
                            matched_evidence = ep
                            break

            # If still no direct match, check text overlap
            if not matched_evidence:
                for ep in evidence_packets:
                    # If sensor numbers overlap
                    if ep.metric_value is not None and ep.metric_value in nums_in_claim:
                        matched_evidence = ep
                        break

            if not matched_evidence:
                # Factual claim without matching evidence
                verified_list.append(VerifiedClaim(
                    claim_text=claim_text,
                    citation_tag=citation,
                    verification_status=ClaimVerificationStatus.UNSUPPORTED,
                    discrepancy_note="No retrieved evidence packet matches this assertion.",
                    extracted_numbers=nums_in_claim
                ))
                unsupported_cnt += 1
                continue

            # Verification Logic against matched_evidence
            status = ClaimVerificationStatus.SUPPORTED
            note = "Verified against underlying evidence."

            # Check 1: Evacuation Policy Hallucination on Sensor Citations
            # Sensors (USGS, Open-Meteo) report physical numbers; they NEVER order evacuations
            if matched_evidence.source_type == SourceType.TELEMETRY:
                if "mandatory evacuation" in claim_lower or "evacuation is required" in claim_lower:
                    if "no mandatory evacuation" not in claim_lower and "not required" not in claim_lower:
                        status = ClaimVerificationStatus.UNANCHORED_CITATION
                        note = "Citation Error: Physical sensor stream cited for administrative evacuation mandate."

            # Check 2: Numeric Consistency
            if matched_evidence.metric_value is not None:
                # If claim asserts a specific river height, verify it matches
                if "gage height" in claim_lower or "stage" in claim_lower or "reading" in claim_lower:
                    if nums_in_claim:
                        # Allow 0.1 tolerance for rounding
                        if not any(abs(n - matched_evidence.metric_value) <= 0.1 for n in nums_in_claim):
                            status = ClaimVerificationStatus.CONTRADICTED
                            note = (
                                f"Numeric Contradiction: Claim asserts {nums_in_claim}, "
                                f"but sensor observation is {matched_evidence.metric_value} {matched_evidence.metric_unit or ''}."
                            )

            # Check 3: Polarity / Flood Threshold Consistency
            if "exceeded flood stage" in claim_lower or "above flood stage" in claim_lower:
                flood_stage = matched_evidence.metadata.get("flood_stage_ft", 10.0)
                if matched_evidence.metric_value is not None and matched_evidence.metric_value < flood_stage:
                    status = ClaimVerificationStatus.CONTRADICTED
                    note = (
                        f"Polarity Contradiction: Claim states water exceeded flood stage, "
                        f"but live gage ({matched_evidence.metric_value} ft) is below flood stage ({flood_stage} ft)."
                    )

            if status == ClaimVerificationStatus.SUPPORTED:
                supported_cnt += 1
            elif status == ClaimVerificationStatus.CONTRADICTED:
                contradicted_cnt += 1
            else:
                unsupported_cnt += 1

            verified_list.append(VerifiedClaim(
                claim_text=claim_text,
                citation_tag=citation,
                linked_evidence_id=matched_evidence.evidence_id,
                verification_status=status,
                source_timestamp=matched_evidence.observed_at.isoformat(),
                discrepancy_note=note,
                extracted_numbers=nums_in_claim
            ))

        total = len(verified_list)
        ratio = round(supported_cnt / total, 3) if total > 0 else 1.0
        is_safe = contradicted_cnt == 0

        return VerificationReport(
            total_claims=total,
            supported_count=supported_cnt,
            contradicted_count=contradicted_cnt,
            unsupported_count=unsupported_cnt,
            faithfulness_ratio=ratio,
            is_safe=is_safe,
            verified_claims=verified_list
        )


claim_verifier = ClaimVerifier()
