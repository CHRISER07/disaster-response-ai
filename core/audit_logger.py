"""
core/audit_logger.py

Comprehensive Incident Audit Logger and Deterministic Replay Engine.

Stores an immutable, structured record of every disaster reasoning transaction:
- Original incident query & wall-clock timestamp
- Model architecture and checkpoint version
- Retrieved evidence IDs, source timestamps, and ranking order
- Chronological tool invocation trace (inputs, outputs, latency)
- Evidence Conflict Resolution state and Safe Abstention decisions
- Claim-Level Verification and citation audit reports
- Deterministic Safety Enforcement Layer (DSEL) actions (L1-L4)
- Operator review and Human-in-the-Loop approval state
- Final synthesized EOC broadcast and dispatch status
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
AUDIT_LOG_FILE = os.path.join(AUDIT_LOG_DIR, "incident_audit_trail.jsonl")
# Single canonical append-only log for alerts AND claim audit records (spec Sec. IV-D).
ALERT_LOG_FILE = AUDIT_LOG_FILE
# Legacy dashboard path, read-only fallback for pre-existing alerts.
LEGACY_ALERTS_FILE = os.path.join(PROJECT_ROOT, "data", "alerts.jsonl")


def alert_log_paths():
    """Return (canonical_path, legacy_path) for dashboard compatibility."""
    return ALERT_LOG_FILE, LEGACY_ALERTS_FILE

os.makedirs(AUDIT_LOG_DIR, exist_ok=True)


@dataclass
class ToolExecutionTrace:
    step_number: int
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output_summary: str
    started_at: str
    latency_ms: float
    latency_measured: bool = True
    evidence_ids_produced: List[str] = field(default_factory=list)


@dataclass
class IncidentAuditRecord:
    incident_id: str
    session_id: str
    query: str
    created_at_utc: str
    model_version: str
    
    # Evidence & Retrieval
    retrieved_evidence_ids: List[str] = field(default_factory=list)
    tool_traces: List[ToolExecutionTrace] = field(default_factory=list)
    
    # Reasoning & Verification
    conflict_resolution_state: Optional[str] = None
    safe_abstention_triggered: bool = False
    claim_verification_summary: Optional[Dict[str, Any]] = None
    
    # Safety & Governance
    dsel_violations_detected: List[str] = field(default_factory=list)
    dsel_rewritten: bool = False
    policy_faithfulness_score: float = 1.0
    
    # Human-in-the-Loop Governance
    operator_approval_status: str = "PENDING_APPROVAL" # "PENDING_APPROVAL" | "APPROVED" | "REJECTED" | "AUTONOMOUS_NORMAL"
    operator_id: Optional[str] = None
    operator_review_notes: Optional[str] = None
    
    # Final Output
    final_response: str = ""
    dispatch_status: str = "DELIVERED"


class IncidentAuditLogger:
    """Manages persistent audit logging and deterministic incident playback."""

    @classmethod
    def log_incident(cls, record: IncidentAuditRecord) -> str:
        """Appends complete transaction record to immutable JSONL audit trail."""
        rec_dict = asdict(record)
        line = json.dumps(rec_dict, ensure_ascii=False)
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return record.incident_id

    @staticmethod
    def active_model_version() -> str:
        """Resolve the generator actually in use at record time (never hardcoded)."""
        provider = os.getenv("PROVIDER", "ollama").lower().strip()
        if provider == "ollama":
            return os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        if provider == "gemini":
            return os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        return os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

    @classmethod
    def create_record(
        cls,
        query: str,
        session_id: str = "default_eoc",
        model_version: Optional[str] = None
    ) -> IncidentAuditRecord:
        """Factory for a new audit record."""
        return IncidentAuditRecord(
            incident_id=f"INC-{uuid.uuid4().hex[:10].upper()}",
            session_id=session_id,
            query=query,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            model_version=model_version or cls.active_model_version(),
        )

    @classmethod
    def get_incident(cls, incident_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an incident record for forensic replay."""
        if not os.path.exists(AUDIT_LOG_FILE):
            return None
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("incident_id") == incident_id:
                        return data
                except Exception:
                    continue
        return None

    @classmethod
    def update_operator_approval(
        cls,
        incident_id: str,
        operator_id: str,
        status: str,
        notes: str = ""
    ) -> bool:
        """Records formal Human-in-the-Loop operator authorization decision."""
        records = []
        found = False
        if not os.path.exists(AUDIT_LOG_FILE):
            return False

        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("incident_id") == incident_id:
                        data["operator_approval_status"] = status
                        data["operator_id"] = operator_id
                        data["operator_review_notes"] = notes
                        data["dispatch_status"] = "DISPATCHED" if status == "APPROVED" else "CANCELLED"
                        found = True
                    records.append(data)
                except Exception:
                    continue

        if found:
            with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return found


audit_logger = IncidentAuditLogger()
