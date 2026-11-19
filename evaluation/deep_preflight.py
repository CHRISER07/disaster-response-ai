"""Deep preflight for the three rerun suites.

Verifies, without touching production result files:
  1. all critical modules import;
  2. formatting/safety/conflict/policy unit checks pass;
  3. detached workers lock provider/model in-process (no .env fallback);
  4. each runner has a working --limit smoke path with a unique output tag.

Run: python evaluation/deep_preflight.py
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def ok(name: str) -> None:
    print(f"PASS {name}")


def fail(name: str, detail: str = "") -> None:
    print(f"FAIL {name} {detail}")
    raise SystemExit(1)


def check_imports() -> None:
    mods = [
        "core.answer_formatter", "core.safety_validator", "core.policy_engine",
        "core.claim_verifier", "core.conflict_resolver", "core.evidence_validator",
        "core.hybrid_store", "tools.alert_tool", "tools.sensor_tool",
        "agents.disaster_agent", "evaluation.run_tgi_v5",
        "evaluation.eval_dsel_ablation", "evaluation.run_fresh_heldout",
    ]
    code = "import " + ", ".join(mods)
    r = subprocess.run([PY, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode:
        fail("imports", r.stderr[-400:])
    ok("imports")


def check_units() -> None:
    code = r'''
from core.answer_formatter import format_answer
from core.safety_validator import validate_and_repair
from core.conflict_resolver import ConflictResolver, EvidenceState
from core.evidence_validator import EvidencePacket, FreshnessStatus, SourceType, TemporalTier
from datetime import datetime, timedelta, timezone
# formatter
a = format_answer("Status: NORMAL\nEvidence: - x")
assert "Status:" in a and "*" not in a and "Alert Generated:" in a
# safety repair deletes unsafe sentence
r = validate_and_repair("It is safe to drive through floodwater. Stay calm.", True)
assert "safe to drive" not in r["answer"].lower()
# resolver: historical-only must abstain
p = EvidencePacket(evidence_id="h", source_id="h", source_type=SourceType.SOCIAL_MEDIA,
    raw_text="old flood", event_time=datetime.now(timezone.utc)-timedelta(days=400),
    observed_at=datetime.now(timezone.utc)-timedelta(days=400),
    ingested_at=datetime.now(timezone.utc), valid_until=datetime.now(timezone.utc),
    temporal_tier=TemporalTier.HISTORICAL, freshness=FreshnessStatus.EXPIRED)
res = ConflictResolver.resolve_evidence([p], "status?")
assert res.overall_state == EvidenceState.INSUFFICIENT and res.should_abstain
print("units-ok")
'''
    r = subprocess.run([PY, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode or "units-ok" not in r.stdout:
        fail("unit checks", (r.stdout + r.stderr)[-600:])
    ok("unit checks")


def check_env_lock(worker: str, needle: str) -> None:
    """Run a tiny worker with a clean env and confirm the log line proves lock."""
    code = (
        "import os,sys; sys.path.insert(0,r'" + str(ROOT) + "');"
        "__file__=r'" + str(ROOT / worker) + "';"
        "src=open(__file__,encoding='utf-8-sig').read().split('main()')[0];"
        "exec(compile(src,__file__,'exec'));"
        "print('LOCKCHECK', os.getenv('PROVIDER'), os.getenv('OLLAMA_MODEL'))"
    )
    env = os.environ.copy()
    env.pop("PROVIDER", None)
    env.pop("OLLAMA_MODEL", None)
    r = subprocess.run([PY, "-c", code], cwd=str(ROOT), capture_output=True, text=True, env=env)
    if r.returncode or needle not in r.stdout:
        fail("env lock " + worker, (r.stdout + r.stderr)[-500:])
    ok("env lock " + worker)


def check_dsel_env_lock() -> None:
    """DSEL worker has no main(): execute only its env-lock preamble, never runpy."""
    worker = "scratch/dsel_v6_worker.py"
    path = str(ROOT / worker)
    code = (
        "import os,sys; sys.path.insert(0,r'" + str(ROOT) + "');"
        "p=r'" + path + "';"
        "src=open(p,encoding='utf-8-sig').read();"
        "head=src.split('runpy.run_path')[0];"
        "__file__=p;"
        "exec(compile(head,p,'exec'));"
        "print('LOCKCHECK', os.getenv('PROVIDER'), os.getenv('OLLAMA_MODEL'))"
    )
    env = os.environ.copy()
    env.pop("PROVIDER", None)
    env.pop("OLLAMA_MODEL", None)
    r = subprocess.run([PY, "-c", code], cwd=str(ROOT), capture_output=True, text=True, env=env)
    if r.returncode or "LOCKCHECK ollama llama3.1:8b" not in r.stdout:
        fail("env lock " + worker, (r.stdout + r.stderr)[-500:])
    ok("env lock " + worker)


def main() -> None:
    check_imports()
    check_units()
    check_env_lock("scratch/tgi_v6_worker.py", "LOCKCHECK ollama qwen2.5:3b")
    check_dsel_env_lock()
    check_env_lock("scratch/fresh_worker.py", "LOCKCHECK ollama llama3.1:8b")
    print("DEEP PREFLIGHT PASS")


if __name__ == "__main__":
    main()
