"""Verify that alert dispatch fails closed and cannot bypass authorization.

Run: python evaluation/test_alert_authorization.py
"""
from __future__ import annotations

import builtins
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.alert_tool as at

MSG = ("Gage height 12.4 ft exceeds 10 ft flood stage. "
       "Mandatory evacuation for low-lying Zone A. USGS NWIS live reading.")


def run_case(name: str, block_import: bool, confidence: str) -> tuple[bool, str, int]:
    with tempfile.TemporaryDirectory() as td:
        at.ALERTS_FILE = str(Path(td) / "alerts.jsonl")
        at.LEGACY_MIRROR = str(Path(td) / "legacy.jsonl")
        real_import = builtins.__import__

        def guarded(import_name, *args, **kwargs):
            if block_import and import_name == "core.policy_engine":
                raise ImportError("simulated policy engine outage")
            return real_import(import_name, *args, **kwargs)

        try:
            builtins.__import__ = guarded
            result = at.generate_structured_alert.func(
                severity="CRITICAL", message=MSG, zone="Zone A",
                confidence=confidence, state=None)
        finally:
            builtins.__import__ = real_import

        written = sum(
            1 for p in (at.ALERTS_FILE, at.LEGACY_MIRROR)
            if os.path.exists(p) and os.path.getsize(p) > 0
        )
        blocked = "BLOCKED" in result
        print(f"{name:28s} blocked={blocked} side_effect_files={written}")
        return blocked, result, written


def main() -> int:
    ok = True

    b1, r1, w1 = run_case("policy engine unavailable", True, "HIGH")
    if not (b1 and w1 == 0):
        ok = False
        print("  FAIL: outage must block dispatch with no side effects")

    b2, r2, w2 = run_case("low confidence", False, "LOW")
    if not (b2 and w2 == 0):
        ok = False
        print("  FAIL: low confidence must block dispatch")

    b3, r3, w3 = run_case("authorized high confidence", False, "HIGH")
    if b3 and w3 > 0:
        print("  NOTE: policy also blocked the grounded case; see policy rationale:")
        print("   ", r3.replace("\n", " ")[:220])

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
