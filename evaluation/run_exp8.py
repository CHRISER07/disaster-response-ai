"""EXP-8: evaluate a revised safety prompt on an INDEPENDENT conflict set.

Three configurations (new system version, original kept as baseline):
  Original          - production prompt, no consistency validation
  RevisedPrompt     - preregistered prompt separating current-hazard
                      classification from precautionary advice
  Revised+Validator - revised prompt plus deterministic decision/reason
                      consistency enforcement

Only the prompt/validation changes; evidence fixture, generator, decoding and
the structured decision schema are identical. Production files untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUTF8", "1")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:3b")

from evaluation.run_dev_eval_v2 import run_full_agent
from evaluation.run_tgi_v5 import call_with_backoff, _sensor_index
from core.temporal_grounding import temporal_tag, parse_timestamp
import tools as tools_pkg
import agents.disaster_agent as da
from langchain_core.tools import StructuredTool

SCENARIOS = ROOT / "evaluation" / "exp8_scenarios.json"
OUT = ROOT / "evaluation" / "exp8_results.csv"
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
SEEDS = [42, 7, 99]
FIELDS = ["scenario_id", "config", "seed", "age_code", "condition", "site",
          "evidence_order", "prompt_hash", "status", "consumption_ok",
          "parse_ok", "decision", "decision_before_validation", "reason",
          "selected_evidence_ids", "expected_label", "correct", "stuck_on_stale",
          "consistent", "raw_answer"]

CONFIGS = ["Original", "RevisedPrompt", "RevisedValidator"]

DECISION_INSTR = (
    '\n\nReply with ONLY a JSON object:\n'
    '{"decision": "CURRENT_HAZARD" | "NO_CURRENT_HAZARD" | "ABSTAIN", '
    '"selected_evidence_ids": ["E001" or "E002"], '
    '"reason": "<short evidence-based reason>"}\n'
    "Classify the CURRENT condition from the most recent valid observation. Do not call tools."
)

# Preregistered prompt revision: inserted after the Temporal Context Bridge.
REVISION_BLOCK = """

## Current-Hazard Classification (v2, preregistered)
You must separate two distinct outputs:
(1) CURRENT-HAZARD CLASSIFICATION: determined ONLY by the most recent valid observation compared with the warning threshold.
(2) PRECAUTIONARY ADVICE: optional caution or monitoring language, which is NOT a hazard classification.

Rules:
- If the most recent observation is below the threshold, the classification is NO_CURRENT_HAZARD, even if an older record exceeded the threshold.
- An older above-threshold reading may be cited ONLY as historical context; it must not change the current classification.
- You may give precautionary advice without classifying the current condition as hazardous.
- The JSON decision field MUST match the classification above and MUST agree with your reason. If they disagree, correct the decision field before replying.
"""

_ORIGINAL_BUILD = None
_ACTIVE_MODE = "original"


def _patched_build(no_tgi=False):
    base = _ORIGINAL_BUILD(no_tgi=no_tgi)
    if _ACTIVE_MODE == "original":
        return base
    # Insert revision immediately before the Grounding Examples section.
    marker = "## Grounding Examples"
    assert marker in base
    return base.replace(marker, REVISION_BLOCK + "\n" + marker, 1)


def _install_prompt_patch():
    global _ORIGINAL_BUILD
    if _ORIGINAL_BUILD is None:
        _ORIGINAL_BUILD = da._build_system_prompt
        da._build_system_prompt = _patched_build
        # run_dev_eval_v2 calls _build_system_prompt via get_agent in the module.
        import evaluation.run_dev_eval_v2 as rdev
        rdev._build_system_prompt = _patched_build  # if referenced there


def _context_capture_patch():
    """Expose final messages through run_full_agent (same method as EXP-7D)."""
    src = inspect.getsource(da.run_agent)
    old = 'return {\n        "answer": final_answer,'
    if old in src and not getattr(da.run_agent, "_ctx2", False):
        src2 = src.replace(old, 'return {\n        "final_messages": messages,\n        "answer": final_answer,', 1)
        ns = dict(vars(da)); exec(compile(src2, "<ra2>", "exec"), ns)
        fn = ns["run_agent"]; fn._ctx2 = True
        da.run_agent = fn
        import evaluation.run_dev_eval_v2 as rdev
        rdev.run_agent = fn
    import evaluation.run_dev_eval_v2 as rdev2
    rs = inspect.getsource(rdev2.run_full_agent)
    anchor = '        answer = ""\n        status = STATUS_OK'
    if anchor in rs and not getattr(rdev2.run_full_agent, "_rf2", False):
        rs2 = rs.replace(anchor,
            '        final_messages = result.get("final_messages", []) if isinstance(result, dict) else []\n'
            '        answer = ""\n        status = STATUS_OK', 1)
        rs2 = rs2.replace('return {\n            "status"',
                          'return {\n            "final_messages": final_messages,\n            "status"')
        ns2 = dict(vars(rdev2)); exec(compile(rs2, "<rf2>", "exec"), ns2)
        fn2 = ns2["run_full_agent"]; fn2._rf2 = True
        rdev2.run_full_agent = fn2
        globals()["run_full_agent"] = fn2


def install_only_sensor(text):
    old_all = list(tools_pkg.ALL_TOOLS); old_da = list(da.ALL_TOOLS)
    idx = _sensor_index(); original = tools_pkg.ALL_TOOLS[idx]
    def pinned_call(location: str = "") -> str: return text
    pinned = StructuredTool.from_function(func=pinned_call, name="query_water_sensor",
                                          description="Frozen sensor record.")
    tools_pkg.ALL_TOOLS[:] = [pinned]
    if da.ALL_TOOLS is not tools_pkg.ALL_TOOLS: da.ALL_TOOLS[:] = [pinned]
    da.get_agent.cache_clear()
    return old_all, old_da, original


def restore_tools(old_all, old_da, original):
    tools_pkg.ALL_TOOLS[:] = old_all
    if da.ALL_TOOLS is not tools_pkg.ALL_TOOLS: da.ALL_TOOLS[:] = old_da
    da.get_agent.cache_clear()


def render(item, rec, show_tier):
    tag = "[RECORD]"
    ts = f" | observed_at: {rec['observed_at']}" if show_tier else ""
    return (f"Evidence {rec['id']}:\n  record_type: {tag}\n"
            f"  station_id: {item['site_id']} ({item['site_name']})\n"
            f"  parameter: {item['parameter']}\n"
            f"  value: {rec['value']:.2f} {item['unit']}\n"
            f"  warning_threshold: {item['threshold_ft']:.1f} {item['unit']}{ts}")


def build_prefetch(item, show_tier=True):
    f, s = render(item, item["fresh"], show_tier), render(item, item["stale"], show_tier)
    return "\n\n".join([s, f] if item["evidence_order"] == "stale_first" else [f, s])


def parse_decision(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return None, None, ""
    try: obj = json.loads(m.group(0))
    except Exception: return None, None, ""
    dec, ids = obj.get("decision"), obj.get("selected_evidence_ids")
    reason = str(obj.get("reason", ""))
    if dec not in ("CURRENT_HAZARD", "NO_CURRENT_HAZARD", "ABSTAIN"): dec = None
    if not isinstance(ids, list): ids = [] if ids is None else [str(ids)]
    return dec, ids, reason


def reason_says_no_hazard(reason):
    r = reason.lower()
    return any(p in r for p in ("no current", "no flood", "below", "no hazard",
                                "not above", "safe", "below the", "below flood"))


def deterministic_validate(dec, reason):
    """Preregistered consistency rule: decision must match the reason."""
    if dec is None: return dec
    if dec == "CURRENT_HAZARD" and reason_says_no_hazard(reason):
        return "NO_CURRENT_HAZARD"
    if dec == "NO_CURRENT_HAZARD" and any(p in reason.lower() for p in ("above", "exceeds", "hazard exists")):
        return "CURRENT_HAZARD"
    return dec


def main():
    global OUT, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out)
        if not OUT.is_absolute(): OUT = ROOT / OUT
    if args.seeds: SEEDS = [int(s) for s in args.seeds.split(",") if s.strip()]

    _install_prompt_patch()
    _context_capture_patch()

    items = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    if args.limit: items = items[:args.limit]
    done = set()
    if OUT.exists():
        for r in csv.DictReader(open(OUT, encoding="utf-8")):
            if r.get("status") == "OK": done.add((r["config"], str(r["seed"]), r["scenario_id"]))
    header = not OUT.exists() or OUT.stat().st_size == 0
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if header: w.writeheader()
        total = len(CONFIGS) * len(SEEDS) * len(items)
        n = 0
        for config in CONFIGS:
            global _ACTIVE_MODE
            _ACTIVE_MODE = "original" if config == "Original" else "revised"
            ph = hashlib.sha256((_ACTIVE_MODE + MODEL).encode()).hexdigest()[:12]
            for seed in SEEDS:
                for item in items:
                    if (config, str(seed), item["query_id"]) in done: continue
                    if args.max_new is not None and n >= args.max_new: f.flush(); return
                    n += 1
                    pf = build_prefetch(item, show_tier=True)
                    orig = install_only_sensor(pf)
                    try:
                        res = call_with_backoff(
                            lambda q, s, pref=None: run_full_agent(q, s, no_tgi=False,
                                                                   prefetched_observation=pref),
                            item["query"] + DECISION_INSTR, seed, pf)
                    finally: restore_tools(*orig)
                    answer = str(res.get("full_answer", ""))
                    status = str(res.get("status", "OTHER_ERROR"))
                    msgs = res.get("final_messages", [])
                    ctx = "\n".join(str(getattr(m, "content", m)) for m in msgs)
                    consume = int(all(e in ctx for e in ("E001", "E002")))
                    dec, ids, reason = parse_decision(answer)
                    parse_ok = int(dec is not None)
                    dec_before = dec
                    if config == "RevisedValidator" and dec is not None:
                        dec = deterministic_validate(dec, reason)
                    assess = status == "OK" and consume and parse_ok
                    if assess:
                        want = item["expected_label"]
                        correct = int((dec == "CURRENT_HAZARD" and want == "CURRENT_HAZARD") or
                                      (dec == "NO_CURRENT_HAZARD" and want == "NO_CURRENT_HAZARD"))
                        stale_danger = item["stale"]["value"] >= 1.3 * item["threshold_ft"]
                        stuck = int((stale_danger and dec == "CURRENT_HAZARD") or
                                    ((not stale_danger) and dec == "NO_CURRENT_HAZARD"))
                        consistent = int(dec == dec_before)
                    else: correct = stuck = consistent = 0
                    w.writerow({"scenario_id": item["query_id"], "config": config, "seed": seed,
                        "age_code": item["age_code"], "condition": item["condition"],
                        "site": item["site_name"], "evidence_order": item["evidence_order"],
                        "prompt_hash": ph, "status": status, "consumption_ok": consume,
                        "parse_ok": parse_ok, "decision": dec or "",
                        "decision_before_validation": dec_before or "", "reason": reason,
                        "selected_evidence_ids": "|".join(ids), "expected_label": item["expected_label"],
                        "correct": correct, "stuck_on_stale": stuck, "consistent": consistent,
                        "raw_answer": answer})
                    f.flush()
                    if n % 30 == 0 or n == total:
                        print(f"[{n}/{total}] {config} {item['query_id']} {dec or status}", flush=True)
                    time.sleep(0.05)
    print("Done", OUT)


if __name__ == "__main__":
    main()
