"""EXP-7D: controlled temporal freshness evaluation.

Three arms, identical unified evidence fixture (same schema/station/variable/
unit/source/threshold, no severity text), differing only in temporal signal:
  TGI           : full system prompt + production temporal_tag on each record; raw timestamps shown
  TimestampOnly : no_tgi system prompt; raw timestamps shown; tags hidden
  NoTGI         : no_tgi system prompt; timestamps and tags hidden

Mandatory consumption assertion: both evidence IDs must appear in the actual
contexts returned by run_agent, or the row is marked non-assessable.
Uncontrolled live tools are disabled; only the pinned sensor tool is exposed.
Decision is parsed from a constrained JSON response.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

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
import agents.disaster_agent as _da_module
import inspect as _inspect


def _enable_context_capture():
    """Expose run_agent's final message state for the consumption assertion.

    Modifies only this process's in-memory function; production files untouched.
    """
    if getattr(_da_module.run_agent, "_ctx_capture", False):
        return
    src = _inspect.getsource(_da_module.run_agent)
    old = 'return {\n        "answer": final_answer,'
    assert old in src
    src = src.replace(old,
                      'return {\n        "final_messages": messages,\n        "answer": final_answer,', 1)
    ns = dict(vars(_da_module))
    exec(compile(src, "<run_agent_ctx_capture>", "exec"), ns)
    fn = ns["run_agent"]
    fn._ctx_capture = True
    _da_module.run_agent = fn
    import evaluation.run_dev_eval_v2 as _rdev
    _rdev.run_agent = fn

    # run_full_agent builds its own result dict and drops final_messages.
    # Patch it to propagate final_messages from the inner run_agent result.
    import evaluation.run_dev_eval_v2 as _rdev2
    rf_src = _inspect.getsource(_rdev2.run_full_agent)
    anchor = '        answer = ""\n        status = STATUS_OK'
    assert anchor in rf_src
    rf_src = rf_src.replace(anchor,
        '        final_messages = result.get("final_messages", []) if isinstance(result, dict) else []\n'
        '        answer = ""\n        status = STATUS_OK', 1)
    # add to every "return {" in run_full_agent
    rf_src = rf_src.replace('return {\n            "status"',
                            'return {\n            "final_messages": final_messages,\n            "status"')
    rf_ns = dict(vars(_rdev2))
    exec(compile(rf_src, "<run_full_ctx_capture>", "exec"), rf_ns)
    rf_fn = rf_ns["run_full_agent"]
    rf_fn._ctx_capture = True
    _rdev2.run_full_agent = rf_fn
    # Rebind in this module's namespace (the local name imported at top).
    globals()["run_full_agent"] = rf_fn


def _context_text(res):
    msgs = res.get("final_messages", []) if isinstance(res, dict) else []
    return "\n".join(str(getattr(m, "content", m)) for m in msgs)

_enable_context_capture()
import tools as tools_pkg
import agents.disaster_agent as da
from langchain_core.tools import StructuredTool

SCENARIOS = ROOT / "evaluation" / "exp7d_scenarios.json"
OUT = ROOT / "evaluation" / "exp7d_results.csv"
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
SEEDS = [42, 7, 99]
FIELDS = ["experiment_id", "scenario_id", "arm", "seed", "group", "condition",
          "age_code", "site", "model_digest", "prompt_hash", "evidence_ids",
          "evidence_order", "temporal_tags", "context_hash", "status",
          "consumption_ok", "parse_ok", "decision", "selected_evidence_ids",
          "expected_label", "correct", "stuck_on_stale", "latency_s", "raw_answer"]

ARMS = [
    ("TGI", "tgi"),
    ("TimestampOnly", "timestamp"),
    ("NoTGI", "notgi"),
]
EXPERIMENT_ID = "EXP-7D-v1"

DECISION_INSTR = (
    "\n\nOUTPUT FORMAT (mandatory): reply with ONLY a JSON object, no other text:\n"
    '{"decision": "CURRENT_HAZARD" | "NO_CURRENT_HAZARD" | "ABSTAIN", '
    '"selected_evidence_ids": ["E001" or "E002"], '
    '"confidence": "HIGH"|"MEDIUM"|"LOW", "reason": "<short evidence-based reason>"}\n'
    "Choose the record that represents the CURRENT condition. Do not call tools; "
    "use only the evidence already supplied."
)


def model_digest():
    import subprocess
    try:
        out = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.startswith(MODEL):
                parts = line.split()
                if len(parts) >= 3:
                    return parts[2][:16]
    except Exception:
        pass
    return hashlib.sha256(MODEL.encode()).hexdigest()[:16]


def prompt_hash(arm):
    no_tgi = arm != "TGI"
    sp = da._build_system_prompt(no_tgi=no_tgi) + DECISION_INSTR
    return hashlib.sha256(sp.encode()).hexdigest()[:16]


def render_record(item, rec, arm):
    ref = parse_timestamp(item["reference_now"])
    if arm == "TGI":
        tag = temporal_tag({"timestamp": rec["observed_at"],
                            "source": "sensor archive", "modality": "text"}, now=ref)
        ts = f" | observed_at: {rec['observed_at']}"
    elif arm == "TimestampOnly":
        tag = "[OBSERVATION]"
        ts = f" | observed_at: {rec['observed_at']}"
    else:  # NoTGI
        tag = "[OBSERVATION]"
        ts = ""
    return (f"Evidence {rec['id']}:\n"
            f"  record_type: {tag}\n"
            f"  station_id: {item['site_id']} ({item['site_name']})\n"
            f"  parameter: {item['parameter']}\n"
            f"  value: {rec['value']:.2f} {item['unit']}\n"
            f"  warning_threshold: {item['threshold_ft']:.1f} {item['unit']}{ts}")


def build_prefetch(item, arm):
    fresh, stale = item["fresh"], item["stale"]
    parts = [render_record(item, fresh, arm), render_record(item, stale, arm)]
    if item["evidence_order"] == "stale_first":
        parts = [render_record(item, stale, arm), render_record(item, fresh, arm)]
    return "\n\n".join(parts)


def install_only_sensor(text):
    """Expose exactly one tool: the frozen sensor returning the fixture."""
    old_all = list(tools_pkg.ALL_TOOLS)
    old_da = list(da.ALL_TOOLS)
    idx = _sensor_index()
    original = tools_pkg.ALL_TOOLS[idx]
    def pinned_call(location: str = "") -> str:
        return text
    pinned = StructuredTool.from_function(func=pinned_call, name="query_water_sensor",
                                          description="Frozen sensor observation record.")
    tools_pkg.ALL_TOOLS[:] = [pinned]
    if da.ALL_TOOLS is not tools_pkg.ALL_TOOLS:
        da.ALL_TOOLS[:] = [pinned]
    da.get_agent.cache_clear()
    return old_all, old_da, original


def restore_tools(old_all, old_da, original):
    tools_pkg.ALL_TOOLS[:] = old_all
    if da.ALL_TOOLS is not tools_pkg.ALL_TOOLS:
        da.ALL_TOOLS[:] = old_da
    da.get_agent.cache_clear()


def parse_decision(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, None
    dec = obj.get("decision")
    ids = obj.get("selected_evidence_ids")
    if dec not in ("CURRENT_HAZARD", "NO_CURRENT_HAZARD", "ABSTAIN"):
        return None, None
    if not isinstance(ids, list):
        ids = [] if ids is None else [str(ids)]
    return dec, ids


def summarize(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print("no rows"); return
    print("=== PRIMARY ===")
    for arm, _ in ARMS:
        s = [r for r in rows if r["arm"] == arm and r["group"] == "primary"
             and r["consumption_ok"] == "1" and r["parse_ok"] == "1"]
        if not s: continue
        c = sum(int(r["correct"]) for r in s)
        st = sum(int(r["stuck_on_stale"]) for r in s)
        ab = sum(r["decision"] == "ABSTAIN" for r in s)
        print(f"{arm:14s} n={len(s):3d} acc={c/len(s):.3f} stuck={st} abstain={ab}")
    print("=== DIAGNOSTIC (separate) ===")
    for arm, _ in ARMS:
        s = [r for r in rows if r["arm"] == arm and r["group"] == "diagnostic"
             and r["consumption_ok"] == "1" and r["parse_ok"] == "1"]
        if s:
            c = sum(int(r["correct"]) for r in s)
            print(f"{arm:14s} n={len(s):3d} acc={c/len(s):.3f}")
    bad = Counter((r["status"], r["consumption_ok"], r["parse_ok"]) for r in rows
                  if r["consumption_ok"] != "1" or r["parse_ok"] != "1")
    print("rejected rows (status,consumption,parse):", dict(bad))


def main():
    global OUT, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out)
        if not OUT.is_absolute(): OUT = ROOT / OUT
    if args.seeds:
        SEEDS = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.summary:
        summarize(OUT); return

    items = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    if args.limit: items = items[:args.limit]
    done = set()
    if OUT.exists():
        for r in csv.DictReader(open(OUT, encoding="utf-8")):
            if r.get("status") == "OK":
                done.add((r["arm"], str(r["seed"]), r["scenario_id"]))
    header = not OUT.exists() or OUT.stat().st_size == 0
    md = model_digest()
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if header: w.writeheader()
        total = len(ARMS) * len(SEEDS) * len(items)
        n = 0
        for arm, mode in ARMS:
            no_tgi = arm != "TGI"
            ph = prompt_hash(arm)
            for seed in SEEDS:
                for item in items:
                    if (arm, str(seed), item["query_id"]) in done: continue
                    if args.max_new is not None and n >= args.max_new:
                        f.flush(); return
                    n += 1
                    prefetch = build_prefetch(item, arm)
                    query = item.get("query") or (
                        f"Two sensor records for {item['site_name']} are supplied. "
                        f"Based only on these records, is there a current flood hazard requiring evacuation?")
                    full_query = query + DECISION_INSTR
                    old_all, old_da, orig = install_only_sensor(prefetch)
                    t0 = time.perf_counter()
                    try:
                        res = call_with_backoff(
                            lambda q, s, pref=None: run_full_agent(q, s, no_tgi=no_tgi,
                                                                   prefetched_observation=pref),
                            full_query, seed, prefetch)
                    finally:
                        restore_tools(old_all, old_da, orig)
                    lat = round(time.perf_counter() - t0, 3)
                    answer = str(res.get("full_answer", ""))
                    status = str(res.get("status", "OTHER_ERROR"))
                    contexts = _context_text(res)
                    # MANDATORY consumption assertion on the ACTUAL final model context.
                    consumption = int(all(eid in contexts for eid in ("E001", "E002")))
                    dec, ids = parse_decision(answer)
                    parse_ok = int(dec is not None)
                    assessable = status == "OK" and consumption and parse_ok
                    if assessable:
                        want = item["expected_label"]
                        correct = int((dec == "CURRENT_HAZARD" and want == "CURRENT_HAZARD") or
                                      (dec == "NO_CURRENT_HAZARD" and want == "NO_CURRENT_HAZARD"))
                        stale_says_danger = item["stale"]["value"] >= 1.3 * item["threshold_ft"]
                        stuck = int((stale_says_danger and dec == "CURRENT_HAZARD") or
                                    ((not stale_says_danger) and dec == "NO_CURRENT_HAZARD"))
                    else:
                        correct = stuck = 0
                    w.writerow({
                        "experiment_id": EXPERIMENT_ID, "scenario_id": item["query_id"],
                        "arm": arm, "seed": seed, "group": item["group"],
                        "condition": item["condition"], "age_code": item["age_code"],
                        "site": item["site_name"], "model_digest": md, "prompt_hash": ph,
                        "evidence_ids": "E001|E002", "evidence_order": item["evidence_order"],
                        "temporal_tags": "tgi" if arm == "TGI" else ("timestamp" if arm == "TimestampOnly" else "none"),
                        "context_hash": hashlib.sha256(contexts.encode()).hexdigest()[:12],
                        "status": status, "consumption_ok": consumption, "parse_ok": parse_ok,
                        "decision": dec or "", "selected_evidence_ids": "|".join(ids),
                        "expected_label": item["expected_label"], "correct": correct,
                        "stuck_on_stale": stuck, "latency_s": lat, "raw_answer": answer,
                    })
                    f.flush()
                    if n % 30 == 0 or n == total:
                        print(f"[{n}/{total}] {arm} seed={seed} {item['query_id']} "
                              f"consume={consumption} parse={parse_ok} dec={dec or status}", flush=True)
                    time.sleep(0.1)
    print("Done:", OUT)
    summarize(OUT)


if __name__ == "__main__":
    main()
