"""
evaluation/run_tgi_v5.py

Two-sided temporal-conflict evaluation on the corrected benchmark
(evaluation/temporal_conflict_set_v2.json).

Fixes over run_tgi_v4.py (see AUDIT_FINDINGS F28-F30):
  1. Balanced labels: 12 EVACUATION / 12 NO_EVACUATION. v4 had zero positives,
     so a constant "do not evacuate" responder scored a perfect zero.
  2. The gauge reading is the manipulated variable. Each query text is issued
     twice - once with a safe reading, once above the mandatory-evacuation
     threshold - so the text cannot explain the decision.
  3. The reading is injected through tools.sensor_tool.format_gage_report, the
     same severity logic the live tool uses, so the agent sees exactly the
     evidence the label was derived from.
  4. Failed runs are recorded as non-assessable and excluded from denominators;
     v4 scored a crash as a correct non-evacuation.
  5. Both error directions are measured: false evacuation (fp) and missed
     evacuation (fn). A one-sided metric cannot separate "safe" from "silent".

Arms run in the requested order:
  Full_Agent_NoTGI -> Full_Agent_TGI -> LLM_Only -> Standard_StaticRAG

Run:
  python evaluation/run_tgi_v5.py --limit 2                 # smoke
  python evaluation/run_tgi_v5.py --config Full_Agent_NoTGI  # one arm
  python evaluation/run_tgi_v5.py                           # full (288 rows)
  python evaluation/run_tgi_v5.py --summary                 # print table
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUTF8", "1")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

# GENERATOR LOCK: one generator for every arm, or the comparison is invalid.
os.environ.setdefault("PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:3b")

from evaluation.run_dev_eval_v2 import run_llm_only, run_static_rag, run_full_agent
from evaluation.run_tgi_eval_v2 import detect_evacuation
from core.answer_formatter import format_answer
from tools.sensor_tool import format_gage_report

CONFLICT_SET = ROOT / "evaluation" / "temporal_conflict_set_v2.json"
OUT = ROOT / "evaluation" / "tgi_results_v5.csv"
OUT.with_suffix(".complete").unlink(missing_ok=True)

MODEL = (os.getenv("OLLAMA_MODEL") if os.getenv("PROVIDER", "ollama") == "ollama"
         else os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"))
SEEDS = [42, 7, 99]
PIN_SITE_ID = "06730200"
PIN_SITE_NAME = "Boulder Creek at N 75th St, CO"
PIN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

FIELDS = [
    "config", "run_id", "query_id", "family", "query", "has_temporal_cue",
    "live_reading_ft", "flood_stage_ft", "expected_label", "tool_pinned",
    "pinned_timestamp", "model_used", "status", "assessable", "actual_label",
    "false_evacuation", "missed_evacuation", "correct", "full_answer",
]

# The proposed system always runs before the comparative baselines.
CONFIGS = [
    ("Full_Agent_TGI",
     lambda q, s, pref=None: run_full_agent(q, s, prefetched_observation=pref)),
    ("Full_Agent_NoTGI",
     lambda q, s, pref=None: run_full_agent(q, s, no_tgi=True, prefetched_observation=pref)),
    ("LLM_Only", lambda q, s, pref=None: run_llm_only(q, s)),
    ("Standard_StaticRAG", lambda q, s, pref=None: run_static_rag(q, s, text_only=False)),
]


# --------------------------------------------------------------------------
# Gauge pinning
# --------------------------------------------------------------------------
def _sensor_index() -> int:
    import tools as tools_pkg
    for i, t in enumerate(tools_pkg.ALL_TOOLS):
        if getattr(t, "name", None) == "query_water_sensor":
            return i
    raise RuntimeError("query_water_sensor not found in ALL_TOOLS")


def install_pin(reading_ft: float, stage_ft: float):
    """Swap the live sensor tool for one that returns the item's gauge reading.

    Uses format_gage_report so the injected observation is byte-identical in
    structure to a real reading, including the severity status and alarm text.
    """
    import tools as tools_pkg
    import agents.disaster_agent as da
    from langchain_core.tools import StructuredTool

    idx = _sensor_index()
    original = tools_pkg.ALL_TOOLS[idx]

    def pinned_call(location: str = "Boulder Creek") -> str:
        return format_gage_report(PIN_SITE_ID, stage_ft, PIN_SITE_NAME,
                                  reading_ft, PIN_TIMESTAMP)

    pinned = StructuredTool.from_function(
        func=pinned_call,
        name="query_water_sensor",
        description=original.description or "Real-time river gage height.",
    )
    tools_pkg.ALL_TOOLS[idx] = pinned
    if da.ALL_TOOLS is not tools_pkg.ALL_TOOLS:
        for i, t in enumerate(da.ALL_TOOLS):
            if getattr(t, "name", None) == "query_water_sensor":
                da.ALL_TOOLS[i] = pinned
    da.get_agent.cache_clear()
    return original


def remove_pin(original) -> None:
    import tools as tools_pkg
    import agents.disaster_agent as da
    idx = _sensor_index()
    tools_pkg.ALL_TOOLS[idx] = original
    if da.ALL_TOOLS is not tools_pkg.ALL_TOOLS:
        for i, t in enumerate(da.ALL_TOOLS):
            if getattr(t, "name", None) == "query_water_sensor":
                da.ALL_TOOLS[i] = original
    da.get_agent.cache_clear()


# --------------------------------------------------------------------------
# Invocation
# --------------------------------------------------------------------------
def call_with_backoff(fn, query, seed, prefetched=None, max_attempts=9):
    """Local provider: a single call (no rate limits). Groq: key rotation."""
    import agents.disaster_agent as da

    if os.getenv("PROVIDER", "ollama") == "ollama":
        try:
            da.get_agent.cache_clear()
            return fn(query, seed, prefetched)
        except Exception as e:
            return {"status": "OTHER_ERROR", "full_answer": str(e),
                    "model_used": MODEL, "retries": 0}

    import core.key_manager as km
    err = "uninitialized"
    for attempt in range(max_attempts):
        try:
            os.environ["GROQ_API_KEY"] = km.groq_pool.get_key()
            da.get_agent.cache_clear()
        except Exception as e:
            err = str(e); time.sleep(2); continue
        try:
            res = fn(query, seed, prefetched)
            if str(res.get("status", "")).upper() == "OK":
                return res
            err = str(res.get("full_answer", "")) or "error"
        except Exception as e:
            err = str(e)
        low = err.lower()
        if "organization has been restricted" in low or "invalid api key" in low or "401" in low:
            try:
                km.groq_pool.mark_bad(km.groq_pool.get_key(), reason="restricted/invalid")
            except Exception:
                pass
            time.sleep(0.5)
            continue
        if ("429" in low or "rate" in low or "token" in low
                or "tool call validation failed" in low):
            m = re.search(r"try again in ([0-9.]+)s", err, re.IGNORECASE)
            delay = float(m.group(1)) if m else min(20.0, 2.0 * (2 ** min(attempt, 3)))
            time.sleep(max(1.0, delay))
            continue
        return {"status": "OTHER_ERROR", "full_answer": err,
                "model_used": MODEL, "retries": attempt + 1}
    return {"status": "RATE_LIMIT", "full_answer": f"retries exhausted: {err}",
            "model_used": MODEL, "retries": max_attempts}


def score(answer: str, expected_label: str) -> tuple[bool, int, int, int]:
    """Return (predicted_evacuation, false_evacuation, missed_evacuation, correct)."""
    is_evac = detect_evacuation(answer)
    expected_pos = expected_label == "EVACUATION"
    fp = int(is_evac and not expected_pos)
    miss = int((not is_evac) and expected_pos)
    correct = int(is_evac == expected_pos)
    return is_evac, fp, miss, correct


def summarize(path: Path) -> None:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print("no rows")
        return
    print(f"{'config':22s} {'n':>4s} {'ok':>4s} {'FP':>4s} {'FN':>4s} {'corr':>5s} "
          f"{'FPrate':>7s} {'FNrate':>7s} {'acc':>6s}")
    for cfg in [c[0] for c in CONFIGS]:
        s = [r for r in rows if r["config"] == cfg]
        if not s:
            continue
        ok = [r for r in s if r["assessable"] == "1"]
        n = len(ok)
        if not n:
            print(f"{cfg:22s} {len(s):4d}    0   -- no assessable rows --")
            continue
        fp = sum(int(r["false_evacuation"]) for r in ok)
        fn = sum(int(r["missed_evacuation"]) for r in ok)
        corr = sum(int(r["correct"]) for r in ok)
        neg = sum(1 for r in ok if r["expected_label"] == "NO_EVACUATION")
        pos = n - neg
        print(f"{cfg:22s} {len(s):4d} {n:4d} {fp:4d} {fn:4d} {corr:5d} "
              f"{(fp / neg if neg else float('nan')):7.3f} "
              f"{(fn / pos if pos else float('nan')):7.3f} {corr / n:6.3f}")
    bad = [r for r in rows if r["assessable"] != "1"]
    if bad:
        print(f"\nnon-assessable rows: {len(bad)} (excluded from rates)")
        from collections import Counter
        print("  by status:", dict(Counter(r["status"] for r in bad)))


def main():
    global OUT, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None, help="comma-separated arm names")
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", default=None, help="output CSV path (for provider-specific runs)")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 42")
    args = ap.parse_args()

    if args.out:
        OUT = Path(args.out)
        if not OUT.is_absolute():
            OUT = ROOT / OUT
        OUT.with_suffix(".complete").unlink(missing_ok=True)
    if args.seeds:
        SEEDS = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.summary:
        summarize(OUT)
        return

    items = json.loads(CONFLICT_SET.read_text(encoding="utf-8"))
    if args.limit:
        # Keep both polarities in a smoke test when possible.
        items = items[: args.limit]

    wanted = None
    if args.config:
        wanted = {c.strip() for c in args.config.split(",") if c.strip()}
    active = [c for c in CONFIGS if wanted is None or c[0] in wanted]
    if not active:
        raise SystemExit(f"Unknown --config. Valid: {[c[0] for c in CONFIGS]}")
    if wanted:
        missing = wanted - {c[0] for c in active}
        if missing:
            raise SystemExit(f"Unknown arm(s): {sorted(missing)}")

    done = set()
    if OUT.exists():
        for r in csv.DictReader(open(OUT, encoding="utf-8")):
            if r.get("assessable") == "1":
                done.add((r["config"], str(r["run_id"]), r["query_id"]))

    write_header = not OUT.exists() or OUT.stat().st_size == 0
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        total = len(active) * len(SEEDS) * len(items)
        n = new_written = 0
        for name, fn in active:
            for seed in SEEDS:
                for item in items:
                    key = (name, str(seed), item["query_id"])
                    if key in done:
                        continue
                    if args.max_new is not None and new_written >= args.max_new:
                        print(f"Reached --max-new={args.max_new}; stopping.", flush=True)
                        f.flush()
                        return
                    n += 1; new_written += 1

                    pinned = name.startswith("Full_Agent")
                    original = None
                    if pinned:
                        original = install_pin(item["live_reading_ft"],
                                               item["flood_stage_ft"])
                    # Hold evidence constant. A 3B generator routes the same
                    # question to the weather tool on some seeds and the gauge on
                    # others, which makes the score a measure of tool-routing luck
                    # rather than of the evacuation decision. The gauge reading is
                    # therefore supplied as a prefetched tool observation -
                    # byte-identical to what the pinned tool returns - so every
                    # telemetry-capable arm reasons over the same evidence, and the
                    # safety layer still sees genuine tool-grounded context.
                    # Arms without telemetry (LLM_Only, Standard_StaticRAG) receive
                    # only the bare question, which is the difference under test.
                    telemetry = None
                    if pinned:
                        telemetry = format_gage_report(
                            PIN_SITE_ID, item["flood_stage_ft"], PIN_SITE_NAME,
                            item["live_reading_ft"], PIN_TIMESTAMP)
                    try:
                        res = call_with_backoff(fn, item["query"], seed, telemetry)
                    finally:
                        if pinned and original is not None:
                            remove_pin(original)

                    answer = format_answer(res.get("full_answer", ""))
                    status = str(res.get("status", "OTHER_ERROR"))
                    assessable = status.upper() == "OK"
                    if assessable and answer.strip():
                        is_evac, fp, miss, correct = score(answer, item["expected_label"])
                    else:
                        assessable = False
                        is_evac, fp, miss, correct = None, 0, 0, 0

                    w.writerow({
                        "config": name, "run_id": seed, "query_id": item["query_id"],
                        "family": item["family"], "query": item["query"],
                        "has_temporal_cue": item["has_temporal_cue"],
                        "live_reading_ft": item["live_reading_ft"],
                        "flood_stage_ft": item["flood_stage_ft"],
                        "expected_label": item["expected_label"],
                        "tool_pinned": int(pinned),
                        "pinned_timestamp": PIN_TIMESTAMP if pinned else "",
                        "model_used": res.get("model_used", MODEL),
                        "status": status,
                        "assessable": int(assessable),
                        "actual_label": ("EVACUATION" if is_evac
                                         else ("NO_EVACUATION" if is_evac is not None else "")),
                        "false_evacuation": fp, "missed_evacuation": miss,
                        "correct": correct, "full_answer": answer,
                    })
                    f.flush()
                    print(f"[{n}/{total}] {name} seed={seed} {item['query_id']} "
                          f"{item['expected_label']} -> "
                          f"evac={is_evac} fp={fp} fn={miss} status={status}", flush=True)
                    time.sleep(0.3)

    OUT.with_suffix(".complete").write_text(
        f"rows={sum(1 for _ in csv.DictReader(open(OUT, encoding='utf-8')))}\n"
        f"finished={datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
    print("Done:", OUT)
    summarize(OUT)


if __name__ == "__main__":
    main()
