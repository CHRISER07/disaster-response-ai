"""EXP-2: 50-scenario two-sided temporal stress test.

One generator (qwen2.5:3b), two arms (Full Agent with and without TGI),
three seeds, 100 items (50 query texts x safe/critical). Evidence is injected
through format_gage_report, the same severity logic as the live sensor tool.
McNemar's test compares paired correctness over scenario x seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import binomtest

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
from evaluation.run_tgi_v5 import call_with_backoff, score, _sensor_index
from core.answer_formatter import format_answer
from tools.sensor_tool import format_gage_report

SCENARIOS = ROOT / "evaluation" / "temporal_exp2_scenarios.json"
OUT = ROOT / "evaluation" / "tgi_exp2_results.csv"
MODEL = (os.getenv("OLLAMA_MODEL") if os.getenv("PROVIDER", "ollama") == "ollama"
         else os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"))
SEEDS = [42, 7, 99]
TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

CONFIGS = [
    ("Full_Agent_TGI", lambda q, s, pref=None: run_full_agent(q, s, prefetched_observation=pref)),
    ("Full_Agent_NoTGI", lambda q, s, pref=None: run_full_agent(q, s, no_tgi=True, prefetched_observation=pref)),
]

FIELDS = ["config", "run_id", "query_id", "family", "site", "query",
          "has_temporal_cue", "live_reading_ft", "flood_stage_ft",
          "expected_label", "tool_pinned", "pinned_timestamp", "model_used",
          "status", "assessable", "actual_label", "false_evacuation",
          "missed_evacuation", "correct", "full_answer"]


def install_pin(item):
    import tools as tools_pkg
    import agents.disaster_agent as da
    from langchain_core.tools import StructuredTool

    idx = _sensor_index()
    original = tools_pkg.ALL_TOOLS[idx]

    def pinned_call(location: str = "") -> str:
        return report(item)

    pinned = StructuredTool.from_function(
        func=pinned_call, name="query_water_sensor",
        description=original.description or "Real-time river gage height.")
    tools_pkg.ALL_TOOLS[idx] = pinned
    if da.ALL_TOOLS is not tools_pkg.ALL_TOOLS:
        for i, t in enumerate(da.ALL_TOOLS):
            if getattr(t, "name", None) == "query_water_sensor":
                da.ALL_TOOLS[i] = pinned
    da.get_agent.cache_clear()
    return original


def remove_pin(original):
    import tools as tools_pkg
    import agents.disaster_agent as da
    idx = _sensor_index()
    tools_pkg.ALL_TOOLS[idx] = original
    if da.ALL_TOOLS is not tools_pkg.ALL_TOOLS:
        for i, t in enumerate(da.ALL_TOOLS):
            if getattr(t, "name", None) == "query_water_sensor":
                da.ALL_TOOLS[i] = original
    da.get_agent.cache_clear()


def report(item):
    return format_gage_report(item["site_id"], item["flood_stage_ft"],
                              item["site_name"], item["live_reading_ft"], TS)


def summarize(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print("no rows")
        return
    print(f"{'config':20s} {'n':>4s} {'FP':>4s} {'FN':>4s} {'corr':>5s} {'acc':>6s}")
    for cfg, _ in CONFIGS:
        sub = [r for r in rows if r["config"] == cfg and r["assessable"] == "1"]
        if not sub:
            continue
        fp = sum(int(r["false_evacuation"]) for r in sub)
        fn = sum(int(r["missed_evacuation"]) for r in sub)
        corr = sum(int(r["correct"]) for r in sub)
        print(f"{cfg:20s} {len(sub):4d} {fp:4d} {fn:4d} {corr:5d} {corr/len(sub):6.3f}")
    bad = [r for r in rows if r["assessable"] != "1"]
    if bad:
        print("non-assessable:", len(bad), dict(Counter(r["status"] for r in bad)))

    # McNemar on paired correctness by (query_id, seed).
    by = {}
    for r in rows:
        if r["assessable"] != "1":
            continue
        by.setdefault((r["query_id"], str(r["run_id"])), {})[r["config"]] = int(r["correct"])
    pairs = [(v[CONFIGS[0][0]], v[CONFIGS[1][0]])
             for v in by.values() if len(v) == 2]
    b = sum(1 for x, y in pairs if x == 1 and y == 0)  # TGI right, NoTGI wrong
    c = sum(1 for x, y in pairs if x == 0 and y == 1)
    if b + c:
        p = binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
        print(f"McNemar paired n={len(pairs)} b={b} c={c} p={p:.4g}")


def main():
    global OUT, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--scenarios", default=None)
    args = ap.parse_args()

    global SCENARIOS
    if args.scenarios:
        SCENARIOS = Path(args.scenarios)
        if not SCENARIOS.is_absolute():
            SCENARIOS = ROOT / SCENARIOS
    if args.out:
        OUT = Path(args.out)
        if not OUT.is_absolute():
            OUT = ROOT / OUT
    if args.seeds:
        SEEDS = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.summary:
        summarize(OUT)
        return

    items = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    if args.limit:
        items = items[:args.limit]
    wanted = None
    if args.config:
        wanted = {c.strip() for c in args.config.split(",") if c.strip()}
    active = [c for c in CONFIGS if wanted is None or c[0] in wanted]
    if not active:
        raise SystemExit(f"Unknown --config. Valid: {[c[0] for c in CONFIGS]}")

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
                    n += 1
                    new_written += 1
                    original = install_pin(item)
                    telemetry = report(item)
                    try:
                        res = call_with_backoff(fn, item["query"], seed, telemetry)
                    finally:
                        remove_pin(original)
                    answer = format_answer(res.get("full_answer", ""))
                    status = str(res.get("status", "OTHER_ERROR"))
                    assessable = status.upper() == "OK" and bool(answer.strip())
                    if assessable:
                        is_evac, fp, miss, correct = score(answer, item["expected_label"])
                    else:
                        is_evac, fp, miss, correct = None, 0, 0, 0
                    w.writerow({
                        "config": name, "run_id": seed, "query_id": item["query_id"],
                        "family": item["family"], "site": item["site_name"],
                        "query": item["query"],
                        "has_temporal_cue": item["has_temporal_cue"],
                        "live_reading_ft": item["live_reading_ft"],
                        "flood_stage_ft": item["flood_stage_ft"],
                        "expected_label": item["expected_label"],
                        "tool_pinned": 1, "pinned_timestamp": TS,
                        "model_used": res.get("model_used", MODEL),
                        "status": status, "assessable": int(assessable),
                        "actual_label": ("EVACUATION" if is_evac else
                                         ("NO_EVACUATION" if is_evac is not None else "")),
                        "false_evacuation": fp, "missed_evacuation": miss,
                        "correct": correct, "full_answer": answer,
                    })
                    f.flush()
                    if n % 20 == 0 or n == total:
                        print(f"[{n}/{total}] {name} seed={seed} {item['query_id']} "
                              f"{item['expected_label']} evac={is_evac} status={status}",
                              flush=True)
                    time.sleep(0.15)
    print("Done:", OUT)
    summarize(OUT)


if __name__ == "__main__":
    main()
