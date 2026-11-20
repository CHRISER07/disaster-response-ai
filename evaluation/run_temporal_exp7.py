"""EXP-7: temporal freshness conflict benchmark.

A stale archived document contradicts a fresh live gauge reading. Three arms:
  Full_Agent_TGI   - normal TGI: stale doc tagged [HISTORICAL - 2013]
  Full_Agent_NoTGI - production no_tgi strip: all temporal tags -> [DOCUMENT]
  Timestamp_Only   - TGI tags present but source timestamps hidden via metadata
                     scrub; tests whether the tier label or the raw date drives
                     the decision (no production code changed)

All arms receive byte-identical evidence text except the temporal tier label,
so the only manipulated variable is the temporal grounding signal.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import binomtest, wilcoxon

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
import tools.kb_search_tool as kb

SCENARIOS = ROOT / "evaluation" / "temporal_exp7_scenarios.json"
OUT = ROOT / "evaluation" / "tgi_exp7_results.csv"
MODEL = (os.getenv("OLLAMA_MODEL") if os.getenv("PROVIDER", "ollama") == "ollama"
         else os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"))
SEEDS = [42, 7, 99]
TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
FIELDS = ["config", "run_id", "query_id", "condition", "site", "query",
          "live_reading_ft", "flood_stage_ft", "expected_label", "model_used",
          "status", "assessable", "actual_label", "false_evacuation",
          "missed_evacuation", "correct", "full_answer"]

CONFIGS = [
    ("Full_Agent_TGI", False),
    ("Full_Agent_NoTGI", True),
    ("Timestamp_Only", "timestamp"),
]


def stale_doc(item):
    """Render the stale document through the real temporal_tag path."""
    from core.temporal_grounding import temporal_tag, parse_timestamp
    ref = parse_timestamp(item.get("reference_now")) if item.get("reference_now") else None
    era = temporal_tag({"timestamp": item["stale_timestamp"],
                        "source": item["stale_source"], "modality": "text"}, now=ref)
    return (f"[KB SEARCH | retrieval=injected_historical,k=1]\n"
            f"[1] [TEXT] {era} Source: {item['stale_source']} | {item['stale_timestamp']}\n"
            f"    {item['stale_text']}")


def install_kb(item, mode):
    original = kb._search
    if mode == "timestamp":
        # Scrub the tier label but keep the raw source date visible.
        doc = stale_doc(item)
        import re
        doc = re.sub(r"\[(LIVE|RECENT|HISTORICAL|UNDATED)[^\]]*\]", "[DOCUMENT]", doc)
    else:
        doc = stale_doc(item)

    def injected_search(query, modality_filter=None, k=5):
        return doc

    kb._search = injected_search
    return original


def restore_kb(original):
    kb._search = original


def install_pin(item):
    import tools as tools_pkg
    import agents.disaster_agent as da
    from langchain_core.tools import StructuredTool
    idx = _sensor_index()
    original = tools_pkg.ALL_TOOLS[idx]

    def pinned_call(location: str = "") -> str:
        return report(item)

    pinned = StructuredTool.from_function(func=pinned_call, name="query_water_sensor",
                                          description=original.description or "River gage height.")
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
    print(f"{'config':18s} {'n':>4s} {'acc':>6s} {'FP':>4s} {'FN':>4s}")
    for cfg, _ in CONFIGS:
        sub = [r for r in rows if r["config"] == cfg and r["assessable"] == "1"]
        if not sub:
            continue
        fp = sum(int(r["false_evacuation"]) for r in sub)
        fn = sum(int(r["missed_evacuation"]) for r in sub)
        corr = sum(int(r["correct"]) for r in sub)
        print(f"{cfg:18s} {len(sub):4d} {corr/len(sub):6.3f} {fp:4d} {fn:4d}")
    bad = [r for r in rows if r["assessable"] != "1"]
    if bad:
        print("non-assessable", len(bad), dict(Counter(r["status"] for r in bad)))
    by = {}
    for r in rows:
        if r["assessable"] == "1":
            by.setdefault((r["query_id"], str(r["run_id"])), {})[r["config"]] = int(r["correct"])
    for ca, cb in (("Full_Agent_TGI", "Full_Agent_NoTGI"), ("Full_Agent_TGI", "Timestamp_Only")):
        pairs = [(v.get(ca), v.get(cb)) for v in by.values()]
        pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        b = sum(1 for a, c in pairs if a == 1 and c == 0)
        c = sum(1 for a, d in pairs if a == 0 and d == 1)
        if b + c:
            p = binomtest(b, b + c, 0.5).pvalue
            print(f"McNemar {ca} vs {cb}: n={len(pairs)} b={b} c={c} p={p:.4g}")


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
        for name, mode in active:
            no_tgi = (mode is True)
            for seed in SEEDS:
                for item in items:
                    key = (name, str(seed), item["query_id"])
                    if key in done:
                        continue
                    if args.max_new is not None and new_written >= args.max_new:
                        f.flush()
                        return
                    n += 1
                    new_written += 1
                    kb_orig = install_kb(item, mode)
                    pin_orig = install_pin(item)
                    try:
                        res = call_with_backoff(
                            lambda q, s, pref=None: run_full_agent(q, s, no_tgi=no_tgi,
                                                                   prefetched_observation=pref),
                            item["query"], seed, report(item))
                    finally:
                        restore_kb(kb_orig)
                        remove_pin(pin_orig)
                    answer = format_answer(res.get("full_answer", ""))
                    status = str(res.get("status", "OTHER_ERROR"))
                    assessable = status.upper() == "OK" and bool(answer.strip())
                    if assessable:
                        is_evac, fp, fn, correct = score(answer, item["expected_label"])
                    else:
                        is_evac, fp, fn, correct = None, 0, 0, 0
                    w.writerow({
                        "config": name, "run_id": seed, "query_id": item["query_id"],
                        "condition": item["condition"], "site": item["site_name"],
                        "query": item["query"], "live_reading_ft": item["live_reading_ft"],
                        "flood_stage_ft": item["flood_stage_ft"],
                        "expected_label": item["expected_label"],
                        "model_used": res.get("model_used", MODEL), "status": status,
                        "assessable": int(assessable),
                        "actual_label": ("EVACUATION" if is_evac else
                                         ("NO_EVACUATION" if is_evac is not None else "")),
                        "false_evacuation": fp, "missed_evacuation": fn,
                        "correct": correct, "full_answer": answer,
                    })
                    f.flush()
                    if n % 25 == 0 or n == total:
                        print(f"[{n}/{total}] {name} seed={seed} {item['query_id']} "
                              f"{item['expected_label']} evac={is_evac} status={status}", flush=True)
                    time.sleep(0.1)
    print("Done:", OUT)
    summarize(OUT)


if __name__ == "__main__":
    main()
