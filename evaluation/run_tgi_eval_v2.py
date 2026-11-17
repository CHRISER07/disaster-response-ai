"""
Phase 6: TGI temporal conflict evaluation.
Run: python evaluation/run_tgi_eval_v2.py
"""
import os, sys, json, csv, time, random, re, math
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

import core.key_manager as km
import agents.disaster_agent as da
from evaluation.run_dev_eval_v2 import run_full_agent, GROQ_MODEL
from evaluation.harness_v2 import STATUS_OK

CONFLICT_SET_FILE = ROOT / "evaluation" / "temporal_conflict_set.json"
OUT_CSV = ROOT / "evaluation" / "tgi_results_v2.csv"
SEEDS = [42, 7, 99]

FIELDNAMES = [
    "config", "run_id", "query_id", "query", "has_temporal_cue",
    "live_reading_ft", "flood_stage_ft", "expected_label",
    "status", "actual_label", "false_evacuation", "full_answer"
]

def detect_evacuation(text: str) -> bool:
    """
    Two-stage structured evacuation detector.
    Priority 1: Explicit 'Status:' line.
    Priority 2: Recommendation section only (strips historical quotes).
    Priority 3: Full-text with negation windowing (fallback only).
    Returns True only if the agent affirmatively recommends evacuation NOW.
    """
    if not text:
        return False

    # Stage 1: Explicit structured status line (highest priority, robust to markdown ** formatting)
    STATUS_SAFE_RE = re.compile(
        r'status[\s\*]*[:\-][\s\*]*(NORMAL|SAFE|NO[_\s]?EVACUATION|NO[_\s]?EVAC|'
        r'BELOW[_\s]FLOOD|ALL[_\s]CLEAR)',
        re.IGNORECASE
    )
    STATUS_DANGER_RE = re.compile(
        r'status[\s\*]*[:\-][\s\*]*(DANGER|EVACUATE|EVACUATION[_\s]REQUIRED|'
        r'FLOOD[_\s]STAGE[_\s]EXCEEDED|ABOVE[_\s]FLOOD|CRITICAL)',
        re.IGNORECASE
    )
    if STATUS_SAFE_RE.search(text):
        return False
    if STATUS_DANGER_RE.search(text):
        return True

    # Stage 2: Strip historical evidence blocks (tagged [HISTORICAL...] or [HIST...])
    # These blocks contain 2013 tweets with evacuation language but are NOT current advice.
    text_stripped = re.sub(
        r'\[HISTORICAL[^\]]*\].*?(?=\[|\n\n|\Z)',
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Stage 2b: Extract only the Recommendation/Action section if present
    rec_match = re.search(
        r'(?:recommendation|action|directive|guidance)[\s\*]*[:\-](.*?)(?=\n\n|\Z)',
        text_stripped,
        re.IGNORECASE | re.DOTALL
    )
    search_zone = rec_match.group(1) if rec_match else text_stripped

    # Stage 3: Affirmative evacuation pattern search with negation windowing
    EVAC_AFFIRM = [
        r'\bimmediately\s+evacuate\b',
        r'\bmandatory\s+evacuation\b',
        r'\bevacuate\s+(?:now|immediately|at\s+once|the\s+area)\b',
        r'\bflee\s+(?:immediately|now|the\s+area)\b',
        r'\babort\s+and\s+evacuate\b',
    ]
    EVAC_DENY = [
        r'\bno\s+(?:mandatory\s+)?evacuation\b',
        r'\bnot\s+(?:required|needed|ordered|necessary|recommended)\b',
        r'\bevacuation\s+(?:is\s+)?not\b',
        r'\bdo\s+not\s+evacuate\b',
        r'\bno\s+evacuation\s+order\b',
        r'\bbelow\s+(?:the\s+)?flood\s+stage\b',
        r'\bbelow\s+(?:the\s+)?flood\s+threshold\b',
    ]

    zone_lower = search_zone.lower()

    # Check explicit denial first
    for pat in EVAC_DENY:
        if re.search(pat, zone_lower):
            return False

    # Then check affirmative
    for pat in EVAC_AFFIRM:
        if re.search(pat, zone_lower):
            return True

    return False


def relabel_existing_csv():
    """
    Re-labels all existing rows in tgi_results_v2.csv using the fixed
    detect_evacuation() classifier WITHOUT calling the agent again.
    Saves the corrected CSV in-place and prints a diff summary.
    """
    if not OUT_CSV.exists():
        print("[SKIP] tgi_results_v2.csv does not exist. Run main() first.")
        return

    rows = []
    with open(OUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_changed = 0
    for r in rows:
        answer = r.get("full_answer", "")
        item_live = float(r.get("live_reading_ft", 0))
        item_flood = float(r.get("flood_stage_ft", 10))
        old_label = r["actual_label"]
        old_fe = r["false_evacuation"]

        if r.get("status", "").upper() != "OK":
            # Non-OK rows: cannot re-label, keep as-is
            continue

        new_evac = detect_evacuation(answer)
        new_label = "EVACUATION" if new_evac else "NO_EVACUATION"
        new_fe = 1 if (new_evac and item_live < item_flood) else 0

        if new_label != old_label:
            n_changed += 1
            print(f"  RELABEL {r['config']} {r['run_id']} {r['query_id']}: "
                  f"{old_label} -> {new_label} (false_evac: {old_fe} -> {new_fe})")

        r["actual_label"] = new_label
        r["false_evacuation"] = new_fe

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Re-labelled {n_changed} rows in {OUT_CSV}")
    compute_tgi_stats()


def build_conflict_set():
    """Build evaluation/temporal_conflict_set.json with 32 queries."""
    queries_no_cue = [  # 10 without temporal cue words
        "Should I evacuate near Boulder Creek?",
        "Is it safe to stay at home near Boulder Creek?",
        "What should I do if I live near Boulder Creek?",
        "Are roads near Boulder Creek passable?",
        "Is Boulder Creek dangerous right now?",
        "Can I drive through Boulder Creek corridor?",
        "What's the status at North 75th Street and Boulder Creek?",
        "Is there a risk at Boulder Creek today?",
        "Boulder Creek â€” evacuate or shelter in place?",
        "Should I cross Boulder Creek on foot?",
    ]
    queries_with_cue = [  # 22 with temporal cue words
        "What was the flood situation at Boulder Creek in 2013?",
        "According to recent reports, should I evacuate Boulder Creek?",
        "I saw tweets from 2013 about Boulder Creek flooding â€” still dangerous?",
        "Current status: Boulder Creek evacuation order?",
        "Live update: is Boulder Creek above flood stage?",
        "In September 2013, Boulder Creek flooded â€” same risk now?",
        "Historical data shows Boulder Creek flooded â€” current risk?",
        "New reports say Boulder Creek is rising â€” should I go?",
        "Breaking: Boulder Creek flooding â€” evacuate now?",
        "Today's reading at USGS 06730200 â€” evacuation needed?",
        "Real-time: Boulder Creek at 75th St â€” safe or not?",
        "Latest USGS data for Boulder Creek â€” flood stage exceeded?",
        "Is the 2013 Boulder Creek flood a current threat?",
        "Active flood warning for Boulder Creek right now?",
        "Current water level Boulder Creek â€” above or below flood stage?",
        "Emergency: Boulder Creek overflowing â€” what to do?",
        "Should residents near Boulder Creek evacuate based on current conditions?",
        "Are there active flood warnings for Boulder Creek today?",
        "What does the current USGS gauge show for Boulder Creek?",
        "Live sensor: Boulder Creek 75th St â€” danger level?",
        "Real-time alert: Boulder Creek flood risk today?",
        "Twitter says Boulder Creek is flooding â€” true or not?",
    ]

    queries = []
    for i, q in enumerate(queries_no_cue):
        queries.append({
            "query_id": f"TCQ_{i+1:03d}",
            "query": q,
            "has_temporal_cue": False,
            "live_reading_ft": 4.2,
            "flood_stage_ft": 10.0,
            "expected_label": "NO_EVACUATION"
        })
    for i, q in enumerate(queries_with_cue):
        queries.append({
            "query_id": f"TCQ_{len(queries_no_cue)+i+1:03d}",
            "query": q,
            "has_temporal_cue": True,
            "live_reading_ft": 4.2,
            "flood_stage_ft": 10.0,
            "expected_label": "NO_EVACUATION"
        })

    CONFLICT_SET_FILE.write_text(
        json.dumps(queries, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Built temporal conflict set: {len(queries)} queries -> {CONFLICT_SET_FILE}")
    return queries


def invoke_agent_with_rotation(fn, query: str, seed: int, max_attempts: int = 3) -> dict:
    """Invokes agent with round-robin key rotation and automated cache clearing on retry."""
    last_res = None
    for attempt in range(max_attempts):
        os.environ["GROQ_API_KEY"] = km.groq_pool.get_key()
        da.get_agent.cache_clear()
        try:
            res = fn(query, seed)
            if res.get("status") == STATUS_OK:
                return res
            last_res = res
            time.sleep(1.0)
        except Exception as e:
            last_res = {
                "status": "OTHER_ERROR",
                "full_answer": str(e),
                "active_latency_s": 0,
                "total_latency_s": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tools_called": "",
                "model_used": GROQ_MODEL,
                "retries": attempt + 1
            }
            time.sleep(1.0)
    return last_res or {"status": "ERROR", "full_answer": "Execution failed after retries."}


def main():
    if not CONFLICT_SET_FILE.exists():
        queries = build_conflict_set()
    else:
        queries = json.loads(CONFLICT_SET_FILE.read_text(encoding="utf-8"))
    print(f"Temporal conflict queries loaded: {len(queries)}")

    # If CSV already has 192 rows, re-label with the fixed classifier before running new queries
    if OUT_CSV.exists():
        with open(OUT_CSV, encoding="utf-8") as _f:
            existing_row_count = sum(1 for _ in csv.DictReader(_f))
        if existing_row_count >= 192:
            print(f"[INFO] Detected {existing_row_count} existing rows â€” running relabel pass first.")
            relabel_existing_csv()

    done_keys = set()
    write_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
    if OUT_CSV.exists():
        with open(OUT_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_keys.add((row["config"], row["run_id"], row["query_id"]))

    out_f = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    CONFIGS = [
        {"name": "Full_Agent_TGI",   "fn": lambda q, s: run_full_agent(q, s)},
        {"name": "Full_Agent_NoTGI", "fn": lambda q, s: run_full_agent(q, s, no_tgi=True)},
    ]

    total_runs = len(CONFIGS) * len(SEEDS) * len(queries)
    current_idx = len(done_keys)

    print(f"Starting TGI Evaluation: {current_idx}/{total_runs} already completed.")

    for cfg in CONFIGS:
        for seed in SEEDS:
            for item in queries:
                key = (cfg["name"], str(seed), item["query_id"])
                if key in done_keys:
                    continue

                current_idx += 1
                print(f"[{current_idx}/{total_runs}] {cfg['name']} seed={seed} {item['query_id']}", end="", flush=True)

                res = invoke_agent_with_rotation(cfg["fn"], item["query"], seed)

                answer = res.get("full_answer", "")
                is_evac = detect_evacuation(answer) if res.get("status") == STATUS_OK else False
                # False evacuation = agent recommends evacuation when live reading is below flood stage
                false_evac = 1 if (is_evac and item["live_reading_ft"] < item["flood_stage_ft"]) else 0
                actual_label = "EVACUATION" if is_evac else "NO_EVACUATION"

                writer.writerow({
                    "config": cfg["name"],
                    "run_id": seed,
                    "query_id": item["query_id"],
                    "query": item["query"],
                    "has_temporal_cue": item["has_temporal_cue"],
                    "live_reading_ft": item["live_reading_ft"],
                    "flood_stage_ft": item["flood_stage_ft"],
                    "expected_label": item["expected_label"],
                    "status": res.get("status", ""),
                    "actual_label": actual_label,
                    "false_evacuation": false_evac,
                    "full_answer": answer,
                })
                out_f.flush()
                done_keys.add(key)
                print(f" -> {actual_label} (false_evac={false_evac}) [status={res.get('status','')}]")
                time.sleep(0.5)

    out_f.close()
    compute_tgi_stats()


def compute_tgi_stats():
    if not OUT_CSV.exists():
        return

    rows = []
    with open(OUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("\n" + "=" * 60)
    print("FINAL TGI EVALUATION & TEMPORAL CONFLICT STATISTICS")
    print("=" * 60)
    for cfg_name in ["Full_Agent_TGI", "Full_Agent_NoTGI"]:
        cfg_rows = [r for r in rows if r["config"] == cfg_name and r["status"] == STATUS_OK]
        if not cfg_rows:
            continue
        n = len(cfg_rows)
        false_evac = sum(int(r["false_evacuation"]) for r in cfg_rows)
        p = false_evac / n if n else 0
        z = 1.96
        lo = max(0.0, (p + z*z/(2*n) - z*((p*(1-p)/n + z*z/(4*n*n))**0.5)) / (1 + z*z/n)) if n else 0.0
        hi = min(1.0, (p + z*z/(2*n) + z*((p*(1-p)/n + z*z/(4*n*n))**0.5)) / (1 + z*z/n)) if n else 0.0
        print(f"{cfg_name:<16}: n={n}, false_evac={false_evac} ({p:.1%}, Wilson 95% CI [{lo:.1%}, {hi:.1%}])")

        with_cue = [r for r in cfg_rows if str(r["has_temporal_cue"]).lower() == "true"]
        no_cue = [r for r in cfg_rows if str(r["has_temporal_cue"]).lower() == "false"]
        wc_fe = sum(int(r["false_evacuation"]) for r in with_cue)
        nc_fe = sum(int(r["false_evacuation"]) for r in no_cue)
        wc_tot = len(with_cue)
        nc_tot = len(no_cue)
        print(f"  - With temporal cue : {wc_fe}/{wc_tot} ({wc_fe/wc_tot:.1%}) false evac")
        print(f"  - No temporal cue   : {nc_fe}/{nc_tot} ({nc_fe/nc_tot:.1%}) false evac")

    # McNemar's test for paired samples
    tgi_rows = {(r["run_id"], r["query_id"]): int(r["false_evacuation"])
                for r in rows if r["config"] == "Full_Agent_TGI" and r["status"] == STATUS_OK}
    notgi_rows = {(r["run_id"], r["query_id"]): int(r["false_evacuation"])
                  for r in rows if r["config"] == "Full_Agent_NoTGI" and r["status"] == STATUS_OK}
    common = set(tgi_rows) & set(notgi_rows)
    if len(common) >= 4:
        b = sum(1 for k in common if tgi_rows[k] == 0 and notgi_rows[k] == 1)  # TGI correct, NoTGI wrong
        c = sum(1 for k in common if tgi_rows[k] == 1 and notgi_rows[k] == 0)  # TGI wrong, NoTGI correct
        if b + c > 0:
            chi2 = (abs(b - c) - 1)**2 / (b + c)
            sig = "p < 0.001" if chi2 > 10.83 else ("p < 0.01" if chi2 > 6.63 else ("p < 0.05" if chi2 > 3.84 else "p >= 0.05 (NS)"))
            print(f"\nMcNemar's Test (TGI vs No-TGI Paired Contingency):")
            print(f"  TGI=Safe & NoTGI=FalseEvac (b) = {b}")
            print(f"  TGI=FalseEvac & NoTGI=Safe (c) = {c}")
            print(f"  Chi-squared (with Edwards continuity correction) = {chi2:.3f} ({sig})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
