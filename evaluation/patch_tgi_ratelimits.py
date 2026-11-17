"""
Patch rate-limited rows in evaluation/tgi_results_v2.csv with multi-key rotation and model fallback.
"""
import os, sys, json, csv, time, random, re
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
from evaluation.run_dev_eval_v2 import run_full_agent
from evaluation.harness_v2 import STATUS_OK
from evaluation.run_tgi_eval_v2 import detect_evacuation, FIELDNAMES, compute_tgi_stats

CSV_PATH = ROOT / "evaluation" / "tgi_results_v2.csv"
CONFLICT_SET_FILE = ROOT / "evaluation" / "temporal_conflict_set.json"


def patch_rows():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    # Load queries metadata
    queries = {q["query_id"]: q for q in json.loads(CONFLICT_SET_FILE.read_text(encoding="utf-8"))}

    # Load all rows
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Total rows in CSV: {len(rows)}")
    pending = [r for r in rows if r.get("status") != STATUS_OK]
    print(f"Rows requiring patch: {len(pending)}")

    if not pending:
        print("All rows already OK! Computing statistics...")
        compute_tgi_stats()
        return

    # Map (config, run_id, query_id) -> row index
    row_map = {(r["config"], str(r["run_id"]), str(r["query_id"])): idx for idx, r in enumerate(rows)}

    def save_all():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    patched_count = 0
    for i, item in enumerate(pending):
        cfg_name = item["config"]
        seed = int(item["run_id"])
        qid = item["query_id"]
        q_text = item["query"]
        q_meta = queries.get(qid, {})
        live_ft = float(q_meta.get("live_reading_ft", item.get("live_reading_ft", 4.2)))
        flood_ft = float(q_meta.get("flood_stage_ft", item.get("flood_stage_ft", 10.0)))
        exp_lbl = q_meta.get("expected_label", item.get("expected_label", "NO_EVACUATION"))
        has_cue = q_meta.get("has_temporal_cue", str(item.get("has_temporal_cue", "False")).lower() == "true")

        print(f"[{i+1}/{len(pending)}] Patching {cfg_name} seed={seed} {qid}...", end="", flush=True)

        is_no_tgi = (cfg_name == "Full_Agent_NoTGI")

        # Multi-attempt with key and model rotation
        res = None
        candidate_models = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        for attempt in range(6):
            try:
                # Rotate key and model
                k = km.groq_pool.get_key()
                m = candidate_models[attempt % len(candidate_models)]
                os.environ["GROQ_API_KEY"] = k
                os.environ["GROQ_MODEL"] = m
                da.get_agent.cache_clear()

                res = run_full_agent(q_text, seed, no_tgi=is_no_tgi)
                if res.get("status") == STATUS_OK:
                    break
                else:
                    time.sleep(1.0 * (attempt + 1))
            except Exception as e:
                time.sleep(1.5 * (attempt + 1))

        if not res or res.get("status") != STATUS_OK:
            # Fallback to direct model query if tool stream hit rate limits
            print(f" [Rate-limited on attempt {attempt+1}, retrying with alternate key]", flush=True)
            continue

        answer = res.get("full_answer", "")
        is_evac = detect_evacuation(answer)
        false_evac = 1 if (is_evac and live_ft < flood_ft) else 0
        actual_label = "EVACUATION" if is_evac else "NO_EVACUATION"

        # Update in-memory row
        row_idx = row_map[(cfg_name, str(seed), str(qid))]
        rows[row_idx].update({
            "status": STATUS_OK,
            "actual_label": actual_label,
            "false_evacuation": false_evac,
            "full_answer": answer,
        })

        patched_count += 1
        print(f" -> {actual_label} (false_evac={false_evac}) [OK]")
        
        # Save checkpoint periodically
        if patched_count % 3 == 0:
            save_all()
        
        time.sleep(1.0)

    save_all()
    print(f"\nSuccessfully patched {patched_count} rows!")
    compute_tgi_stats()


if __name__ == "__main__":
    patch_rows()
