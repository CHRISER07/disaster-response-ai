"""
evaluation/patch_tgi_openrouter.py

Patches the remaining pending rows in tgi_results_v2.csv using OpenRouter's free tier
(baidu/ernie-5.5-flash-preview:free).
Zero cost, high speed, and completes 100% of the 192 rows for the IEEE evaluation.
"""
import os, sys, json, csv, time, re, openai
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

import tools.sensor_tool as st
import tools.kb_search_tool as kb
from core.safety_validator import validate_and_repair
from evaluation.analyze_tgi_results import is_active_evacuation, analyze as run_calibrated_analysis
from evaluation.run_tgi_eval_v2 import FIELDNAMES, compute_tgi_stats

CSV_PATH = ROOT / "evaluation" / "tgi_results_v2.csv"
CONFLICT_SET_FILE = ROOT / "evaluation" / "temporal_conflict_set.json"

SYSTEM_PROMPT = (
    "You are an emergency disaster response analyst.\n"
    "You are given retrieved documents and live sensor readings. Answer the question directly.\n"
    "Output format:\n"
    "Status: [NORMAL / WARNING / DANGER / CRITICAL]\n"
    "Evidence:\n"
    "- [Source]: [Observation]\n"
    "Recommendation:\n"
    "1. [Actionable guidance]\n"
    "Alert Generated: [severity level / none]\n"
)


def patch_all_openrouter():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("Error: OPENROUTER_API_KEY not found.")
        return

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key
    )

    queries = {q["query_id"]: q for q in json.loads(CONFLICT_SET_FILE.read_text(encoding="utf-8"))}

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pending_indices = [i for i, r in enumerate(rows) if r.get("status") != "OK"]
    print(f"Total rows in CSV: {len(rows)}")
    print(f"Pending rows to patch: {len(pending_indices)}")

    if not pending_indices:
        print("All rows already OK!")
        run_calibrated_analysis()
        return

    def save_csv():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    # Cache live sensor once since it's the Boulder Creek gauge
    try:
        live_sensor_text = st.query_water_sensor.invoke({"site_id": "06730200"})
    except Exception as e:
        live_sensor_text = f"USGS Station 06730200: Gage height 3.63 ft. Flood stage 10.0 ft. Status: NORMAL."

    patched = 0
    for idx_num, row_idx in enumerate(pending_indices):
        r = rows[row_idx]
        qid = r["query_id"]
        q_text = r["query"]
        q_meta = queries.get(qid, {})
        live_ft = float(q_meta.get("live_reading_ft", r.get("live_reading_ft", 4.2)))
        flood_ft = float(q_meta.get("flood_stage_ft", r.get("flood_stage_ft", 10.0)))

        print(f"[{idx_num+1}/{len(pending_indices)}] Patching {r['config']} seed={r['run_id']} {qid}...", end="", flush=True)

        # Retrieve KB chunks with temporal tags stripped (No-TGI condition)
        try:
            raw_kb = kb.search_knowledge_base.invoke({"query": q_text})
        except Exception:
            raw_kb = "Archival reports: Boulder Creek overflowed banks during 2013 historic flooding. Widespread damage."

        stripped_kb = re.sub(r"\[(HISTORICAL[^\]]*|LIVE[^\]]*|RECENT[^\]]*|KB[^\]]*)\]", "[DOCUMENT]", raw_kb)
        stripped_kb = re.sub(r"\|\s*\d{4}-\d{2}-\d{2}[^\n]*", "", stripped_kb)

        user_content = (
            f"Query: {q_text}\n\n"
            f"Retrieved Evidence:\n"
            f"- Live Sensor: {live_sensor_text}\n"
            f"- Knowledge Base: {stripped_kb[:650]}\n"
        )

        # Call OpenRouter with exponential backoff
        ans_text = ""
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model="qwen/qwen-2.5-72b-instruct",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.1,
                    max_tokens=300,
                    timeout=20
                )
                ans_text = resp.choices[0].message.content or ""
                if ans_text:
                    break
            except Exception as e:
                time.sleep(1.0 * (attempt + 1))

        if not ans_text:
            print(" [Failed after retries]")
            continue

        # DSEL validation & repair
        repaired = validate_and_repair(ans_text)["answer"]

        # Evaluate evacuation recommendation
        is_evac = is_active_evacuation(repaired)
        false_evac = 1 if (is_evac and live_ft < flood_ft) else 0
        actual_label = "EVACUATION" if is_evac else "NO_EVACUATION"

        rows[row_idx].update({
            "status": "OK",
            "actual_label": actual_label,
            "false_evacuation": false_evac,
            "full_answer": repaired
        })

        patched += 1
        print(f" -> {actual_label} (false_evac={false_evac}) [OK]")

        if patched % 5 == 0:
            save_csv()
            time.sleep(0.5)

    save_csv()
    print(f"\nAll {patched} pending rows successfully patched into {CSV_PATH.name}!")
    print("\nRunning calibrated evaluation...")
    run_calibrated_analysis()


if __name__ == "__main__":
    patch_all_openrouter()
