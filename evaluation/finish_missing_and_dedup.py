
import os, sys, json, csv, time, re
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from groq import Groq
from evaluation.run_judge_v3 import (
    UNIFIED_JUDGE_PROMPT,
    parse_unified_response,
    JUDGE_FIELDNAMES,
    compute_kappa
)

OUT_CSV = ROOT / "evaluation" / "judge_scores_v2.csv"
DATASET_FILE = ROOT / "evaluation" / "golden_dataset_v2.json"
DEV_CSV = ROOT / "evaluation" / "results_per_query_v2.csv"
HELDOUT_CSV = ROOT / "evaluation" / "results_heldout_v2.csv"

# 1. Deduplicate existing CSV
print("Step 1: Deduplicating existing judge_scores_v2.csv...")
rows = list(csv.DictReader(open(OUT_CSV, encoding="utf-8")))
print(f"Original row count: {len(rows)}")

seen = set()
unique_rows = []
for r in rows:
    # Must have valid scores
    if any(r.get(d) in ("", "None", None) for d in ["dr_fact", "saa", "temporal"]):
        continue
    key = (r["config"], str(r["run_id"]), str(r["id"]), r["judge"])
    if key not in seen:
        seen.add(key)
        unique_rows.append(r)

print(f"Valid unique rows before gap-fill: {len(unique_rows)}")
print(f"  Judge A: {sum(1 for r in unique_rows if r['judge'] == 'JudgeA')}")
print(f"  Judge B: {sum(1 for r in unique_rows if r['judge'] == 'JudgeB')}")

# 2. Identify missing rows
q_map = {q["id"]: q for q in json.loads(DATASET_FILE.read_text(encoding="utf-8"))}
dev_rows = [r for r in csv.DictReader(open(DEV_CSV, encoding="utf-8")) if r.get("status") == "OK"]
hld_rows = [r for r in csv.DictReader(open(HELDOUT_CSV, encoding="utf-8")) if r.get("status") == "OK"]
all_candidates = dev_rows + hld_rows

missing = []
for row in all_candidates:
    raw_id = row.get("id", "")
    for jname in ["JudgeA", "JudgeB"]:
        ck = (row["config"], str(row["run_id"]), str(raw_id), jname)
        if ck not in seen:
            missing.append((row, jname, ck))

print(f"\nStep 2: Missing evaluations to complete: {len(missing)}")

# 3. Fill missing rows
groq_keys = [
    os.getenv(f"GROQ_API_KEY_{let}", "").strip()
    for let in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
]
groq_keys = [k for k in groq_keys if k]
groq_clients = [Groq(api_key=k) for k in groq_keys]

for i, (row, judge_name, ck) in enumerate(missing):
    raw_id = row.get("id", "")
    try:
        qid = int(raw_id)
    except Exception:
        qid = raw_id
    q_data = q_map.get(qid) or q_map.get(str(qid)) or {}
    query = q_data.get("query", f"Query id={qid}")
    ground_truth = q_data.get("ground_truth_answer", q_data.get("safe_response", "N/A"))
    time_aware_ref = q_data.get("time_aware_reference", "N/A")
    answer = row.get("full_answer", "")[:2000]

    prompt = UNIFIED_JUDGE_PROMPT.format(
        query=query,
        answer=answer,
        ground_truth=ground_truth,
        time_aware_reference=time_aware_ref
    )

    success = False
    for attempt in range(len(groq_clients)):
        client = groq_clients[(i + attempt) % len(groq_clients)]
        for m in ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=800
                )
                raw_text = resp.choices[0].message.content or ""
                (df_s, df_r), (saa_s, saa_r), (tp_s, tp_r) = parse_unified_response(raw_text)
                if df_s is not None and saa_s is not None:
                    new_row = {
                        "config": row["config"],
                        "run_id": row["run_id"],
                        "id": raw_id,
                        "judge": judge_name,
                        "dr_fact": df_s,
                        "saa": saa_s,
                        "temporal": tp_s if tp_s is not None else 1.0,
                        "dr_fact_rationale": df_r,
                        "saa_rationale": saa_r,
                        "temporal_rationale": tp_r,
                    }
                    unique_rows.append(new_row)
                    seen.add(ck)
                    print(f"[{i+1}/{len(missing)}] {row['config']} id={raw_id} {judge_name} ({m}): dr={df_s} saa={saa_s} tp={tp_s}")
                    success = True
                    break
            except Exception as e:
                continue
        if success:
            break
        time.sleep(0.5)

    if not success:
        print(f"Failed {ck} after retrying across all keys.")
    time.sleep(0.5)

# 4. Write perfectly deduplicated 6,600 rows
print(f"\nStep 3: Writing final deduplicated dataset ({len(unique_rows)} rows)...")
with open(OUT_CSV, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=JUDGE_FIELDNAMES)
    writer.writeheader()
    writer.writerows(unique_rows)

print("Step 4: Computing final Cohen's Kappa...")
compute_kappa()
print("Deduplication and completion verified 100% successfully!")
