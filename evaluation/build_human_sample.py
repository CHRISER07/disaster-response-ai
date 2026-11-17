"""
Phase 4b: Human label sample (60 rows, stratified, identity-blinded).
Run: python evaluation/build_human_sample.py
"""
import os, sys, json, csv, random
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_SAMPLE = ROOT / "evaluation" / "human_label_sample.csv"
OUT_KEY    = ROOT / "evaluation" / "human_label_key.json"

SAMPLE_FIELDNAMES = [
    "blind_id","query","full_answer",
    "human_dr_fact","human_saa","human_safe_yn"
]
KEY_FIELDNAMES = ["blind_id","config","run_id","id","source_file"]


def main():
    dev_csv    = ROOT / "evaluation" / "results_per_query_v2.csv"
    heldout_csv = ROOT / "evaluation" / "results_heldout_v2.csv"

    all_rows = []
    for src_file in [dev_csv, heldout_csv]:
        if not src_file.exists():
            print(f"WARNING: {src_file} not found, skipping.")
            continue
        with open(src_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status","") == "OK":
                    row["_source"] = src_file.name
                    all_rows.append(row)

    if not all_rows:
        print("No OK rows found. Run dev/heldout evals first.")
        return

    # Stratified sample: 10 rows per config (up to 6 configs), shuffle within each
    from collections import defaultdict
    by_config = defaultdict(list)
    for row in all_rows:
        by_config[row["config"]].append(row)

    sample = []
    target_per_config = max(1, 60 // len(by_config))
    for cfg, rows in by_config.items():
        random.seed(42)
        selected = random.sample(rows, min(target_per_config, len(rows)))
        sample.extend(selected)

    # Trim to exactly 60
    random.seed(42)
    random.shuffle(sample)
    sample = sample[:60]

    # Assign blind IDs
    random.seed(99)
    blind_ids = [f"RESP_{i+1:03d}" for i in range(len(sample))]
    random.shuffle(blind_ids)

    key = []
    with open(OUT_SAMPLE, "w", newline="", encoding="utf-8") as sf, \
         open(OUT_KEY, "w", encoding="utf-8") as kf:

        writer = csv.DictWriter(sf, fieldnames=SAMPLE_FIELDNAMES)
        writer.writeheader()

        for bid, row in zip(blind_ids, sample):
            writer.writerow({
                "blind_id": bid,
                "query": row.get("full_answer","")[:50],  # don't put query in answer col
                "full_answer": row.get("full_answer",""),
                "human_dr_fact": "",
                "human_saa": "",
                "human_safe_yn": "",
            })
            key.append({
                "blind_id": bid,
                "config": row["config"],
                "run_id": row["run_id"],
                "id": row["id"],
                "source_file": row["_source"]
            })

        json.dump(key, kf, indent=2, ensure_ascii=False)

    print(f"Human sample: {OUT_SAMPLE}")
    print(f"Key: {OUT_KEY}")
    print(f"Rows sampled: {len(sample)}")


if __name__ == "__main__":
    main()

