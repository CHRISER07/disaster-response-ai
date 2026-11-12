import json
from collections import Counter

with open("evaluation/golden_dataset.json") as f:
    data = json.load(f)

total = len(data)
by_scenario = Counter(d["scenario"] for d in data)
by_type = Counter(d["eval_type"] for d in data)

print(f"Total queries: {total}")
print()
print("--- By Scenario ---")
for s, c in sorted(by_scenario.items(), key=lambda x: -x[1]):
    pct = c / total * 100
    print(f"  {s:25s}: {c:3d} ({pct:.0f}%)")

print()
print("--- By Eval Type ---")
for t, c in by_type.items():
    print(f"  {t:20s}: {c}")

print()
print("--- Behavioral Queries (no fixed ground truth) ---")
for d in data:
    if d["eval_type"] == "behavioral":
        print(f"  Q{d['id']:02d}: [{d['scenario']}] {d['query'][:65]}")

print()
print("--- New non-Colorado queries (id >= 51) ---")
for d in data:
    if d["id"] >= 51:
        print(f"  Q{d['id']}: {d['query'][:65]}")

print()
print("--- Has ground_truth_answer=null (behavioral, correct) ---")
nulls = [d for d in data if d.get("ground_truth_answer") is None]
print(f"  {len(nulls)} queries have null ground truth (all should be behavioral)")
wrong = [d for d in nulls if d.get("eval_type") != "behavioral"]
if wrong:
    print(f"  WARNING: {len(wrong)} null-GT queries are NOT marked behavioral!")
else:
    print("  All null-GT queries correctly marked as behavioral. OK.")
