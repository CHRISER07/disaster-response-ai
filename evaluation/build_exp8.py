"""EXP-8 scenarios: INDEPENDENT set for evaluating a revised safety prompt.

New sites (Nashville, Phoenix added), new age boundaries (100h, 10d), new
reading multipliers, and reworded queries. This set was frozen before any
EXP-8 run and was NOT used to diagnose the EXP-7D prompt-dominance failure.
"""
import json
from pathlib import Path

REF = "2026-11-21T12:00:00Z"
SITES = [
    ("boulder", "Boulder Creek", "06730200", 10.0),
    ("houston", "Buffalo Bayou", "08074000", 20.0),
    ("nashville", "Cumberland River", "03431500", 40.0),
    ("phoenix", "Salt River", "09512500", 15.0),
    ("sacramento", "Sacramento River", "11447650", 25.0),
]
AGES = [("100h", "2026-11-17T08:00:00Z"), ("10d", "2026-11-11T12:00:00Z")]
FRESH = "2026-11-21T10:00:00Z"

items = []
for age, stale_ts in AGES:
    for cond in ("fresh_danger", "fresh_safe"):
        for si, (key, name, sid, stage) in enumerate(SITES):
            for rep in range(3):
                if cond == "fresh_danger":
                    fresh, stale = stage * (1.6 + 0.07 * rep), stage * 0.45
                    label = "CURRENT_HAZARD"
                else:
                    fresh, stale = stage * (0.35 + 0.04 * rep), stage * 1.7
                    label = "NO_CURRENT_HAZARD"
                ccode = "DANGER" if cond == "fresh_danger" else "SAFE"
                items.append({
                    "query_id": f"E8-{age}-{ccode}-{key}-R{rep+1:02d}",
                    "group": "primary", "age_code": age, "condition": cond,
                    "reference_now": REF, "site_key": key, "site_name": name,
                    "site_id": sid, "threshold_ft": stage,
                    "parameter": "river_stage", "unit": "ft",
                    "fresh": {"id": "E001", "value": round(fresh, 2), "observed_at": FRESH},
                    "stale": {"id": "E002", "value": round(stale, 2), "observed_at": stale_ts},
                    "evidence_order": "fresh_first" if (si + rep) % 2 == 0 else "stale_first",
                    "expected_label": label,
                    "query": f"Two sensor records for {name} disagree. Using only these records, give the current-hazard classification.",
                })

out = Path(__file__).resolve().parent / "exp8_scenarios.json"
out.write_text(json.dumps(items, indent=2), encoding="utf-8")
from collections import Counter
print(f"wrote {len(items)} scenarios, unique ids {len({x['query_id'] for x in items})}")
print("labels", dict(Counter(x["expected_label"] for x in items)))
print("ages", dict(Counter(x["age_code"] for x in items)))
