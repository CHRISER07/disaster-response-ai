"""EXP-7D: frozen controlled temporal freshness benchmark.

Builds 80 scenario instances with a UNIFIED evidence fixture: two records for
the same station, parameter, unit, source authority and threshold. No severity
or precomputed decision fields. Evidence IDs are stable:
  E001 = fresh record, E002 = stale record.
Evidence presentation order is assigned at scenario level and is identical
across all three arms.

Labels derive only from the fresh reading against the threshold.
"""
import json
from pathlib import Path

REFERENCE_NOW = "2026-11-21T12:00:00Z"
SITES = [
    ("boulder", "Boulder Creek", "06730200", 10.0),
    ("houston", "Buffalo Bayou", "08074000", 20.0),
    ("neworleans", "Mississippi River", "07374000", 17.0),
    ("miami", "Miami Canal", "02288990", 5.0),
    ("sacramento", "Sacramento River", "11447650", 25.0),
]
AGES = [
    ("80h", "2026-11-18T04:00:00Z"),
    ("8d", "2026-11-13T12:00:00Z"),
]
FRESH_TS = "2026-11-21T10:00:00Z"

items = []
order_toggle = 0

def add(group, age, site, cond, fresh, stale, stale_ts, fresh_label, note=""):
    global order_toggle
    skey, sname, sid, stage = site
    order = "fresh_first" if order_toggle % 2 == 0 else "stale_first"
    order_toggle += 1
    items.append({
        "query_id": f"E7D-{group}-{age}-{cond}-{skey}-R{len([x for x in items if x['site_key']==skey and x['condition']==cond])+1:02d}",
        "group": group, "age_code": age, "condition": cond,
        "reference_now": REFERENCE_NOW,
        "site_key": skey, "site_name": sname, "site_id": sid,
        "threshold_ft": stage, "parameter": "river_stage", "unit": "ft",
        "fresh": {"id": "E001", "value": round(fresh, 2), "observed_at": FRESH_TS},
        "stale": {"id": "E002", "value": round(stale, 2), "observed_at": stale_ts},
        "evidence_order": order,
        "expected_label": fresh_label,
        "note": note,
    })

# PRIMARY: 60 conflict scenarios (30 + 30)
for age, stale_ts in AGES:
    for cond in ("fresh_danger_stale_safe", "fresh_safe_stale_danger"):
        for si, site in enumerate(SITES):
            for rep in range(3):
                stage = site[3]
                if cond == "fresh_danger_stale_safe":
                    fresh, stale = stage * (1.5 + 0.05 * rep), stage * 0.5
                    label = "CURRENT_HAZARD"
                else:
                    fresh, stale = stage * (0.4 + 0.03 * rep), stage * 1.6
                    label = "NO_CURRENT_HAZARD"
                add("primary", age, site, cond, fresh, stale, stale_ts, label)

# DIAGNOSTIC: 20 cases. TGI uses t_source only, so these are reported separately.
for cond in ("delayed_publication", "expired_warning"):
    for si, site in enumerate(SITES):
        for rep in range(2):
            stage = site[3]
            if cond == "delayed_publication":
                fresh, stale = stage * 0.5, stage * 1.6
                label = "NO_CURRENT_HAZARD"  # old event described in a recent article
            else:
                fresh, stale = stage * 0.5, stage * 1.6
                label = "NO_CURRENT_HAZARD"  # warning validity ended
            add("diagnostic", "8d", site, cond, fresh, stale,
                "2013-09-12T06:00:00Z", label,
                note=("published recently but describes a 2013 event" if cond == "delayed_publication"
                      else "advisory validity period has expired"))

out = Path(__file__).resolve().parent / "exp7d_scenarios.json"
out.write_text(json.dumps(items, indent=2), encoding="utf-8")
from collections import Counter
print(f"wrote {len(items)} scenarios, unique ids {len({x['query_id'] for x in items})}")
print("groups", dict(Counter(x["group"] for x in items)))
print("labels", dict(Counter(x["expected_label"] for x in items)))
print("orders", dict(Counter(x["evidence_order"] for x in items)))
