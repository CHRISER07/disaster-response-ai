"""EXP-7b: boundary-age temporal freshness benchmark.

EXP-7a used a 13-year-old archive that all arms rejected trivially (100%
each), so it could not discriminate TGI. This suite targets the actual TGI
decision boundary in core/temporal_grounding.py: 72 hours and 7 days.

The stale document is given an age just OUTSIDE each tier boundary and states
the OPPOSITE of a fresh live gauge:
  age 80 hours  -> RECENT tier (just outside 72h LIVE window)
  age 8 days    -> HISTORICAL tier (just outside 7-day RECENT window)
  age 2013      -> far HISTORICAL (control, as in EXP-7a)

TGI-tagged arms should reject the tier-labelled stale claim; No-TGI strips all
tags; Timestamp_Only hides the tier label but keeps the raw source date. The
only manipulated variable is the temporal grounding signal.

A future-dated clock-skew condition is included as a fourth age.

Labels always derive from the live gauge threshold (EVACUATION iff reading >=
1.3*flood stage), independent of any model output.
"""
import json
from pathlib import Path

SITES = [
    ("boulder", "Boulder Creek", "06730200", 10.0),
    ("houston", "Buffalo Bayou", "08074000", 20.0),
    ("neworleans", "Mississippi River", "07374000", 17.0),
    ("miami", "Miami Canal", "02288990", 5.0),
    ("sacramento", "Sacramento River", "11447650", 25.0),
]

REFERENCE_NOW = "2026-11-21T12:00:00Z"
AGES = [
    ("80h",  "2026-11-18T04:00:00Z", "80 hours old, just outside the 72-hour live window."),
    ("8d",   "2026-11-13T12:00:00Z", "8 days old, just outside the 7-day recent window."),
    ("2013", "2013-09-12T06:00:00Z", "Archived 2013 flood report."),
    ("future", "2026-11-25T12:00:00Z", "Clock-skewed future-dated sensor snapshot."),
]

STALE_SAFE = "Sensor archive from the incident window: conditions were reported normal and no evacuation was ordered at this location."
STALE_CRIT = "Sensor archive from the incident window: residents were ordered to evacuate immediately as the creek exceeded flood stage."

items = []
for age_code, ts, age_desc in AGES:
    for cond in ("live_critical", "live_safe"):
        for si, (skey, sname, sid, stage) in enumerate(SITES):
            for rep in range(3):  # 4 ages * 2 conditions * 5 sites * 3 = 120
                if cond == "live_critical":
                    reading = round(stage * (1.5 + 0.05 * rep), 2)
                    label = "EVACUATION"
                    stale_text = STALE_SAFE   # stale says safe, live critical
                else:
                    reading = round(stage * (0.4 + 0.03 * rep), 2)
                    label = "NO_EVACUATION"
                    stale_text = STALE_CRIT   # stale says evacuate, live safe
                cond_code = "CRIT" if cond == "live_critical" else "SAFE"
                qid = f"E7B-{age_code}-{cond_code}-S{si+1:02d}R{rep+1:02d}"
                items.append({
                    "query_id": qid,
                    "age_code": age_code,
                    "reference_now": REFERENCE_NOW,
                    "stale_timestamp": ts,
                    "age_description": age_desc,
                    "condition": cond,
                    "site_key": skey, "site_name": sname, "site_id": sid,
                    "query": f"An archived sensor report and the current sensor feed disagree. Based ONLY on current conditions, should I evacuate near {sname} right now?",
                    "flood_stage_ft": stage, "live_reading_ft": reading,
                    "expected_label": label, "stale_text": stale_text,
                    "stale_source": "sensor archive snapshot",
                    "label_rule": "EVACUATION iff live reading >= 1.3*flood_stage",
                })

out = Path(__file__).resolve().parent / "temporal_exp7b_scenarios.json"
out.write_text(json.dumps(items, indent=2), encoding="utf-8")
print(f"wrote {len(items)} scenarios to {out}")
from collections import Counter
print("ages", dict(Counter(x["age_code"] for x in items)))
print("unique ids", len({x["query_id"] for x in items}))
