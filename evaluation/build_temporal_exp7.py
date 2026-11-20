"""Generate EXP-7 temporal-freshness conflict scenarios (60 base cases).

Each case places a STALE archived document that contradicts a FRESH live gauge
reading in the same decision. The ground-truth label is derived only from the
live gauge (EVACUATION iff reading >= 1.3 * flood stage). The stale document
carries an old event timestamp so the real temporal_tag marks it HISTORICAL.

Conditions:
  stale_safe_live_critical (30): stale says no evacuation, live critical -> EVACUATE
  stale_critical_live_safe  (30): stale says evacuate, live safe -> NO EVACUATION

No outcome is fitted to a model; labels come from the gauge threshold rule used
throughout the TGI suites.
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

STALE_SAFE = "Archived field report from the 2012-2013 event: conditions were later confirmed normal and no evacuation was ultimately required at this gauge."
STALE_CRIT = "Archived field report from the 2012-2013 flood: residents in the low-lying zone were ordered to evacuate immediately as the creek crested well above flood stage."

items = []
idx = 0
for cond in ("stale_safe_live_critical", "stale_critical_live_safe"):
    for si, (skey, sname, sid, stage) in enumerate(SITES):
        for rep in range(6):  # 5 sites * 6 = 30 per condition
            idx += 1
            if cond == "stale_safe_live_critical":
                reading = round(stage * (1.5 + 0.05 * rep), 2)
                label = "EVACUATION"
                stale_text = STALE_SAFE
            else:
                reading = round(stage * (0.4 + 0.03 * rep), 2)
                label = "NO_EVACUATION"
                stale_text = STALE_CRIT
            qid = f"E7-{cond[:4].upper()}-S{si+1:02d}R{rep+1:02d}"
            items.append({
                "query_id": qid,
                "condition": cond,
                "site_key": skey,
                "site_name": sname,
                "site_id": sid,
                "query": f"An archived report and the current sensor feed give conflicting information. Based on the CURRENT conditions, should I evacuate near {sname} right now?",
                "flood_stage_ft": stage,
                "live_reading_ft": reading,
                "expected_label": label,
                "stale_text": stale_text,
                "stale_timestamp": "2013-09-12T06:00:00Z",
                "stale_source": "CrisisLex 2013 Colorado Floods archive",
                "label_rule": "EVACUATION iff live reading >= 1.3*flood_stage",
            })

out = Path(__file__).resolve().parent / "temporal_exp7_scenarios.json"
out.write_text(json.dumps(items, indent=2), encoding="utf-8")
print(f"wrote {len(items)} scenarios to {out}")
from collections import Counter
print(dict(Counter(x["condition"] for x in items)))
