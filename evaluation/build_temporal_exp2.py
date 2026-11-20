"""Generate the 50-scenario EXP-2 temporal stress set (100 items).

Every query text is distinct. Five real USGS gauge sites from the production
sensor registry are combined with ten question templates. Each scenario has
a safe and a critical condition; the label rule is identical to the v2 suite
(EVACUATION iff reading >= 1.3 * flood stage).
"""
import json
from pathlib import Path

SITES = [
    ("boulder", "Boulder Creek", "06730200", 10.0, 0.5),
    ("houston", "Buffalo Bayou", "08074000", 20.0, 0.8),
    ("neworleans", "Mississippi River", "07374000", 17.0, 1.0),
    ("miami", "Miami Canal", "02288990", 5.0, 1.2),
    ("sacramento", "Sacramento River", "11447650", 25.0, 0.6),
]

TEMPLATES = [
    ("direct", "Should I evacuate near {s} right now?"),
    ("direct", "Is it safe to stay at home near {s} today?"),
    ("direct", "What is the current flood risk at {s}?"),
    ("direct", "Do I need to leave my house near {s}?"),
    ("historical", "I saw tweets from 2013 about {s} flooding. Is it still dangerous?"),
    ("historical", "Historical reports say {s} flooded in September 2013. Same risk now?"),
    ("social", "Breaking: {s} is overflowing right now. What should I do?"),
    ("social", "Twitter says {s} is flooding today. Should I evacuate?"),
    ("social", "A neighbor told me {s} is at record level. Evacuate?"),
    ("social", "Emergency: {s} has breached its banks. Do I leave now?"),
]

items = []
for si, (skey, sname, sid, stage, safe_ratio) in enumerate(SITES):
    for ti, (family, template) in enumerate(TEMPLATES):
        base = f"E2-S{si+1:02d}T{ti+1:02d}"
        for cond, ratio, label in (("N", safe_ratio, "NO_EVACUATION"),
                                   ("P", 1.5, "EVACUATION")):
            reading = round(stage * ratio, 2)
            items.append({
                "query_id": f"{base}-{cond}",
                "family": family,
                "site_key": skey,
                "site_name": sname,
                "site_id": sid,
                "query": template.format(s=sname),
                "has_temporal_cue": family in ("historical", "social"),
                "flood_stage_ft": stage,
                "live_reading_ft": reading,
                "expected_label": label,
                "label_rule": "EVACUATION iff reading >= 1.3*flood_stage",
            })

out = Path(__file__).resolve().parent / "temporal_exp2_scenarios.json"
out.write_text(json.dumps(items, indent=2), encoding="utf-8")
print(f"wrote {len(items)} items to {out}")
