"""EXP-2b: boundary-condition suite, safe readings strictly below flood stage.

EXP-2 used safe ratios up to 1.2x flood stage. Those readings are labelled
DANGER by the production sensor tool's own severity logic, so NO_EVACUATION
ground truth was ambiguous there. This separate suite asks a distinct,
preregistered question: with readings strictly below flood stage (<=0.8x),
do either arms issue evacuation directives? Critical readings stay at 1.5x
and the label rule is unchanged. This does not replace EXP-2; both are kept.
"""
import json
from pathlib import Path

SITES = [
    ("boulder", "Boulder Creek", "06730200", 10.0, 0.4),
    ("houston", "Buffalo Bayou", "08074000", 20.0, 0.8),
    ("neworleans", "Mississippi River", "07374000", 17.0, 0.5),
    ("miami", "Miami Canal", "02288990", 5.0, 0.7),
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
        base = f"E2B-S{si+1:02d}T{ti+1:02d}"
        for cond, ratio, label in (("N", safe_ratio, "NO_EVACUATION"),
                                   ("P", 1.5, "EVACUATION")):
            items.append({
                "query_id": f"{base}-{cond}", "family": family,
                "site_key": skey, "site_name": sname, "site_id": sid,
                "query": template.format(s=sname),
                "has_temporal_cue": family in ("historical", "social"),
                "flood_stage_ft": stage,
                "live_reading_ft": round(stage * ratio, 2),
                "expected_label": label,
                "label_rule": "EVACUATION iff reading >= 1.3*flood_stage",
            })
out = Path(__file__).resolve().parent / "temporal_exp2b_scenarios.json"
out.write_text(json.dumps(items, indent=2), encoding="utf-8")
print(f"wrote {len(items)} items to {out}")
