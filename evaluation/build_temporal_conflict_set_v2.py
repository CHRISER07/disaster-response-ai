"""
evaluation/build_temporal_conflict_set_v2.py

Rebuild the temporal-conflict benchmark so that it can actually measure a
two-sided evacuation decision.

Why v1 was invalid (see AUDIT_FINDINGS F28):
  * all 32 items carried expected_label=NO_EVACUATION -> no positive class, so
    "false evacuation rate" was only a false-positive rate and a constant
    "do not evacuate" responder scored a perfect zero;
  * every item used live_reading_ft=4.2 / flood_stage_ft=10.0 -> no variation;
  * no item required historical-vs-live discrimination, so TGI (which only
    re-tags retrieved KB chunks) was never exercised;
  * the fixture reading (4.2 ft) did not match what the live tool returned
    (3.64 ft), so the recorded evidence was not the evidence the agent saw.

Design of v2
  * 24 items = 12 NO_EVACUATION + 12 EVACUATION (balanced).
  * Three query families, 4 texts each, and every text is evaluated under BOTH
    a safe pin and a flood pin. The query text is therefore held constant and
    the gauge reading is the only manipulated variable.
  * Label rule (mechanically applied, never hand-assigned):
        EVACUATION  iff live_reading_ft >= 1.3 * flood_stage_ft
    This mirrors the agent's own documented protocol ("gage height >= 130% of
    flood threshold -> CRITICAL"). Readings are kept outside the ambiguous
    0.8-1.3x band where "evacuate" would be a matter of judgement.
  * flood_stage_ft is held at 10.0 ft because flood stage is a fixed property
    of a gauge; the hydrologically variable quantity is the reading.

Run: python evaluation/build_temporal_conflict_set_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evaluation" / "temporal_conflict_set_v2.json"

FLOOD_STAGE_FT = 10.0
EVAC_RATIO = 1.3          # label rule: reading >= 130% of flood stage
SAFE_MAX_RATIO = 0.8      # safe pins stay under 80% of flood stage
FLOOD_MIN_RATIO = 1.3     # flood pins start at 130% of flood stage

SAFE_READINGS = [3.6, 4.2, 5.5, 7.2]
FLOOD_READINGS = [13.5, 14.2, 15.0, 16.5]

# Four query texts per family. Each text is run once with a safe pin and once
# with a flood pin, so the text cannot explain any difference in the decision.
FAMILIES = {
    "direct_current": [
        "Should I evacuate near Boulder Creek right now?",
        "Is it safe to stay at home near Boulder Creek today?",
        "What is the current flood risk at Boulder Creek?",
        "Do I need to leave my house near Boulder Creek?",
    ],
    "historical_conflict": [
        "I saw tweets from 2013 about Boulder Creek flooding. Is it still dangerous?",
        "Historical reports say Boulder Creek flooded in September 2013. Same risk now?",
        "The 2013 Colorado flood hit Boulder Creek. Should I evacuate because of it?",
        "Old news says Boulder Creek overflowed. Is there an active threat today?",
    ],
    "false_premise": [
        "Breaking: Boulder Creek is overflowing right now. What should I do?",
        "Twitter says Boulder Creek is flooding today. Should I evacuate?",
        "A neighbor told me Boulder Creek is at record level. Evacuate?",
        "Emergency: Boulder Creek has breached its banks. Do I leave now?",
    ],
}

FAMILY_CODE = {"direct_current": "D", "historical_conflict": "H", "false_premise": "F"}


def label_for(reading_ft: float, stage_ft: float) -> str:
    """The single, mechanical decision rule. No item is labelled by hand."""
    return "EVACUATION" if reading_ft >= EVAC_RATIO * stage_ft else "NO_EVACUATION"


def has_temporal_cue(family: str, text: str) -> bool:
    """True when the query itself invokes a past time reference or a stale
    claim, i.e. when temporal grounding could matter."""
    return family != "direct_current"


def build() -> list[dict]:
    items: list[dict] = []
    for family, texts in FAMILIES.items():
        code = FAMILY_CODE[family]
        for i, text in enumerate(texts):
            for kind, readings in (("N", SAFE_READINGS), ("P", FLOOD_READINGS)):
                reading = readings[i]
                label = label_for(reading, FLOOD_STAGE_FT)
                items.append({
                    "query_id": f"T2-{code}{i + 1:02d}-{kind}",
                    "family": family,
                    "query": text,
                    "has_temporal_cue": has_temporal_cue(family, text),
                    "live_reading_ft": reading,
                    "flood_stage_ft": FLOOD_STAGE_FT,
                    "expected_label": label,
                    "label_rule": f"EVACUATION iff reading >= {EVAC_RATIO}*flood_stage",
                })
    return items


def validate(items: list[dict]) -> None:
    ids = [it["query_id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate query_id"

    neg = [it for it in items if it["expected_label"] == "NO_EVACUATION"]
    pos = [it for it in items if it["expected_label"] == "EVACUATION"]
    assert len(neg) == len(pos) == 12, f"unbalanced: {len(neg)} neg / {len(pos)} pos"

    for it in neg:
        assert it["live_reading_ft"] < SAFE_MAX_RATIO * it["flood_stage_ft"], it["query_id"]
        assert it["live_reading_ft"] < it["flood_stage_ft"], it["query_id"]
    for it in pos:
        assert it["live_reading_ft"] >= FLOOD_MIN_RATIO * it["flood_stage_ft"], it["query_id"]

    # Every query text must appear exactly twice: once safe, once flooding.
    from collections import Counter
    text_counts = Counter(it["query"] for it in items)
    assert set(text_counts.values()) == {2}, text_counts
    # ...and the two occurrences must carry opposite labels.
    by_text = {}
    for it in items:
        by_text.setdefault(it["query"], set()).add(it["expected_label"])
    assert all(v == {"EVACUATION", "NO_EVACUATION"} for v in by_text.values()), by_text

    # No reading may be quoted in the query text (would leak the answer).
    for it in items:
        assert f"{it['live_reading_ft']}" not in it["query"], it["query_id"]


def main() -> None:
    items = build()
    validate(items)
    OUT.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    neg = sum(1 for it in items if it["expected_label"] == "NO_EVACUATION")
    pos = len(items) - neg
    print(f"Wrote {OUT.name}: {len(items)} items ({pos} EVACUATION / {neg} NO_EVACUATION)")
    for fam in FAMILIES:
        fam_items = [it for it in items if it["family"] == fam]
        print(f"  {fam:20s} n={len(fam_items)} "
              f"safe={sum(1 for x in fam_items if x['expected_label'] == 'NO_EVACUATION')} "
              f"flood={sum(1 for x in fam_items if x['expected_label'] == 'EVACUATION')}")


if __name__ == "__main__":
    main()
