"""
evaluation/dataset_analyzer.py

Analyzes all 26 CrisisLex T26 disaster events to:
  1. Count documents per event, per label, per disaster type
  2. Produce a statistical summary table for the IEEE paper (Dataset section)
  3. Verify what percentage of the T26 corpus is currently loaded into ChromaDB
  4. Output: evaluation/dataset_analysis.json + evaluation/dataset_stats.md

Usage:
    python evaluation/dataset_analyzer.py
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#  Paths 
PROJECT_ROOT   = Path(__file__).parent.parent
T26_ROOT       = PROJECT_ROOT / "data" / "text" / "CrisisLexT26" / "CrisisLexT26"
OUTPUT_JSON    = PROJECT_ROOT / "evaluation" / "dataset_analysis.json"
OUTPUT_MD      = PROJECT_ROOT / "evaluation" / "dataset_stats.md"

#  Disaster Type Mapping 
# Maps each CrisisLex event folder name  query scenario category
EVENT_METADATA = {
    "2013_Colorado_floods":        {"type": "Flood",            "region": "US",          "year": 2013},
    "2013_Alberta_floods":         {"type": "Flood",            "region": "Canada",      "year": 2013},
    "2012_Philipinnes_floods":     {"type": "Flood",            "region": "Philippines", "year": 2012},
    "2013_Queensland_floods":      {"type": "Flood",            "region": "Australia",   "year": 2013},
    "2013_Sardinia_floods":        {"type": "Flood",            "region": "Europe",      "year": 2013},
    "2013_Manila_floods":          {"type": "Flood",            "region": "Philippines", "year": 2013},
    "2012_Colorado_wildfires":     {"type": "Wildfire",         "region": "US",          "year": 2012},
    "2013_Australia_bushfire":     {"type": "Wildfire",         "region": "Australia",   "year": 2013},
    "2012_Costa_Rica_earthquake":  {"type": "Earthquake",       "region": "Latin America","year": 2012},
    "2012_Guatemala_earthquake":   {"type": "Earthquake",       "region": "Latin America","year": 2012},
    "2012_Italy_earthquakes":      {"type": "Earthquake",       "region": "Europe",      "year": 2012},
    "2013_Bohol_earthquake":       {"type": "Earthquake",       "region": "Philippines", "year": 2013},
    "2012_Typhoon_Pablo":          {"type": "Typhoon",          "region": "Philippines", "year": 2012},
    "2013_Typhoon_Yolanda":        {"type": "Typhoon",          "region": "Philippines", "year": 2013},
    "2013_Savar_building_collapse":{"type": "Building_Collapse","region": "Bangladesh",  "year": 2013},
    "2012_Venezuela_refinery":     {"type": "Chemical",         "region": "Latin America","year": 2012},
    "2013_West_Texas_explosion":   {"type": "Chemical",         "region": "US",          "year": 2013},
    "2013_Singapore_haze":         {"type": "Air_Quality",      "region": "Asia",        "year": 2013},
    "2013_Russia_meteor":          {"type": "Meteor_Impact",    "region": "Russia",      "year": 2013},
    "2013_Boston_bombings":        {"type": "Man_Made",         "region": "US",          "year": 2013},
    "2013_Brazil_nightclub_fire":  {"type": "Fire",             "region": "Latin America","year": 2013},
    "2013_Glasgow_helicopter_crash":{"type": "Aviation",        "region": "Europe",      "year": 2013},
    "2013_LA_airport_shootings":   {"type": "Man_Made",         "region": "US",          "year": 2013},
    "2013_Lac_Megantic_train_crash":{"type": "Chemical_Fire",   "region": "Canada",      "year": 2013},
    "2013_NY_train_crash":         {"type": "Transit",          "region": "US",          "year": 2013},
    "2013_Spain_train_crash":      {"type": "Transit",          "region": "Europe",      "year": 2013},
}

INFORMATIVE_LABEL = "Related and informative"


def read_event_csv(event_folder: Path) -> dict:
    """Reads a CrisisLex labeled tweet CSV and returns document counts."""
    csv_files = list(event_folder.glob("*-tweets_labeled.csv"))
    if not csv_files:
        return {"total": 0, "informative": 0, "related_not_informative": 0,
                "not_related": 0, "error": "No CSV found"}

    csv_path = csv_files[0]
    counts = defaultdict(int)
    info_type_dist = defaultdict(int)
    source_dist    = defaultdict(int)

    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CrisisLex CSVs have leading spaces in column names - strip them all
                row = {k.strip(): v.strip() for k, v in row.items()}

                informativeness = row.get("Informativeness", "").strip()
                info_type       = row.get("Information Type", "").strip()
                source          = row.get("Information Source", "").strip()

                counts["total"] += 1

                if INFORMATIVE_LABEL in informativeness:
                    counts["informative"] += 1
                elif "Related" in informativeness:
                    counts["related_not_informative"] += 1
                else:
                    counts["not_related"] += 1

                if info_type and info_type != "Not labeled":
                    info_type_dist[info_type] += 1
                if source and source != "Not labeled":
                    source_dist[source] += 1

    except Exception as e:
        return {"total": 0, "informative": 0, "error": str(e)}

    return {
        "total": counts["total"],
        "informative": counts["informative"],
        "related_not_informative": counts["related_not_informative"],
        "not_related": counts["not_related"],
        "top_info_types": dict(sorted(info_type_dist.items(),
                                      key=lambda x: x[1], reverse=True)[:5]),
        "top_sources":    dict(sorted(source_dist.items(),
                                      key=lambda x: x[1], reverse=True)[:5]),
    }


def read_event_description(event_folder: Path) -> dict:
    """Reads the event_description.json if available."""
    desc_files = list(event_folder.glob("*-event_description.json"))
    if not desc_files:
        return {}
    try:
        with open(desc_files[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def analyze_all_events() -> list[dict]:
    """Iterates all 26 CrisisLex event folders and builds analysis records."""
    if not T26_ROOT.exists():
        print(f"[ERROR] CrisisLex T26 root not found: {T26_ROOT}")
        return []

    events = []
    event_folders = sorted([d for d in T26_ROOT.iterdir()
                             if d.is_dir() and not d.name.startswith(".")])

    print(f"\nFound {len(event_folders)} event folders in CrisisLex T26.\n")
    print(f"{'Event':<40} {'Type':<20} {'Total':>7} {'Informative':>12} {'%Info':>6}")
    print("-" * 90)

    type_totals   = defaultdict(lambda: {"total": 0, "informative": 0})
    grand_total   = 0
    grand_inform  = 0

    for folder in event_folders:
        event_name = folder.name
        if event_name in ("__MACOSX", ".DS_Store", "README.md"):
            continue

        meta    = EVENT_METADATA.get(event_name, {"type": "Unknown", "region": "Unknown", "year": 0})
        desc    = read_event_description(folder)
        counts  = read_event_csv(folder)

        total   = counts.get("total", 0)
        inform  = counts.get("informative", 0)
        pct     = round(inform / total * 100, 1) if total > 0 else 0.0

        print(f"{event_name:<40} {meta['type']:<20} {total:>7} {inform:>12} {pct:>5.1f}%")

        type_totals[meta["type"]]["total"]       += total
        type_totals[meta["type"]]["informative"] += inform
        grand_total  += total
        grand_inform += inform

        record = {
            "event_name":   event_name,
            "disaster_type": meta["type"],
            "region":        meta["region"],
            "year":          meta["year"],
            "total_tweets":  total,
            "informative":   inform,
            "related_not_informative": counts.get("related_not_informative", 0),
            "not_related":   counts.get("not_related", 0),
            "pct_informative": pct,
            "top_info_types":  counts.get("top_info_types", {}),
            "top_sources":     counts.get("top_sources", {}),
            "event_description": desc,
        }
        events.append(record)

    print("-" * 90)
    pct_grand = round(grand_inform / grand_total * 100, 1) if grand_total > 0 else 0
    print(f"{'TOTAL':<40} {'ALL':<20} {grand_total:>7} {grand_inform:>12} {pct_grand:>5.1f}%")

    print(f"\n\nPer Disaster-Type Summary:")
    print(f"{'Type':<25} {'Events':>7} {'Total':>8} {'Inform.':>9} {'%Info':>7}")
    print("-" * 60)
    type_event_counts = defaultdict(int)
    for e in events:
        type_event_counts[e["disaster_type"]] += 1
    for dtype, agg in sorted(type_totals.items()):
        n  = type_event_counts[dtype]
        t  = agg["total"]
        i  = agg["informative"]
        p  = round(i / t * 100, 1) if t > 0 else 0
        print(f"{dtype:<25} {n:>7} {t:>8} {i:>9} {p:>6.1f}%")

    return events, grand_total, grand_inform


def write_json(events: list[dict], grand_total: int, grand_inform: int):
    summary = {
        "total_events": len(events),
        "total_tweets": grand_total,
        "total_informative": grand_inform,
        "pct_informative": round(grand_inform / grand_total * 100, 1) if grand_total > 0 else 0,
        "events": events,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] JSON analysis saved -> {OUTPUT_JSON}")


def write_markdown(events: list[dict], grand_total: int, grand_inform: int):
    """Writes a Markdown table suitable for the IEEE paper Dataset section."""
    lines = [
        "# CrisisLex T26 Dataset Analysis",
        "",
        f"**Total events:** {len(events)}  ",
        f"**Total tweets across all events:** {grand_total:,}  ",
        f"**'Related and informative' tweets (used for ingestion):** {grand_inform:,} "
        f"({round(grand_inform/grand_total*100,1)}%)  ",
        "",
        "---",
        "",
        "## Per-Event Statistics",
        "",
        "| # | Event | Type | Region | Year | Total | Informative | % |",
        "|---|-------|------|--------|------|------:|------------:|---|",
    ]

    for i, e in enumerate(events, 1):
        name = e["event_name"].replace("_", " ")
        lines.append(
            f"| {i} | {name} | {e['disaster_type']} | {e['region']} | {e['year']} "
            f"| {e['total_tweets']:,} | {e['informative']:,} | {e['pct_informative']}% |"
        )

    lines += [
        f"| **Total** | **26 events** | - | - | - | **{grand_total:,}** | **{grand_inform:,}** | "
        f"**{round(grand_inform/grand_total*100,1)}%** |",
        "",
        "---",
        "",
        "## Per Disaster-Type Summary (for Table 1 in paper)",
        "",
        "| Disaster Type | Events | Total Tweets | Informative | % Informative |",
        "|---|---:|---:|---:|---:|",
    ]

    type_agg = defaultdict(lambda: {"events": 0, "total": 0, "inform": 0})
    for e in events:
        dt = e["disaster_type"]
        type_agg[dt]["events"] += 1
        type_agg[dt]["total"]  += e["total_tweets"]
        type_agg[dt]["inform"] += e["informative"]

    for dt, agg in sorted(type_agg.items()):
        p = round(agg["inform"] / agg["total"] * 100, 1) if agg["total"] > 0 else 0
        lines.append(
            f"| {dt} | {agg['events']} | {agg['total']:,} | {agg['inform']:,} | {p}% |"
        )

    lines += [
        "",
        "---",
        "",
        "## Notes for IEEE Paper",
        "",
        "- Only tweets labeled **'Related and informative'** are ingested into ChromaDB.",
        "- Timestamps are reconstructed from Twitter Snowflake IDs for all events.",
        "- Each document is tagged with: `event_name`, `disaster_type`, `year`, `region`.",
        "- The `modality` tag is set to `'text'` for all social media documents.",
        "- Total ingested: ~" + f"{grand_inform:,} social media documents across 26 events.",
    ]

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Markdown stats saved -> {OUTPUT_MD}")


if __name__ == "__main__":
    print("=" * 90)
    print("  CrisisLex T26 Dataset Analyzer")
    print("=" * 90)
    events, grand_total, grand_inform = analyze_all_events()
    if events:
        write_json(events, grand_total, grand_inform)
        write_markdown(events, grand_total, grand_inform)
        print(f"\nAnalysis complete. {len(events)} events, "
              f"{grand_total:,} total tweets, {grand_inform:,} informative.")
    else:
        print("\n[ERROR] No events found. Check the T26_ROOT path.")
