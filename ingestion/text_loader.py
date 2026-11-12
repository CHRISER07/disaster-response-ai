"""
ingestion/text_loader.py  (v2 - All 26 CrisisLex T26 Events)

Loads social media tweets from ALL 26 CrisisLex T26 disaster events into
LangChain Document objects with rich, per-event metadata tags.

Key changes from v1:
  - Scans all 26 event sub-directories (not just Colorado floods)
  - Tags each document with: event_name, disaster_type, region, year
  - Reads event_description.json for geo/temporal metadata
  - Per-event document cap (max_per_event) prevents any single event from
    dominating the knowledge base
  - Snowflake timestamp reconstruction applied to all events

Column format (CrisisLex T26 labeled CSV):
    Tweet ID | Tweet Text | Information Source | Information Type | Informativeness
"""

import os
import csv
import json
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document

# === Constants ===
TWITTER_EPOCH    = 1288834974657   # ms since epoch (Nov 4, 2010 01:42:54 UTC)
INFORMATIVE_LABEL = "Related and informative"

T26_ROOT = Path(__file__).parent.parent / "data" / "text" / "CrisisLexT26" / "CrisisLexT26"

# Maps event folder name  disaster_type and region (for ChromaDB metadata)
EVENT_METADATA = {
    "2013_Colorado_floods":         {"disaster_type": "Flood",            "region": "US"},
    "2013_Alberta_floods":          {"disaster_type": "Flood",            "region": "Canada"},
    "2012_Philipinnes_floods":      {"disaster_type": "Flood",            "region": "Philippines"},
    "2013_Queensland_floods":       {"disaster_type": "Flood",            "region": "Australia"},
    "2013_Sardinia_floods":         {"disaster_type": "Flood",            "region": "Europe"},
    "2013_Manila_floods":           {"disaster_type": "Flood",            "region": "Philippines"},
    "2012_Colorado_wildfires":      {"disaster_type": "Wildfire",         "region": "US"},
    "2013_Australia_bushfire":      {"disaster_type": "Wildfire",         "region": "Australia"},
    "2012_Costa_Rica_earthquake":   {"disaster_type": "Earthquake",       "region": "Latin America"},
    "2012_Guatemala_earthquake":    {"disaster_type": "Earthquake",       "region": "Latin America"},
    "2012_Italy_earthquakes":       {"disaster_type": "Earthquake",       "region": "Europe"},
    "2013_Bohol_earthquake":        {"disaster_type": "Earthquake",       "region": "Philippines"},
    "2012_Typhoon_Pablo":           {"disaster_type": "Typhoon",          "region": "Philippines"},
    "2013_Typhoon_Yolanda":         {"disaster_type": "Typhoon",          "region": "Philippines"},
    "2013_Savar_building_collapse": {"disaster_type": "Building_Collapse","region": "Bangladesh"},
    "2012_Venezuela_refinery":      {"disaster_type": "Chemical",         "region": "Latin America"},
    "2013_West_Texas_explosion":    {"disaster_type": "Chemical",         "region": "US"},
    "2013_Singapore_haze":          {"disaster_type": "Air_Quality",      "region": "Asia"},
    "2013_Russia_meteor":           {"disaster_type": "Meteor_Impact",    "region": "Russia"},
    "2013_Boston_bombings":         {"disaster_type": "Man_Made",         "region": "US"},
    "2013_Brazil_nightclub_fire":   {"disaster_type": "Fire",             "region": "Latin America"},
    "2013_Glasgow_helicopter_crash":{"disaster_type": "Aviation",         "region": "Europe"},
    "2013_LA_airport_shootings":    {"disaster_type": "Man_Made",         "region": "US"},
    "2013_Lac_Megantic_train_crash":{"disaster_type": "Chemical_Fire",    "region": "Canada"},
    "2013_NY_train_crash":          {"disaster_type": "Transit",          "region": "US"},
    "2013_Spain_train_crash":       {"disaster_type": "Transit",          "region": "Europe"},
}


# === Snowflake decoder ===
def snowflake_to_datetime(tweet_id) -> datetime | None:
    """
    Decodes a Twitter Snowflake ID into a precise UTC datetime.
    Right-shifts 22 bits (removes machine ID + sequence), then adds Twitter epoch.
    """
    try:
        tweet_id_int = int(str(tweet_id).replace('"', "").strip())
        timestamp_ms = (tweet_id_int >> 22) + TWITTER_EPOCH
        return datetime.utcfromtimestamp(timestamp_ms / 1000.0)
    except Exception:
        return None


def _read_event_description(event_folder: Path) -> dict:
    """Reads event_description.json if present; returns {} on failure."""
    for desc_file in event_folder.glob("*-event_description.json"):
        try:
            with open(desc_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# === Single-event loader ===
def load_event_tweets(event_folder: Path, max_per_event: int = 300) -> list[Document]:
    """
    Loads informative tweets from a single CrisisLex event folder.

    Args:
        event_folder:   Path to one event folder (e.g., .../2013_Colorado_floods/)
        max_per_event:  Maximum number of 'Related and informative' docs to keep.
                        Default 300 prevents any single event from dominating.

    Returns:
        List of LangChain Document objects with rich metadata.
    """
    event_name = event_folder.name
    meta       = EVENT_METADATA.get(event_name, {"disaster_type": "Unknown", "region": "Unknown"})
    desc       = _read_event_description(event_folder)

    # Year from description JSON or event folder name
    year = desc.get("time", {}).get("year", int(event_name[:4]) if event_name[:4].isdigit() else 0)
    location_desc = desc.get("location", {}).get("location_description", meta["region"])

    # Temporal tag for the Temporal Context Bridge
    temporal_tag = f"[HISTORICAL -- {year}]"

    # Find the labeled CSV
    csv_files = list(event_folder.glob("*-tweets_labeled.csv"))
    if not csv_files:
        print(f"  [WARN] No labeled CSV in {event_name}")
        return []

    documents  = []
    row_count  = 0
    skip_count = 0

    with open(csv_files[0], "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalise column names (CrisisLex uses leading spaces)
            row = {k.strip(): v.strip() for k, v in row.items()}

            informativeness = row.get("Informativeness", "")
            if INFORMATIVE_LABEL not in informativeness:
                skip_count += 1
                continue

            tweet_text = row.get("Tweet Text", "").strip()
            if not tweet_text or len(tweet_text) < 10:
                skip_count += 1
                continue

            tweet_id   = row.get("Tweet ID", "")
            info_type  = row.get("Information Type", "Unknown")
            source_cat = row.get("Information Source", "Unknown")

            # Reconstruct UTC timestamp from Snowflake ID
            tweet_dt = snowflake_to_datetime(tweet_id)
            time_str = tweet_dt.isoformat() + "Z" if tweet_dt else f"{year}-01-01T00:00:00Z"

            # Pre-pend temporal tag (Temporal Context Bridge)
            content = f"{temporal_tag} [{event_name}] [{info_type}] {tweet_text}"

            doc = Document(
                page_content=content,
                metadata={
                    "modality":      "text",
                    "source":        "CrisisLex_T26",
                    "event_name":    event_name,
                    "disaster_type": meta["disaster_type"],
                    "region":        location_desc,
                    "year":          str(year),
                    "timestamp":     time_str,
                    "info_type":     info_type,
                    "source_category": source_cat,
                    "tweet_id":      str(tweet_id),
                    "temporal_tag":  temporal_tag,
                }
            )
            documents.append(doc)
            row_count += 1

            if row_count >= max_per_event:
                break

    return documents


# === All-events loader (primary entry point) ===
def load_all_crisis_tweets(max_per_event: int = 300,
                            include_types: list[str] | None = None) -> list[Document]:
    """
    Loads informative tweets from ALL 26 CrisisLex T26 disaster events.

    Args:
        max_per_event:  Cap on documents per event (default 300).
        include_types:  If provided, only load events of these disaster_type values.
                        E.g., ["Flood", "Typhoon"]. None = load all types.

    Returns:
        Combined list of Document objects tagged with per-event metadata.
    """
    if not T26_ROOT.exists():
        print(f"[ERROR] CrisisLex T26 root not found: {T26_ROOT}")
        return []

    event_folders = sorted([d for d in T26_ROOT.iterdir()
                             if d.is_dir() and not d.name.startswith(".")])

    all_docs = []
    type_counts = {}

    for folder in event_folders:
        event_name = folder.name
        if event_name in ("__MACOSX",):
            continue

        meta = EVENT_METADATA.get(event_name, {"disaster_type": "Unknown", "region": "Unknown"})

        # Optional type filter
        if include_types and meta["disaster_type"] not in include_types:
            continue

        docs = load_event_tweets(folder, max_per_event=max_per_event)
        all_docs.extend(docs)

        dtype = meta["disaster_type"]
        type_counts[dtype] = type_counts.get(dtype, 0) + len(docs)

        print(f"  [{event_name}] -> {len(docs)} informative docs  "
              f"(type={meta['disaster_type']}, region={meta['region']})")

    # Summary
    print(f"\n  === CrisisLex T26 Ingestion Summary ===")
    print(f"  Events processed : {len(event_folders)}")
    print(f"  Total documents  : {len(all_docs):,}")
    for dtype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {dtype:<25} {cnt:>5} docs")

    return all_docs


# === Legacy single-event loader (backward-compatible) ===
def load_crisis_tweets(csv_path: str, max_documents: int = 1500) -> list[Document]:
    """
    Backward-compatible loader for a single CSV path.
    Used by old code that points directly to the Colorado floods CSV.
    """
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        print(f"[WARN] Tweet CSV not found: {csv_path}")
        return []
    event_folder = csv_path_obj.parent
    return load_event_tweets(event_folder, max_per_event=max_documents)


# === CLI test ===
if __name__ == "__main__":
    print("Loading all 26 CrisisLex T26 events...\n")
    docs = load_all_crisis_tweets(max_per_event=50)   # small cap for test
    print(f"\nTotal documents loaded: {len(docs)}")
    if docs:
        sample = docs[0]
        print(f"\nSample document:")
        print(f"  Content  : {sample.page_content[:120]}...")
        print(f"  Metadata : {sample.metadata}")
