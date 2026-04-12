"""
core/populate_db.py

Master data ingestion pipeline. Loads all real modalities into ChromaDB.
Run this once (or with --reset) before starting the dashboard.

Usage:
  python core/populate_db.py           # Incremental — adds only new documents
  python core/populate_db.py --reset   # Wipes and repopulates from scratch
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ingestion.text_loader import load_crisis_tweets
from ingestion.iot_loader import load_iot_data
from ingestion.drone_loader import load_drone_data
from core.knowledge_loader import load_and_chunk_knowledge_base
from core.vector_store import add_documents_to_store, get_vector_store, clear_collection

TEXT_CSV = os.path.join("data", "text", "CrisisLexT26", "CrisisLexT26",
                        "2013_Colorado_floods", "2013_Colorado_floods-tweets_labeled.csv")
IOT_CSV  = os.path.join("data", "iot", "water_levels.csv")
DRONE_JSON = os.path.join("data", "imagery", "real_drone_metadata.json")
KB_DIR   = os.path.join("data", "kb")


def populate(reset: bool = False):
    if reset:
        print("==> Resetting vector database...")
        clear_collection()
    
    all_docs = []
    counters = {}

    # 1. CrisisLex Tweets (real social media data)
    print("\n[1/4] Loading CrisisLex tweets...")
    if os.path.exists(TEXT_CSV):
        tweets = load_crisis_tweets(TEXT_CSV, max_documents=500)
        all_docs.extend(tweets)
        counters["text"] = len(tweets)
    else:
        print(f"  WARNING: Tweet CSV not found at {TEXT_CSV}")
        counters["text"] = 0

    # 2. USGS IoT sensors (real live data from API)
    print("\n[2/4] Loading USGS sensor data (live API)...")
    iot_docs = load_iot_data(IOT_CSV)
    all_docs.extend(iot_docs)
    counters["telemetry"] = len(iot_docs)

    # 3. Drone/Satellite VLM descriptions (real Ollama LLaVA output)
    print("\n[3/4] Loading VLM satellite image descriptions...")
    if os.path.exists(DRONE_JSON):
        drone_docs = load_drone_data(DRONE_JSON)
        all_docs.extend(drone_docs)
        counters["imagery"] = len(drone_docs)
    else:
        print(f"  WARNING: {DRONE_JSON} not found.")
        print("  Run: python ingestion/extract_xview2_subset.py")
        print("  Then: python ingestion/vlm_processor.py")
        counters["imagery"] = 0

    # 4. FEMA/CDC Knowledge Base PDFs
    print("\n[4/4] Processing knowledge base PDFs...")
    kb_docs = load_and_chunk_knowledge_base(KB_DIR)
    all_docs.extend(kb_docs)
    counters["knowledge_base"] = len(kb_docs)

    print(f"\n{'='*50}")
    print(f"INGESTION SUMMARY")
    print(f"{'='*50}")
    for modality, count in counters.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {modality:20s}: {count:,} documents")
    print(f"  {'TOTAL':20s}: {len(all_docs):,} documents")
    print(f"{'='*50}")

    if not all_docs:
        print("ERROR: No documents to ingest. Check your data sources.")
        return

    print(f"\nEmbedding and indexing {len(all_docs):,} documents into ChromaDB...")
    add_documents_to_store(all_docs)

    # Verify
    store = get_vector_store()
    final_count = store._collection.count()
    print(f"\n✓ Vector store now contains {final_count:,} documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate the disaster response vector database.")
    parser.add_argument("--reset", action="store_true", help="Wipe and repopulate from scratch")
    args = parser.parse_args()
    populate(reset=args.reset)
