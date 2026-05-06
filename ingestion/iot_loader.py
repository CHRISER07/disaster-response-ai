"""
ingestion/iot_loader.py

Real USGS water sensor ingestion — no synthetic data.
Fetches recent instantaneous values for known flood gauge sites and stores
WARNING/CRITICAL readings as embedded documents for the RAG knowledge base.

Fix log:
  - FIXED: os.makedirs(os.path.dirname(csv_path)) crashes when csv_path has no
    directory component (dirname returns ""). Added guard for empty string.
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from langchain_core.documents import Document

USGS_SITE_ID = "06730200"       # Boulder Creek at N 75th St, CO (primary reference site)
DANGER_THRESHOLD_FT = 10.0


def fetch_usgs_historical(site_id: str = USGS_SITE_ID, days_back: int = 7) -> pd.DataFrame:
    """
    Fetches historical gage height readings from USGS NWIS for the given site.
    Returns a DataFrame of timestamped readings with status classification.
    Falls back to an empty DataFrame on API failure (not an exception).
    """
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    params = {
        "sites":       site_id,
        "parameterCd": "00065",
        "startDT":     start.strftime("%Y-%m-%dT%H:%M+00:00"),
        "endDT":       end.strftime("%Y-%m-%dT%H:%M+00:00"),
        "format":      "json"
    }

    try:
        resp = requests.get(
            "https://waterservices.usgs.gov/nwis/iv/",
            params=params,
            timeout=15
        )
        resp.raise_for_status()
        ts_data = resp.json().get("value", {}).get("timeSeries", [])

        if not ts_data:
            print(f"  [IoT] No time series data returned for site {site_id}.")
            return pd.DataFrame()

        readings  = ts_data[0]["values"][0]["value"]
        site_name = ts_data[0]["sourceInfo"]["siteName"]

        records = []
        for r in readings:
            try:
                val = float(r["value"])
                records.append({
                    "timestamp":       r["dateTime"],
                    "sensor_id":       site_id,
                    "site_name":       site_name,
                    "gage_height_ft":  val,
                    "status": (
                        "CRITICAL_FLOOD" if val >= DANGER_THRESHOLD_FT * 1.2 else
                        "DANGER"         if val >= DANGER_THRESHOLD_FT          else
                        "WARNING"        if val >= DANGER_THRESHOLD_FT * 0.8   else
                        "NORMAL"
                    )
                })
            except (ValueError, KeyError):
                continue

        df = pd.DataFrame(records)
        print(f"  [IoT] Fetched {len(df)} real USGS readings for {site_name}")
        return df

    except requests.RequestException as e:
        print(f"  [IoT] USGS API error — {e}. Will use cached CSV if available.")
        return pd.DataFrame()


def load_iot_data(csv_path: str) -> list[Document]:
    """
    Loads USGS sensor data into LangChain Documents.

    Strategy:
      1. Try to fetch fresh data from USGS live API (7 days of readings)
      2. Fall back to cached CSV if API is unreachable
      3. Embed last 24 readings as current context (any status)
      4. Embed top 10 WARNING/CRITICAL peaks as historical flood context

    Args:
        csv_path: Path to the sensor cache CSV file (auto-created/updated).
    """
    # 1. Try live API
    df = fetch_usgs_historical()

    if not df.empty:
        # FIX: Guard against dirname("") which raises FileNotFoundError
        parent_dir = os.path.dirname(csv_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  [IoT] Saved real USGS data to {csv_path}")

    elif os.path.exists(csv_path):
        print(f"  [IoT] API unavailable — using cached data from {csv_path}")
        df = pd.read_csv(csv_path)

    else:
        print(
            "  [IoT] No data available — USGS API unreachable and no cached CSV found.\n"
            "  Ensure network connectivity and try running populate_db.py again."
        )
        return []

    documents = []

    # (a) Last 24 readings → current conditions context
    recent_df = df.tail(24)
    for _, row in recent_df.iterrows():
        site  = row.get("site_name", row["sensor_id"])
        gage  = row["gage_height_ft"]
        status = row["status"]
        text = (
            f"SENSOR READING [{status}]: {site} (USGS #{row['sensor_id']}) "
            f"recorded gage height of {gage:.2f} ft at {row['timestamp']}. "
            f"Flood danger threshold is {DANGER_THRESHOLD_FT} ft. "
            f"Current river status: {status}."
        )
        documents.append(Document(
            page_content=text,
            metadata={
                "source":     "USGS_Sensor",
                "sensor_id":  str(row["sensor_id"]),
                "status":     status,
                "modality":   "telemetry",
                "timestamp":  str(row["timestamp"]),
            }
        ))

    # (b) Historical WARNING/CRITICAL peaks — flood pattern context
    alert_df = df[df["status"] != "NORMAL"]
    peak_docs_count = 0
    if not alert_df.empty:
        peak_df = alert_df.nlargest(10, "gage_height_ft")
        for _, row in peak_df.iterrows():
            site = row.get("site_name", row["sensor_id"])
            text = (
                f"HISTORICAL FLOOD ALERT [{row['status']}]: {site} "
                f"(USGS #{row['sensor_id']}) reached a peak of {row['gage_height_ft']:.2f} ft "
                f"at {row['timestamp']}. Flood danger threshold: {DANGER_THRESHOLD_FT} ft."
            )
            documents.append(Document(
                page_content=text,
                metadata={
                    "source":    "USGS_Sensor",
                    "sensor_id": str(row["sensor_id"]),
                    "status":    row["status"],
                    "modality":  "telemetry",
                    "timestamp": str(row["timestamp"]),
                }
            ))
        peak_docs_count = len(peak_df)

    print(
        f"  [IoT] Embedded {len(recent_df)} recent readings "
        f"+ {peak_docs_count} historical peaks = {len(documents)} total sensor documents."
    )
    return documents
