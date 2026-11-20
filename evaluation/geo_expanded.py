"""EXP-4: Expanded geospatial routing evaluation using real USGS stations.

Fetches Colorado NWIS sites (cached offline), builds a test registry and
incident zones at runtime, and evaluates the real GeoSpatialRouter against a
Haversine-only baseline. No production code is modified and no data is invented.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core.geo_router as geo_mod
from core.geo_router import GeoSpatialRouter, IncidentZone, KNOWN_INCIDENT_ZONES

URL = "https://waterservices.usgs.gov/nwis/site/?stateCd=CO&format=rdb"
CACHE = ROOT / "data" / "usgs_co_stations_raw.rdb"
OUT_JSON = ROOT / "evaluation" / "geospatial_expanded_results.json"
OUT_CSV = ROOT / "evaluation" / "geospatial_expanded_results.csv"


def fetch_raw(refresh: bool = False) -> tuple[str, dict]:
    if CACHE.exists() and not refresh:
        text = CACHE.read_text(encoding="utf-8", errors="replace")
        meta = {"source": "cache", "cache_file": str(CACHE.relative_to(ROOT)),
                "url": URL}
        return text, meta
    req = urllib.request.Request(URL, headers={"User-Agent": "disaster-rag-eval/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8", errors="replace")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(text, encoding="utf-8")
    meta = {"source": "live_api", "url": URL,
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "cache_file": str(CACHE.relative_to(ROOT))}
    return text, meta


def parse_rdb(text: str) -> list[dict]:
    stations = []
    header = None
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if parts[0] == "agency_cd" and "site_no" in parts:
            header = parts
            continue
        if header and re.match(r"^\d+s$", parts[0]):
            continue
        if header and len(parts) >= len(header):
            row = dict(zip(header, parts))
            site = row.get("site_no", "").strip()
            huc = row.get("huc_cd", "").strip()
            try:
                lat = float(row.get("dec_lat_va", "").strip())
                lon = float(row.get("dec_long_va", "").strip())
            except ValueError:
                continue
            if not site or not huc or not re.match(r"^\d{8,12}$", huc):
                continue
            stations.append({"site_no": site, "name": row.get("station_nm", "").strip(),
                             "lat": lat, "lon": lon, "huc": huc,
                             "basin": huc[:10], "huc_level": len(huc)})
    return stations


def build_registry(stations):
    reg = {}
    for s in stations:
        reg[f"USGS_{s['site_no']}"] = {
            "name": s["name"], "lat": s["lat"], "lon": s["lon"],
            "watershed": s["basin"], "huc_level": s["huc_level"],
        }
    return reg


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0, c - m) * 100, 1), round(min(1, c + m) * 100, 1))


def geomedian(lat, lon, pairs):
    """Pick a central station from candidate pairs for stable zones."""
    return pairs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--max-cases", type=int, default=80)
    args = ap.parse_args()

    text, meta = fetch_raw(args.refresh)
    stations = parse_rdb(text)
    print(f"stations parsed: {len(stations)} (source={meta['source']})")
    if len(stations) < 60:
        raise SystemExit("fewer than 60 usable stations; refusing to pad data")

    # Use a spatial subset around the Front Range to keep pairs realistic.
    front = [s for s in stations if 39.4 <= s["lat"] <= 40.5 and -106.2 <= s["lon"] <= -104.8]
    print(f"front-range stations: {len(front)}")
    pool = front if len(front) >= 40 else stations
    # Cap the working pool so the O(n^2) pair scan stays bounded and deterministic.
    pool = pool[:400]
    print(f"working pool capped at: {len(pool)}")

    reg = build_registry(pool)
    original_reg = dict(geo_mod.WATERSHED_REGISTRY)
    original_zones = dict(KNOWN_INCIDENT_ZONES)
    geo_mod.WATERSHED_REGISTRY.clear()
    geo_mod.WATERSHED_REGISTRY.update(reg)

    cases = []
    # Generate pairs deterministically by scanning station combinations.
    for i, inc in enumerate(pool):
        zone_key = f"INC_{inc['site_no']}"
        zone = IncidentZone(zone_key, inc["name"], inc["lat"], inc["lon"],
                            inc["basin"], radius_km=25.0)
        KNOWN_INCIDENT_ZONES[zone_key] = zone
        for j, ev in enumerate(pool):
            if i == j:
                continue
            d = GeoSpatialRouter.haversine_distance_km(inc["lat"], inc["lon"], ev["lat"], ev["lon"])
            same = inc["basin"] == ev["basin"]
            if d <= 25 and same:
                cls = "same_basin_local" if d <= 5 else "same_basin_near"
                expect = "accept"
            elif d <= 25 and not same:
                cls, expect = "cross_watershed", "reject"
            elif d > 25:
                continue
            else:
                continue
            cases.append({"incident": inc["site_no"], "evidence": ev["site_no"],
                          "distance_km": round(d, 3), "class": cls, "expect": expect,
                          "incident_basin": inc["basin"], "evidence_basin": ev["basin"]})
        if len(cases) >= args.max_cases:
            break

    # Evaluate router and Haversine baseline per case.
    for c in cases:
        st_id = f"USGS_{c['evidence']}"
        zone_key = f"INC_{c['incident']}"
        rel, _, _ = GeoSpatialRouter.evaluate_spatial_relevance(
            None, None, incident_zone_key=zone_key, station_id=st_id)
        c["router_result"] = rel.value
        c["router_accept"] = rel in ("DIRECT_LOCAL", "CATCHMENT_RELEVANT",
                                     "DISTANCE_LOCAL", "DISTANCE_NEAR")
        c["baseline_accept"] = c["distance_km"] <= 25.0
        c["truth_accept"] = c["expect"] == "accept"
        c["router_correct"] = c["router_accept"] == c["truth_accept"]
        c["baseline_correct"] = c["baseline_accept"] == c["truth_accept"]

    # Restore production state.
    geo_mod.WATERSHED_REGISTRY.clear()
    geo_mod.WATERSHED_REGISTRY.update(original_reg)
    KNOWN_INCIDENT_ZONES.clear()
    KNOWN_INCIDENT_ZONES.update(original_zones)

    # Metrics.
    results = {}
    for method in ("router", "baseline"):
        accept = [c[f"{method}_accept"] for c in cases]
        correct = [c[f"{method}_correct"] for c in cases]
        tp = sum(a and t for a, t in zip(accept, [c["truth_accept"] for c in cases]))
        fp = sum(a and not t for a, t in zip(accept, [c["truth_accept"] for c in cases]))
        fn = sum((not a) and t for a, t in zip(accept, [c["truth_accept"] for c in cases]))
        tn = sum((not a) and not t for a, t in zip(accept, [c["truth_accept"] for c in cases]))
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        far = fp / (fp + tn) if fp + tn else 0
        results[method] = {"n": len(cases), "accuracy": sum(correct) / len(cases),
                           "ci95_pct": wilson(sum(correct), len(cases)),
                           "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                           "sensitivity": sens, "specificity": spec, "far": far}

    cross = [c for c in cases if c["class"] == "cross_watershed"]
    cross_stats = {}
    for method in ("router", "baseline"):
        correct = sum(c[f"{method}_correct"] for c in cross)
        cross_stats[method] = {"n": len(cross), "accuracy": correct / len(cross) if cross else 0,
                               "ci95_pct": wilson(correct, len(cross))}
    # Fisher exact on paired correctness (router vs baseline) over cross cases.
    try:
        from scipy.stats import fisher_exact
        a = sum(c["router_correct"] for c in cross)
        b = len(cross) - a
        c_ = sum(c["baseline_correct"] for c in cross)
        d = len(cross) - c_
        odds, p = fisher_exact([[a, b], [c_, d]])
        cross_stats["fisher_exact"] = {"odds_ratio": odds, "p_value": p,
                                        "table": [[a, b], [c_, d]]}
    except Exception as exc:
        cross_stats["fisher_exact"] = {"error": str(exc)}

    out = {"metadata": {**meta, "stations_total": len(stations),
                        "stations_pool": len(pool),
                        "generated_utc": datetime.now(timezone.utc).isoformat(),
                        "basin_key": "huc_cd[:10]; fallback to HUC-8 where source has 8 digits"},
           "cases": cases, "overall": results, "cross_watershed": cross_stats}
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
        w.writeheader()
        w.writerows(cases)
    print("overall", json.dumps(results, indent=2))
    print("cross", json.dumps(cross_stats, indent=2))
    print("saved", OUT_JSON, OUT_CSV)


if __name__ == "__main__":
    main()
