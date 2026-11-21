"""EXP-4b: geospatial generalization to an independent state.

Runs the IDENTICAL GeoSpatialRouter logic and Haversine baseline used in EXP-4,
but on a second state's USGS stations. Geographic holdout: EXP-4 was configured
on Colorado; this evaluation uses a different state and shares no stations or
basins with the Colorado working pool. No production code is modified.

Outputs: evaluation/geospatial_generalize_results.{json,csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import fisher_exact, binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import core.geo_router as geo_mod
from core.geo_router import GeoSpatialRouter, IncidentZone, KNOWN_INCIDENT_ZONES

CACHE_DIR = ROOT / "data"
OUT_JSON = ROOT / "evaluation" / "geospatial_generalize_results.json"
OUT_CSV = ROOT / "evaluation" / "geospatial_generalize_results.csv"


def fetch(state, refresh=False):
    cache = CACHE_DIR / f"usgs_{state.lower()}_stations_raw.rdb"
    url = f"https://waterservices.usgs.gov/nwis/site/?stateCd={state}&format=rdb"
    if cache.exists() and not refresh:
        return cache.read_text(encoding="utf-8", errors="replace"), {
            "source": "cache", "url": url, "cache_file": str(cache.relative_to(ROOT))}
    req = urllib.request.Request(url, headers={"User-Agent": "disaster-rag-eval/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8", errors="replace")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text, {"source": "live_api", "url": url,
                  "fetched_utc": datetime.now(timezone.utc).isoformat(),
                  "cache_file": str(cache.relative_to(ROOT))}


def parse(text):
    out, header = [], None
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if parts[0] == "agency_cd" and "site_no" in parts:
            header = parts; continue
        if header and re.match(r"^\d+s$", parts[0]):
            continue
        if header and len(parts) >= len(header):
            row = dict(zip(header, parts))
            site, huc = row.get("site_no", "").strip(), row.get("huc_cd", "").strip()
            try:
                lat, lon = float(row.get("dec_lat_va", "").strip()), float(row.get("dec_long_va", "").strip())
            except ValueError:
                continue
            if not site or not huc or not re.match(r"^\d{8,12}$", huc):
                continue
            out.append({"site_no": site, "name": row.get("station_nm", "").strip(),
                        "lat": lat, "lon": lon, "basin": huc[:10], "huc_level": len(huc)})
    return out


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return (round(max(0,c-m)*100,1), round(min(1,c+m)*100,1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="UT")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--max-cases", type=int, default=93)
    args = ap.parse_args()

    text, meta = fetch(args.state, args.refresh)
    stations = parse(text)
    print(f"{args.state}: {len(stations)} usable stations (source={meta['source']})")
    if len(stations) < 60:
        raise SystemExit(f"only {len(stations)} usable stations; refusing to pad")
    rng = __import__("random").Random(42)
    pool = list(stations)
    rng.shuffle(pool)
    pool = pool[:400]

    reg = {f"USGS_{s['site_no']}": {"name": s["name"], "lat": s["lat"],
           "lon": s["lon"], "watershed": s["basin"], "huc_level": s["huc_level"]}
           for s in pool}
    orig_reg = dict(geo_mod.WATERSHED_REGISTRY)
    orig_zones = dict(KNOWN_INCIDENT_ZONES)
    geo_mod.WATERSHED_REGISTRY.clear(); geo_mod.WATERSHED_REGISTRY.update(reg)

    cases = []
    for i, inc in enumerate(pool):
        zkey = f"INC_{inc['site_no']}"
        KNOWN_INCIDENT_ZONES[zkey] = IncidentZone(zkey, inc["name"], inc["lat"],
                                                   inc["lon"], inc["basin"], radius_km=25.0)
        for ev in pool:
            if ev["site_no"] == inc["site_no"]: continue
            d = GeoSpatialRouter.haversine_distance_km(inc["lat"], inc["lon"], ev["lat"], ev["lon"])
            if d > 25: continue
            same = inc["basin"] == ev["basin"]
            cls, expect = ("same_basin_local" if d <= 5 else "same_basin_near", "accept") if same else ("cross_watershed", "reject")
            cases.append({"incident": inc["site_no"], "evidence": ev["site_no"],
                          "distance_km": round(d,3), "class": cls, "expect": expect,
                          "incident_basin": inc["basin"], "evidence_basin": ev["basin"]})
        if len(cases) >= args.max_cases: break

    for c in cases:
        rel, _, _ = GeoSpatialRouter.evaluate_spatial_relevance(
            None, None, incident_zone_key=f"INC_{c['incident']}",
            station_id=f"USGS_{c['evidence']}")
        c["router_accept"] = rel.value in ("DIRECT_LOCAL","CATCHMENT_RELEVANT","DISTANCE_LOCAL","DISTANCE_NEAR")
        c["baseline_accept"] = c["distance_km"] <= 25.0
        c["truth_accept"] = c["expect"] == "accept"
        c["router_correct"] = c["router_accept"] == c["truth_accept"]
        c["baseline_correct"] = c["baseline_accept"] == c["truth_accept"]

    geo_mod.WATERSHED_REGISTRY.clear(); geo_mod.WATERSHED_REGISTRY.update(orig_reg)
    KNOWN_INCIDENT_ZONES.clear(); KNOWN_INCIDENT_ZONES.update(orig_zones)

    results = {}
    for method in ("router","baseline"):
        n = len(cases); k = sum(c[f"{method}_correct"] for c in cases)
        tp = sum(c[f"{method}_accept"] and c["truth_accept"] for c in cases)
        fp = sum(c[f"{method}_accept"] and not c["truth_accept"] for c in cases)
        results[method] = {"n": n, "accuracy": k/n, "ci95_pct": wilson(k,n), "TP": tp, "FP": fp}

    cross = [c for c in cases if c["class"] == "cross_watershed"]
    a = sum(c["router_correct"] for c in cross); b = len(cross)-a
    cc = sum(c["baseline_correct"] for c in cross); d = len(cross)-cc
    odds, pf = fisher_exact([[a,b],[cc,d]])
    mc = binomtest(a, a+cc, 0.5).pvalue if a+cc else None
    cross_stats = {"n": len(cross), "router_correct": a, "baseline_correct": cc,
                   "fisher_p": pf, "mcnemar_p": mc,
                   "router_ci": wilson(a,len(cross)), "baseline_ci": wilson(cc,len(cross))}

    out = {"metadata": {**meta, "state": args.state, "stations_total": len(stations),
                        "stations_pool": len(pool),
                        "generated_utc": datetime.now(timezone.utc).isoformat(),
                        "note": "geographic holdout: independent state, no CO stations/basins"},
           "cases": cases, "overall": results, "cross_watershed": cross_stats}
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
        w.writeheader(); w.writerows(cases)
    print("overall", json.dumps(results, indent=2))
    print("cross", json.dumps(cross_stats, indent=2))
    print("saved", OUT_JSON, OUT_CSV)


if __name__ == "__main__":
    main()
