import csv, math, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "evaluation" / "tgi_results_v2.csv"
STATUS_OK = "OK"

ACTIVE_EVAC_PATTERNS = [
    r"\bevacuate immediately\b",
    r"\bmandatory evacuation (?:is )?ordered\b",
    r"\bevacuation order (?:is )?(?:issued|active|effective|in effect)\b",
    r"\bimmediate(?:ly)? evacuate\b",
    r"\bmust leave (?:the area|immediately)\b",
    r"\bimmediate evacuation\b",
    r"\bevacuate to high(?:er)? ground\b",
    r"\border(?:ed)?(?: a)? mandatory evacuation\b",
    r"\binitiate (?:immediate )?evacuation now\b",
]

def is_active_evacuation(text: str) -> bool:
    if not text:
        return False
    text_l = text.lower()
    
    # 1. Check Status line
    status_match = re.search(r"\bstatus:\s*(\w+)", text_l)
    if status_match:
        status_val = status_match.group(1)
        if status_val in ("normal", "low", "safe"):
            return False
            
    # 2. Extract Recommendation section
    rec_section = text_l
    rec_match = re.search(r"\brecommendation:\s*(.*?)(?:\balert generated|\Z)", text_l, re.DOTALL)
    if rec_match:
        rec_section = rec_match.group(1)
        
    first_recs = rec_section.split("\n")[:3]
    first_block = " ".join(first_recs)
    if re.search(r"\b(no\s+(?:mandatory\s+)?evacuation|safe to remain|do not evacuate|no need to evacuate|shelter in place)\b", first_block):
        return False
        
    for pat in ACTIVE_EVAC_PATTERNS:
        if re.search(pat, rec_section):
            m = re.search(pat, rec_section)
            start = max(0, m.start() - 30)
            context = rec_section[start:m.end()]
            if not re.search(r"\b(no|not|don't|do not|never|unnecessary)\b", context):
                return True
                
    return False


def analyze():
    if not CSV_PATH.exists():
        print("CSV file does not exist yet.")
        return

    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\nLoaded {len(rows)} rows from {CSV_PATH.name}")

    # Re-evaluate false evacuations using calibrated ground truth
    for r in rows:
        ans = r.get("full_answer", "")
        is_evac = is_active_evacuation(ans) if r.get("status") == STATUS_OK else False
        live_ft = float(r.get("live_reading_ft", 4.2))
        flood_ft = float(r.get("flood_stage_ft", 10.0))
        r["calibrated_false_evac"] = 1 if (is_evac and live_ft < flood_ft) else 0
        r["calibrated_label"] = "EVACUATION" if is_evac else "NO_EVACUATION"

    print("\n" + "=" * 65)
    print("CALIBRATED TGI EVALUATION & TEMPORAL CONFLICT ANALYSIS")
    print("=" * 65)

    for cfg_name in ["Full_Agent_TGI", "Full_Agent_NoTGI"]:
        cfg_rows = [r for r in rows if r["config"] == cfg_name and r["status"] == STATUS_OK]
        if not cfg_rows:
            print(f"{cfg_name:<16}: No completed OK rows yet.")
            continue

        n = len(cfg_rows)
        false_evac = sum(r["calibrated_false_evac"] for r in cfg_rows)
        p = false_evac / n if n else 0
        z = 1.96
        lo = max(0.0, (p + z*z/(2*n) - z*((p*(1-p)/n + z*z/(4*n*n))**0.5)) / (1 + z*z/n)) if n else 0.0
        hi = min(1.0, (p + z*z/(2*n) + z*((p*(1-p)/n + z*z/(4*n*n))**0.5)) / (1 + z*z/n)) if n else 0.0
        print(f"{cfg_name:<16}: n={n}, false_evac={false_evac} ({p:.1%}, Wilson 95% CI [{lo:.1%}, {hi:.1%}])")

        with_cue = [r for r in cfg_rows if str(r["has_temporal_cue"]).lower() == "true"]
        no_cue = [r for r in cfg_rows if str(r["has_temporal_cue"]).lower() == "false"]
        if with_cue:
            wc_fe = sum(r["calibrated_false_evac"] for r in with_cue)
            print(f"  - With temporal cue : {wc_fe}/{len(with_cue)} ({wc_fe/len(with_cue):.1%}) false evac")
        if no_cue:
            nc_fe = sum(r["calibrated_false_evac"] for r in no_cue)
            print(f"  - No temporal cue   : {nc_fe}/{len(no_cue)} ({nc_fe/len(no_cue):.1%}) false evac")

    # McNemar's test
    tgi_rows = {(r["run_id"], r["query_id"]): r["calibrated_false_evac"]
                for r in rows if r["config"] == "Full_Agent_TGI" and r["status"] == STATUS_OK}
    notgi_rows = {(r["run_id"], r["query_id"]): r["calibrated_false_evac"]
                  for r in rows if r["config"] == "Full_Agent_NoTGI" and r["status"] == STATUS_OK}
    common = set(tgi_rows) & set(notgi_rows)
    if len(common) >= 4:
        b = sum(1 for k in common if tgi_rows[k] == 0 and notgi_rows[k] == 1)
        c = sum(1 for k in common if tgi_rows[k] == 1 and notgi_rows[k] == 0)
        if b + c > 0:
            chi2 = (abs(b - c) - 1)**2 / (b + c)
            sig = "p < 0.001" if chi2 > 10.83 else ("p < 0.01" if chi2 > 6.63 else ("p < 0.05" if chi2 > 3.84 else "p >= 0.05 (NS)"))
            print(f"\nMcNemar's Paired Test (n={len(common)} pairs):")
            print(f"  TGI=Safe & NoTGI=FalseEvac (b) = {b}")
            print(f"  TGI=FalseEvac & NoTGI=Safe (c) = {c}")
            print(f"  Chi-squared (with Edwards continuity correction) = {chi2:.3f} ({sig})")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    analyze()
