"""
api/routes/eval_routes.py

Serves evaluation results and alert log as JSON so the Streamlit dashboard
can fetch them from the API instead of reading files directly.

Endpoints:
  GET /eval/results      - latest results_per_query.csv as list of dicts
  GET /eval/summary      - aggregated per-config summary
  GET /eval/alerts       - latest alerts.jsonl as list of dicts
  DELETE /eval/alerts    - clear alerts.jsonl
"""
import os
import sys
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException

router = APIRouter()

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_CSV   = os.path.join(PROJECT_ROOT, "evaluation", "results_per_query.csv")
from core.audit_logger import ALERT_LOG_FILE as ALERTS_FILE, LEGACY_ALERTS_FILE as LEGACY_ALERTS_FILE

if not os.path.exists(ALERTS_FILE) and os.path.exists(LEGACY_ALERTS_FILE):
    ALERTS_FILE = LEGACY_ALERTS_FILE


@router.get("/results")
async def get_eval_results():
    """
    Returns the latest per-query evaluation results as a JSON array.
    Each element is one row from results_per_query.csv.
    Returns an empty list if evaluations have not been run yet.
    """
    if not os.path.exists(RESULTS_CSV):
        return {"results": [], "count": 0, "message": "No evaluation results found. Run evaluation/run_evals.py first."}
    try:
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return {"results": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read results CSV: {e}")


@router.get("/summary")
async def get_eval_summary():
    """
    Returns aggregated per-config summary statistics from the results CSV.
    Computes mean KW-F1, DR-Fact, SAA per configuration.
    """
    if not os.path.exists(RESULTS_CSV):
        return {"summary": {}, "message": "No evaluation results found."}
    try:
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        from collections import defaultdict
        by_config: dict[str, list] = defaultdict(list)
        for row in rows:
            by_config[row["config"]].append(row)

        summary = {}
        for cfg, cfg_rows in by_config.items():
            def mean(field):
                vals = [float(r[field]) for r in cfg_rows if r.get(field)]
                return round(sum(vals) / len(vals), 4) if vals else 0.0

            summary[cfg] = {
                "n":          len(cfg_rows),
                "kw_f1":      mean("kw_f1"),
                "dr_fact":    mean("dr_fact"),
                "saa":        mean("saa"),
                "p_at_k":     mean("p_at_k"),
                "latency_s":  mean("latency_s"),
                "svc_total":  sum(int(r.get("svc", 0)) for r in cfg_rows),
            }
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize results: {e}")


@router.get("/alerts")
async def get_alerts():
    """
    Returns the latest alerts from alerts.jsonl as a JSON array.
    Returns most recent 50 alerts, newest first.
    """
    if not os.path.exists(ALERTS_FILE):
        return {"alerts": [], "count": 0}
    try:
        alerts = []
        with open(ALERTS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        alerts.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        alerts = list(reversed(alerts[-50:]))
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read alerts: {e}")


@router.delete("/alerts")
async def clear_alerts():
    """Clears the alerts log. Idempotent - returns 200 even if file doesn't exist."""
    if os.path.exists(ALERTS_FILE):
        os.remove(ALERTS_FILE)
    return {"status": "cleared"}
