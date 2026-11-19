"""
evaluation/audit_outputs.py

Continuous quality audit of every generated evaluation answer.

Status counts alone are not enough: a run can be "OK" yet emit markdown,
empty prose, leaked tool plumbing, a missing operational template, or an
out-of-range metric. This script checks the actual text and scores, writes a
machine-readable report, and exits non-zero if any check fails.

Run:
    python evaluation/audit_outputs.py [csv_path]
    python evaluation/audit_outputs.py --watch    # audit on a 60 s interval
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.answer_formatter import TEMPLATE_SECTIONS
from evaluation.run_dev_eval_v2 import KW_SCORER_VERSION

DEFAULT_CSV = ROOT / "evaluation" / "results_heldout_fresh.csv"
REPORT = ROOT / "evaluation" / "output_audit.json"
LEAK_MARKERS = (
    "tool call validation failed", "no json output", "json object",
    "requires two parameters", "<tool_call>", "traceback (most recent call last)",
    "[sensor tool error]", "unexpected usgs api response format",
)
AGENT_PREFIX = "Full_Agent"


def audit(path: Path) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    problems: list[dict] = []
    by_config: dict[str, dict] = {}

    def flag(row, kind, detail):
        problems.append({"config": row.get("config"), "id": row.get("id"),
                         "run_id": row.get("run_id"), "kind": kind,
                         "detail": detail[:200]})

    seen = set()
    for row in rows:
        key = (row.get("config"), row.get("run_id"), row.get("id"))
        if key in seen:
            flag(row, "duplicate_key", str(key))
        seen.add(key)

        ans = row.get("full_answer", "") or ""
        status = row.get("status", "")
        if status != "OK":
            flag(row, "bad_status", status)
        if not ans.strip():
            flag(row, "empty_answer", "")
        if "*" in ans:
            flag(row, "markdown_asterisk", f"count={ans.count('*')}")
        low = ans.lower()
        for marker in LEAK_MARKERS:
            if marker in low:
                flag(row, "leaked_plumbing", marker)
                break
        if row.get("config", "").startswith(AGENT_PREFIX):
            missing = [s for s in TEMPLATE_SECTIONS if s not in ans]
            if missing:
                flag(row, "missing_template", ",".join(missing))
        try:
            f1 = float(row.get("kw_f1") or 0)
            if not 0.0 <= f1 <= 1.0:
                flag(row, "kw_f1_out_of_range", str(f1))
        except (TypeError, ValueError):
            if status == "OK":
                flag(row, "kw_f1_not_numeric", str(row.get("kw_f1")))

    models = sorted({r.get("model_used", "") for r in rows})
    if len(models) > 1:
        problems.append({"kind": "mixed_generators", "detail": str(models)})

    # Coverage and provenance: PASS must mean a complete, single-generator,
    # single-scorer artifact, not merely "no problem in the rows that exist".
    expected_configs = {
        "Full_Agent", "Full_Agent_NoTGI", "Full_Agent_NoDSEL",
        "Full_Agent_NoTelem", "LLM_Only", "BM25_Baseline", "Standard_StaticRAG",
    }
    expected_rows = len(expected_configs) * 150
    if len(rows) != expected_rows:
        problems.append({"kind": "wrong_row_count",
                         "detail": f"{len(rows)} != {expected_rows}"})
    cfg_counts = Counter(r.get("config", "") for r in rows)
    for cfg in expected_configs:
        if cfg_counts.get(cfg, 0) != 150:
            problems.append({"kind": "wrong_config_count",
                             "detail": f"{cfg}={cfg_counts.get(cfg, 0)} != 150"})
    unexpected = set(cfg_counts) - expected_configs
    if unexpected:
        problems.append({"kind": "unexpected_config", "detail": str(sorted(unexpected))})

    held = set(json.loads((ROOT / "evaluation" / "heldout_ids.json").read_text(encoding="utf-8")))
    dev = set(json.loads((ROOT / "evaluation" / "dev_ids.json").read_text(encoding="utf-8")))
    expected_ids = held - dev
    for cfg in expected_configs:
        got_ids = {r.get("id") for r in rows if r.get("config") == cfg}
        if got_ids != expected_ids:
            problems.append({"kind": "wrong_id_coverage", "detail": cfg})

    marker = path.with_suffix(".complete")
    locked = os.getenv("OLLAMA_MODEL", "")
    if marker.exists():
        for line in marker.read_text(encoding="utf-8").splitlines():
            if line.startswith("generator="):
                locked = line.split("=", 1)[1]
    if locked and models != [locked]:
        problems.append({"kind": "unexpected_generator",
                         "detail": f"{models} != {locked}"})
    scorers = sorted({r.get("kw_scorer_version", "") for r in rows
                      if r.get("kw_scorer_version", "")})
    if scorers != [KW_SCORER_VERSION]:
        problems.append({"kind": "scorer_mismatch", "detail": str(scorers)})

    for row in rows:
        if row.get("status") == "OK":
            svc = row.get("svc_regex", "")
            if svc not in ("0", "1"):
                problems.append({"kind": "bad_svc_regex", "detail": svc})
        elif any(row.get(k, "") not in ("", None) for k in
                 ("kw_precision", "kw_recall", "kw_f1", "svc_regex")):
            problems.append({"kind": "scored_error_row",
                             "detail": str(row.get("id"))})

    for cfg in sorted({r.get("config", "") for r in rows}):
        s = [r for r in rows if r.get("config") == cfg]
        ok = [r for r in s if r.get("status") == "OK"]
        f1 = [float(r["kw_f1"]) for r in ok if r.get("kw_f1")]
        by_config[cfg] = {
            "rows": len(s), "ok": len(ok),
            "mean_kw_f1": round(sum(f1) / len(f1), 4) if f1 else None,
        }

    kinds = Counter(p["kind"] for p in problems)
    return {
        "path": str(path), "rows": len(rows), "models": models,
        "status_counts": dict(Counter(r.get("status", "") for r in rows)),
        "problem_counts": dict(kinds), "by_config": by_config,
        "problems": problems, "passed": not problems,
    }


def main() -> int:
    watch = "--watch" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else DEFAULT_CSV
    while True:
        result = audit(path)
        REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"rows={result['rows']} passed={result['passed']} "
              f"problems={result['problem_counts']} report={REPORT.name}")
        for p in result["problems"][:20]:
            print("  ", p)
        if not watch:
            return 0 if result["passed"] else 1
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
