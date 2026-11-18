"""Replace stale BM25 rows in the fresh held-out CSV and re-measure them.

The old BM25 baseline ignored the query and retrieved a constant context. This
script removes all BM25 rows (including the single timeout row) and re-runs only
the BM25 arm with the corrected FTS5 retriever, on the same 150 queries, same
generator (llama3.1:8b), seed 42 and scorer as the rest of the fresh run.

Run only after the main held-out worker has finished.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "evaluation" / "results_heldout_fresh.csv"
MARKER = CSV.with_suffix(".complete")


def main() -> None:
    if MARKER.exists():
        print("Held-out run already complete; use targeted repair instead.")
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    kept = [r for r in rows if r.get("config") != "BM25_Baseline"]
    removed = len(rows) - len(kept)
    with open(CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(kept)
    print(f"Removed {removed} stale BM25 rows; kept {len(kept)}")

    env = os.environ.copy()
    env.update({
        "PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "llama3.1:8b",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    })
    cmd = [sys.executable, "-u", "evaluation/run_fresh_heldout.py",
           "--config", "BM25_Baseline"]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT), env=env))


if __name__ == "__main__":
    main()
