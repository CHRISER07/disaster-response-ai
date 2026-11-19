"""Supervise the fresh held-out worker: alert + auto-restart on death or stall.

The worker is checkpointed per row, so a restart never loses completed rows.
Alerts are appended to evaluation/heldout_alerts.log with a UTC timestamp.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "scratch" / "fresh_worker.py"
OUT = ROOT / "evaluation" / "results_heldout_fresh.csv"
MARKER = OUT.with_suffix(".complete")
ALERTS = ROOT / "evaluation" / "heldout_alerts.log"
LOG = ROOT / "evaluation" / "heldout_supervisor.log"
TARGET = 1050
STALL_MINUTES = 10
CHECK_SECONDS = 60


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_alert(reason: str, rows: int, extra: str = "") -> None:
    line = f"{utc()} | {reason} | rows={rows}/{TARGET} | {extra}\n"
    with open(ALERTS, "a", encoding="utf-8") as f:
        f.write(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("ALERT " + line)


def row_count() -> int:
    if not OUT.exists():
        return 0
    try:
        with open(OUT, encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return -1


def spawn():
    env = os.environ.copy()
    env.update({
        "PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "llama3.1:8b",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    })
    # Do NOT use DETACHED_PROCESS here: the Intel Fortran runtime used by the
    # numerical stack aborts with "window-CLOSE event" when launched detached.
    # The supervisor is itself hidden via Start-Process, so the child inherits
    # a stable hidden console.
    p = subprocess.Popen(
        [sys.executable, "-u", str(WORKER)],
        cwd=str(ROOT), env=env,
        stdout=open(ROOT / "evaluation" / "fresh_heldout_supervised.log", "a", encoding="utf-8"),
        stderr=open(ROOT / "evaluation" / "fresh_heldout_supervised.err", "a", encoding="utf-8"),
    )
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{utc()} | START worker pid={p.pid} rows={row_count()}\n")
    return p


def main() -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{utc()} | SUPERVISOR start\n")
    proc = None
    last_rows = row_count()
    last_change = time.time()

    while True:
        if MARKER.exists():
            count = row_count()
            if count != TARGET:
                # A stale or partial marker must not stop the supervisor.
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(f"{utc()} | MARKER INVALID rows={count} expected={TARGET}\n")
                MARKER.unlink(missing_ok=True)
            else:
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(f"{utc()} | COMPLETE rows={count}\n")
                return

        rows = row_count()
        if rows < 0:
            log_alert("output_unreadable", last_rows)

        if rows > last_rows:
            last_rows = rows
            last_change = time.time()

        if proc is not None and proc.poll() is not None:
            log_alert("worker_exited", rows, f"exit_code={proc.returncode}")
            proc = None

        stalled = (time.time() - last_change) > STALL_MINUTES * 60
        if stalled:
            log_alert("worker_stalled", rows, f"no progress for {STALL_MINUTES} min")
            if proc is not None:
                proc.kill()
            proc = None
            last_change = time.time()

        if proc is None:
            proc = spawn()
            last_change = time.time()

        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
