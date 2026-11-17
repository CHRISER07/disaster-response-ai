"""
evaluation/verify_eval_io.py

Pre-flight input/output verification for every evaluation configuration.

Runs one fixed query through each config runner and checks:
  - the input query is the one requested (echoed back for audit)
  - status is OK
  - the answer is non-empty
  - the answer contains NO asterisks (clean plain-text template)
  - no leaked tool-calling / JSON plumbing text
  - full-agent answers contain the operational template sections

This is deliberately small (one query per config) so it can be run before any
expensive full evaluation. No results are hardcoded; every check is derived
from the actual returned answer.

Run:
    python evaluation/verify_eval_io.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUTF8", "1")

from evaluation.run_dev_eval_v2 import (
    run_llm_only, run_bm25, run_static_rag, run_full_agent,
)
from core.answer_formatter import TEMPLATE_SECTIONS, format_answer

DATASET = ROOT / "evaluation" / "golden_dataset_v2.json"
VERIFY_ID = 13
SEED = 42

LEAK_MARKERS = (
    "tool call validation failed",
    "no json output",
    "json object",
    "requires two parameters",
    "<tool_call>",
)


def main() -> None:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    item = next((q for q in dataset if q["id"] == VERIFY_ID), None)
    if item is None:
        print(f"Verification query id {VERIFY_ID} not found in dataset")
        sys.exit(2)
    query = item["query"]
    print(f"Verification input (id={VERIFY_ID}): {query}\n")

    configs = [
        ("LLM_Only", lambda q, s: run_llm_only(q, s)),
        ("BM25_Baseline", lambda q, s: run_bm25(q, s)),
        ("Standard_StaticRAG", lambda q, s: run_static_rag(q, s, text_only=False)),
        ("Full_Agent", lambda q, s: run_full_agent(q, s)),
        ("Full_Agent_NoTGI", lambda q, s: run_full_agent(q, s, no_tgi=True)),
        ("Full_Agent_NoDSEL", lambda q, s: run_full_agent(q, s, no_dsel=True)),
    ]

    failures = 0
    for name, fn in configs:
        try:
            res = fn(query, SEED)
        except Exception as exc:
            print(f"{name:22s} FAIL exception: {exc}")
            failures += 1
            continue

        status = str(res.get("status", "")).upper()
        answer = format_answer(str(res.get("full_answer", "")))
        problems = []
        if status != "OK":
            problems.append(f"status={status}")
        if not answer.strip():
            problems.append("empty answer")
        if "*" in answer:
            problems.append(f"asterisks={answer.count('*')}")
        if any(m in answer.lower() for m in LEAK_MARKERS):
            problems.append("tool/json leak")
        if name.startswith("Full_Agent"):
            missing = [s for s in TEMPLATE_SECTIONS if s not in answer]
            if missing:
                problems.append("missing " + ",".join(missing))

        verdict = "PASS" if not problems else "FAIL"
        if problems:
            failures += 1
        print(f"{name:22s} {verdict:4s} chars={len(answer):5d} tools={res.get('tools_called','')}"
              + (f"  [{'; '.join(problems)}]" if problems else ""))

    print(f"\n{len(configs) - failures}/{len(configs)} configs passed I/O verification.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
