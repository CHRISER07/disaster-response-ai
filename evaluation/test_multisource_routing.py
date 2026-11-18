"""Smoke-test multi-source tool routing after the prompt fix (no CSV writes)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["PROVIDER"] = "ollama"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "llama3.1:8b"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, str(ROOT))

from evaluation.run_dev_eval_v2 import run_full_agent

gold = {q["id"]: q for q in json.load(open(ROOT / "evaluation/golden_dataset_v2.json", encoding="utf-8"))}
for qid in (25, 41, 89):
    item = gold[qid]
    res = run_full_agent(item["query"], 42)
    tools = res.get("tools_called", "")
    ans = res.get("full_answer", "").replace("\n", " ")
    print(f"ID {qid} tools={tools} status={res.get('status')}")
    print("  gold:", item["keywords"][:8])
    print("  ans :", ans[:320])
    print()
