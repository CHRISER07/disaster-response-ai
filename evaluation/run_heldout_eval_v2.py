"""
Phase 3: Held-out evaluation runner (configs 1,2,4,5 on heldout_ids).
Run: python evaluation/run_heldout_eval_v2.py
"""
import os, sys, json, csv, time, random, re
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

# Reuse runners from run_dev_eval_v2
from evaluation.run_dev_eval_v2 import (
    run_llm_only, run_bm25, run_static_rag, run_full_agent,
    kw_scores, svc_regex_check, FIELDNAMES, GROQ_MODEL, STATUS_OK, load_done_keys
)
from core.answer_formatter import format_answer

OUT_CSV = ROOT / "evaluation" / "results_heldout_v2.csv"
HELDOUT_IDS_FILE = ROOT / "evaluation" / "heldout_ids.json"
DATASET_FILE = ROOT / "evaluation" / "golden_dataset_v2.json"

SEEDS = [42, 7, 99]

CONFIGS = [
    {"name": "LLM_Only",           "fn": lambda q,s: run_llm_only(q,s),                   "runs": SEEDS},
    {"name": "BM25_Baseline",      "fn": lambda q,s: run_bm25(q,s),                        "runs": SEEDS},
    {"name": "Standard_StaticRAG", "fn": lambda q,s: run_static_rag(q,s,text_only=False),  "runs": SEEDS},
    {"name": "Full_Agent",         "fn": lambda q,s: run_full_agent(q,s),                  "runs": SEEDS},
    {"name": "Full_Agent_NoTGI",   "fn": lambda q,s: run_full_agent(q,s,no_tgi=True),      "runs": [42]},  # 1 run if quota tight
    {"name": "Full_Agent_NoDSEL",  "fn": lambda q,s: run_full_agent(q,s,no_dsel=True),     "runs": [42]},
]


def main():
    heldout_ids = set(json.loads(HELDOUT_IDS_FILE.read_text(encoding="utf-8")))
    dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    heldout_queries = [q for q in dataset if q["id"] in heldout_ids]
    print(f"Heldout queries: {len(heldout_queries)}, Configs: {len(CONFIGS)}")

    done_keys = load_done_keys(OUT_CSV)
    write_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
    out_f = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    total = sum(len(c["runs"]) * len(heldout_queries) for c in CONFIGS)
    done_count = len(done_keys)
    errors = 0

    for cfg in CONFIGS:
        for seed in cfg["runs"]:
            for q in heldout_queries:
                key = (cfg["name"], str(seed), str(q["id"]))
                if key in done_keys:
                    continue
                print(f"  [{done_count+1}/{total}] cfg={cfg['name']} id={q['id']}", end="", flush=True)
                try:
                    res = cfg["fn"](q["query"], seed)
                except Exception as e:
                    res = {"status":"OTHER_ERROR","full_answer":str(e),
                           "active_latency_s":0,"total_latency_s":0,
                           "prompt_tokens":0,"completion_tokens":0,
                           "tools_called":"","model_used":GROQ_MODEL,"retries":0}

                gold_kw = q.get("keywords", [])
                expected_svc = q.get("expected_service","")
                if res["status"] == STATUS_OK:
                    p,r,f1 = kw_scores(res["full_answer"], gold_kw)
                    svc = svc_regex_check(res["full_answer"], expected_svc)
                else:
                    p=r=f1=svc=""
                    errors+=1

                row = {
                    "config":cfg["name"],"run_id":seed,"id":q["id"],
                    "scenario":q.get("scenario",""),"category":q.get("category",""),
                    "status":res["status"],"kw_precision":p,"kw_recall":r,"kw_f1":f1,
                    "svc_regex":svc,"p_at_k":"","ndcg_at_7":"","rr":"",
                    "active_latency_s":res["active_latency_s"],"total_latency_s":res["total_latency_s"],
                    "prompt_tokens":res["prompt_tokens"],"completion_tokens":res["completion_tokens"],
                    "tools_called":res["tools_called"],"model_used":res["model_used"],
                    "retries":res["retries"],"full_answer":format_answer(res["full_answer"]),
                }
                writer.writerow(row)
                out_f.flush()
                done_count += 1
                done_keys.add(key)
                print(f" → {res['status']}")
                time.sleep(0.8)

    out_f.close()
    print(f"\nDone. {done_count} rows. Errors: {errors}. Output: {OUT_CSV}")


if __name__ == "__main__":
    main()

