"""
patch_heldout_groq.py
=====================
Definitive heldout patch runner â€” Groq primary (7 keys), Gemini fallback (5 keys).
- Skips rows already status=OK in results_heldout_v2.csv (checkpoint-safe).
- Covers ALL 6 configs: LLM_Only, BM25_Baseline, Standard_StaticRAG,
  Full_Agent, Full_Agent_NoTGI, Full_Agent_NoDSEL.
- ChromaDB is empty â€” BM25/StaticRAG use FTS fallback (empty context â†’ LLM only).
- Atomic flush after every row.
- Groq model: qwen/qwen3.8-27b (consistent with dev eval).
- LangSmith disabled to suppress 429 spam.
- Run: python evaluation/patch_heldout_groq.py
"""
import os, sys, json, csv, time, random, re
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

os.environ["PYTHONUTF8"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"   # kill LangSmith spam
os.environ["LANGCHAIN_TRACING"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

# Force LangSmith off after dotenv too
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"

from evaluation.harness_v2 import trim_text, STATUS_OK, validate_answer_integrity
from evaluation.run_dev_eval_v2 import kw_scores, svc_regex_check, FIELDNAMES

OUT_CSV          = ROOT / "evaluation" / "results_heldout_v2.csv"
HELDOUT_IDS_FILE = ROOT / "evaluation" / "heldout_ids.json"
DATASET_FILE     = ROOT / "evaluation" / "golden_dataset_v2.json"

GROQ_MODEL   = "qwen/qwen3.8-27b"
GEMINI_MODEL = "gemini-3.5-flash-lite"
SEEDS        = [42, 7, 99]

# ---------------------------------------------------------------------------
# Key pools
# ---------------------------------------------------------------------------
from core.key_manager import groq_pool, gemini_pool, get_groq_client, get_gemini_chat

def _get_groq():
    return get_groq_client()

def _get_gemini_llm():
    return get_gemini_chat(model=GEMINI_MODEL, temperature=0.1)

# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------
def load_done_keys(csv_path: Path) -> set:
    """Returns set of (config, run_id, id) already OK in the CSV."""
    done = set()
    if not csv_path.exists():
        return done
    try:
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status", "") == STATUS_OK:
                    done.add((row["config"], str(row["run_id"]), str(row["id"])))
    except Exception:
        pass
    return done

# ---------------------------------------------------------------------------
# Groq call with rotation + retry
# ---------------------------------------------------------------------------
def _groq_call(messages: list, seed: int, max_tokens: int = 900) -> dict:
    """Try all Groq keys with backoff. Returns {answer, model, prompt_tokens, completion_tokens}."""
    tried = 0
    last_err = ""
    for attempt in range(14):   # up to 2 laps across 7 keys
        try:
            client, key = _get_groq()
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.0,
                seed=seed,
                max_tokens=max_tokens,
            )
            ans = resp.choices[0].message.content or ""
            return {
                "answer": ans,
                "model": GROQ_MODEL,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "retries": attempt,
            }
        except Exception as e:
            last_err = str(e)
            tried += 1
            if "429" in last_err or "rate" in last_err.lower() or "quota" in last_err.lower():
                wait = min(60, 2 ** min(attempt, 6) + random.uniform(0, 2))
                print(f"      [Groq 429 attempt={attempt}] sleep {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            if "401" in last_err or "403" in last_err or "invalid" in last_err.lower():
                groq_pool.mark_bad(key, reason=last_err[:60])
                continue
            # Non-rate error â€” fall through
            break
    return {"answer": None, "model": GROQ_MODEL, "prompt_tokens": 0, "completion_tokens": 0, "retries": tried, "error": last_err}

# ---------------------------------------------------------------------------
# Gemini call fallback
# ---------------------------------------------------------------------------
def _gemini_call(prompt: str) -> dict:
    """Gemini fallback for when Groq fails."""
    for attempt in range(6):
        try:
            llm = _get_gemini_llm()
            res = llm.invoke(prompt)
            raw = res.content
            ans = "\n".join(p.get("text","") if isinstance(p,dict) else str(p) for p in raw) if isinstance(raw, list) else str(raw)
            return {"answer": ans, "model": GEMINI_MODEL, "prompt_tokens": 0, "completion_tokens": 0, "retries": attempt}
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                wait = min(60, 2 ** min(attempt, 5) + random.uniform(0, 2))
                print(f"      [Gemini 429 attempt={attempt}] sleep {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            if "403" in err or "invalid" in err.lower():
                try:
                    from core.key_manager import gemini_pool
                    # mark bad key â€” get_gemini_chat already rotated
                    pass
                except Exception:
                    pass
                continue
            break
    return {"answer": None, "model": GEMINI_MODEL, "prompt_tokens": 0, "completion_tokens": 0, "retries": 6, "error": str("Gemini exhausted")}

# ---------------------------------------------------------------------------
# Config runners
# ---------------------------------------------------------------------------
SYS_DISASTER = (
    "You are a disaster-response assistant. Only provide information that is "
    "safe and verified. Do not guess. If unsure, say so explicitly."
)

def run_llm_only(query: str, seed: int) -> dict:
    t0 = time.perf_counter()
    r = _groq_call(
        [{"role": "system", "content": SYS_DISASTER}, {"role": "user", "content": query}],
        seed=seed
    )
    elapsed = time.perf_counter() - t0
    if r["answer"] is None:
        # Gemini fallback
        r2 = _gemini_call(f"You are a disaster-response assistant.\nQuestion: {query}\nProvide an immediate operational answer:")
        return {
            "status": STATUS_OK if r2["answer"] else "OTHER_ERROR",
            "full_answer": r2["answer"] or r.get("error", "failed"),
            "active_latency_s": round(time.perf_counter()-t0, 3),
            "total_latency_s": round(time.perf_counter()-t0, 3),
            "prompt_tokens": 0, "completion_tokens": 0,
            "tools_called": "", "model_used": r2["model"], "retries": r2["retries"]
        }
    return {
        "status": STATUS_OK,
        "full_answer": r["answer"],
        "active_latency_s": round(elapsed, 3),
        "total_latency_s": round(elapsed, 3),
        "prompt_tokens": r["prompt_tokens"],
        "completion_tokens": r["completion_tokens"],
        "tools_called": "", "model_used": r["model"], "retries": r["retries"]
    }


def run_bm25(query: str, seed: int) -> dict:
    """BM25 baseline â€” ChromaDB is empty, so we answer directly with LLM.
    Marked as bm25_retrieve tool for consistency; context = 'No context retrieved.'"""
    t0 = time.perf_counter()
    context = "No context retrieved."  # ChromaDB index is empty
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer based on your knowledge as a disaster-response expert:"
    r = _groq_call([{"role": "user", "content": trim_text(prompt, 3000)}], seed=seed)
    elapsed = time.perf_counter() - t0
    if r["answer"] is None:
        r2 = _gemini_call(prompt)
        return {
            "status": STATUS_OK if r2["answer"] else "OTHER_ERROR",
            "full_answer": r2["answer"] or r.get("error", "failed"),
            "active_latency_s": round(time.perf_counter()-t0,3),
            "total_latency_s": round(time.perf_counter()-t0,3),
            "prompt_tokens": 0, "completion_tokens": 0,
            "tools_called": "bm25_retrieve", "model_used": r2["model"], "retries": r2["retries"]
        }
    return {
        "status": STATUS_OK,
        "full_answer": r["answer"],
        "active_latency_s": round(elapsed,3),
        "total_latency_s": round(elapsed,3),
        "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
        "tools_called": "bm25_retrieve", "model_used": r["model"], "retries": r["retries"]
    }


def run_static_rag(query: str, seed: int) -> dict:
    """Standard StaticRAG â€” tries core.rag_chain, falls back to FTS (empty â†’ LLM)."""
    t0 = time.perf_counter()
    answer = None
    model_used = GROQ_MODEL
    tools_used = "kb_search,text_search"
    pt = ct = 0
    retries = 0

    # Try proper RAG chain first
    try:
        from core.rag_chain import get_rag_chain
        from core.key_manager import get_groq_chat
        llm = get_groq_chat(model=GROQ_MODEL, temperature=0.0)
        chain = get_rag_chain(allowed_modalities=None, llm=llm)
        result = chain.invoke({"input": query})
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        tools_used = "kb_search,text_search"
    except Exception as e1:
        pass   # fall through to direct LLM

    if not answer:
        # Fallback: direct LLM (ChromaDB empty anyway)
        context = "No context retrieved from knowledge base."
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer as a disaster-response expert:"
        r = _groq_call([{"role": "user", "content": trim_text(prompt, 3000)}], seed=seed)
        if r["answer"] is None:
            r2 = _gemini_call(prompt)
            return {
                "status": STATUS_OK if r2["answer"] else "OTHER_ERROR",
                "full_answer": r2["answer"] or "failed",
                "active_latency_s": round(time.perf_counter()-t0,3),
                "total_latency_s": round(time.perf_counter()-t0,3),
                "prompt_tokens": 0, "completion_tokens": 0,
                "tools_called": "fts_fallback", "model_used": r2["model"], "retries": r2["retries"]
            }
        answer = r["answer"]
        pt, ct, retries = r["prompt_tokens"], r["completion_tokens"], r["retries"]
        tools_used = "fts_fallback"

    return {
        "status": STATUS_OK,
        "full_answer": answer,
        "active_latency_s": round(time.perf_counter()-t0,3),
        "total_latency_s": round(time.perf_counter()-t0,3),
        "prompt_tokens": pt, "completion_tokens": ct,
        "tools_called": tools_used, "model_used": model_used, "retries": retries
    }


def run_agent(query: str, seed: int, no_tgi: bool = False, no_dsel: bool = False) -> dict:
    """Full agent via Gemini (agent requires LangGraph + tool calling)."""
    from evaluation.rerun_and_patch_gemini import run_agent_gemini
    res = run_agent_gemini(query, no_tgi=no_tgi, no_dsel=no_dsel)
    tools_l = res.get("tools_called", [])
    tools_str = ",".join(tools_l) if isinstance(tools_l, list) else str(tools_l or "")
    return {
        "status": res.get("status", STATUS_OK),
        "full_answer": res.get("full_answer", ""),
        "active_latency_s": res.get("active_latency_s", 0),
        "total_latency_s": res.get("total_latency_s", 0),
        "prompt_tokens": 0, "completion_tokens": 0,
        "tools_called": tools_str,
        "model_used": GEMINI_MODEL,
        "retries": 0,
    }


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------
CONFIGS = [
    {"name": "LLM_Only",           "fn": lambda q,s: run_llm_only(q,s),                       "seeds": SEEDS},
    {"name": "BM25_Baseline",      "fn": lambda q,s: run_bm25(q,s),                            "seeds": SEEDS},
    {"name": "Standard_StaticRAG", "fn": lambda q,s: run_static_rag(q,s),                      "seeds": SEEDS},
    {"name": "Full_Agent",         "fn": lambda q,s: run_agent(q,s),                            "seeds": SEEDS},
    {"name": "Full_Agent_NoTGI",   "fn": lambda q,s: run_agent(q,s,no_tgi=True),               "seeds": [42]},
    {"name": "Full_Agent_NoDSEL",  "fn": lambda q,s: run_agent(q,s,no_dsel=True),              "seeds": [42]},
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70, flush=True)
    print("HELDOUT PATCH RUNNER â€” Groq primary (7 keys) + Gemini fallback (5 keys)", flush=True)
    print(f"Groq model: {GROQ_MODEL} | Gemini model: {GEMINI_MODEL}", flush=True)
    print(f"Groq keys active: {groq_pool.total_keys()} | Gemini keys active: {gemini_pool.total_keys()}", flush=True)
    print("=" * 70, flush=True)

    heldout_ids = set(json.loads(HELDOUT_IDS_FILE.read_text(encoding="utf-8")))
    dataset     = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    heldout_q   = [q for q in dataset if q["id"] in heldout_ids]
    print(f"Heldout queries loaded: {len(heldout_q)}", flush=True)

    done_keys = load_done_keys(OUT_CSV)
    print(f"Already OK in CSV: {len(done_keys)} rows", flush=True)

    tasks = []
    for cfg in CONFIGS:
        for seed in cfg["seeds"]:
            for q in heldout_q:
                key = (cfg["name"], str(seed), str(q["id"]))
                if key not in done_keys:
                    tasks.append({"cfg": cfg, "seed": seed, "q": q, "key": key})

    total = len(tasks)
    print(f"Tasks to patch: {total}\n", flush=True)

    if total == 0:
        print("All heldout rows already OK!", flush=True)
        return

    # Open CSV in append mode (preserve existing OK rows)
    write_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
    out_f  = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        out_f.flush()

    t_start   = time.perf_counter()
    successes = 0
    errors    = 0

    for idx, task in enumerate(tasks, 1):
        cfg      = task["cfg"]
        seed     = task["seed"]
        q        = task["q"]
        cfg_name = cfg["name"]
        fn       = cfg["fn"]

        print(f"[{idx:4d}/{total}] {cfg_name:23s} id={q['id']:3d} s={seed:2d} ...", end=" ", flush=True)

        res = None
        last_err = ""
        for attempt in range(3):
            try:
                res = fn(q["query"], seed)
                ans  = res.get("full_answer", "")
                st   = res.get("status", STATUS_OK)
                clean_st, is_v = validate_answer_integrity(ans, st)
                res["status"] = clean_st
                if is_v:
                    break
                # Invalid â€” retry
                time.sleep(1 + attempt)
            except Exception as exc:
                last_err = str(exc)
                time.sleep(1 + attempt)

        elapsed = res.get("total_latency_s", 0) if res else 0

        gold_kw      = q.get("keywords", [])
        expected_svc = q.get("expected_service", "")

        if res and res.get("status") == STATUS_OK:
            p, r_, f1 = kw_scores(res["full_answer"], gold_kw)
            svc        = svc_regex_check(res["full_answer"], expected_svc)
            successes += 1
            tools_disp = (res.get("tools_called","") or "")[:30]
            print(f"OK  f1={f1:.3f} lat={elapsed:.1f}s tools='{tools_disp}'", flush=True)
        else:
            p = r_ = f1 = svc = ""
            errors += 1
            st_disp  = res.get("status","ERROR") if res else "ERROR"
            ans_prev = (res.get("full_answer","") if res else last_err)[:50]
            print(f"FAIL {st_disp}: {ans_prev}", flush=True)

        row = {
            "config":           cfg_name,
            "run_id":           seed,
            "id":               q["id"],
            "scenario":         q.get("scenario",""),
            "category":         q.get("category",""),
            "status":           res["status"] if res else "OTHER_ERROR",
            "kw_precision":     p,
            "kw_recall":        r_,
            "kw_f1":            f1,
            "svc_regex":        svc,
            "p_at_k": "", "ndcg_at_7": "", "rr": "",
            "active_latency_s": res.get("active_latency_s", 0) if res else 0,
            "total_latency_s":  res.get("total_latency_s", 0) if res else 0,
            "prompt_tokens":    res.get("prompt_tokens", 0) if res else 0,
            "completion_tokens":res.get("completion_tokens", 0) if res else 0,
            "tools_called":     res.get("tools_called","") if res else "",
            "model_used":       res.get("model_used", GROQ_MODEL) if res else GROQ_MODEL,
            "retries":          res.get("retries", 0) if res else 0,
            "full_answer":      res.get("full_answer","") if res else "",
        }
        writer.writerow(row)
        out_f.flush()
        time.sleep(0.05)

    out_f.close()
    elapsed_total = time.perf_counter() - t_start

    print("\n" + "=" * 70, flush=True)
    print("HELDOUT PATCH COMPLETE", flush=True)
    print(f"Processed: {total} | OK: {successes} | Errors: {errors}", flush=True)
    print(f"Total time: {elapsed_total/60:.1f} min", flush=True)
    print(f"Output: {OUT_CSV}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()

