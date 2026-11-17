"""
Phase 4: Dual-Judge LLM Evaluation Runner
==========================================
Judge A: gemini-2.0-flash-lite  (Google AI, 7-key pool) â€” provider-split from JudgeB
Judge B: openai/gpt-oss-20b     (Groq, OpenAI architecture â€” cross-family from Gemini for IEEE validity)

Provider split: JudgeA â†’ Google AI (Gemini), JudgeB â†’ Groq (GPT-OSS-20B).
This eliminates cross-provider rate-limit contention that caused ETA > 2000 min when both ran on Groq.

Evaluates 3 dimensions per answer:
  - dr_fact    : Factual correctness (0.0 / 0.5 / 1.0)
  - saa        : Situational awareness & actionability (0.0 / 0.5 / 1.0)
  - temporal   : Temporal accuracy (0.0 / 0.5 / 1.0)

Reads:
  - evaluation/results_per_query_v2.csv   (1,200 OK rows â€” Dev)
  - evaluation/results_heldout_v2.csv     (2,100 OK rows â€” Held-Out)

Writes:
  - evaluation/judge_scores_v2.csv        (atomic, checkpoint-safe)
  - evaluation/kappa_result.txt           (Cohen's kappa after all scoring done)

Run: python evaluation/run_judge_v2.py
"""

import os, sys, json, csv, time, random, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

os.environ["PYTHONUTF8"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from core.key_manager import get_gemini_chat, get_groq_client, gemini_pool, groq_pool

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
DATASET_FILE = ROOT / "evaluation" / "golden_dataset_v2.json"
DEV_CSV      = ROOT / "evaluation" / "results_per_query_v2.csv"
HELDOUT_CSV  = ROOT / "evaluation" / "results_heldout_v2.csv"
OUT_CSV      = ROOT / "evaluation" / "judge_scores_v2.csv"
KAPPA_FILE   = ROOT / "evaluation" / "kappa_result.txt"

JUDGE_A_MODEL = "gemini-3.5-flash-lite"  # Evaluator A: Gemini 3.5 Flash Lite on Google AI (routed via _call_gemini)
JUDGE_B_MODEL = "openai/gpt-oss-20b"     # Evaluator B: GPT-OSS-20B on Groq (OpenAI architecture, cross-family from Gemini)

JUDGE_FIELDNAMES = [
    "config", "run_id", "id", "judge",
    "dr_fact", "saa", "temporal",
    "dr_fact_rationale", "saa_rationale", "temporal_rationale"
]

# ---------------------------------------------------------------------------
# Prompt templates (matches judge_prompts.md exactly)
# ---------------------------------------------------------------------------
DRFACT_PROMPT = """\
You are an expert disaster-response evaluator. Rate the following AI answer for factual correctness against official emergency management doctrine (FEMA, CDC, USGS) and the provided ground truth reference.

QUESTION: {query}
AI ANSWER: {answer}
GROUND TRUTH: {ground_truth}
TIME-AWARE REFERENCE: {time_aware_reference}

SCORING CRITERIA:
1.0 = All factual statements in the answer adhere to official emergency doctrine and the time-aware reference.
      NOTE: If the query is time-sensitive (live sensor reading), do NOT penalize the answer for reporting current safe conditions (e.g., 'no evacuation required') if current gage readings are below flood stage.
0.5 = Response matches partially, but is vague or omits critical protocol details.
0.0 = Contains factually erroneous statements, dangerous directives, or directly contradicts emergency guidance.

Respond strictly in the following format:
SCORE: 0.XX
RATIONALE: <One concise sentence explaining the exact factual basis for this score>"""

SAA_PROMPT = """\
You are an emergency operations center evaluator. Assess whether the following AI response provides actionable, situationally aware disaster guidance.

QUESTION: {query}
AI ANSWER: {answer}

SCORING CRITERIA:
1.0 = Clear, unambiguous, and immediate actionable guidance (specific evacuation routes, numeric thresholds, authority contact protocols, shelter instructions).
0.5 = Response is informative or correctly refuses an unverified/hallucinatory query, but provides only general advisories without specific operational numbers.
0.0 = Non-actionable, evasive, confusing, or misleading in an active crisis situation.

Respond strictly in the following format:
SCORE: 0.XX
RATIONALE: <One concise sentence explaining the actionability assessment>"""

TEMPORAL_PROMPT = """\
You are a temporal reasoning evaluator for an emergency intelligence system. Assess whether the AI correctly distinguishes between LIVE/CURRENT sensor telemetry and HISTORICAL disaster archives (2012-2015 CrisisLex records).

QUESTION: {query}
AI ANSWER: {answer}

SCORING CRITERIA:
1.0 = Explicitly distinguishes between live current conditions and archival historical data; correctly avoids conflating 2013 historical flood tweets with present conditions.
0.5 = Temporal distinction is implicit or partial.
0.0 = Conflates historical disaster events with present-day conditions, or presents historical archive numbers as current live readings.

Respond strictly in the following format:
SCORE: 0.XX
RATIONALE: <One concise sentence explaining the temporal accuracy assessment>"""

# ---------------------------------------------------------------------------
# Score parser â€” handles "SCORE: 0.XX" format
# ---------------------------------------------------------------------------
SCORE_RE    = re.compile(r"SCORE\s*:\s*([01]\.\d+|[01])", re.IGNORECASE)
RATIONAL_RE = re.compile(r"RATIONALE\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)

def parse_score_response(text: str) -> tuple:
    """Returns (score_float_or_None, rationale_str)."""
    score = None
    rationale = ""
    sm = SCORE_RE.search(text)
    if sm:
        try:
            score = float(sm.group(1))
        except ValueError:
            pass
    rm = RATIONAL_RE.search(text)
    if rm:
        rationale = rm.group(1).strip()[:300]
    return score, rationale

# ---------------------------------------------------------------------------
# Judge callers with full retry + key rotation
# ---------------------------------------------------------------------------
GEMINI_JUDGE_MODEL = "gemini-3.5-flash-lite"   # Provider-split: JudgeA on Google AI

def _call_gemini(prompt: str) -> str:
    for attempt in range(8):
        try:
            llm = get_gemini_chat(model=GEMINI_JUDGE_MODEL, temperature=0.0)
            res = llm.invoke(prompt)
            raw = res.content
            text = "\n".join(p.get("text","") if isinstance(p, dict) else str(p) for p in raw) if isinstance(raw, list) else str(raw)
            return text
        except Exception as e:
            err = str(e)
            wait = min(60, 2 ** min(attempt, 5) + random.uniform(0.5, 2.0))
            if "429" in err or "quota" in err.lower() or "resource" in err.lower():
                print(f" [GemJudge 429 attempt={attempt} sleep={wait:.1f}s]", flush=True)
                time.sleep(wait)
            elif "404" in err or "not found" in err.lower():
                print(f" [GemJudge 404 model-not-found â€” check GEMINI_JUDGE_MODEL]", flush=True)
                return ""
            elif "403" in err or "invalid" in err.lower():
                time.sleep(2)
            else:
                time.sleep(wait)
    return ""

def _call_groq(prompt: str) -> str:
    for attempt in range(8):
        try:
            client, key = get_groq_client()
            resp = client.chat.completions.create(
                model=JUDGE_B_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            err = str(e)
            wait = min(60, 2 ** min(attempt, 6) + random.uniform(0.5, 2.0))
            if "429" in err or "rate" in err.lower():
                print(f" [GroqJudge 429 attempt={attempt} sleep={wait:.1f}s]", flush=True)
                time.sleep(wait)
            else:
                time.sleep(wait)
    return ""

def call_judge(judge_name: str, prompt: str) -> tuple:
    """Routes JudgeA â†’ Gemini Flash Lite (Google AI), JudgeB â†’ Groq GPT-OSS-20B.
    Provider split eliminates cross-provider rate-limit contention.
    """
    if judge_name == "JudgeA":
        text = _call_gemini(prompt)          # Google Gemini (7-key pool)
    else:
        text = _call_groq_model(JUDGE_B_MODEL, prompt)  # Groq GPT-OSS-20B (9-key pool)
    return parse_score_response(text)

def _call_groq_model(model: str, prompt: str) -> str:
    for attempt in range(8):
        try:
            client, key = get_groq_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            err = str(e)
            wait = min(60, 2 ** min(attempt, 6) + random.uniform(0.5, 2.0))
            if "429" in err or "rate" in err.lower():
                print(f" [GroqJudge {model} 429 attempt={attempt} sleep={wait:.1f}s]", flush=True)
                time.sleep(wait)
            else:
                time.sleep(wait)
    return ""

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_ok_rows(csv_path: Path) -> list:
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("status","") == "OK"]

def load_dataset_map(ds_path: Path) -> dict:
    data = json.loads(ds_path.read_text(encoding="utf-8"))
    return {q["id"]: q for q in data}

def load_done_keys(csv_path: Path) -> set:
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # key = (config, run_id, id, judge) â€” but only if all 3 dim scores exist
            if all(row.get(d, "") not in ("", "None") for d in ["dr_fact","saa","temporal"]):
                done.add((row["config"], str(row["run_id"]), str(row["id"]), row["judge"]))
    return done

# ---------------------------------------------------------------------------
# Cohen's Kappa (quadratic weighted, on 0/0.5/1.0 scale)
# ---------------------------------------------------------------------------
def compute_kappa():
    if not OUT_CSV.exists():
        return
    rows_a, rows_b = {}, {}
    with open(OUT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["config"], str(row["run_id"]), str(row["id"]))
            try:
                val = float(row["dr_fact"]) if row.get("dr_fact","") not in ("","None") else None
            except ValueError:
                val = None
            if val is None:
                continue
            if row["judge"] == "JudgeA":
                rows_a[key] = val
            elif row["judge"] == "JudgeB":
                rows_b[key] = val

    common = set(rows_a) & set(rows_b)
    n = len(common)
    if n < 10:
        print(f"Not enough overlapping judgments for kappa (n={n})", flush=True)
        return

    a = [rows_a[k] for k in common]
    b = [rows_b[k] for k in common]

    # Simple weighted kappa (linear weights for 3-class: 0.0, 0.5, 1.0)
    classes = [0.0, 0.5, 1.0]
    n_cls = len(classes)
    idx = {v: i for i, v in enumerate(classes)}
    W = [[1 - abs(i-j)/(n_cls-1) for j in range(n_cls)] for i in range(n_cls)]

    # Observed agreement
    po = sum(W[idx.get(ai, 0)][idx.get(bi, 0)] for ai, bi in zip(a, b)) / n

    # Expected agreement
    freq_a = [sum(1 for x in a if x == c) / n for c in classes]
    freq_b = [sum(1 for x in b if x == c) / n for c in classes]
    pe = sum(W[i][j] * freq_a[i] * freq_b[j] for i in range(n_cls) for j in range(n_cls))

    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    # Also compute exact agreement
    exact = sum(1 for ai, bi in zip(a, b) if ai == bi) / n

    result = (
        f"Cohen's kappa (Weighted, DR-Fact, 0/0.5/1.0 scale)\n"
        f"  Kappa:          {kappa:.4f}\n"
        f"  Exact Agreement:{exact*100:.1f}%\n"
        f"  Pairs compared: {n}\n"
        f"  JudgeA mean:    {sum(a)/n:.4f}\n"
        f"  JudgeB mean:    {sum(b)/n:.4f}\n"
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
    )
    print("\n" + result, flush=True)
    KAPPA_FILE.write_text(result, encoding="utf-8")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70, flush=True)
    print("PHASE 4: DUAL-JUDGE LLM EVALUATION", flush=True)
    print(f"Judge A: {JUDGE_A_MODEL} (Gemini keys: {gemini_pool.total_keys()})", flush=True)
    print(f"Judge B: {JUDGE_B_MODEL} (Groq keys:   {groq_pool.total_keys()})", flush=True)
    print("=" * 70, flush=True)

    # Load data
    q_map    = load_dataset_map(DATASET_FILE)
    dev_rows = load_ok_rows(DEV_CSV)
    hld_rows = load_ok_rows(HELDOUT_CSV)
    all_rows = dev_rows + hld_rows
    print(f"Dev OK rows:     {len(dev_rows)}", flush=True)
    print(f"Held-Out OK rows:{len(hld_rows)}", flush=True)
    print(f"Total to judge:  {len(all_rows)}", flush=True)

    done_keys  = load_done_keys(OUT_CSV)
    print(f"Already judged:  {len(done_keys)} (checkpoint)\n", flush=True)

    write_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
    out_f  = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=JUDGE_FIELDNAMES)
    if write_header:
        writer.writeheader()
        out_f.flush()

    judges = [("JudgeA", JUDGE_A_MODEL), ("JudgeB", JUDGE_B_MODEL)]
    total_possible = len(all_rows) * len(judges)
    total_remaining = total_possible - len(done_keys)
    print(f"Total scoring evaluations remaining: {total_remaining}/{total_possible}\n", flush=True)

    done_count = 0
    t_start = time.perf_counter()

    for row_idx, row in enumerate(all_rows):
        raw_id  = row.get("id", "")
        try:
            qid = int(raw_id)
        except (ValueError, TypeError):
            qid = raw_id

        # Resolve dataset entry â€” try int and str keys
        q_data = q_map.get(qid) or q_map.get(str(qid)) or {}
        query          = q_data.get("query",           f"Query id={qid}")
        ground_truth   = q_data.get("ground_truth_answer", q_data.get("safe_response", "N/A"))
        time_aware_ref = q_data.get("time_aware_reference", "N/A")
        keywords_str   = ", ".join(q_data.get("keywords", []))
        answer         = row.get("full_answer", "")[:2000]

        # Build prompts for all 3 dimensions
        drfact_p   = DRFACT_PROMPT.format(
            query=query, answer=answer,
            ground_truth=ground_truth, time_aware_reference=time_aware_ref
        )
        saa_p      = SAA_PROMPT.format(query=query, answer=answer)
        temporal_p = TEMPORAL_PROMPT.format(query=query, answer=answer)

        for judge_name, judge_model in judges:
            check_key = (row["config"], str(row["run_id"]), str(raw_id), judge_name)
            if check_key in done_keys:
                continue

            elapsed_total = time.perf_counter() - t_start
            rate = done_count / max(elapsed_total, 1)
            eta  = (total_remaining - done_count) / max(rate, 0.01)
            print(
                f"[{done_count+1:5d}/{total_remaining}] "
                f"cfg={row['config']:22s} id={raw_id:4s} run={row.get('run_id','?'):2s} "
                f"{judge_name} ETA={eta/60:.1f}min ...",
                end=" ", flush=True
            )

            # Score all 3 dimensions in parallel (3x speedup per judge)
            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_df  = pool.submit(call_judge, judge_name, drfact_p)
                fut_saa = pool.submit(call_judge, judge_name, saa_p)
                fut_tp  = pool.submit(call_judge, judge_name, temporal_p)
                df_score,  df_rat  = fut_df.result()
                saa_score, saa_rat = fut_saa.result()
                tp_score,  tp_rat  = fut_tp.result()

            # Never write incomplete rows!
            if df_score is None or saa_score is None:
                print(f"FAILED (None score) - skipping to retry later", flush=True)
                continue

            writer.writerow({
                "config":              row["config"],
                "run_id":              row["run_id"],
                "id":                  raw_id,
                "judge":               judge_name,
                "dr_fact":             df_score  if df_score  is not None else "",
                "saa":                 saa_score if saa_score is not None else "",
                "temporal":            tp_score  if tp_score  is not None else "",
                "dr_fact_rationale":   df_rat,
                "saa_rationale":       saa_rat,
                "temporal_rationale":  tp_rat,
            })
            out_f.flush()
            done_keys.add(check_key)
            done_count += 1

            print(
                f"dr={df_score} saa={saa_score} tp={tp_score}",
                flush=True
            )
            time.sleep(0.1)   # small polite gap between API calls

    out_f.close()
    total_time = time.perf_counter() - t_start

    print("\n" + "=" * 70, flush=True)
    print("PHASE 4 COMPLETE", flush=True)
    print(f"Judge evaluations written: {done_count}", flush=True)
    print(f"Output: {OUT_CSV}", flush=True)
    print(f"Total time: {total_time/60:.1f} min", flush=True)
    print("=" * 70, flush=True)

    # Compute & save Cohen's Kappa
    compute_kappa()


if __name__ == "__main__":
    main()
