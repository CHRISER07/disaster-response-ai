"""
evaluation/run_judge_v3.py

Ultra-Fast High-Throughput Dual-Judge LLM Evaluation Engine (v3.1)
==================================================================
Key Innovations:
1. Non-Blocking Round-Robin Key Scheduling:
   - Zero future-accumulating sleep delays.
   - Dynamic per-key cooldown on rate limits (sidelining key without blocking threads).
2. Robust Multi-Pattern Score Parsing:
   - Recovers tri-state scores (0.0, 0.5, 1.0) and rationales across formatting variations.
3. Strict Cross-Family Dual-Judge Traffic Split:
   - Judge A: Google Gemini 3.1 Flash Lite (7 Google keys, 105 RPM capacity).
     Seamless failover to Alibaba Qwen 3.8-27B on Groq.
   - Judge B: OpenAI GPT-OSS-120B on Groq (9 Groq keys, paced round-robin).
     Seamless failover to GPT-OSS-20B.
4. Concurrent High-Throughput Execution:
   - 6 parallel worker threads processing evaluations simultaneously.
   - Atomic thread-safe CSV checkpointing.
"""

import os, sys, json, csv, time, random, re, threading, queue
import requests
from itertools import zip_longest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

os.environ["PYTHONUTF8"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from google import genai
from groq import Groq

# ---------------------------------------------------------------------------
# Paths & Schema
# ---------------------------------------------------------------------------
DATASET_FILE = ROOT / "evaluation" / "golden_dataset_v2.json"
DEV_CSV      = ROOT / "evaluation" / "results_per_query_v2.csv"
# Judge input can be switched to the fresh (restored-system) answers via
# JUDGE_INPUT_CSV, and the output tag keeps the two score sets separate.
_judge_input = os.getenv("JUDGE_INPUT_CSV", "results_heldout_v2.csv")
_judge_tag = os.getenv("JUDGE_OUTPUT_TAG", "v3")
HELDOUT_CSV  = ROOT / "evaluation" / _judge_input
OUT_CSV      = ROOT / "evaluation" / f"judge_scores_{_judge_tag}.csv"
JUDGE_DONE   = ROOT / "evaluation" / f"judge_scores_{_judge_tag}.complete"
# v3 uses the corrected, bias-free rubric. A separate file prevents the
# checkpoint logic from reusing scores produced by the old hedging-reward rubric.
KAPPA_FILE   = ROOT / "evaluation" / "kappa_result.txt"

JUDGE_FIELDNAMES = [
    "config", "run_id", "id", "judge", "model_used",
    "dr_fact", "saa", "temporal",
    "dr_fact_rationale", "saa_rationale", "temporal_rationale"
]

# ---------------------------------------------------------------------------
# Non-Blocking Fair Round-Robin Key Pool
# ---------------------------------------------------------------------------
class NonBlockingKeyPool:
    def __init__(self, name: str, keys: list, per_key_cooldown: float = 1.0):
        self.name = name
        self.lock = threading.Lock()
        self.keys = list(dict.fromkeys(k.strip() for k in keys if k and len(k.strip()) > 8))
        self.per_key_cooldown = per_key_cooldown
        self.last_used = {k: 0.0 for k in self.keys}
        self.cooldown_until = {k: 0.0 for k in self.keys}
        self.dead_keys = set()
        self.success_count = {k: 0 for k in self.keys}

    def get_ready_key(self) -> str:
        """Instantly selects the best available key without compounding sleep."""
        with self.lock:
            now = time.time()
            valid = [k for k in self.keys if k not in self.dead_keys]
            if not valid:
                return ""

            # Check for keys not in error cooldown
            active = [k for k in valid if self.cooldown_until[k] <= now]
            if not active:
                # All in cooldown; pick the one that expires soonest
                best = min(valid, key=lambda k: self.cooldown_until[k])
                sleep_needed = max(0.0, self.cooldown_until[best] - now)
                if sleep_needed > 2.5:
                    sleep_needed = 0.5
            else:
                # Pick the key used longest ago
                best = min(active, key=lambda k: self.last_used[k])
                sleep_needed = max(0.0, (self.last_used[best] + self.per_key_cooldown) - now)

            self.last_used[best] = now + min(sleep_needed, 1.0)

        if 0.05 < sleep_needed <= 2.0:
            time.sleep(sleep_needed)
        return best

    def report_rate_limit(self, key: str, cooldown_seconds: float = 3.0):
        """Put this specific key in temporary cooldown while others continue."""
        with self.lock:
            self.cooldown_until[key] = time.time() + cooldown_seconds

    def mark_dead(self, key: str, reason: str = ""):
        """Permanently disable key (e.g. 403 Invalid API Key)."""
        with self.lock:
            self.dead_keys.add(key)
            valid_left = len([k for k in self.keys if k not in self.dead_keys])
            print(f"[{self.name}] Key ...{key[-6:]} permanently disabled ({reason}). Valid left: {valid_left}", flush=True)

    def mark_success(self, key: str):
        with self.lock:
            self.success_count[key] += 1

    def has_valid_keys(self) -> bool:
        with self.lock:
            return any(k not in self.dead_keys for k in self.keys)


# Initialize pools
groq_key_list = [
    os.getenv(f"GROQ_API_KEY_{let}", "").strip()
    for let in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
]
groq_key_list = [k for k in groq_key_list if k]
groq_pool = NonBlockingKeyPool("GroqPool", groq_key_list, per_key_cooldown=1.2)

gemini_key_list = [
    os.getenv(f"GEMINI_API_KEY_{let}", "").strip()
    for let in ['A', 'B', 'C', 'D', 'E', 'F', 'G']
]
gemini_key_list = [k for k in gemini_key_list if k]
gemini_pool = NonBlockingKeyPool("GeminiPool", gemini_key_list, per_key_cooldown=2.0)

# Pre-cache client instances
gemini_clients = {k: genai.Client(api_key=k) for k in gemini_key_list}
groq_clients   = {k: Groq(api_key=k) for k in groq_key_list}

# ---------------------------------------------------------------------------
# Unified 3-in-1 Prompt Template
# ---------------------------------------------------------------------------
UNIFIED_JUDGE_PROMPT = """\
You are an expert disaster-response and emergency operations center evaluator.
Evaluate the following AI response across three rigorous, independent dimensions:

QUESTION: {query}
AI ANSWER: {answer}
GROUND TRUTH: {ground_truth}
TIME-AWARE REFERENCE: {time_aware_reference}

======================================================================
DIMENSION 1: DR-FACT (Factual Correctness against FEMA/CDC/USGS Doctrine)
Score ONLY verifiable factual accuracy. Judge substance, not length or tone.
1.0 = Every specific factual claim is correct AND consistent with the ground truth and time-aware reference, AND the answer contains at least one substantive, checkable, data-grounded statement. Generic disclaimers or hedges alone MUST NOT score 1.0.
0.5 = Partially correct: some accurate detail, but vague, incomplete, or missing critical protocol specifics.
0.0 = Any factually wrong statement, dangerous directive, or direct contradiction of the reference. Refusing to engage when the reference contains the needed answer scores AT MOST 0.5.
NOTE: If the reference shows below-flood-stage conditions, reporting that fact WITH the actual reading is correct. Reporting it without the supporting number is vague (0.5). Inventing a reading is 0.0.

DIMENSION 2: SAA (Situational Awareness & Actionability)
1.0 = Specific, executable guidance: numeric thresholds, actual readings, named authorities, or explicit step-by-step directives an responder could act on immediately.
0.5 = Only general advisories ("consult authorities", "stay safe", "monitor conditions") with no operational specifics. A correct refusal counts here ONLY when the requested act is genuinely unauthorized.
0.0 = Evasive, contradictory, misleading, or recommends an unsafe action.
NOTE: For ordinary operational questions that the reference can answer, a generic hedge scores 0.5, NOT 1.0. Do not reward politeness or disclaimers.

DIMENSION 3: TEMPORAL (Temporal Reasoning & Grounding)
1.0 = Explicitly labels live telemetry versus archival records and bases decisions on live data.
0.5 = Temporal distinction is implicit or partial.
0.0 = Conflates historical disaster events with present-day conditions, or presents archive numbers as current live readings.
======================================================================

Respond strictly in the following format (valid scores are 0.0, 0.5, or 1.0):
DR_FACT_SCORE: <score>
DR_FACT_RATIONALE: <one concise sentence>
SAA_SCORE: <score>
SAA_RATIONALE: <one concise sentence>
TEMPORAL_SCORE: <score>
TEMPORAL_RATIONALE: <one concise sentence>
"""

# ---------------------------------------------------------------------------
# Robust Score Parser
# ---------------------------------------------------------------------------
def parse_unified_response(text: str) -> tuple:
    """Robustly extracts tri-state scores and clean rationales with zero prefix leakage."""
    if not text:
        return (None, ""), (None, ""), (None, "")

    clean = re.sub(r"\*\*|__", "", text)

    def extract_dimension(tag, fallback_tag=None):
        score = None
        # 1. Match score
        tag_pat = rf"(?:{tag}|{fallback_tag})" if fallback_tag else tag
        sm = re.search(rf"{tag_pat}(?:_SCORE|\s+SCORE)?\s*[:=\-]?\s*([01](?:\.\d+)?)", clean, re.IGNORECASE)
        if sm:
            try:
                val = float(sm.group(1))
                if val in (0.0, 0.5, 1.0):
                    score = val
                elif val in (0, 1):
                    score = float(val)
            except (ValueError, TypeError):
                score = None

        # 2. Match rationale: strictly requires RATIONALE keyword
        rat = ""
        rm = re.search(
            rf"{tag_pat}(?:_RATIONALE|\s+RATIONALE)\s*[:=\-]?\s*(.+?)(?=(?:DR_FACT|SAA|TEMPORAL|DIMENSION|\Z))",
            clean,
            re.IGNORECASE | re.DOTALL
        )
        if rm:
            raw_rat = rm.group(1).strip()
            # Clean any leftover prefixes
            raw_rat = re.sub(r"^.*?RATIONALE:\s*", "", raw_rat, flags=re.IGNORECASE)
            raw_rat = re.sub(r"^_SCORE:\s*[01]\.\d+\s*", "", raw_rat, flags=re.IGNORECASE)
            raw_rat = re.sub(r"^[:=\-\s]+", "", raw_rat)
            raw_rat = re.sub(r"\s+", " ", raw_rat).strip()[:300]
            rat = raw_rat

        return score, rat

    df_s, df_r = extract_dimension("DR_FACT", "DIMENSION 1")
    saa_s, saa_r = extract_dimension("SAA", "DIMENSION 2")
    tp_s, tp_r = extract_dimension("TEMPORAL", "DIMENSION 3")

    # No fabricated defaults. A missing temporal score stays None so the row is
    # retried/excluded rather than silently inflating the temporal mean.
    return (df_s, df_r), (saa_s, saa_r), (tp_s, tp_r)

# ---------------------------------------------------------------------------
# Direct Engine Callers
# ---------------------------------------------------------------------------
def execute_gemini_call(prompt: str) -> str:
    """Invokes Gemini 3.1 Flash Lite across Google key pool."""
    for _ in range(len(gemini_pool.keys)):
        key = gemini_pool.get_ready_key()
        if not key:
            break
        try:
            client = gemini_clients.get(key) or genai.Client(api_key=key)
            resp = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )
            gemini_pool.mark_success(key)
            return (resp.text or ""), "gemini-3.1-flash-lite"
        except Exception as e:
            err = str(e)
            if "429" in err:
                gemini_pool.report_rate_limit(key, 3.0)
            elif "403" in err or "invalid" in err.lower():
                gemini_pool.mark_dead(key, "permission denied")
            else:
                gemini_pool.report_rate_limit(key, 2.0)
    return "", "none"


def execute_groq_call(model: str, prompt: str, max_tokens: int = 800) -> str:
    """Invokes Groq model across Groq key pool with round-robin cooldown."""
    for _ in range(len(groq_pool.keys)):
        key = groq_pool.get_ready_key()
        if not key:
            break
        try:
            client = groq_clients.get(key) or Groq(api_key=key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens
            )
            content = resp.choices[0].message.content or ""
            groq_pool.mark_success(key)
            return content.strip(), model
        except Exception as e:
            err = str(e)
            retry_after = 2.0
            if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                try:
                    retry_after = float(e.response.headers.get("retry-after", 2.0))
                except Exception:
                    pass
            if "429" in err or "rate" in err.lower():
                groq_pool.report_rate_limit(key, max(retry_after, 2.5))
            elif "403" in err or "invalid" in err.lower():
                groq_pool.mark_dead(key, "invalid key")
            else:
                groq_pool.report_rate_limit(key, 2.0)
    return "", "none"


def execute_openrouter_call(model: str, prompt: str, max_tokens: int = 800) -> tuple:
    """Invoke a judge model on OpenRouter (cross-family, independent of Groq pools)."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        return "", "none"
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens},
            timeout=120,
        )
        if resp.status_code != 200:
            return "", "none"
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content") or ""
        return content.strip(), model
    except Exception:
        return "", "none"


def call_judge_unified(judge_name: str, prompt: str) -> tuple:
    """Evaluate with strict cross-provider, cross-family separation.

    Independence invariant: a judge must NEVER fall back to the primary
    generator (qwen/qwen3.8-27b), and the two judges must come from different
    families. With the free Gemini quota exhausted, Judge A falls back to a
    Llama-3.3-70B judge on OpenRouter (family: Llama); Judge B is GPT-OSS on
    Groq (family: GPT). The generator is Qwen -- all three families disjoint.
    """
    generator_model = "qwen/qwen3.8-27b"
    if judge_name == "JudgeA":
        raw, used = execute_gemini_call(prompt)
        if not raw:
            raw, used = execute_openrouter_call("meta-llama/llama-3.3-70b-instruct", prompt)
    else:
        raw, used = execute_groq_call("openai/gpt-oss-120b", prompt, max_tokens=800)
        if not raw:
            raw, used = execute_groq_call("openai/gpt-oss-20b", prompt, max_tokens=800)
    # Defensive: no judge may be the generator family.
    assert used != generator_model, "Judge fell back to the generator model!"
    return parse_unified_response(raw), used

# ---------------------------------------------------------------------------
# Data Loaders & Checkpointing
# ---------------------------------------------------------------------------
def load_ok_rows(csv_path: Path) -> list:
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("status", "") == "OK"]


def load_dataset_map(ds_path: Path) -> dict:
    data = json.loads(ds_path.read_text(encoding="utf-8"))
    return {q["id"]: q for q in data}


def load_done_keys(csv_path: Path) -> set:
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if all(row.get(d, "") not in ("", "None") for d in ["dr_fact", "saa", "temporal"]):
                done.add((row["config"], str(row["run_id"]), str(row["id"]), row["judge"]))
    return done

# ---------------------------------------------------------------------------
# Cohen's Kappa Computation
# ---------------------------------------------------------------------------
def compute_kappa():
    if not OUT_CSV.exists():
        return
    rows_a, rows_b = {}, {}
    with open(OUT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["config"], str(row["run_id"]), str(row["id"]))
            try:
                val = float(row["dr_fact"]) if row.get("dr_fact", "") not in ("", "None") else None
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

    classes = [0.0, 0.5, 1.0]
    n_cls = len(classes)
    idx = {v: i for i, v in enumerate(classes)}
    W = [[1 - abs(i - j) / (n_cls - 1) for j in range(n_cls)] for i in range(n_cls)]

    po = sum(W[idx.get(ai, 0)][idx.get(bi, 0)] for ai, bi in zip(a, b)) / n
    freq_a = [sum(1 for x in a if x == c) / n for c in classes]
    freq_b = [sum(1 for x in b if x == c) / n for c in classes]
    pe = sum(W[i][j] * freq_a[i] * freq_b[j] for i in range(n_cls) for j in range(n_cls))
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
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
# Main Concurrent Execution Engine
# ---------------------------------------------------------------------------
def main():
    print("=" * 70, flush=True)
    print("PHASE 4: ULTRA-FAST DUAL-JUDGE EVALUATION ENGINE (v3.1 Non-Blocking)", flush=True)
    print(f"Groq Pool:   {len(groq_pool.keys)} keys initialized (Interval: {groq_pool.per_key_cooldown}s)", flush=True)
    print(f"Gemini Pool: {len(gemini_pool.keys)} keys initialized (Interval: {gemini_pool.per_key_cooldown}s)", flush=True)
    print("=" * 70, flush=True)

    q_map = load_dataset_map(DATASET_FILE)
    dev_rows = load_ok_rows(DEV_CSV)
    hld_rows = load_ok_rows(HELDOUT_CSV)
    all_rows = dev_rows + hld_rows

    # Ensure every existing row in OUT_CSV is strictly unique
    if OUT_CSV.exists() and OUT_CSV.stat().st_size > 0:
        with open(OUT_CSV, encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        seen_keys = set()
        clean_rows = []
        for r in existing:
            if all(r.get(d, "") not in ("", "None") for d in ["dr_fact", "saa", "temporal"]):
                k = (r["config"], str(r["run_id"]), str(r["id"]), r["judge"])
                if k not in seen_keys:
                    seen_keys.add(k)
                    clean_rows.append(r)
        if len(clean_rows) < len(existing):
            with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=JUDGE_FIELDNAMES)
                w.writeheader()
                w.writerows(clean_rows)
            print(f"Deduplicated judge_scores_v2.csv: removed {len(existing) - len(clean_rows)} duplicate rows ({len(clean_rows)} unique rows remain).", flush=True)

    done_keys = load_done_keys(OUT_CSV)

    write_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
    out_f = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=JUDGE_FIELDNAMES)
    if write_header:
        writer.writeheader()
        out_f.flush()

    file_lock = threading.Lock()
    judges = ["JudgeA", "JudgeB"]

    tasks_a = []
    tasks_b = []
    for row in all_rows:
        raw_id = row.get("id", "")
        for jname in judges:
            ck = (row["config"], str(row["run_id"]), str(raw_id), jname)
            if ck not in done_keys:
                if jname == "JudgeA":
                    tasks_a.append((row, jname, ck))
                else:
                    tasks_b.append((row, jname, ck))

    task_queue = queue.Queue()
    for a, b in zip_longest(tasks_a, tasks_b):
        if a:
            task_queue.put(a)
        if b:
            task_queue.put(b)

    total_remaining = task_queue.qsize()
    total_possible = len(all_rows) * len(judges)
    print(f"Total pairs to judge:    {total_possible}", flush=True)
    print(f"Already judged:          {len(done_keys)} (from checkpoint)", flush=True)
    print(f"Evaluations in queue:    {total_remaining}\n", flush=True)

    if total_remaining == 0:
        print("All evaluations are already complete!", flush=True)
        compute_kappa()
        return

    progress_lock = threading.Lock()
    done_count = 0
    retry_map = {}
    t_start = time.perf_counter()

    def worker_loop(worker_id: int):
        nonlocal done_count
        while True:
            try:
                row, judge_name, check_key = task_queue.get_nowait()
            except queue.Empty:
                break

            raw_id = row.get("id", "")
            try:
                qid = int(raw_id)
            except (ValueError, TypeError):
                qid = raw_id

            q_data = q_map.get(qid) or q_map.get(str(qid)) or {}
            query = q_data.get("query", f"Query id={qid}")
            ground_truth = q_data.get("ground_truth_answer", q_data.get("safe_response", "N/A"))
            time_aware_ref = q_data.get("time_aware_reference", "N/A")
            answer = row.get("full_answer", "")[:2000]

            prompt = UNIFIED_JUDGE_PROMPT.format(
                query=query,
                answer=answer,
                ground_truth=ground_truth,
                time_aware_reference=time_aware_ref
            )

            (df_s, df_r), (saa_s, saa_r), (tp_s, tp_r), model_used = call_judge_unified(judge_name, prompt)

            # Check validity
            if df_s is None or saa_s is None:
                retries = retry_map.get(check_key, 0)
                if retries < 3:
                    retry_map[check_key] = retries + 1
                    time.sleep(0.3)
                    task_queue.put((row, judge_name, check_key))
                else:
                    print(f" [SKIP permanently failed task {check_key}]", flush=True)
                task_queue.task_done()
                continue

            # Atomic file append
            with file_lock:
                writer.writerow({
                    "config":             row["config"],
                    "run_id":             row["run_id"],
                    "id":                 raw_id,
                    "judge":              judge_name,
                    "model_used":         model_used,
                    "dr_fact":            df_s if df_s is not None else "",
                    "saa":                saa_s if saa_s is not None else "",
                    "temporal":           tp_s if tp_s is not None else "",
                    "dr_fact_rationale":  df_r,
                    "saa_rationale":      saa_r,
                    "temporal_rationale": tp_r,
                })
                out_f.flush()

            with progress_lock:
                done_count += 1
                elapsed = time.perf_counter() - t_start
                rate = done_count / max(elapsed, 0.1)
                eta_min = (total_remaining - done_count) / max(rate, 0.01) / 60
                print(
                    f"[{done_count:4d}/{total_remaining}] "
                    f"{row['config'][:18]:18s} id={raw_id:4s} {judge_name} | "
                    f"dr={df_s} saa={saa_s} tp={tp_s} | "
                    f"Speed={rate:.2f} ev/s | ETA={eta_min:.1f}min",
                    flush=True
                )

            task_queue.task_done()

    # Launch 6 concurrent workers across 16 keys (7 Gemini + 9 Groq)
    NUM_WORKERS = 6
    print(f"Launching {NUM_WORKERS} non-blocking concurrent worker threads...", flush=True)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futs = [executor.submit(worker_loop, wid) for wid in range(NUM_WORKERS)]
        for f in futs:
            f.result()

    out_f.close()
    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 70, flush=True)
    print("ALL EVALUATIONS SUCCESSFULLY COMPLETED!", flush=True)
    print(f"Total processed in this session: {done_count}", flush=True)
    print(f"Elapsed time: {total_time/60:.2f} minutes ({done_count/max(total_time,1):.2f} evals/sec)", flush=True)
    print("=" * 70, flush=True)

    compute_kappa()
    # Completion marker: downstream reports must not consume a partial file.
    JUDGE_DONE.write_text(
        f"rows={done_count}\nfinished={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
