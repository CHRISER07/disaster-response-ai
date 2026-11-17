"""
evaluation/run_evals.py

IEEE evaluation harness for the Multimodal Agentic RAG Disaster Intelligence system.

Metrics computed:
    KW-F1       Keyword-based F1 (objective, deterministic, no LLM involvement)
    DR-Fact     LLM-as-Judge Disaster-Response Factuality score (0.0-1.0)
    SAA         LLM-as-Judge Situational Awareness Accuracy score (0.0-1.0)
    SVC         Safety Violation Count (deterministic regex, no LLM involvement)
    P@K         Retrieval Precision at K=7
    Latency     Wall-clock seconds per query

All aggregate values are reported as mean +/- population std with 95% CI (1.96 * SEM).

Breakdown tables generated:
    Table 1: Ablation study across all four system configurations
    Table 2: Per-scenario breakdown for Multimodal RAG
    Table 3: Per-category breakdown for Multimodal RAG
    Table 4: Per-difficulty breakdown for Multimodal RAG
    Table 5: Per-CrisisLex-event breakdown for Multimodal RAG

System configurations evaluated:
    LLM Only       No retrieval, direct LLM generation
    Text RAG       Dense retrieval from text + knowledge base only
    Multimodal RAG Full system: all modalities + live tools
    BM25 Baseline  Sparse keyword retrieval, no LLM generation (lower bound)

Dataset options:
    --dataset 50q   evaluation/golden_dataset.json        (original 50-query set)
    --dataset 200q  evaluation/golden_dataset_200q.json   (extended 200-query set)

Usage:
    python evaluation/run_evals.py                       # Full 200-query run
    python evaluation/run_evals.py --dataset 50q         # Original 50-query run
    python evaluation/run_evals.py --quick               # First 10 queries only (smoke test)
    python evaluation/run_evals.py --config bm25         # BM25 baseline only
    python evaluation/run_evals.py --config multimodal   # Multimodal RAG only

Resume logic: if a results CSV already exists, previously completed rows are loaded
and skipped automatically. Safe to interrupt and restart.
"""

import json
import time
import os
import sys
import csv
import math
import argparse
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_chain import get_rag_chain
from core.safety_validator import check_and_count
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(override=False)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def population_std(values):
    """Population standard deviation. Returns 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return round(math.sqrt(variance), 4)


def confidence_interval_95(values):
    """
    95% confidence interval half-width using normal approximation: 1.96 * SEM.
    SEM = population_std / sqrt(n).
    Returns 0.0 for fewer than 2 values.
    """
    if len(values) < 2:
        return 0.0
    std = population_std(values)
    sem = std / math.sqrt(len(values))
    return round(1.96 * sem, 4)


# ---------------------------------------------------------------------------
# Groq rate-limit retry wrapper
# ---------------------------------------------------------------------------

def groq_call_with_retry(fn, max_retries=5):
    """
    Calls fn() with exponential backoff on Groq 429 rate-limit errors.
    Backoff schedule: 60s, 120s, 240s, 480s, 960s.
    Raises RuntimeError after max_retries consecutive failures.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait_s = 60 * (2 ** attempt)
                print(
                    f"    [Rate limit] attempt {attempt + 1}/{max_retries}, "
                    f"waiting {wait_s}s before retry."
                )
                time.sleep(wait_s)
            else:
                raise
    raise RuntimeError(
        f"Groq API call failed after {max_retries} retries. Check rate limits."
    )


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_50Q      = os.path.join(PROJECT_ROOT, "evaluation", "golden_dataset.json")
DATASET_200Q     = os.path.join(PROJECT_ROOT, "evaluation", "golden_dataset_200q.json")
RESULTS_50Q_CSV  = os.path.join(PROJECT_ROOT, "evaluation", "results_per_query.csv")
RESULTS_200Q_CSV = os.path.join(PROJECT_ROOT, "evaluation", "results_200q.csv")
RESULTS_50Q_MD   = os.path.join(PROJECT_ROOT, "evaluation", "results.md")
RESULTS_200Q_MD  = os.path.join(PROJECT_ROOT, "evaluation", "results_200q.md")
RESULTS_EVENT    = os.path.join(PROJECT_ROOT, "evaluation", "results_per_event.csv")

# CSV columns written per query row
CSV_FIELDS = [
    "config", "id", "eval_type", "scenario", "category", "event_source", "difficulty", "query",
    "kw_precision", "kw_recall", "kw_f1",
    "behavioral_score", "dr_fact", "saa", "svc", "p_at_k", "latency_s", "matched_keywords"
]


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

# modalities: None means all modalities (Multimodal RAG)
# use_retriever: whether to call the dense retrieval chain
# bm25: whether to use the BM25 sparse baseline instead
CONFIGS = {
    "LLM Only":       {"modalities": [],                         "use_retriever": False, "bm25": False},
    "Text RAG":       {"modalities": ["text", "knowledge_base"], "use_retriever": True,  "bm25": False},
    "Multimodal RAG": {"modalities": None,                       "use_retriever": True,  "bm25": False},
    "BM25 Baseline":  {"modalities": ["text", "knowledge_base"], "use_retriever": False, "bm25": True},
}


# ---------------------------------------------------------------------------
# BM25 sparse retrieval baseline
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Sparse keyword retrieval baseline using a TF-IDF-style BM25 approximation.

    Reads all documents from the ChromaDB collection into memory and scores
    them using term frequency and unique query-term intersection. No neural
    embeddings and no LLM are involved. This represents the lower bound that
    dense retrieval + LLM generation must exceed to justify its cost.

    Term scoring formula (simplified BM25):
        score(doc, query) = tf(query_terms in doc) + 2.0 * |unique_hits|
    where unique_hits = query_terms that appear in the document term set.
    """

    def __init__(self, k=7):
        self.k = k
        self._corpus = []   # list of {"text": str, "metadata": dict}
        self._built = False

    def build_from_collection(self):
        """
        Load all documents from the persistent ChromaDB collection.
        Called once per evaluation run before the per-query loop.
        """
        try:
            import chromadb
            db_path = os.path.join(PROJECT_ROOT, "chroma_db")
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_or_create_collection("disaster_response")
            result = collection.get(include=["documents", "metadatas"])
            self._corpus = [
                {"text": doc, "metadata": meta}
                for doc, meta in zip(result["documents"], result["metadatas"])
            ]
            self._built = True
            print(
                f"  [BM25] Index built: {len(self._corpus):,} documents "
                f"from ChromaDB collection."
            )
        except Exception as exc:
            print(f"  [BM25] Failed to build index: {exc}")
            self._corpus = []

    def search(self, query):
        """Return top-K documents by BM25 score."""
        if not self._built:
            self.build_from_collection()
        if not self._corpus:
            return []

        query_terms = set(query.lower().split())
        scored = []
        for doc in self._corpus:
            doc_words = doc["text"].lower().split()
            doc_term_set = set(doc_words)
            tf = sum(1 for w in doc_words if w in query_terms)
            unique_hits = len(query_terms & doc_term_set)
            score = tf + 2.0 * unique_hits
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[: self.k]]

    def get_answer(self, query):
        """
        Concatenate top-K retrieved document texts as the BM25 system answer.
        Returns (answer_str, docs_list).
        """
        docs = self.search(query)
        if not docs:
            return "No relevant documents found in knowledge base.", []
        lines = ["Based on retrieved documents:"]
        for doc in docs:
            src = doc["metadata"].get("source", "unknown")
            text_snippet = doc["text"][:300]
            lines.append(f"[{src}] {text_snippet}")
        return "\n\n".join(lines), docs


# ---------------------------------------------------------------------------
# Metric 1: Keyword-Based F1 (fully deterministic, no LLM)
# ---------------------------------------------------------------------------

def keyword_f1(response, expected_keywords):
    """
    Computes Precision, Recall, and F1 from keyword presence in the response.

    True Positives (TP): expected keywords found verbatim (case-insensitive)
                         as substrings in the response.
    False Negatives (FN): expected keywords not found.
    False Positives (FP): length-penalty estimate = max(0, response_words//10 - TP).
                          This penalises verbose responses that miss keywords
                          without inflating FP for concise, accurate responses.

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN) = TP / len(expected_keywords)
    F1        = 2 * P * R / (P + R)

    Multi-word phrases are matched as exact case-insensitive substrings.
    No LLM is involved; result is deterministic and reproducible.
    """
    if not expected_keywords:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched_keywords": []}

    response_lower = response.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in response_lower]
    tp = len(matched)
    fn = len(expected_keywords) - tp
    fp = max(0, len(response.split()) // 10 - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1":               round(f1, 4),
        "matched_keywords": matched,
    }


# ---------------------------------------------------------------------------
# Metric 2: Safety Violation Count (deterministic regex)
# ---------------------------------------------------------------------------

def get_safety_violation_count(answer, item):
    """
    Delegates to core/safety_validator.py check_and_count().
    Returns integer SVC only; does not use the rewritten answer.
    The original answer is always sent to the LLM judge.
    """
    _, svc = check_and_count(answer, item)
    return svc


# ---------------------------------------------------------------------------
# Metric 2b: Behavioral Citation Check (for live-API queries)
# ---------------------------------------------------------------------------

def behavioral_citation_check(answer: str, item: dict) -> float:
    """
    Evaluates a behavioral query (eval_type='behavioral') by checking whether
    the agent's response contains the expected citation patterns.

    For live-API queries, we cannot validate the specific numeric value returned
    (it changes in real time). Instead, we validate WHAT the agent did:
      - Did it cite a live source? (e.g. '[LIVE', 'USGS', 'Open-Meteo')
      - Does the response contain the expected informational patterns?
      - Did it include a severity status?

    Returns a score from 0.0 to 1.0:
      1.0 = all expected citation patterns found
      0.0 = none found
      Partial credit proportional to patterns matched.

    This metric is deterministic and requires no LLM judge.
    """
    patterns = item.get("expected_citation_patterns", [])
    if not patterns:
        # No citation patterns defined - fall back to keyword check only
        return 0.0

    answer_lower = answer.lower()
    matched = sum(1 for p in patterns if p.lower() in answer_lower)
    return round(matched / len(patterns), 4)


# ---------------------------------------------------------------------------
# Metric 3: LLM-as-Judge (holistic quality evaluation)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = PromptTemplate.from_template(
    """You are an impartial academic reviewer evaluating an AI disaster response system.

Question: {query}
Ground Truth Answer: {ground_truth}
AI System Answer: {answer}

Score the AI System Answer on these metrics. Return ONLY a valid JSON object, no other text.

Metrics:
  dr_fact       Disaster-Response Factuality: are the key facts correct and present? (float 0.0-1.0)
  saa           Situational Awareness Accuracy: does the answer correctly capture the disaster situation? (float 0.0-1.0)
  relevant_docs How many of the retrieved documents (if any) appear relevant to the answer? (integer 0-7)

Required format (no markdown, no explanation):
{{"dr_fact": 0.0, "saa": 0.0, "relevant_docs": 0}}"""
)


def judge_response(judge_chain, query, ground_truth, answer):
    """
    Calls the LLM judge and parses its JSON output.
    Returns dict with dr_fact, saa, relevant_docs.
    On any parse error, returns zeros and logs the error.
    """
    try:
        raw = judge_chain.invoke({
            "query":        query,
            "ground_truth": ground_truth,
            "answer":       answer,
        }).content

        # Strip markdown code fences if the judge wraps its output
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].strip()

        scores = json.loads(raw)
        return {
            "dr_fact":      float(scores.get("dr_fact", 0.0)),
            "saa":          float(scores.get("saa", 0.0)),
            "relevant_docs": int(scores.get("relevant_docs", 0)),
        }
    except Exception as exc:
        print(f"    [Judge parse error]: {exc}")
        return {"dr_fact": 0.0, "saa": 0.0, "relevant_docs": 0}


# ---------------------------------------------------------------------------
# Aggregate statistics helpers
# ---------------------------------------------------------------------------

def agg_mean(rows, key):
    """Mean of a numeric field across a list of row dicts."""
    return round(sum(r[key] for r in rows) / len(rows), 4) if rows else 0.0


def agg_std(rows, key):
    """Population std of a numeric field across a list of row dicts."""
    return population_std([r[key] for r in rows]) if rows else 0.0


def agg_ci(rows, key):
    """95% CI half-width of a numeric field across a list of row dicts."""
    return confidence_interval_95([r[key] for r in rows]) if rows else 0.0


def build_summary_record(cfg_name, rows):
    """Build a summary statistics dict for one configuration."""
    return {
        "System":       cfg_name,
        "n":            len(rows),
        "KW-F1":        agg_mean(rows, "kw_f1"),
        "KW-F1-std":    agg_std(rows, "kw_f1"),
        "KW-F1-CI95":   agg_ci(rows, "kw_f1"),
        "KW-Precision": agg_mean(rows, "kw_precision"),
        "KW-Recall":    agg_mean(rows, "kw_recall"),
        "DR-Fact":      agg_mean(rows, "dr_fact"),
        "DR-Fact-std":  agg_std(rows, "dr_fact"),
        "DR-Fact-CI95": agg_ci(rows, "dr_fact"),
        "SAA":          agg_mean(rows, "saa"),
        "SAA-std":      agg_std(rows, "saa"),
        "SAA-CI95":     agg_ci(rows, "saa"),
        "SVC":          sum(r["svc"] for r in rows),
        "P@K":          agg_mean(rows, "p_at_k"),
        "P@K-std":      agg_std(rows, "p_at_k"),
        "Latency(s)":   agg_mean(rows, "latency_s"),
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_markdown_report(
    results_md, dataset_version, dataset, summary,
    scenario_summaries, cat_summary, diff_summary, event_summary
):
    """
    Writes the IEEE-formatted Markdown evaluation report.
    All output is strict ASCII; no Unicode special characters.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(results_md, "w", encoding="utf-8") as f:

        f.write(
            f"# Disaster Response AI -- IEEE Evaluation Results ({dataset_version})\n"
            f"*Generated: {ts} | Queries evaluated: {len(dataset)}*\n\n"
        )

        # Table 1: Ablation study
        f.write("---\n\n")
        f.write("## Table 1: Ablation Study -- All Metrics (mean +/- std)\n\n")
        f.write(
            "| System | n | KW-F1 | std | CI95 | DR-Fact | std | SAA | std | SVC | P@K | Latency(s) |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for m in summary:
            f.write(
                f"| **{m['System']}** | {m['n']} "
                f"| {m['KW-F1']:.4f} | {m['KW-F1-std']:.4f} | {m['KW-F1-CI95']:.4f} "
                f"| {m['DR-Fact']:.4f} | {m['DR-Fact-std']:.4f} "
                f"| {m['SAA']:.4f} | {m['SAA-std']:.4f} "
                f"| {m['SVC']} | {m['P@K']:.4f} | {m['Latency(s)']:.2f} |\n"
            )
        f.write(
            "\nNote: KW-F1 is deterministic (no LLM). DR-Fact and SAA use LLM-as-Judge "
            "(Llama-3.3-70B, temperature=0). SVC uses deterministic regex. "
            "CI95 = 1.96 * SEM. All n values reflect queries with non-empty responses.\n\n"
        )

        # Table 2: Per-scenario breakdown
        f.write("---\n\n")
        f.write("## Table 2: Multimodal RAG -- Per-Scenario Breakdown\n\n")
        f.write("| Scenario | n | KW-F1 | DR-Fact | SAA | SVC |\n")
        f.write("|---|---|---|---|---|---|\n")
        mm_sc = scenario_summaries.get("Multimodal RAG", {})
        for sc in sorted(mm_sc):
            ss = mm_sc[sc]
            f.write(
                f"| {sc} | {ss['n']} | {ss['KW-F1']:.4f} "
                f"| {ss['DR-Fact']:.4f} | {ss['SAA']:.4f} | {ss['SVC']} |\n"
            )

        # Table 3: Per-category breakdown
        f.write("\n---\n\n")
        f.write("## Table 3: Multimodal RAG -- Per-Category Breakdown\n\n")
        f.write("| Category | n | KW-F1 | DR-Fact | SAA | SVC |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cat in sorted(cat_summary):
            cs = cat_summary[cat]
            f.write(
                f"| {cat} | {cs['n']} | {cs['KW-F1']:.4f} "
                f"| {cs['DR-Fact']:.4f} | {cs['SAA']:.4f} | {cs['SVC']} |\n"
            )

        # Table 4: Per-difficulty breakdown
        f.write("\n---\n\n")
        f.write("## Table 4: Multimodal RAG -- Per-Difficulty Breakdown\n\n")
        f.write("| Difficulty | n | KW-F1 | DR-Fact | SAA |\n")
        f.write("|---|---|---|---|---|\n")
        for d in sorted(diff_summary):
            ds = diff_summary[d]
            f.write(
                f"| {d} | {ds['n']} | {ds['KW-F1']:.4f} "
                f"| {ds['DR-Fact']:.4f} | {ds['SAA']:.4f} |\n"
            )

        # Table 5: Per-event breakdown (only if Multimodal RAG rows exist)
        if event_summary:
            f.write("\n---\n\n")
            f.write("## Table 5: Multimodal RAG -- Per-CrisisLex-Event Breakdown\n\n")
            f.write("| Event | n | KW-F1 | DR-Fact | SAA | SVC |\n")
            f.write("|---|---|---|---|---|---|\n")
            for ev in sorted(event_summary):
                es = event_summary[ev]
                f.write(
                    f"| {ev.replace('_', ' ')} | {es['n']} | {es['KW-F1']:.4f} "
                    f"| {es['DR-Fact']:.4f} | {es['SAA']:.4f} | {es['SVC']} |\n"
                )

        # Key findings (computed deltas)
        f.write("\n---\n\n")
        f.write("## Key Findings\n\n")
        llm_row  = next((m for m in summary if m["System"] == "LLM Only"), None)
        mm_row   = next((m for m in summary if m["System"] == "Multimodal RAG"), None)
        bm25_row = next((m for m in summary if m["System"] == "BM25 Baseline"), None)

        if llm_row and mm_row:
            f1_delta  = round(mm_row["KW-F1"]  - llm_row["KW-F1"],  4)
            saa_delta = round(mm_row["SAA"]     - llm_row["SAA"],    4)
            pct_saa   = round(saa_delta / max(llm_row["SAA"], 0.001) * 100, 1)
            f.write(
                f"- KW-F1 gain over LLM Only: {f1_delta:+.4f} "
                f"({mm_row['KW-F1']:.4f} +/- {mm_row['KW-F1-std']:.4f} "
                f"vs {llm_row['KW-F1']:.4f} +/- {llm_row['KW-F1-std']:.4f})\n"
            )
            f.write(
                f"- SAA gain over LLM Only: {pct_saa:+.1f}% "
                f"({mm_row['SAA']:.4f} +/- {mm_row['SAA-std']:.4f} "
                f"vs {llm_row['SAA']:.4f} +/- {llm_row['SAA-std']:.4f})\n"
            )
            f.write(
                f"- Safety: Multimodal RAG {mm_row['SVC']} violations "
                f"vs {llm_row['SVC']} for LLM Only\n"
            )
            f.write(
                f"- Retrieval P@K: {mm_row['P@K']:.4f} +/- {mm_row['P@K-std']:.4f} "
                f"(Multimodal RAG)\n"
            )
        if bm25_row and mm_row:
            f1_over_bm25 = round(mm_row["KW-F1"] - bm25_row["KW-F1"], 4)
            f.write(
                f"- Dense vs Sparse: Multimodal RAG {f1_over_bm25:+.4f} KW-F1 "
                f"over BM25 baseline "
                f"({mm_row['KW-F1']:.4f} vs {bm25_row['KW-F1']:.4f})\n"
            )

        # Methodology notes
        f.write("\n---\n\n")
        f.write("## Methodology\n\n")
        ds_file = (
            "golden_dataset_200q.json" if dataset_version == "200q"
            else "golden_dataset.json"
        )
        f.write(
            f"- Dataset: {len(dataset)} queries from {ds_file} "
            f"grounded in 26 real CrisisLex T26 disaster events.\n"
            "- KW-F1: verbatim keyword matching (case-insensitive substring). "
            "No LLM involvement. Fully deterministic.\n"
            "- DR-Fact / SAA: LLM-as-Judge (Llama-3.3-70B, temperature=0.0). "
            "Acknowledged limitation: self-referential bias when judge model "
            "matches generation model.\n"
            "- SVC: deterministic regex pattern matching. Zero LLM involvement.\n"
            "- BM25 Baseline: TF-IDF keyword retrieval without LLM generation. "
            "Lower bound for demonstrating dense retrieval value.\n"
            "- Statistics: mean +/- population std. "
            "CI95 computed as 1.96 * (std / sqrt(n)).\n"
            f"- Full per-query scores: {os.path.basename(results_md).replace('.md', '.csv')}\n"
            "- Per-event scores: results_per_event.csv\n"
        )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(quick=False, dataset_version="200q", configs_filter=None, local=False, model_name=None):
    """
    Runs the full evaluation pipeline.

    Args:
        quick:           If True, evaluates only the first 10 queries (smoke test).
        dataset_version: "50q" or "200q".
        configs_filter:  String substring to select specific configurations.
                         None = run all four configurations.
    """
    # Select dataset and output paths
    if dataset_version == "50q":
        dataset_path = DATASET_50Q
        results_csv  = os.path.join(PROJECT_ROOT, "evaluation", "results_local.csv") if local else RESULTS_50Q_CSV
        results_md   = os.path.join(PROJECT_ROOT, "evaluation", "results_local.md") if local else RESULTS_50Q_MD
    else:
        dataset_path = DATASET_200Q
        results_csv  = os.path.join(PROJECT_ROOT, "evaluation", "results_local_200q.csv") if local else RESULTS_200Q_CSV
        results_md   = os.path.join(PROJECT_ROOT, "evaluation", "results_local_200q.md") if local else RESULTS_200Q_MD

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print(
            "  Generate it first:\n"
            "    python evaluation/generate_extended_dataset.py"
        )
        return

    with open(dataset_path, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)

    if quick:
        dataset = dataset[:10]
        print(f"[Smoke test] Running first {len(dataset)} queries only.")

    # Select which configurations to run
    active_configs = dict(CONFIGS)
    if configs_filter:
        active_configs = {
            k: v for k, v in CONFIGS.items()
            if configs_filter.lower() in k.lower()
        }
        if not active_configs:
            print(
                f"[WARN] No configuration matched '{configs_filter}'. "
                f"Running all configurations."
            )
            active_configs = dict(CONFIGS)

    print(f"\nDataset: {os.path.basename(dataset_path)} ({len(dataset)} queries)")
    print(f"Configurations: {list(active_configs.keys())}\n")

    # Rate-limit sleeps configuration
    sleep_before_judge = 0.1 if local else 2.0
    sleep_after_query = 0.1 if local else 2.0

    # Initialise models
    if local:
        m_name = model_name or "llama3.2"
        print(f"[Ollama] Initializing local model: {m_name}")
        llm         = ChatOllama(model=m_name, temperature=0.0)
        judge_llm   = ChatOllama(model=m_name, temperature=0.0)
    else:
        m_name = model_name or "llama-3.3-70b-versatile"
        print(f"[Groq] Initializing cloud model: {m_name}")
        llm         = ChatGroq(model_name=m_name, temperature=0.0)
        judge_llm   = ChatGroq(model_name=m_name, temperature=0.0)

    judge_chain = JUDGE_PROMPT | judge_llm
    bm25        = BM25Retriever(k=7)

    all_rows   = []
    config_agg = {cfg: [] for cfg in active_configs}
    completed  = set()   # set of (config_name, query_id) already in CSV

    # Resume from prior run if results CSV exists
    if os.path.exists(results_csv):
        try:
            with open(results_csv, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    row["id"]               = int(row["id"])
                    row["kw_precision"]     = float(row["kw_precision"])
                    row["kw_recall"]        = float(row["kw_recall"])
                    row["kw_f1"]            = float(row["kw_f1"])
                    row["dr_fact"]          = float(row["dr_fact"])
                    row["saa"]              = float(row["saa"])
                    row["svc"]              = int(row["svc"])
                    row["p_at_k"]           = float(row["p_at_k"])
                    row["latency_s"]        = float(row["latency_s"])
                    # Back-fill fields added in newer schema versions
                    row.setdefault("event_source",     "unknown")
                    row.setdefault("difficulty",       "medium")
                    row.setdefault("eval_type",        "static")
                    row.setdefault("behavioral_score", "0.0")
                    row["behavioral_score"] = float(row["behavioral_score"])
                    all_rows.append(row)
                    if row["config"] in config_agg:
                        config_agg[row["config"]].append(row)
                    completed.add((row["config"], row["id"]))
            print(
                f"[Resume] Loaded {len(all_rows)} completed rows "
                f"from {os.path.basename(results_csv)}."
            )
        except Exception as exc:
            print(f"[Resume failed] Starting from scratch. Reason: {exc}")
            all_rows   = []
            config_agg = {cfg: [] for cfg in active_configs}
            completed  = set()


    # Per-configuration evaluation loop
    for cfg_name, cfg_params in active_configs.items():
        print(f"\n{'=' * 60}")
        print(f"  Configuration: {cfg_name}")
        print(f"{'=' * 60}")

        # Build retrieval chain or BM25 index once per configuration
        chain = None
        if cfg_params.get("bm25"):
            if not bm25._built:
                bm25.build_from_collection()
        elif cfg_params["use_retriever"]:
            chain = get_rag_chain(allowed_modalities=cfg_params["modalities"], llm=llm)

        for item in dataset:
            qid        = item["id"]
            query      = item["query"]
            category   = item.get("category", "general")
            scenario   = item.get("scenario", "")
            event_src  = item.get("event_source", "unknown")
            difficulty = item.get("difficulty", "medium")
            eval_type  = item.get("eval_type", "static")   # NEW: static or behavioral
            gt         = item.get("ground_truth_answer", "")
            keywords   = item.get("expected_keywords", [])

            if (cfg_name, qid) in completed:
                print(f"  [Q{qid:03d}] SKIP (cached)")
                continue

            print(f"  [Q{qid:03d}] [{eval_type.upper()}] {query[:65]}")

            # Generate answer
            t0 = time.time()
            try:
                if cfg_params.get("bm25"):
                    answer, retrieved_docs = bm25.get_answer(query)
                elif not cfg_params["use_retriever"]:
                    answer = groq_call_with_retry(lambda: llm.invoke(query).content)
                    retrieved_docs = []
                else:
                    cur_chain = chain
                    res = groq_call_with_retry(
                        lambda c=cur_chain: c.invoke({"input": query})
                    )
                    answer         = res["answer"]
                    retrieved_docs = res.get("context", [])
            except Exception as exc:
                print(f"    [Generation error]: {exc}")
                answer         = ""
                retrieved_docs = []
            latency = round(time.time() - t0, 2)

            # Metric 1: Keyword F1 (works for both static and behavioral)
            kw = keyword_f1(answer, keywords)

            # Metric 2: Safety Violation Count (always run)
            svc = get_safety_violation_count(answer, item)

            # Metric 2b: Behavioral Citation Score (behavioral queries only)
            behavioral_score = 0.0
            if eval_type == "behavioral":
                behavioral_score = behavioral_citation_check(answer, item)

            # Metric 3: LLM-as-Judge
            # For static queries: judge against fixed ground_truth_answer
            # For behavioral queries: ground_truth_answer is null - skip judge to avoid
            # meaningless evaluation of live sensor values against a fixed reference.
            if eval_type == "static" and gt:
                time.sleep(sleep_before_judge)
                judge_scores = groq_call_with_retry(
                    lambda: judge_response(judge_chain, query, gt, answer)
                )
            else:
                # Behavioral: no fixed ground truth to judge against
                judge_scores = {"dr_fact": 0.0, "saa": 0.0, "relevant_docs": 0}
                if eval_type == "behavioral":
                    # Use behavioral_score as a proxy for both DR-Fact and SAA
                    judge_scores["dr_fact"] = behavioral_score
                    judge_scores["saa"]     = behavioral_score

            # Metric 4: Precision@K
            p_at_k = (
                round(min(judge_scores["relevant_docs"], 7) / 7, 4)
                if retrieved_docs else 0.0
            )

            row = {
                "config":            cfg_name,
                "id":                qid,
                "eval_type":         eval_type,
                "scenario":          scenario,
                "category":          category,
                "event_source":      event_src,
                "difficulty":        difficulty,
                "query":             query,
                "kw_precision":      kw["precision"],
                "kw_recall":         kw["recall"],
                "kw_f1":             kw["f1"],
                "behavioral_score":  behavioral_score,
                "dr_fact":           judge_scores["dr_fact"],
                "saa":               judge_scores["saa"],
                "svc":               svc,
                "p_at_k":            p_at_k,
                "latency_s":         latency,
                "matched_keywords":  "|".join(kw["matched_keywords"]),
            }
            all_rows.append(row)
            config_agg[cfg_name].append(row)

            print(
                f"         KW-F1={kw['f1']:.3f}  Behav={behavioral_score:.3f}"
                f"  DR-Fact={judge_scores['dr_fact']:.3f}"
                f"  SAA={judge_scores['saa']:.3f}  SVC={svc}  {latency:.1f}s"
            )

            # Atomic CSV write
            tmp_path = results_csv + ".tmp"
            with open(tmp_path, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(all_rows)
            os.replace(tmp_path, results_csv)

            time.sleep(sleep_after_query)

    print(f"\n[DONE] Per-query results saved to {results_csv}")

    # Build aggregate statistics
    summary = [
        build_summary_record(cfg_name, rows)
        for cfg_name, rows in config_agg.items()
        if rows
    ]

    # Per-scenario breakdown (Multimodal RAG)
    scenario_summaries = {}
    for cfg_name, rows in config_agg.items():
        if not rows:
            continue
        by_scenario = defaultdict(list)
        for r in rows:
            by_scenario[r["scenario"]].append(r)
        scenario_summaries[cfg_name] = {
            sc: {
                "n":       len(rs),
                "KW-F1":   agg_mean(rs, "kw_f1"),
                "DR-Fact": agg_mean(rs, "dr_fact"),
                "SAA":     agg_mean(rs, "saa"),
                "SVC":     sum(r["svc"] for r in rs),
            }
            for sc, rs in by_scenario.items()
        }

    mm_rows = config_agg.get("Multimodal RAG", [])

    # Per-event breakdown
    by_event = defaultdict(list)
    for r in mm_rows:
        by_event[r.get("event_source", "unknown")].append(r)
    event_summary = {
        ev: {
            "n":       len(rs),
            "KW-F1":   agg_mean(rs, "kw_f1"),
            "DR-Fact": agg_mean(rs, "dr_fact"),
            "SAA":     agg_mean(rs, "saa"),
            "SVC":     sum(r["svc"] for r in rs),
        }
        for ev, rs in by_event.items()
    }

    # Per-difficulty breakdown
    by_diff = defaultdict(list)
    for r in mm_rows:
        by_diff[r.get("difficulty", "medium")].append(r)
    diff_summary = {
        d: {
            "n":       len(rs),
            "KW-F1":   agg_mean(rs, "kw_f1"),
            "DR-Fact": agg_mean(rs, "dr_fact"),
            "SAA":     agg_mean(rs, "saa"),
        }
        for d, rs in by_diff.items()
    }

    # Per-category breakdown
    by_cat = defaultdict(list)
    for r in mm_rows:
        by_cat[r["category"]].append(r)
    cat_summary = {
        cat: {
            "n":       len(rs),
            "KW-F1":   agg_mean(rs, "kw_f1"),
            "DR-Fact": agg_mean(rs, "dr_fact"),
            "SAA":     agg_mean(rs, "saa"),
            "SVC":     sum(r["svc"] for r in rs),
        }
        for cat, rs in by_cat.items()
    }

    # Write per-event CSV
    if event_summary:
        with open(RESULTS_EVENT, "w", newline="", encoding="utf-8") as ef:
            ew = csv.DictWriter(
                ef, fieldnames=["event", "n", "KW-F1", "DR-Fact", "SAA", "SVC"]
            )
            ew.writeheader()
            for ev in sorted(event_summary):
                ew.writerow({"event": ev, **event_summary[ev]})
        print(f"[OK] Per-event breakdown saved to {RESULTS_EVENT}")

    # Write Markdown report
    write_markdown_report(
        results_md, dataset_version, dataset, summary,
        scenario_summaries, cat_summary, diff_summary, event_summary
    )

    # Print final console table
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY  (mean +/- std)")
    print(f"{'=' * 80}")
    header = (
        f"{'System':<22}  {'KW-F1':>12}  {'DR-Fact':>12}  "
        f"{'SAA':>12}  {'SVC':>5}  {'P@K':>8}"
    )
    print(header)
    print("-" * 80)
    for m in summary:
        print(
            f"{m['System']:<22}  "
            f"{m['KW-F1']:>6.4f}+/-{m['KW-F1-std']:<5.4f}  "
            f"{m['DR-Fact']:>6.4f}+/-{m['DR-Fact-std']:<5.4f}  "
            f"{m['SAA']:>6.4f}+/-{m['SAA-std']:<5.4f}  "
            f"{m['SVC']:>5}  "
            f"{m['P@K']:>8.4f}"
        )

    print(f"\n[DONE] Report written to {results_md}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IEEE evaluation harness for the Multimodal Disaster RAG system."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke test: evaluate only the first 10 queries."
    )
    parser.add_argument(
        "--dataset", choices=["50q", "200q"], default="200q",
        help="Benchmark version to use (default: 200q)."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Run only configurations whose name contains this substring. "
            "Examples: 'bm25', 'multimodal', 'text', 'llm'. Default: run all."
        )
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use local Ollama model instead of Groq API."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Local Ollama model name (default: llama3.2)."
    )
    args = parser.parse_args()
    run_evaluation(
        quick=args.quick,
        dataset_version=args.dataset,
        configs_filter=args.config,
        local=args.local,
        model_name=args.model,
    )
