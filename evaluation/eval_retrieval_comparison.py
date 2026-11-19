"""
evaluation/eval_retrieval_comparison.py

Rigorously evaluates retrieval across DAB-200 with strict Train/Dev/Test separation:
1. Evaluates BM25 Sparse (SQLite FTS5), Dense Semantic (BGE-small), and Hybrid RRF.
2. Conducts an Alpha Sweep (alpha in [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]) STRICTLY on Dev queries (N=50).
3. Freezes the optimal alpha and evaluates on Held-Out queries (N=150) to prevent data leakage.
4. Per-query latency measured with controlled warm/cold state (N_LATENCY_REPS=3 runs, median).
5. Reports warm_latency_ms (median of runs 2+3) and cold_latency_ms (run 1) separately.
"""
import os, sys, json, math, time, random
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.hybrid_store import hybrid_store

DATASET_FILE = ROOT / "evaluation" / "golden_dataset_v2.json"
DEV_IDS_FILE = ROOT / "evaluation" / "dev_ids.json"
HELDOUT_IDS_FILE = ROOT / "evaluation" / "heldout_ids.json"
OUT_JSON = ROOT / "evaluation" / "retrieval_benchmark_results.json"
OUT_FULL_JSON = ROOT / "evaluation" / "retrieval_full_results.json"

# Number of latency repetitions per query per method
N_LATENCY_REPS = 3


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def dcg_at_k(relevance_list, k=10):
    dcg = 0.0
    for i, rel in enumerate(relevance_list[:k], 1):
        dcg += rel / math.log2(i + 1)
    return dcg


def evaluate_retrieval(chunks, gold_keywords, top_k=10):
    """
    Returns (recall@5, recall@10, MRR, NDCG@10) using BINARY relevance consistently.

    A chunk is relevant (rel=1) if ANY gold keyword appears in its content.
    Both Recall and NDCG use this binary definition.

    Parameters
    ----------
    chunks       : ranked list of chunk objects (up to top_k)
    gold_keywords: list of keyword strings that define a relevant chunk
    top_k        : cutoff (default 10)
    """
    gold_set = set(gk.lower() for gk in gold_keywords if gk)
    if not gold_set:
        return 0.0, 0.0, 0.0, 0.0

    hit_ranks = []
    # Binary relevance: 1 if any gold keyword matched, 0 otherwise
    relevance_scores = []
    for rank, c in enumerate(chunks[:top_k], 1):
        content_lower = c.content.lower()
        matched = any(gk in content_lower for gk in gold_set)
        if matched:
            hit_ranks.append(rank)
            relevance_scores.append(1.0)
        else:
            relevance_scores.append(0.0)

    # Pad to top_k if fewer chunks returned
    relevance_scores += [0.0] * (top_k - len(relevance_scores))

    r_at_5  = 1.0 if any(r <= 5  for r in hit_ranks) else 0.0
    r_at_10 = 1.0 if any(r <= 10 for r in hit_ranks) else 0.0
    mrr     = (1.0 / hit_ranks[0]) if hit_ranks else 0.0

    # NDCG with binary relevance — IDCG is the best achievable ordering
    idcg       = dcg_at_k(sorted(relevance_scores, reverse=True), k=top_k)
    actual_dcg = dcg_at_k(relevance_scores, k=top_k)
    ndcg_10    = (actual_dcg / idcg) if idcg > 0 else 0.0

    return r_at_5, r_at_10, mrr, ndcg_10


# ---------------------------------------------------------------------------
# Bootstrap CI + Wilcoxon significance test
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(scores, n_boot=1000, ci=0.95, seed=42):
    """Return (mean, lower_ci, upper_ci) via bootstrap."""
    rng = random.Random(seed)
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = []
    for _ in range(n_boot):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    alpha = 1.0 - ci
    lo = boot_means[int(alpha / 2 * n_boot)]
    hi = boot_means[int((1 - alpha / 2) * n_boot)]
    return sum(scores) / n, lo, hi


def wilcoxon_signed_rank(x, y):
    """
    Two-sided paired Wilcoxon signed-rank test.
    Returns (W_statistic, approx_p_value).
    Uses normal approximation for n > 10.
    """
    diffs = [a - b for a, b in zip(x, y)]
    nonzero = [(abs(d), d > 0) for d in diffs if d != 0.0]
    if not nonzero:
        return 0.0, 1.0
    nonzero.sort(key=lambda t: t[0])
    # Assign ranks (handle ties with average rank)
    ranks = []
    i = 0
    while i < len(nonzero):
        j = i
        while j < len(nonzero) and nonzero[j][0] == nonzero[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks.append((avg_rank, nonzero[k][1]))
        i = j
    W_plus  = sum(r for r, pos in ranks if pos)
    n       = len(ranks)
    mu      = n * (n + 1) / 4.0
    sigma   = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return W_plus, 1.0
    z       = (W_plus - mu) / sigma
    # Approximate two-sided p from standard normal
    # Using the Abramowitz & Stegun approximation for the normal CDF
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    p_one_tail = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly
    p_value    = min(1.0, 2 * p_one_tail)
    return W_plus, round(p_value, 6)


# ---------------------------------------------------------------------------
# Per-query latency helper (controlled warm/cold)
# ---------------------------------------------------------------------------

def measure_latency(fn, n_reps=N_LATENCY_REPS):
    """
    Run fn() n_reps times. Returns:
        cold_ms  : elapsed time of run 1 (ms)
        warm_ms  : median of runs 2..n_reps (ms); NaN if n_reps < 2
    Also returns the result of the FIRST call (for correctness evaluation).
    """
    times = []
    result = None
    for i in range(n_reps):
        t0 = time.perf_counter()
        out = fn()
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)
        if i == 0:
            result = out
    cold_ms = times[0]
    warm_runs = times[1:]
    if warm_runs:
        warm_runs_sorted = sorted(warm_runs)
        mid = len(warm_runs_sorted) // 2
        warm_ms = warm_runs_sorted[mid] if len(warm_runs_sorted) % 2 == 1 \
                  else (warm_runs_sorted[mid - 1] + warm_runs_sorted[mid]) / 2.0
    else:
        warm_ms = float("nan")
    return result, cold_ms, warm_ms


def weighted_rrf_fused(q, alpha, k=10):
    """Build weighted RRF fusion and return ranked chunks."""
    dense  = hybrid_store._search_dense(q, k=20)
    sparse = hybrid_store._search_sparse_fts(q, k=20)
    fused  = {}
    for item_d in dense:
        key = item_d.content[:150]
        item_d.rrf_score = alpha * (1.0 / (60 + item_d.dense_rank))
        fused[key] = item_d
    for item_s in sparse:
        key = item_s.content[:150]
        if key in fused:
            fused[key].rrf_score += (1.0 - alpha) * (1.0 / (60 + item_s.sparse_rank))
        else:
            item_s.rrf_score = (1.0 - alpha) * (1.0 / (60 + item_s.sparse_rank))
            fused[key] = item_s
    return sorted(fused.values(), key=lambda c: c.rrf_score, reverse=True)[:k]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation():
    dataset    = {q["id"]: q for q in json.loads(DATASET_FILE.read_text(encoding="utf-8"))}
    dev_ids    = json.loads(DEV_IDS_FILE.read_text(encoding="utf-8"))
    heldout_ids = json.loads(HELDOUT_IDS_FILE.read_text(encoding="utf-8"))

    print("=" * 72)
    print(f"STEP 1: ALPHA SWEEP ON DEVELOPMENT PARTITION ONLY (N={len(dev_ids)} QUERIES)")
    print("=" * 72)

    # Warmup dense model (one call, not part of measurement)
    _ = hybrid_store._search_dense("flood risk at Boulder Creek", k=5)
    _ = hybrid_store._search_sparse_fts("flood risk at Boulder Creek", k=5)

    # ------------------------------------------------------------------
    # Alpha Sweep on Dev partition only
    # ------------------------------------------------------------------
    alphas = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]
    alpha_results = {}

    for a in alphas:
        r5_list, mrr_list = [], []
        for qid in dev_ids:
            item = dataset[qid]
            q    = item["query"]
            gold = item.get("expected_keywords", item.get("keywords", []))
            ranked = weighted_rrf_fused(q, alpha=a, k=10)
            r5, r10, mrr, ndcg = evaluate_retrieval(ranked, gold)
            r5_list.append(r5)
            mrr_list.append(mrr)

        mean_r5  = sum(r5_list)  / len(r5_list)
        mean_mrr = sum(mrr_list) / len(mrr_list)
        alpha_results[a] = {"recall_at_5": mean_r5, "mrr": mean_mrr}
        print(f"  Alpha={a:.2f} -> Dev Recall@5={mean_r5:.4f}, Dev MRR={mean_mrr:.4f}")

    best_alpha = max(alpha_results.keys(),
                     key=lambda a: (alpha_results[a]["recall_at_5"], alpha_results[a]["mrr"]))
    print(f"\n[OK] Optimal Alpha selected strictly on Dev partition: alpha = {best_alpha:.2f}")

    # ------------------------------------------------------------------
    # Step 2: Full 200-query benchmark with controlled per-query latency
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"STEP 2: BENCHMARK ACROSS ALL 200 QUERIES (BM25 vs DENSE vs UNWEIGHTED vs WEIGHTED alpha={best_alpha:.2f})")
    print(f"        Latency: {N_LATENCY_REPS} reps/query, reporting cold (run 1) and warm (median runs 2-{N_LATENCY_REPS})")
    print("=" * 72)

    configs = ["BM25_Sparse", "Dense_BGE_Small", "Standard_RRF_Unweighted", "Proposed_Weighted_RRF"]
    metrics = {
        cfg: {
            "r5": [], "r10": [], "mrr": [], "ndcg10": [],
            "cold_latencies_ms": [], "warm_latencies_ms": []
        }
        for cfg in configs
    }
    split_metrics = {
        split: {cfg: {"r5": [], "r10": [], "mrr": [], "ndcg10": []}
                for cfg in configs}
        for split in ("dev", "heldout")
    }

    all_ids = dev_ids + heldout_ids

    for idx, qid in enumerate(all_ids, 1):
        item = dataset[qid]
        q    = item["query"]
        gold = item.get("expected_keywords", item.get("keywords", []))

        # ---- Each method measured INDEPENDENTLY per query ----

        # 1. BM25 Sparse
        bm25_result, cold_bm25, warm_bm25 = measure_latency(
            lambda: hybrid_store._search_sparse_fts(q, k=10))
        r5, r10, mrr, ndcg = evaluate_retrieval(bm25_result, gold)
        metrics["BM25_Sparse"]["r5"].append(r5)
        metrics["BM25_Sparse"]["r10"].append(r10)
        metrics["BM25_Sparse"]["mrr"].append(mrr)
        metrics["BM25_Sparse"]["ndcg10"].append(ndcg)
        metrics["BM25_Sparse"]["cold_latencies_ms"].append(cold_bm25)
        metrics["BM25_Sparse"]["warm_latencies_ms"].append(warm_bm25)

        # 2. Dense Semantic (BGE-Small)
        dense_result, cold_dense, warm_dense = measure_latency(
            lambda: hybrid_store._search_dense(q, k=10))
        r5, r10, mrr, ndcg = evaluate_retrieval(dense_result, gold)
        metrics["Dense_BGE_Small"]["r5"].append(r5)
        metrics["Dense_BGE_Small"]["r10"].append(r10)
        metrics["Dense_BGE_Small"]["mrr"].append(mrr)
        metrics["Dense_BGE_Small"]["ndcg10"].append(ndcg)
        metrics["Dense_BGE_Small"]["cold_latencies_ms"].append(cold_dense)
        metrics["Dense_BGE_Small"]["warm_latencies_ms"].append(warm_dense)

        # 3. Standard Unweighted RRF (alpha = 0.50)
        unweighted_result, cold_uw, warm_uw = measure_latency(
            lambda: hybrid_store.hybrid_search(q, k=10, alpha_dense=0.50))
        r5, r10, mrr, ndcg = evaluate_retrieval(unweighted_result, gold)
        metrics["Standard_RRF_Unweighted"]["r5"].append(r5)
        metrics["Standard_RRF_Unweighted"]["r10"].append(r10)
        metrics["Standard_RRF_Unweighted"]["mrr"].append(mrr)
        metrics["Standard_RRF_Unweighted"]["ndcg10"].append(ndcg)
        metrics["Standard_RRF_Unweighted"]["cold_latencies_ms"].append(cold_uw)
        metrics["Standard_RRF_Unweighted"]["warm_latencies_ms"].append(warm_uw)

        # 4. Proposed Weighted RRF (alpha = best_alpha)
        weighted_result, cold_w, warm_w = measure_latency(
            lambda: hybrid_store.hybrid_search(q, k=10, alpha_dense=best_alpha))
        r5, r10, mrr, ndcg = evaluate_retrieval(weighted_result, gold)
        metrics["Proposed_Weighted_RRF"]["r5"].append(r5)
        metrics["Proposed_Weighted_RRF"]["r10"].append(r10)
        metrics["Proposed_Weighted_RRF"]["mrr"].append(mrr)
        metrics["Proposed_Weighted_RRF"]["ndcg10"].append(ndcg)
        metrics["Proposed_Weighted_RRF"]["cold_latencies_ms"].append(cold_w)
        metrics["Proposed_Weighted_RRF"]["warm_latencies_ms"].append(warm_w)

        if idx % 50 == 0:
            print(f"  ... processed {idx}/{len(all_ids)} queries")

    # ------------------------------------------------------------------
    # Per-split assignment (alpha frozen from dev; held-out untouched)
    # ------------------------------------------------------------------
    for qid, item_pos in [(q, i) for i, q in enumerate(all_ids)]:
        split = "dev" if qid in dev_ids else "heldout"
        for cfg in configs:
            for key, src in (("r5", "r5"), ("r10", "r10"),
                             ("mrr", "mrr"), ("ndcg10", "ndcg10")):
                split_metrics[split][cfg][key].append(metrics[cfg][src][item_pos])

    # ------------------------------------------------------------------
    # Aggregate + print table
    # ------------------------------------------------------------------
    flat_results = {}
    split_results = {}
    for split in ("dev", "heldout"):
        split_results[split] = {}
        for cfg in configs:
            m = split_metrics[split][cfg]
            n = len(m["r5"])
            if not n:
                continue
            split_results[split][cfg] = {
                "n": n,
                "recall_at_5": round(sum(m["r5"]) / n, 4),
                "recall_at_10": round(sum(m["r10"]) / n, 4),
                "mrr": round(sum(m["mrr"]) / n, 4),
                "ndcg_at_10": round(sum(m["ndcg10"]) / n, 4),
            }

    print(f"\n{'Configuration':<25} | {'Recall@5':<9} | {'Recall@10':<10} | {'MRR':<8} | "
          f"{'NDCG@10':<9} | {'Cold Lat':<10} | {'Warm Lat':<10}")
    print("-" * 100)

    for cfg in configs:
        m = metrics[cfg]
        n = len(m["r5"])

        mean_r5   = sum(m["r5"])   / n
        mean_r10  = sum(m["r10"])  / n
        mean_mrr  = sum(m["mrr"])  / n
        mean_ndcg = sum(m["ndcg10"]) / n

        valid_cold = [v for v in m["cold_latencies_ms"] if not math.isnan(v)]
        valid_warm = [v for v in m["warm_latencies_ms"] if not math.isnan(v)]
        mean_cold = sum(valid_cold) / len(valid_cold) if valid_cold else float("nan")
        mean_warm = sum(valid_warm) / len(valid_warm) if valid_warm else float("nan")

        flat_results[cfg] = {
            "recall_at_5":    round(mean_r5,   4),
            "recall_at_10":   round(mean_r10,  4),
            "mrr":            round(mean_mrr,  4),
            "ndcg_at_10":     round(mean_ndcg, 4),
            "cold_latency_ms": round(mean_cold, 1),
            "warm_latency_ms": round(mean_warm, 1),
            # Legacy key kept for paper-table compatibility
            "latency_ms":     round(mean_warm, 1),
        }
        print(f"{cfg:<25} | {mean_r5:.4f}    | {mean_r10:.4f}     | {mean_mrr:.4f}   | "
              f"{mean_ndcg:.4f}    | {mean_cold:>7.1f} ms | {mean_warm:>7.1f} ms")

    # ------------------------------------------------------------------
    # Statistical significance section
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STATISTICAL SIGNIFICANCE (Bootstrap 95% CI + Wilcoxon signed-rank)")
    print("=" * 72)

    for cfg in configs:
        m = metrics[cfg]
        for metric_name, scores in [("Recall@5", m["r5"]), ("MRR", m["mrr"]), ("NDCG@10", m["ndcg10"])]:
            mean_v, lo, hi = bootstrap_mean_ci(scores)
            print(f"  {cfg:<25} {metric_name:<10}: {mean_v:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    # Wilcoxon: Dense vs Proposed Weighted RRF on MRR
    dense_mrr    = metrics["Dense_BGE_Small"]["mrr"]
    weighted_mrr = metrics["Proposed_Weighted_RRF"]["mrr"]
    W, p_val = wilcoxon_signed_rank(dense_mrr, weighted_mrr)
    print(f"\n  Paired Wilcoxon (Dense vs Proposed Weighted RRF on MRR):")
    print(f"    W = {W:.1f},  p = {p_val:.6f}  {'(significant at p<0.05)' if p_val < 0.05 else '(not significant)'}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    # Flat per-method dict (matches paper tables)
    OUT_JSON.write_text(json.dumps(flat_results, indent=2), encoding="utf-8")
    print(f"\n[OK] Per-method results saved to {OUT_JSON}")

    # Full summary (companion file)
    full_summary = {
        "dev_alpha_sweep":  {str(k): v for k, v in alpha_results.items()},
        "selected_alpha":   best_alpha,
        "results_all_200":  flat_results,
        "results_dev":      split_results["dev"],
        "results_heldout":  split_results["heldout"],
        "significance": {
            "wilcoxon_dense_vs_weighted_mrr": {
                "W": W,
                "p_value": p_val,
                "significant_at_0_05": p_val < 0.05
            }
        }
    }
    OUT_FULL_JSON.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    print(f"[OK] Full summary saved to {OUT_FULL_JSON}")


if __name__ == "__main__":
    run_evaluation()
