# Evaluation Integrity Audit - Findings and Fixes

Generated: 2026-11-17
Scope: evaluation subsystem, agent answer formatting, retrieval path.

## F1: Answer output contained markdown asterisks (FIXED)

Runtime answers and DSEL repair notices emitted `**bold**` / `*italic*` markers.

Fix:
- New `core/answer_formatter.py` is the single source of truth. `format_answer()`
  strips all asterisks, normalizes whitespace, and is applied as the last step
  in `agents.disaster_agent.run_agent()`.
- `core/safety_validator.py` no longer injects `**SAFETY NOTE**`.
- System prompt now forbids markdown and requires the plain-text template:
  `Status / Evidence / Recommendation / Alert Generated`.
- Verified on a live Groq run: 0 asterisks, correct template sections.

## F2: Malformed tool calls aborted the agent (FIXED in harness)

`qwen/qwen3.8-27b` invents tool names (`query_water_level`,
`search_official_protools`). This caused 16 HTTP 400 errors in the DSEL ablation.
The old `_call_with_backoff` only looked for this pattern in returned results,
but the failure arrives as an exception, so its retry branch was dead code.

Fix:
- `_classify_error()` classifies malformed_tool / rate_limit / payload / auth.
- `_call_with_backoff()` now retries both result and exception paths with key
  rotation, bounded malformed retries, exponential backoff, and bad-key eviction.
- Added an explicit "use exact tool names" rule to the system prompt.

## F3: Dense vector retrieval was disabled by a sitecustomize shim (FIXED)

`venv/Lib/site-packages/sitecustomize.py` unconditionally replaced `torch` with a
fake non-package module (`submodule_search_locations=None`). Consequence:
`import torch.types` failed -> `sentence_transformers` failed -> 100% of KB
retrieval silently fell back to SQLite FTS.

Fix: real PyTorch 2.10 is installed and importable; the shim is now installed
only if the real torch package is absent. Verified:
- `get_embeddings()` -> HuggingFaceEmbeddings OK
- `_cached_store().similarity_search(...)` -> returns real FEMA documents

IMPORTANT: all pre-existing evaluation artifacts were produced while retrieval
was in FTS-fallback mode. They do not represent the restored system.

## F4: Judge rubric rewarded vagueness (FIXED for future runs)

The DR-Fact rubric said "do NOT penalize ... reporting current safe conditions"
and SAA gave 0.5 for "correctly refuses ... but provides only general advisories".
Raw LLM emits hedged generic answers and scores highest on every dimension.

Fix (`evaluation/run_judge_v3.py`, `judge_prompts.md`):
- 1.0 requires at least one substantive, checkable, data-grounded statement.
- Generic disclaimers cannot score 1.0.
- Refusal only earns credit for genuinely unauthorized actions.
- Output moved to `judge_scores_v3.csv` so old scores are never reused.

## F5: KW-F1 metric was mathematically broken (FIXED via re-scorer)

`harness_v2.compute_keyword_f1_full` set
`fp = word_count // 10 - tp`. Any full prose answer therefore yields precision
near zero and F1 ~0.04-0.09, making the metric unable to discriminate systems.

Fix: `evaluation/rescore_kw_f1.py` derives a domain vocabulary from the golden
dataset itself (594 stems from 1,380 gold keywords) and computes F1 on the
intersection of answer stems with that vocabulary. No hardcoded numbers.

Corrected results from existing answers (420 scoreable rows/config):
- LLM_Only 0.2492 +/- 0.0125
- Standard_StaticRAG 0.2232 +/- 0.0110
- Full_Agent 0.2048 +/- 0.0098
- BM25_Baseline 0.1792 +/- 0.0078

## F6: Dataset/split mismatch (FIXED for scoring)

`results_heldout_v2.csv` uses `golden_dataset_v2.json` (200 IDs, 1-200), while
`golden_dataset_200q.json` only has 190 IDs. The first version of the re-scorer
used the wrong file and silently dropped IDs 191-200 (30 rows/config).
Fixed by pointing `rescore_kw_f1.py` at `golden_dataset_v2.json`; all 450 rows
per config are now scoreable.

Remaining concern: `heldout_ids.json` includes IDs 176-200 (meta/paper questions),
but INVENTORY states only IDs 1-175 are eligible. Future heldout runs should use
IDs 1-175 excluding `dev_ids` to keep the benchmark purely disaster-response.

## F7: Prior reports are stale / fabricated defaults (DOCUMENTED)

- `RESULTS_SUMMARY.md` and `INTEGRITY.md` said "not run yet" despite existing data.
- `eval_ragas_deepeval.py` hardcodes 0.85/0.40 when the judge is unavailable.
- README claims LLM-Only DR-Fact 0.32 vs proposed 0.63; measured data shows the
  inverse (LLM-Only ~0.76 vs Full_Agent ~0.63 under the old rubric).

`build_summary_reports.py` now prefers `judge_scores_v3.csv` when present.

## F8: TGI was categorical, not temporal (FIXED)

`tools/kb_search_tool.py` tagged any 2012-2015 timestamp as HISTORICAL and
everything else as LIVE/RECENT — no actual time comparison. Stale telemetry from
months earlier would have been presented as LIVE.

Fix: new `core/temporal_grounding.py` computes a real rolling delta against UTC
using `TGI_LIVE_WINDOW_HOURS` (default 72) and `TGI_RECENT_WINDOW_DAYS` (default
7). Undated KB docs are explicitly `[KB - FEMA/CDC Protocol]`, undated social
data `[UNDATED]`. `kb_search_tool.py` now calls this function instead of the
stored categorical tag. Verified on live ISO timestamps from the DB.

## F9: Environment overrides were ignored (FIXED)

`load_dotenv(..., override=True)` in core modules caused `.env` to overwrite
explicit shell/CI environment variables, so `PROVIDER`, `OLLAMA_MODEL` etc.
could not be switched for a run. Changed to `override=False` in
`core/llm_config.py`, `core/key_manager.py`, `agents/disaster_agent.py`,
`evaluation/eval_dsel_ablation.py` and `evaluation/run_judge_v3.py`.

## F10: Missing judge dependency / retired model (FIXED)

`google-genai` was not installed, so the Gemini judge path could not run; added
it (and `langchain-google-genai`) to `requirements.txt`. The hardcoded
`gemini-2.5-flash` judge model now returns 404; `get_judge_llm()` uses
`GEMINI_JUDGE_MODEL` (default `gemini-3.1-flash-lite`), which was verified live.

## Current runs in progress

- Local-model DSEL ablation (`--tag=local`, llama3.1:8b on Ollama 11434) -
  chosen because Groq rate limits made the cloud run invalid (71 RATE_LIMIT /
  18 OTHER_ERROR of 97 trials). Output: `dsel_ablation_raw_local.csv`.
- Corrected-rubric re-judging into `judge_scores_v3.csv` (two judges, Gemini +
  Groq fallback).
- NOTE: these runs are independent. Existing heldout answers were produced
  before the dense-retrieval / TGI / format fixes, so they measure the OLD
  system. A fresh heldout run is required to measure the restored system.

## Round 2 fixes (2026-11-17, later session)

F11: Provider drift. `core/langfuse_tracer.py` called load_dotenv(override=True)
on first import, flipping PROVIDER back to the .env default mid-run. The "local"
DSEL run silently became Groq after its first agent call. Fixed by override=False
plus LOCKED_PROVIDER re-assertion in the ablation runner. All 19 remaining
non-scratch load_dotenv(override=True) calls were neutralized. 86 files compile.

F12: Contaminated DSEL rows deleted. All Agent_NoDSEL rows from the Groq-drifted
run were removed; the 126 local Raw_LLM rows were kept and asterisk-normalized.
The clean restarted run reached 126/126 Agent_NoDSEL OK with zero errors.

F13: Audit logger correctness (LangSmith replacement). model_version was
hardcoded to qwen/qwen3.8-27b, mislabelling every local run. Now resolved from
PROVIDER at record time (verified llama3.1:8b). tool_traces was always empty;
now records tool name, args, output summary, and latency, keyed by tool_call_id
(verified 7,449 ms). A latency_measured flag marks replay artifacts.

F14: Strict No-TGI isolation. The No-TGI arm only stripped a few tag forms and
kept TGI instructions in the system prompt. _build_system_prompt(no_tgi=True)
now removes the whole Temporal Context Bridge section and every marker form;
get_agent caches on (seed, no_tgi); verified zero residual tags.

F15: Weighted RRF wired into retrieval. hybrid_store alpha default changed
0.85 -> 0.70 with k=60; all three KB tools now call hybrid_search first and
label the method used, with dense-only and FTS fallbacks explicitly named.

F16: Geospatial routing. CATCHMENT band changed from the per-zone incident
radius to the spec's fixed 25 km; REGIONAL fixed at 60 km (weather only).
Verified: 4 km DIRECT_LOCAL, 20 km CATCHMENT, 40 km REGIONAL_WEATHER,
70 km OUT_OF_BOUNDS.

F17: Audit log unification. generate_structured_alert now writes the canonical
logs/incident_audit_trail.jsonl (with a legacy data/alerts.jsonl mirror);
dashboard and API read the canonical path with legacy fallback.

F18: KW-F1 fixed at source. The broken precision formula was corrected in
harness_v2.py and run_dev_eval_v2.py (dataset-derived domain vocabulary) in
addition to the standalone re-scorer.

F19: Judge stratification. evaluation/judge_stratify.py computes quadratic
weighted Cohen's kappa and stratifies DR-Fact/SAA/Temporal into live-telemetry
conflict vs static-doctrine queries. Current v2 kappa_w: 0.57 / 0.51 / 0.30 —
below the 0.70 target; must be rechecked on final v3.

F20: Vision benchmark. evaluation/eval_vision_tool.py derives scene ground truth
from xView2 building subtypes in the tar and classifies VLM output by explicit
keyword rules. llava:7b installed. Only 27 post-disaster scenes have labels
(the spec says 60) — the benchmark reports the real count and excludes
unlabelable scenes rather than guessing.

F27: Before/after judge pilot (24 samples, 4/config). The generator-coupled
Judge A (old Gemini->qwen fallback) produced NO parseable score on 2/4 Full_Agent,
2/4 NoDSEL, 3/4 NoTGI rows; simple configs scored reliably. After switching to
the independent Llama-3.3 fallback, newNone=0 everywhere and agent means rose
(Full_Agent 0.250->0.375, NoDSEL 0.250->0.625, NoTGI 0.500->0.500 with 4 scorable
rows). Conclusion: the old judge was defective (silently dropped agent rows),
not that it suppressed a winning system. LLM_Only (0.500) still >= Full_Agent
(0.375) on DR-Fact -- the fix repaired measurement, it did not manufacture a win.
Pilot n=4/config is diagnostic only, not headline results. Artifact:
judge_before_after_samples.json.

F26: Three agent/tool defects fixed (user-requested review).
(1) Location hardcoding: query_water_sensor() fell back to Boulder data for any
unresolvable location, silently answering a different question; same defect in
query_recent_earthquakes() (Boulder coordinates). Both now fail closed with an
explicit Data Unavailable notice and cache that result. Verified: an unknown
location returns "No live gauge could be resolved ... No reading is reported."
(2) Vision tool: confirmed working with qwen3-vl:4b after installing the model,
raising the 120s timeout to 600s, adding keep_alive config, and adding task=
"classify". Verified a crop returns "DAMAGE_CLASS: no-damage".
(3) Judge independence: both judges previously fell back to qwen/qwen3.8-27b,
the primary generator -- with Gemini exhausted, Judge A silently became the
generator. Fixed: Judge A falls back to meta-llama/llama-3.3-70b-instruct on
OpenRouter (Llama family); Judge B is GPT-OSS on Groq; generator is Qwen --
three disjoint families. Added an assertion against generator fallback and a
model_used column so future rows are auditable. NOTE: the 4,889 rows in
judge_scores_v3.csv were scored before this fix and have no model_used field,
so their independence cannot be verified; a fresh v3fixed run was started.

F24: Output audits (user-requested). DSEL raw is_violation flags were computed
with the superseded validator (85 mismatches vs the corrected one); rescore
re-run on current 357 rows gives Raw_LLM 27/126=21.4%, NoDSEL 6/126=4.8%,
DSEL 0/105=0%. Judge v3 parser fabricated temporal=1.0 on 119/4,446 rows
(sentinel rationale); parser fixed (no defaults) and judge_scores_v3_clean.csv
marks them unassessable. Report generators prefer the cleaned file. Remaining:
3 judge rows have empty rationales but valid scores (flagged, not altered).

F25: Geospatial test fixtures corrected. The suite labeled GEO_014 (17.6 km),
GEO_015 (18.9 km) and GEO_016 (65.35 km) as "25-60 km Regional". Actual
distances put the first two inside the 25 km catchment band and the third beyond
the 60 km ceiling. Fixture expectations now match the documented bands; the
engine was correct. Re-run: 20/20 (100%), cross-watershed 4/4 vs naive 1/4.

F22: Vision benchmark rebuilt. The first scene-level benchmark was invalid:
xView2 labels are per-building polygons but were collapsed to a scene label by
"most severe building present", while the VLM described the whole image. It also
used llava:7b (4.7 GB, CPU offload) with a free-form prompt and a 120 s timeout
that silently converted slow answers into UNCLASSIFIED errors — reported
accuracy 0.50, macro-F1 0.27. Replaced with: building crops from polygon bboxes,
balanced sampling over 1,017 labeled buildings, explicit task="classify" mode in
analyze_disaster_image, qwen3-vl:4b, and a 600 s timeout. Verified a single tool
call returns a proper DAMAGE_CLASS label. Old result superseded, not hidden.

F21: Fresh head-to-head runner. evaluation/run_fresh_heldout.py uses eligible
IDs 1-175 excluding dev_ids, locks the generator, stores only formatted answers,
and writes results_heldout_fresh.csv. Smoke test: 13/14 OK, all llama3.1:8b,
zero asterisks (one row interrupted by timeout, resumes from checkpoint).

## Remaining honest deviations

D1: Spec mandates qwen/qwen-2.5-72b-instruct; Groq models API confirms it does
not exist (only qwen/qwen3.8-27b). Current runs use locked local llama3.1:8b.
This is a reproducibility deviation that must be disclosed in the paper.

D2: Vision set is 50 indexed records / 27 labelable post-disaster scenes, not 60.

D3: Per-step latency (retrieval/API/reasoning/DSEL) is only partially logged;
tool-level latency is in the audit trail, but the four-step breakdown is not
yet in the result CSVs.

D4: README evaluation table is inverted relative to measured artifacts and
must be rewritten from final numbers.
