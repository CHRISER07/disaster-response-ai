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

F28: Temporal-conflict benchmark v1 was invalid. All 32 items were labelled
NO_EVACUATION (no positive class), every item used the same 4.2 ft reading
against a 10.0 ft flood stage, the fixture reading did not match the live tool
(3.64 ft), and the queries were answerable from the gauge alone, so TGI -- which
only re-tags retrieved KB chunks -- was never exercised. A constant "do not
evacuate" responder scored a perfect 0%. Rebuilt as v2: 24 items, 12 safe /
12 above threshold, each query text issued once with a safe and once with a
critical pin, label assigned mechanically by reading >= 1.3 x flood stage.

F29: The agent answered flood/evacuation questions from weather alone and never
called the river gauge (verified: answers cited no gage height). Added a
mandatory routing rule: any flooding, evacuation, water-level or area-safety
question MUST call query_water_sensor first; weather alone is not sufficient.

F30: Two evaluation-instrument artifacts were found and fixed. (a) Supplying the
pinned gauge in the user message made DSEL's Ragas faithfulness check score 0.00
and block valid alerts; the reading is now injected as a prefetched ToolMessage
so the safety layer sees genuine tool-grounded context. (b) The TGI prompt had a
single worked example ending in "Status: NORMAL" at 3.63 ft; the 3B generator
pattern-matched it and emitted NORMAL even at 14.2 ft. Replaced with paired
CASE A (live safe) / CASE B (live critical) examples plus an explicit rule that
Status follows the live gage.

F31: Judge integrity. judge_scores_v2.csv is a valid independent run (Judge A
Gemini 3.5 Flash Lite, Judge B GPT-OSS-20B; 3,300 paired items). The v3 run is
NOT usable: after Gemini quota exhaustion Judge A fell back to the Qwen
generator itself, so those scores are self-judged; the paper leaves v3 cells
blank and now says so explicitly. Recomputed kappa_w from v2: 0.576 DR-Fact
(exact 64.5%), 0.519 SAA, 0.324 Temporal. Corrected the paper's judge model
labels (3.1 -> 3.5; 120B -> 20B) and its unsourced kappa of 0.4949.

F32: The paper's abstract and conclusion claimed TGI "reduces false evacuation
orders to 0/93 active runs". No file contains 93 runs; the underlying v2 data
was TGI 1/96 and NoTGI 1/96, i.e. no reduction. Replaced with the corrected v5
result: 288 runs, all OK, one generator; TGI 0/36 false evacuations (95.8%
accuracy) vs NoTGI 10/36 (86.1%), with TGI's 3/36 missed evacuations disclosed,
and both no-telemetry baselines missing all 36 critical cases.

F33: Fresh held-out eligibility was wrong. run_fresh_heldout.py selected IDs
1-175 excluding dev, on the claim that 176-200 were meta/paper queries. Inspection
showed 176-200 are ordinary domain queries (hazard, triage, live telemetry,
damage, resource); the filter silently dropped 25 of the 150 held-out items and
biased the category mix to 15 categories. Now selects exactly the 150 IDs in
heldout_ids.json minus dev_ids, with an assertion on the count.

F34: Vision benchmark metric was inflated by dropping failures. Accuracy and
macro-F1 were computed only over rows where the tool returned a parseable label,
so tool errors could raise the reported score. UNCLASSIFIED/error rows are now
counted as wrong in both accuracy and macro-F1; assessable-only accuracy is
reported separately. Also set OLLAMA_VISION_KEEP_ALIVE=30m for the benchmark so
the 3.3 GB VLM is not reloaded on every call. Dataset audit passed: 50 indexed
records, 27 post-disaster scenes, 1,017 labeled buildings across the four
xView2 subtypes, unique crop names, no duplicate query IDs, no dev/heldout
overlap.

F35: Continuous output audit found three more defects. (a) format_answer
advertised a four-section template but only stripped markdown, so five early
held-out rows lacked "Alert Generated:"; the formatter now fills missing
sections with explicit unavailable markers, and those rows were re-normalized
with their KW-F1 recomputed. (b) BM25 was not lexical retrieval: run_bm25 built
an FTS clause from the question and then ignored it, fetching the same first
five documents for every query. It now calls HybridRetrievalStore._search_sparse_fts
with the actual query; all BM25 rows must be regenerated. (c) evaluation/
audit_outputs.py now checks every stored row for status, empty/asterisk/leaked
text, template completeness, duplicate keys, mixed generators and out-of-range
KW-F1, and exits non-zero on any failure.

F36: 384 px crop-size sensitivity check was smoke-tested (4 crops, valid)
and started on the full 100-building sample, then stopped at 43/100. The
partial pattern matched the 192 px baseline (destroyed recognised, minor and
major damage largely missed), so continuing ~50 more minutes would not change
the conclusion. This run is ABORTED, not a result: no numbers from it are
reported, and the published vision result remains the completed 100-crop,
192 px benchmark. The second-VLM comparison was smoke-tested (valid output)
but not run in full, for the same time/value reason.

F37: Second reviewer pass fixes. Alert dispatch now fails closed when the
policy engine is unavailable or raises; evaluation/test_alert_authorization.py
verifies outage and low-confidence cases block with zero side effects, while an
authorized alert still logs. Paper: removed the invalid BM25 comparison from
abstract, main table and discussion; fixed "Optimal" chart label, empty
"Calibrated Rubric" row, ambiguous "0 errors" footnote, missing abstract
disclosure of 3/36 missed evacuations; Open-Meteo described as weather-model
current/forecast rather than AWOS; CrisisLex identified as historical;
conflict resolver no longer suppresses warnings on non-comparable evidence;
Algorithm 2 REQUIRE now lists Q, Q_surge, M; audit claim softened to an
auditable execution log; DAB-200 dataset version note added; vision labelled
exploratory. Removed the manual \textheight override and compressed prose so
the paper compiles cleanly at 8 pages with official IEEE geometry.

F38: Full-system checkup. Confirmed clean: golden dataset (200 items, 50/150
dev/held-out split, no overlap or duplicate query text), temporal set (24
items, 12/12 balanced, unique IDs), all active result CSVs use a single
generator per file, no duplicate keys, no asterisks in current answers.
Found: (a) RESULTS_SUMMARY.md, INTEGRITY.md and DISCREPANCIES.md were stale
(pre-rescore safety rates, "DSEL not run", broken dense retrieval) and are now
marked superseded, with CURRENT_RESULTS.md as the authoritative source;
(b) legacy CSVs with markdown answers/error rows remain for provenance and are
listed as superseded; (c) core/evidence_validator.py implements source-specific
TTLs and four timestamps and is wired into the agent, but uses wall-clock now
as observed_at and is wrapped in a silent except, so the paper's description of
TGI as a single-window tagging mechanism remains accurate; (d) untracked
diagnostic scripts/PNG files from a separate LaTeX debugging session are
present and were left untouched.

F39: Independent v3 judge abandoned after diagnosis. Judge A had no working
provider: REST probe of all 10 Gemini keys returned HTTP 429 (quota exhausted);
OpenRouter returned HTTP 402 (insufficient credits); Groq did not expose any
Llama-family judge (404 for llama-3.3-70b-versatile and llama-3.1-8b-instant),
and the only available Groq judge family (GPT-OSS) is already used by Judge B.
Groq Judge B itself was healthy (~1.3 s/call). With Judge A timing out and
retried three times, only Judge B rows were written (3,453/6,600), so the run
was stopped and its partial data marked non-reportable. The published v2 dual
judge table remains valid and independent; the paper's corrected-rubric cell
stays blank.

F40: Second-review remaining items fixed. core/conflict_resolver.py now
requires incident/time/region/phenomenon/measurement comparability before
returning CONTRADICTED; non-comparable reports return UNVERIFIED with abstention
(test_conflict_resolver.py passes). Paper: removed the exploratory vision table
(reported as one honest limitations sentence), removed the retrieval Wilcoxon
claim and disclosed that tuning queries are included, disclosed that reference
answers were team-authored with no inter-annotator agreement or established
incident-disjointness, added that only median latency was measured (no P95/P99
or failure rates), renamed the figure's "Immutable Audit Ledger" to
"Hash-Linked Audit Ledger", and narrowed novelty to systems integration.
BM25 re-measurement and the held-out run remain pending.

F41: Academic-writing pass. Rewrote the three RAG failure modes to avoid
absolute claims, replaced the contribution list with precise systems-integration
contributions plus RQ1--RQ3, removed the unsupported sub-millisecond comparison,
replaced the unclear text-intersection safety invariant with a detector contract,
removed decorative MDP terminology, aligned the architecture figure's timestamp
tuple with the four implemented fields while stating that validity windows are
not enforced, rewrote Related Work paragraphs to state achieved/unresolved/how
this differs, narrowed the Conclusion to implemented / established / unvalidated,
and added verifiable official references (USGS NWIS, FDSN, Open-Meteo, FEMA NIMS).
Title left unchanged at the user's request.

F42: Fresh held-out run complete: 1,050 rows, one generator (llama3.1:8b),
zero errors, zero unformatted answers, audit PASS. The corrected BM25 retriever
was re-measured (151 stale rows removed, 150 regenerated). Under the published
KW-F1 scorer the controlled re-run does not reproduce the published ranking
(Static RAG 0.1110 > Full Agent 0.0928), so the paper now reports this as a
negative/configuration-specific finding and no longer claims general KW-F1
superiority. The multi-source tool-routing fix (up to 4 calls when multiple
sources are requested) was smoke-tested; its effect is limited to late NoTelem
rows because earlier agent rows were generated before the change.

F43: Paper results table added for the completed controlled re-run
(Table~\ref{tab:fresh}): all seven configurations, published KW-F1 scorer,
1,050 clean runs, with the explicit note that the published ranking was not
reproduced. All ablations retained because they are the only evidence for the
DSEL, TGI and telemetry claims. Compiled to 8 pages.

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
