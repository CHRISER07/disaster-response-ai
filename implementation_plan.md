# Implementation Plan: Fixing IEEE.tex Inconsistencies

This plan outlines the steps to resolve all critical and medium-priority issues identified in the gap audit, aligning the paper (`IEEE.tex`) with the actual codebase and evaluation results.

## Proposed Changes

### `IEEE.tex`

We will make the following corrections to the LaTeX document:

#### 1. Results Tables (Section V)
- **Ablation Table (Table 1):** Add the missing BM25 Baseline row. Add ` std` for all metrics. Add the `Latency(s)` column to highlight the massive speedup of the proposed system (1.21s vs 1668s).
- **Per-Category Table (Table 2):** Update the values to match the actual 50-query run in `results.md`. Add the missing `Earthquake` and `Hallucination` rows. Fix incorrect `n` values.

#### 2. Dataset Claims (Section IV)
- **Social Media Corpus:** Correct the text and Table `kb_comp` to state that ~7,800 documents from all 26 CrisisLex T26 events were ingested, not just 500 from the Colorado Floods. Update total documents to ~10,790.

#### 3. Tools and Architecture (Section III & IV)
- **Tools Table:** Add the missing `query_recent_earthquakes` and `generate_structured_alert` tools to `tab:tools`.
- **Methodology Prose:** Rewrite the fragmented, unpublishable sentences in Section III (e.g., "The Augmented Generation (RAG) and agentic reasoning architecture. The system is specially designed for complex The data relating to disaster were collected...") into coherent, academic prose.

#### 4. Citations and Bibliography
- **Undefined Keys:** Replace `\cite{olteanu2014crisislex}`, `\cite{gupta2019xview2}`, and `\cite{fema2021nrp}` in the text with their corresponding `\cite{bX}` tags.
- **Citation Mismatches:** Fix the mismatch where `\cite{b10}` was used for sensor networks instead of prompt-LLMs. Remove the duplicate entry for Singh & Hoskere (`b7` and `b15`), and add the correct citation for the Earthquake Knowledge Graph paper.

#### 5. Abstract and Introduction
- **Abstract:** Remove the false claim of "simulation-based disaster data" and replace it with "live operational API telemetry".
- **Contributions:** Expand the 3 vague bullet points in the Introduction to 5 precise contributions, explicitly mentioning Temporal Grounding Injection (TGI), the Deterministic Safety Enforcement Layer (DSEL), and the LangGraph ReAct architecture.

#### 6. Formatting and Structure
- **Section Headers:** Change `\section{IMPLEMENTATION}` to `\section{Implementation}`. Use proper `\subsubsection{}` tags instead of manual `1) ` lists.
- **Literature Survey:** Replace the incorrect body text under `Reliability, Safety, and Evaluation in Disaster AI` (which was accidentally copied from a deleted sensor network section) with appropriate text discussing hallucination mitigation and safety validation.

## Verification Plan

- The LaTeX document should compile cleanly without undefined citation warnings.
- All numbers in the paper's tables must exactly match the outputs in `evaluation/results.md` and `evaluation/results_200q.csv`.
- A manual review of the rewritten Methodology section will ensure clarity and academic tone.
