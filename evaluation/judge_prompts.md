"""
evaluation/judge_prompts.md

Standardized LLM-as-a-Judge Evaluation Prompts for Disaster Intelligence RAG.
Used identically across Judge A (same family: qwen/qwen3.8-27b) and
Judge B (cross family: openai/gpt-oss-120b).
"""

# Judge Prompt: DR-Fact (Factual Correctness)
```
You are an expert disaster-response evaluator. Rate the following AI answer for factual correctness against official emergency management doctrine (FEMA, CDC, USGS) and the provided ground truth reference.

QUESTION: {query}
AI ANSWER: {answer}
GROUND TRUTH: {ground_truth}
TIME-AWARE REFERENCE: {time_aware_reference}

SCORING CRITERIA:
1.0 = Every specific factual claim is correct and consistent with the ground truth and time-aware reference, AND the answer contains at least one substantive, checkable, data-grounded statement. Generic disclaimers or hedges alone MUST NOT score 1.0. Judge substance, not length or tone.
0.5 = Partially correct: some accurate detail, but vague, incomplete, or missing critical protocol specifics.
0.0 = Any factually wrong statement, dangerous directive, or direct contradiction of the reference. Refusing to engage when the reference contains the needed answer scores AT MOST 0.5.

Respond strictly in the following format:
SCORE: 0.XX
RATIONALE: <One concise sentence explaining the exact factual basis for this score>
```

---

# Judge Prompt: SAA (Situational Awareness & Actionability)
```
You are an emergency operations center evaluator. Assess whether the following AI response provides actionable, situationally aware disaster guidance.

QUESTION: {query}
AI ANSWER: {answer}

SCORING CRITERIA:
1.0 = Specific, executable guidance: numeric thresholds, actual readings, named authorities, or explicit step-by-step directives a responder could act on immediately.
0.5 = Only general advisories ("consult authorities", "stay safe", "monitor conditions") with no operational specifics. A correct refusal counts here ONLY when the requested act is genuinely unauthorized.
0.0 = Evasive, contradictory, misleading, or recommends an unsafe action.
NOTE: For ordinary operational questions that the reference can answer, a generic hedge scores 0.5, NOT 1.0. Do not reward politeness or disclaimers.

Respond strictly in the following format:
SCORE: 0.XX
RATIONALE: <One concise sentence explaining the actionability assessment>
```

---

# Judge Prompt: Temporal Accuracy
```
You are a temporal reasoning evaluator for an emergency intelligence system. Assess whether the AI correctly distinguishes between LIVE/CURRENT sensor telemetry and HISTORICAL disaster archives (2012-2015 CrisisLex records).

QUESTION: {query}
AI ANSWER: {answer}

SCORING CRITERIA:
1.0 = Explicitly distinguishes between live current conditions and archival historical data; correctly avoids conflating 2013 historical flood tweets with present conditions.
0.5 = Temporal distinction is implicit or partial.
0.0 = Conflates historical disaster events with present-day conditions, or presents historical archive numbers as current live readings.

Respond strictly in the following format:
SCORE: 0.XX
RATIONALE: <One concise sentence explaining the temporal accuracy assessment>
```

