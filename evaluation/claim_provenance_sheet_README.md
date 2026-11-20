# Claim-level provenance annotation sheet (human, single annotator)

`claim_provenance_sheet.csv` contains 120 responses from the locked
qwen2.5:3b generator, sampled deterministically (seed 42): 60 telemetry,
40 doctrine, 20 other. Responses are shuffled and re-identified. All machine
verdicts (claim_verification_summary, DSEL flags, policy scores) are stripped.
The join key is in `claim_provenance_key.csv`. NOTE: every locked-generator
ledger row has dsel_rewritten=False, so this sheet cannot contrast DSEL-rewritten
answers; it measures claim-to-cited-evidence consistency only.

IMPORTANT SCOPE LIMITATION: the audit ledger persisted only capped (<=300
character) tool-output summaries and synthetic evidence IDs that cannot be
joined to full retrieved documents. You therefore judge claim support against
the evidence statements quoted inside the response, NOT against original
source text. This measures internal claim-to-cited-evidence consistency and
citation correctness, which is weaker than external attribution.

For each response, split it into atomic factual claims (one claim per row,
repeating annotation_id). For each claim fill:

- claim_text: the single factual assertion.
- cited_evidence_span: the exact text in the evidence column the claim relies on.
- support_verdict: SUPPORTED | CONTRADICTED | INSUFFICIENT | NOT_JUDGEABLE.
  Use NOT_JUDGEABLE when the capped evidence is too short to decide.
- citation_present: YES | NO (does the claim carry a source/timestamp marker).
- citation_correct: YES | NO | NA (does that marker point to evidence that
  actually supports the claim).
- note: optional.

After labeling, join on annotation_id with claim_provenance_key.csv to compute
claim support precision, contradiction rate, unsupported rate, citation
correctness, and per-family breakdowns.
