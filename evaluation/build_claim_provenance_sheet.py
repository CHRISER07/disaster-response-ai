"""Build a human claim-level provenance annotation sheet from the audit ledger.

HONEST SCOPE: the audit ledger stores only capped (<=300 char) tool-output
summaries and synthetic evidence IDs that cannot be joined to full retrieved
chunks. Full cited passages were never persisted. Therefore this sheet asks
the annotator to judge claim support against the EVIDENCE STATEMENTS QUOTED IN
THE RESPONSE ITSELF, not against original source documents. This measures
internal claim-to-cited-evidence support and citation correctness, which is
weaker than external attribution, and is disclosed as such.

Locked generator qwen2.5:3b only. Machine verdict fields are stripped so they
cannot bias the annotator. Deterministic seed-42 sample, stratified by tool
family (telemetry / doctrine / other). NOTE: all locked-generator ledger rows
have dsel_rewritten=False, so this sheet cannot contrast DSEL-rewritten answers;
it measures claim-to-cited-evidence consistency only.
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "logs" / "incident_audit_trail.jsonl"
OUT = ROOT / "evaluation" / "claim_provenance_sheet.csv"
README = ROOT / "evaluation" / "claim_provenance_sheet_README.md"

TELEMETRY = {"query_water_sensor", "fetch_live_weather", "query_recent_earthquakes"}
DOCTRINE = {"search_official_protocols", "search_knowledge_base"}


def family(rec):
    tools = {t.get("tool_name") for t in rec.get("tool_traces", [])}
    if tools & TELEMETRY:
        return "telemetry"
    if tools & DOCTRINE:
        return "doctrine"
    return "other"


def clean(text):
    return " ".join(str(text).replace("\n", " ").split())


records = []
with open(LEDGER, encoding="utf-8") as f:
    for line in f:
        try:
            o = json.loads(line)
        except Exception:
            continue
        if "qwen2.5" not in str(o.get("model_version", "")):
            continue
        if not o.get("tool_traces") or not o.get("final_response"):
            continue
        rec = {
            "incident_id": o["incident_id"],
            "query": clean(o["query"]),
            "evidence": " || ".join(clean(t.get("tool_output_summary", ""))
                                    for t in o["tool_traces"]
                                    if t.get("tool_output_summary")),
            "response": clean(o["final_response"]),
            "family": family(o),
            "dsel_rewritten": bool(o.get("dsel_rewritten")),
        }
        records.append(rec)

# Stratified deterministic sample: 60 telemetry, 40 doctrine, 20 other = 120.
rng = random.Random(42)
by_family = defaultdict(list)
for r in records:
    by_family[r["family"]].append(r)
for v in by_family.values():
    rng.shuffle(v)

targets = {"telemetry": 60, "doctrine": 40, "other": 20}
sample = []
for fam, n in targets.items():
    pool = by_family.get(fam, [])
    sample.extend(pool[:n])
rng.shuffle(sample)

# Re-ID blind and drop any machine verdict fields.
for i, r in enumerate(sample, 1):
    r["annotation_id"] = f"C{i:03d}"

fields = ["annotation_id", "family", "dsel_rewritten_blind", "query",
          "evidence_quoted_in_response_context", "response",
          "claim_id", "claim_text", "cited_evidence_span",
          "support_verdict", "citation_present", "citation_correct", "note"]
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in sample:
        w.writerow({
            "annotation_id": r["annotation_id"],
            "family": r["family"],
            "dsel_rewritten_blind": "",  # intentionally withheld; reveal after annotation
            "query": r["query"],
            "evidence_quoted_in_response_context": r["evidence"],
            "response": r["response"],
            "claim_id": "", "claim_text": "", "cited_evidence_span": "",
            "support_verdict": "", "citation_present": "",
            "citation_correct": "", "note": "",
        })

# Separate key kept local so DSEL-group analysis is possible after labeling.
key_path = ROOT / "evaluation" / "claim_provenance_key.csv"
with open(key_path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["annotation_id", "incident_id", "family", "dsel_rewritten"])
    w.writeheader()
    for r in sample:
        w.writerow({"annotation_id": r["annotation_id"], "incident_id": r["incident_id"],
                    "family": r["family"], "dsel_rewritten": r["dsel_rewritten"]})

README.write_text("""# Claim-level provenance annotation sheet (human, single annotator)

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
""", encoding="utf-8")

print(f"records available {len(records)}")
print(f"sample {len(sample)} families {dict(Counter(r['family'] for r in sample))}")
print(f"saved {OUT} and {key_path}")
