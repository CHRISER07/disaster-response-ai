# EXP-7D: corrected temporal freshness evaluation

## Distinction from earlier EXP-7 pilots

- `tgi_exp7_results.csv` (EXP-7a) and `tgi_exp7b_results.csv` are INVALID as TGI
  measurements: the injected stale document was never consumed by the agent,
  which answered from a prefetched gauge carrying a severity label. All arms
  scored 100%. Preserved as invalid pilots; not reported as TGI results.
- `exp7d_results.csv` is the corrected experiment. Unified evidence fixture,
  severity text removed, mandatory evidence-consumption assertion on the final
  model context, structured JSON decision, matched evidence order, production
  `temporal_tag` used in the TGI arm.

## Verified result (720/720 executions OK, all consumption assertions passed)

Primary, 60 scenarios x 3 arms x 3 seeds = 540 executions:
- TGI: 53.3% (96/180), stuck-on-stale 84/180
- TimestampOnly: 50.0% (90/180), stuck 90/180
- NoTGI: 50.0% (90/180), stuck 90/180

Scenario-level Wilcoxon (n=60), Holm-corrected:
- TGI vs TimestampOnly: raw p=0.157, Holm p=0.472, delta CI [0.00, 0.08]
- TGI vs NoTGI: raw p=0.157, Holm p=0.472
- TimestampOnly vs NoTGI: identical, p=1

Direction-specific: all arms 100% when fresh=danger/stale=safe; TGI 6.7% (6/90)
and both comparators 0% when fresh=safe/stale=danger.

Diagnostic (20 scenarios, 180 executions), reported separately:
- TGI 40.0%, TimestampOnly 0%, NoTGI 0%

All 60 scenarios were consistent across the 3 seeds (deterministic decoding).
Abstention was 0. Decision/reason inconsistency: 3/180 (TGI), 6/180 (Timestamp),
0/180 (NoTGI).

## Finding

No significant TGI advantage. A safety-prompt dominance effect causes the model
to emit CURRENT_HAZARD even when it selects the correct fresh record and states
that no hazard exists. This is an engineering finding about the interaction of
the production safety prompt with temporal evidence, documented in the paper's
Failure Analysis subsection. No system or metric was changed to favor TGI.
