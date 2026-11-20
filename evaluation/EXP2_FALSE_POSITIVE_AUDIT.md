# EXP-2 false-evacuation audit

The 27 false evacuations reported for each arm in `tgi_exp2_results.csv` were
inspected. They are not 27 unconditional directives:

| Category | Full Agent (TGI) | Full Agent (No-TGI) |
|---|---|---|
| Conditional ("if gauge exceeds stage, evacuate") | 9 | 0 |
| Preparatory ("prepare for potential evacuation") | 9 | 12 |
| Direct imperative ("evacuate immediately", "issue orders") | 9 | 6 |
| Other | 0 | 9 |

Readings cluster at 1.0x and 1.2x flood stage (21 of 27 for TGI), where the
production sensor tool's own severity logic already emits DANGER and
"prepare evacuation resources". Those readings were labelled NO_EVACUATION in
EXP-2, so the ground truth at the boundary was ambiguous. The remaining cases
are conditional or planning statements that `detect_evacuation` counts as
evacuation directives.

EXP-2b was therefore registered as a separate suite using readings strictly
below flood stage (<=0.8x), matching the original v2 suite's clearly submerged
readings. EXP-2 is retained and both are reported; no primary metric was
changed.
