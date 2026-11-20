# EXP-7 temporal freshness benchmark design

## EXP-7a (control), `tgi_exp7a_control_results.csv`

60 scenarios, stale archive from 2013, contradicting a fresh gauge. All three
arms (TGI, No-TGI, Timestamp-Only) scored 180/180 correct with zero false
evacuations and zero missed evacuations. A 13-year-old archive is rejected by
every arm, so this suite is a control showing robustness, not a discriminating
test of TGI. It also exposed a scenario-ID collision (both conditions used the
same 4-character prefix), which was fixed in EXP-7b.

## EXP-7b (discriminating), `tgi_exp7b_results.csv`

120 scenarios, 4 ages x 2 conditions x 5 real USGS sites x 3 repetitions,
3 seeds, 3 arms = 1,080 trials. The stale document age is set just outside the
actual TGI tier boundaries defined in `core/temporal_grounding.py`:

- 80 hours -> RECENT tier (just outside the 72-hour LIVE window)
- 8 days -> HISTORICAL tier (just outside the 7-day RECENT window)
- 2013 -> far HISTORICAL (control)
- future-dated -> ANOMALOUS tier (clock skew)

All timestamps are computed against a pinned reference time
(2026-11-21T12:00:00Z) passed to `temporal_tag(now=...)`, so tier assignment is
reproducible and independent of wall-clock date.

The stale document states the opposite of the live gauge. Ground-truth label
comes only from the gauge threshold (EVACUATION iff reading >= 1.3*flood stage).
The three arms receive byte-identical evidence except the temporal tier label:

- Full_Agent_TGI: real tier tag
- Full_Agent_NoTGI: production no_tgi strip, all tags -> [DOCUMENT]
- Timestamp_Only: tier label hidden, raw source date visible

This design was chosen because EXP-7a tied across arms; it targets the
mechanism's decision boundary, not a post-hoc winner.
