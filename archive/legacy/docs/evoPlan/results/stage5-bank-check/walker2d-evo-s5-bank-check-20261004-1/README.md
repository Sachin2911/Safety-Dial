# Fresh-source bank workflow validation

The prospective six-episode development check passed under declaration commit
`8ba746e`. It collected two independent roots in each of the workflow's fitness,
selection and evaluation roles. All six episodes are development validation data;
none is part of the future main final bank. No visual-controller outcomes were
queried and no world-model rollouts or training were performed.

Collection and initial-history encoding each paused and resumed successfully.
Invoking either completed stage again used zero additional physics, teacher,
render or encode queries. The role ranges were disjoint, every root came from a
separate source episode, and root eligibility preserved the readiness audit's
healthy-recorded-future conditioning. The same frozen 32-actor pool and uniform
0/0.05/0.10 action-noise mixture were used.

## Measured costs

| Item | Count |
|---|---:|
| Attempted / accepted source episodes | 6 / 6 |
| Rejected short episodes | 0 |
| Real simulator steps | 5,440 |
| Teacher action queries | 5,440 |
| World-model predictor rows | 0 |
| History renders / encodes | 18 / 18 |
| Separate renderer fingerprint frames | 64 |
| Gradient updates / candidate-controller queries | 0 / 0 |
| Wall time including model loading | 6.63 s |

The declared upper bound was 24,000 source steps, not a promise that every source
teacher would last exactly 1,000 steps. Source episodes stopped at first dense
health failure or the fixed 1,000-step cap. All attempted steps were charged.

## Other verification in this implementation

Fourteen focused tests passed across collection, encoding, main analysis and
protocol validation. They verify rejected-episode charges, fixed attempt caps,
whole-episode role separation, exact resume, preservation of valid trajectory
prefixes after physics failure, rejection of invalid snapshots, and refusal to
reuse image features after the encoder identity changes.

The main statistical implementation follows the review draft: independent crossed
seed/episode bootstrap with the same resamples across paired arms, two 97.5%
co-primary intervals, descriptive 95% secondary intervals, and separate real-risk
and progress-retention requirements. A changed noisy prediction alone cannot
satisfy its practical-benefit conditions. Undefined progress denominators prevent
a retention claim. Degenerate empirical bootstrap intervals are flagged, and
fixed-policy Wilson intervals retain nonzero uncertainty after zero real failures.
These are code tests, not new scientific findings or a ROSARL experiment.

The query-free [main preflight](main_preflight.json) reproduces the draft accounting:
222,201,600 predictor rows and at most 13,504,000 new simulator steps, including
source attempts. It confirms that the main protocol remains disabled and records
the outstanding scientific review, fixed counts, exact-batch throughput and
archival prerequisites. No main-study launch follows this validation automatically.

The end-to-end main launcher and verified final-query loader still need wiring.
The complete component checks do not establish that the main experiment ran.
See [check.json](check.json) for exact costs, role IDs and completed invariants.
The bundle is local; no upload is claimed.

Full regression verification: 680 tests passed, 49 warnings, 77.59 seconds. Ruff passed.
