# Stage 5 implementation preparation

The main workflow is integrated and its orchestration has passed synthetic
end-to-end validation. The latest full project suite passes 730 tests, including offline sizing and user-authorized pilot checks. The original main candidate remains disabled. The user has now
[explicitly authorized the readiness extension and four-arm study](stage5-user-decision-20261004.json).
A new bounded development variance pilot will inform the final sample-size review.
No supervisor approval, ROSARL effect or new final evaluation bank is claimed.

## Completed components

The readiness CMA core saves complete generations, optimizer/random state,
fitness-only penalty bounds, all nominees, selection picks and costs atomically.
Every nominee is rescored on the independent selection bank when the penalty
changes. Full-model zero-penalty interrupted and uninterrupted searches matched
exactly at k=0 and k=1 in the
[resume check](../results/stage5-resume-check/walker2d-evo-s5-resume-check-20261004-1/README.md).
Synthetic tests also cover the proposed ROSARL-style code path.

The query store pins inputs, outputs and actual costs. Complete queries can be
reused; unresolved partial queries retain their measured costs or uncertainty and
refuse implicit replay. The study scheduler freezes all selected policies before
any final real outcomes. The
[pipeline check](../results/stage5-pipeline-check/walker2d-evo-s5-pipeline-check-20261004-1/README.md)
passed search/evaluation resume and reproduced archived real controls bitwise.

Fresh-bank collection pins disjoint whole source episodes, charges all attempted
steps and rejected sources, and preserves valid trajectory prefixes on failure.
The [six-episode bank check](../results/stage5-bank-check/walker2d-evo-s5-bank-check-20261004-1/README.md)
passed collection and encoding resume with no repeated completed queries. Those
episodes are development data, never the main final bank.

The main estimators use paired crossed search-seed/source-episode resampling,
two 97.5% co-primary intervals, descriptive secondary contrasts and distinct real
risk/progress requirements. Degenerate empirical intervals are flagged. Structural
preflight derives the exact query/physics budgets and rejects the disabled draft.

## Integrated main workflow

The [main integration record](../results/stage5-main-integration/README.md) covers
the guarded launcher, frozen-asset loading, clipped initial-action diagnostics,
verified final-query loading and explicit paired numeric exports. The actual main
orchestration passed with synthetic backends, including restarts after collection,
encoding, search and evaluation, followed by analysis and completed-query reuse.

The new loader verified all 30 completed full-model/simulator pipeline receipts
against their declared requests and physical root order without new experiment
queries. Eleven new focused tests passed; the complete suite passed with 691 tests,
49 warnings, in 83.12 seconds. Repository Ruff checks passed.

The launcher requires committed source and a locked protocol with recorded review
evidence. It checks private archive revisions, embedded controller identity and
source hashes, and preserves the absolute wall cap across restarts. Resuming a
completed run still incurs its recorded startup renderer check. Test doubles do
not demonstrate a full-model four-arm main run.

## Remaining before the main launch

The [bounded batch benchmark](../results/stage5-batch-benchmark/walker2d-evo-s5-batch-benchmark-20261004-1/benchmark.json)
completed all six proposed model-query shapes: 590,400 predictor rows, zero real
steps, and a 2.72-hour model-only projection for the ten-seed candidate. Independent
verification checked all 31 query archives and 105 source snapshots. The largest
query allocated about 1.15 GiB of CUDA memory. This projection excludes simulation,
collection, encoding, archival, analysis and interruptions.

The [independent validation](../results/stage5-independent-audit/validation.json)
closed two prelaunch gaps: unsupported protocol semantics now fail before collection,
and cached query receipts check declared costs and metadata integrity. The final
full regression suite passed 722 tests with 49 warnings in 83.47 seconds; Ruff passed.
The existing 30-query real pipeline archive remains readable without new queries.

The [offline Monte Carlo audit](../results/stage5-independent-audit/mc_precision.json)
verified all 48 saved 32-sample pilot audits against the reported probabilities.
For the k=1 high-minus-low comparison, estimated conditional audit sampling SE is
0.12 percentage points. This supports pooled audit precision only: individual
root probabilities remain noisy, and ROSARL effects and search-seed variance are
not measured by this check.

The [sizing sensitivity review](../results/stage5-independent-audit/sizing_review.json)
compares 10, 20 and 30 paired seeds with the same 320-root batch shapes. Twenty
seeds is a review candidate, with 5.45 model-only hours and an illustrative 11.37
hours under twice the scaled pilot nonfitness time plus a one-hour reserve. The
reserve and scaling are assumptions, not an end-to-end benchmark or runtime bound.
Thirty seeds exceeds the 12-hour cap under that scenario. Five focused arithmetic
tests passed. Plausible unmeasured ROSARL variance scenarios still give less than
80% power for a 10-point effect at twenty seeds, so no power guarantee or final
sample-size lock is claimed. The executable candidate stays unchanged at ten
provisional seeds and 320 evaluation episodes.

The user's scientific decision is recorded separately from supervisor approval.
The exact noise bundle was privately uploaded with explicit permission and all
24 remote files verified at revision `d164248ecb2a2b80c31b9145074528168d9ec2d1`.
Complete the bounded variance pilot, then commit and lock scoring, role splits,
sample sizes, seeds, budgets and endpoints before collecting the main episodes. Historical strict Gate 0 and joint transfer-screen
failures remain failures; no controller repairs or environment change are planned.
