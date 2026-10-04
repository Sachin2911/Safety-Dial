# Stage 5 implementation preparation

The main workflow is integrated and its orchestration has passed synthetic
end-to-end validation. The full project suite passes 691 tests. The actual main
study remains a disabled review candidate; no scientific approval, ROSARL effect
or new final evaluation bank is claimed.

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

The [bounded batch benchmark](stage5-batch-benchmark-20261004.md) is declared for
all six proposed model-query shapes on existing features. After measuring runtime,
review variance between search seeds, including the currently unmeasured ROSARL
contrast. The proposed ten paired seeds and 320 final episodes remain provisional.

Record the scientific decision on the readiness extension and observed-return
ROSARL adaptation. Complete private archival and pin the noise revision. Commit
and lock scoring, role splits, sample sizes, seeds, budgets and endpoints before
collecting the main episodes. Historical strict Gate 0 and joint transfer-screen
failures remain failures; no controller repairs or environment change are planned.
