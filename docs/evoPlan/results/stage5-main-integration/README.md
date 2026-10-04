# Main readiness workflow integration

The guarded main launcher now connects source collection, initial-history encoding,
all-arm search, selection freezing, paired evaluation and the declared analysis.
Its orchestration passed with synthetic backends through every phase and restart.
The real Walker2d main experiment has not run; the candidate protocol remains
explicitly disabled pending scientific review, archival and final sizing.

The results loader also verified all 30 evaluation query archives from the
completed four-root, two-seed pipeline check. Initial qpos and qvel matched the
physical roots in their declared order. Loading and verification used zero new
model queries and zero real steps. See [archive_check.json](archive_check.json).
This reused archive has only the zero extra penalty objective; it is not a
Walker2d ROSARL comparison.

## Verification

Eleven focused tests passed. They cover the disabled execution guard, complete
four-arm synthetic orchestration with pause/resume at each phase, final paired
analysis, corrupted numeric content, changed requests, extra query archives,
unresolved queries, incomplete evaluation, inconsistent costs, incorrect physical
root order and zero/saturated residual action diagnostics.

The synthetic orchestration uses the real bank, query, optimizer, freeze and
analysis implementations, with every model, actor, renderer and simulator backend
replaced by test doubles. It checks exact planned costs and no repeated completed
queries across restarts. Each restart after collection performs its separately
charged renderer fingerprint. It does not establish full-model main throughput
or scientific effects.

The complete regression suite passed: **691 tests, 49 warnings, 83.12 seconds**.
Repository Ruff checks passed. These checks cover the new code and earlier
experiments. Historical run outputs remain unchanged.

## Execution and remaining decisions

`uv run python experiments/scripts/evo_readiness_main.py` performs a query-free
review. `--execute` refuses the current disabled candidate before asset loading
or run creation. The [recorded preflight](main_preflight.json) retains the outstanding
requirements. A reviewed, locked protocol must name a run ID; `--resume` requires
identical protocol and executable source. Its absolute wall-time deadline survives
process restarts, including time while stopped.

The final loader checks study identity, selection and parameter hashes, every
expected query request and numeric hash, paired source order and complete accounting.
It exports explicit condition, seed and source-episode axes. Dense physics truth,
block-end truth and real-image readouts remain separate from imagined outcomes.
A smaller prediction gap alone cannot satisfy the declared real-risk and progress
requirements.

The [prospective batch benchmark](../../protocols/stage5-batch-benchmark-20261004.md)
will measure the six proposed model-query shapes using existing features only.
Scientific review of the readiness extension and ROSARL adaptation, a variance
plan for the unmeasured ROSARL contrast, final counts and the pending private
noise archive remain prerequisites for the main launch.
