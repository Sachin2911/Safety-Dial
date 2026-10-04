# Stage 5 implementation preparation

The main study remains a review draft. A separate bounded development power pilot
completed under protocol commit `9c043c1`; it does not include ROSARL, a new final
bank or a claim that the failed gates passed. Its scientific source files have
not been changed during execution.

## Implemented during the pilot

The new [readiness CMA core](../../../experiments/helpers/evoReadinessCMA.py) stores
the optimiser and its private random stream, fitness-only penalty bounds, every
selection-bank nominee, checkpoint picks and query counters in one atomic
completed-generation checkpoint. It supports the proposed matched-termination
zero-penalty objective and the existing ROSARL-style score as code paths. Testing
the latter on synthetic metrics is implementation validation, not approval of the
scientific adaptation or a Walker2d ROSARL experiment.

Every generation's penalty bounds come only from its fitness outcomes. The
selection bank cannot update them. All stored nominees are rescored at the current
penalty before a checkpoint pick, which permits a previously rejected candidate
to become preferred when that penalty changes. Cross-generation fitness scores
are never substituted for selection-bank scores.

Configuration, initial parameters, package version and caller-supplied input
provenance form the checkpoint identity. Changing any of these rejects resume.
An interrupted, uncommitted generation retains a phase journal with completed
query receipts and the next requested query size. It refuses automatic replay
until partial cost accounting is reconciled. An uncompleted request is not
silently treated as zero cost or as a measured completed query count.

## Verification

Four new focused tests passed. Together with the existing CMA-ES regression suite,
31 tests passed in 3.36 s. Ruff and whitespace checks passed. Tests verify:

- Resuming at a completed generation reproduces the uninterrupted candidate
  sequence, selected policies, penalty state and charged query counts in both modes.
- Large selection-bank returns do not leak into fitness penalty bounds.
- Revised penalties rescore the entire nominee history, rather than only the
  previously preferred candidate.
- A partial selection interruption preserves its journal and cannot silently
  rerun already charged fitness queries.

The full project suite subsequently passed with the new core: 661 tests,
49 warnings, 79.29 seconds. Ruff passed. The new core has not yet been integrated into a main-study
entry point. The subsequent [frozen-model validation](../results/stage5-resume-check/walker2d-evo-s5-resume-check-20261004-1/README.md)
passed at k=0 and k=1: complete and resumed searches matched candidates, nominees,
picks, penalty bounds and query counts exactly. Its 4,400 predictor rows include
both copies of each validation search. It used zero real steps and the zero extra
penalty objective only. These checks do not establish successful locomotion,
transfer or a ROSARL effect.

## Remaining before a main launch

The completed pilot reports candidate behaviour, seed variation and runtime;
its k=0 gap amplification supports fresh-episode confirmation. ROSARL variance
remains unmeasured, and the four-seed variance estimates are provisional.
Integrate the checked core with the frozen residual controller, independent episode
banks and evaluation runner. Full-model generation-boundary resume and its query counters now match.
Keep final roots uncollected during power and
runtime planning. Lock the final config before collection, after the requested
scientific review of the readiness extension and ROSARL adaptation.

## Integrated study scheduling and evaluation

The subsequent [pipeline validation](../results/stage5-pipeline-check/walker2d-evo-s5-pipeline-check-20261004-1/README.md)
passed under declaration commit `c38ef5c`. Four small k=0/k=1 searches and eight
checkpoint selections exercised search pause/resume, an all-selections freeze,
paired real/imagined evaluation, evaluation pause/resume and zero-cost reuse of
completed work. Both the controller and recorded-tape reference reproduced their
archived simulator outcomes bitwise. All 6,400 predictor rows and 4,000 real steps
were charged; no new episodes or model/controller training were used.

`evoReadinessQueries.py` records each complete query as one atomic numeric archive,
including input identity, output hashes and measured costs. Partial queries retain
journals and refuse automatic replay. `evoReadinessStudy.py` schedules every arm,
freezes all picks, verifies bank identity and evaluation settings, and then evaluates
the complete frozen set. Five new focused tests cover leakage, interrupted costs,
corruption, scheduling and trajectory consistency.

Remaining engineering: integrate fresh-bank collection and the main entry point;
implement the declared crossed-bootstrap main analysis; run an exact final-batch
benchmark once sample sizes are locked. The reviewed readiness extension, ROSARL
adaptation and source archival are still required before a main launch. The new
components do not alter those scientific decisions or enable the disabled draft.

After pipeline integration, the full regression suite passed with 666 tests, 49 warnings, in 76.55 seconds. Ruff passed.

## Fresh-bank workflow, analysis and protocol checks

The [six-episode source-bank validation](../results/stage5-bank-check/walker2d-evo-s5-bank-check-20261004-1/README.md)
passed under declaration commit `8ba746e`. Collection and initial-history encoding
paused/resumed without duplicating completed queries. Six independent development
roots cost 5,440 real steps and 18 history encodes, with zero world-model predictor
rows, training updates or candidate-controller outcomes. These episodes cannot
be relabelled as the main final evaluation bank.

The collector stores every source attempt, charges rejected episodes, halts at
fixed role-level attempt caps and retains completed trajectory prefixes on failure.
Encoded banks pin source episodes, root contents and encoder identity. Partial
attempts and unresolved query journals still require reconciliation before replay.

`evoReadinessAnalysis.py` implements the declared crossed-bootstrap estimators,
paired co-primary and descriptive contrasts, baseline changes, common-k0 audits
and real-risk/progress checks. `evoReadinessProtocol.py` derives search specifications,
role ranges and exact budgets, and exposes the outstanding launch prerequisites.
Fourteen focused tests passed. The query-free preflight reproduces the draft costs
and correctly refuses to label the unreviewed draft execution-ready.

Remaining integration: the guarded main entry point, verified loading of the final
query archives into the declared analysis, and initial clipped-residual diagnostics.
An exact main-shape benchmark and a reviewed variance plan are still needed before
locking main counts. Scientific review and archival prerequisites remain unchanged.

After adding bank workflows, estimators and preflight, the full regression suite passed: 680 tests, 49 warnings, 77.59 seconds. Ruff passed.
