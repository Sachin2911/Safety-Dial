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
