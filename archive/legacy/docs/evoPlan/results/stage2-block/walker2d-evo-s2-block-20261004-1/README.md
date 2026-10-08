# Sequential action blocks still fail Gate 0

On 4 October 2026, the revised linear controller predicted ten sequential six-motor
commands per model block instead of holding one command. Gate 0 still failed: **163 of
256 real segments violated the health rule**, against an allowance of 3. Mean forward
progress passed at 79% of the recorded reference. Stages 3 to 7 were not started.

The implementation and configuration were committed as `e62c3f9` before this run.
The encoder, world model, probes, 300 cloning episodes, episode split, ridge grid,
256 evaluation starts and gate thresholds were unchanged. There are now 23,100 policy
parameters, with actions executed at 125 Hz and observations updated at 12.5 Hz.
Existing experiment files and the earlier results were preserved.

## Gate results

Each segment lasts 0.8 seconds. The allowance remains provisional and subject to supervisor
confirmation. These are reused evaluation states, so this is a development gate rerun,
not an untouched confirmatory evaluation.

| Controller | Real health violations | Mean progress | Fraction of reference progress |
|---|---:|---:|---:|
| Exact recorded actions | 0/256 | 1.907 m | 100% |
| Previous held-action linear policy | 171/256 (66.8%) | 1.571 m | 82.4% |
| Sequential-block linear policy | 163/256 (63.7%) | 1.501 m | 78.7% |

The new violation rate has a 95% source-episode bootstrap interval of 55.1% to 71.9%
(64 source episodes). The paired change is -3.1 percentage points, with an interval
of -8.6 to +2.0 points: this does not establish a safety improvement. Of the 256 starts,
26 became safe and 18 became unsafe. Progress decreased by 0.069 m on average
(paired interval -0.107 to -0.033 m). See [gate0.json](gate0.json),
[evaluation_rows.json](evaluation_rows.json) and [diagnostic.json](diagnostic.json).

Imagination predicted violations in only 15/256 segments (5.9%). The probe on real
frames reported 176/256 (68.8%), while true block-end states reported 161/256 (62.9%).
The large imagined-versus-real discrepancy remains before evolutionary selection.

## What the diagnostic tells us

The ridge coefficient was selected using held-out episodes only: alpha 1.0, from
59,100 samples, of which 11,820 were validation samples. The final policy was then
refit on all selected episodes, as declared in the original protocol. Feature scaling
also follows that protocol and uses all selected cloning frames.

A post-hoc comparison refits each policy on fit episodes only, using its already selected
coefficient, and evaluates both against the same full action-block target:

| Validation quantity | Held-action policy | Sequential-block policy |
|---|---:|---:|
| Full action-block MSE | 0.16237 | 0.13330 |
| Block-mean MSE | 0.00742 | 0.00743 |
| Within-block MSE | 0.15494 | 0.12587 |
| Predicted within-block variation RMS | 0 | 0.16783 |

The target within-block variation RMS is 0.39363. The full-block linear map captures
some temporal variation, but substantial imitation error remains. Comparing the new
full-block MSE directly with the old mean-action MSE would compare different targets.
This diagnostic used cached features and saved outcomes, with no new simulator steps.

Failures also depend on the source policy family. PPO-generated starts failed in
101/104 cases (previously 96/104), while PPO-Lagrangian starts failed in 62/152
(previously 75/152). Cloning used only late PPO-Lagrangian episodes. This suggests a
source-distribution mismatch as one contributor; it does not establish causation,
and the substantial PPO-Lagrangian failure rate means that mismatch alone is not
an adequate explanation.

The earlier hold diagnostic did not prove that every possible held-action controller
must fail. This run similarly does not rule out every sequential-block controller.
It establishes that the specified linear latent controller does not pass Gate 0.

## Verification and cost

All 610 tests passed, including seven new tests covering temporal action order,
block normalisation, fixed-tape imagination equivalence, closed-loop feedback,
exact MuJoCo replay and CUDA worker/chunk invariance. Ruff passed. The rendering
fingerprint matches the pinned data. See [validation.json](validation.json).

The run took 53 seconds and used 51,200 new simulator steps: 25,600 for the cloned
policy and 25,600 for recorded-action replay. It made 2,560 imagined predictor rows
and rendered 60,000 cloning frames. The existing set-A collection cost was 2,999,673
steps; probe and root collection costs were 300,347 and 300,038 steps respectively.
These are reused upstream costs, not new collection. Verification tests are separate
from the scientific run. The diagnostic added zero simulator steps.

[manifest.json](manifest.json) records input revisions, data and code hashes, the
implementation commit, packages, seeds and hardware. Its dirty flag includes generated
result files and unrelated workspace files; the implementation hashes identify the
exact code used. Learned weights and the feature cache are excluded from git.

## Next decision and archive status

The preregistered response to failure is to stop before stage 3 and diagnose. A next
controller revision should address imitation capacity and the PPO versus PPO-Lagrangian
starting-state mismatch together. A small MLP with sequential-block outputs is a candidate,
not a demonstrated fix. Its fit should be chosen on independent development episodes,
with a new declaration before another gate evaluation. No MLP was trained in this run.

The local bundle is `runs/walker2d-evo-s2-block-20261004-1/`. The intended private archive
is `Sachioster/safetydial-walker2d`, under `evo/walker2d-evo-s2-block-20261004-1`.
**Upload is pending explicit approval.** Automatic approval review rejected the planned
upload because the generated payload and destination had not been explicitly approved.
The experiment was completed with `--no-upload`; there is no new remote revision yet.

Reproduce the experiment with a fresh run ID:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  uv run python experiments/scripts/evo_block_bc_init.py --run-id NEW_RUN_ID --no-upload
```

The post-hoc report script is `experiments/scripts/evo_block_gate0_report.py`;
its `--help` lists the saved-input and new-output arguments.
