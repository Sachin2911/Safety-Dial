# Broader demonstrations improve development control but leave four failures

On 4 October 2026, a 50/50 mixture of PPO and PPO-Lagrangian demonstrations reduced
real health violations from 9/24 to 4/24 on the development starts. Mean progress
rose from 81.0% to 99.2% of the recorded reference. This supports broader demonstration
coverage as a useful change in this diagnostic, but four failures remain and the
small-sample safety difference is uncertain. **Gate 0 was not rerun.**

The preceding MLP run is archived at revision
`171a651f63306d8f3cd926a51208f34f0f1bdf77`. This experiment's protocol was committed
as `a3b5dbd` before new feature generation, training or outcome inspection.

## Controlled comparison

Both arms used the same 384-256-128-60 ReLU MLP, frozen latent features and feature
scaler, sequential 60-action targets, optimiser settings, three seeds, 100 epochs
per seed, and checkpoint-selection rule. Only the training-data allocation differed:

- Baseline: 45,986 samples from the original PPO-Lagrangian fit pool.
- Mixed: 22,993 samples per family, drawn from 120 fit episodes per family.

Short PPO episodes limited the mixed pool, so the baseline was subsampled to the
same total. Both arms spent 13,500 optimiser updates across their three seed runs.
All sampling was fixed by seed, without replacement, before outcomes were seen.

The common validation pool contained 60 whole episodes from each family, 11,820
samples per family. Each family had half the weight in checkpoint selection.
Validation targets received no gradient updates, and neither arm was refit on them.
The inherited feature scaler had previously used all 300 PPO-Lagrangian cloning
episodes, including their validation features; it was not refitted for this experiment.

The original dataset contains action-noise variants; the targets here are recorded
executed actions, without relabelling them with noiseless teacher outputs. Finite
imitation error therefore should not be interpreted entirely as missing policy capacity.

The selected baseline was seed 20261005, epoch 4; the selected mixed model was the
same seed, epoch 63. The total search budgets were equal, though the selected checkpoints
had different training ages. This baseline is newly trained under common family-balanced
validation and a matched frame budget; it is not the preceding MLP checkpoint.

## Outcomes

All real comparisons use the same 24 S4 development starts. Recorded-action replay
had no violations and mean progress 2.110 m per 0.8-second segment.

| Quantity | PPO-Lagrangian only | Mixed demonstrations |
|---|---:|---:|
| Real health violations | 9/24 (37.5%) | 4/24 (16.7%) |
| PPO-start violations | 8/8 | 2/8 |
| PPO-Lagrangian-start violations | 1/16 | 2/16 |
| Mean real progress | 1.710 m | 2.093 m |
| Progress relative to recorded actions | 81.0% | 99.2% |
| Imagined health violations | 0/24 | 0/24 |
| Family-balanced validation MSE | 0.23395 | 0.08781 |
| PPO validation MSE | 0.33617 | 0.04917 |
| PPO-Lagrangian validation MSE | 0.13173 | 0.12644 |

Seven starts became safe and two became unsafe. The paired violation-rate change
is -20.8 percentage points, with a source-episode bootstrap interval of -41.7 to
+4.2 points. This interval includes zero; the descriptive reduction does not establish
a general safety improvement. The mixed policy's violation-rate Wilson interval is
6.7% to 35.9%. Progress increased by 0.383 m, with a paired interval of 0.119 to 0.686 m.

The remaining mixed-policy failures are two PPO and two PPO-Lagrangian starts. Their
first unsafe environment-step indices are 66, 74, 77 and 87, on a zero-based scale.
All four occur late in the segment. That observation motivates examining accumulated
closed-loop error, but does not establish its cause. Both policies were imagined safe
on every start, so the model's false-safe behavior remains a separate problem.

[diagnostic.json](diagnostic.json) contains the summaries and uncertainty;
[development_rows.json](development_rows.json) records every paired outcome.
[data_plan.json](data_plan.json), [fit.json](fit.json), and
[training_history.json](training_history.json) record sampling and selection.
The run bundle stores dense real states, actions and latents for both controllers.

## Interpretation and next step

Including PPO demonstrations substantially reduced PPO validation error and the observed
PPO-start failure count. One additional PPO-Lagrangian start failed, so retaining both
families in subsequent checks is necessary. These development states have now been
inspected repeatedly; they are for diagnosis, not a new untouched confirmation set.

The next useful diagnostic is to inspect the four saved failed trajectories against
their teacher policies, separating imitation error on visited states from errors caused
by committing to ten actions between observations. A further controller change should
follow that evidence. Four development failures provide no basis to assume the much
stricter 3-of-256 Gate 0 will pass. No new Gate 0, transfer check or evolutionary search
ran here.

## Verification and resources

All 618 tests and Ruff passed. Four new tests cover disjoint whole-episode roles,
matched sample budgets, deterministic sampling, invalid inputs, equal-family validation
weighting and unchanged network/optimiser settings. Existing tests cover sequential
physics, real and imagined feedback, and batching invariance. See
[validation.json](validation.json).

The run took 61.6 seconds and used 7,200 new simulator steps, 480 imagined predictor
rows, 27,000 optimiser updates, and 58,681 newly rendered/encoded PPO frames. It reused
60,000 previously cached PPO-Lagrangian frames. Development histories and block ends
added 552 renders/encodes. Verification tests are separate from the scientific run.
The manifest records earlier source costs as provenance, not new collection.

The local bundle is `runs/walker2d-evo-s2-coverage-20261004-1/`. Its intended private
archive is `Sachioster/safetydial-walker2d`, under
`evo/walker2d-evo-s2-coverage-20261004-1`; an upload receipt records the remote revision
once archived. Learned weights, feature caches and dense arrays are excluded from git.
[manifest.json](manifest.json) pins source revisions, code and data hashes, the
preregistration commit, packages, hardware, seeds and costs.

Reproduce with a fresh run ID:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  uv run python experiments/scripts/evo_coverage_diagnostic.py --run-id NEW_RUN_ID --no-upload
```
