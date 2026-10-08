# MLP improves the demonstrated gait but still fails on PPO starts

On 4 October 2026, a small MLP predicting ten sequential actions reduced health
violations on the 24 development starts from 13 to 9. The change came entirely from
PPO-Lagrangian starts: 1/16 violated with the MLP, compared with 5/16 for the linear
controller. Both failed on all 8 PPO starts. **This was a development diagnostic;
the 256-state Gate 0 was not rerun, and evolutionary search was not started.**

## Fixed protocol

Commit `e8f650b` fixed the protocol before training. The MLP has layers
384, 256, 128 and 60, with ReLU hidden activations and 139,196 parameters. Its features,
normalisation and full sequential-action targets are the same as the preceding
linear-block run. The backbone, predictor and probes remain frozen.

Both controllers fit the same 240 late PPO-Lagrangian episodes (47,280 samples).
The same 60 whole episodes (11,820 samples) provide offline validation. The MLP ran
three fixed seeds for 100 epochs each with AdamW, batch size 1,024, learning rate
0.001 and weight decay 0.0001. Selection used clipped validation MSE only, with
smaller seed then earlier epoch breaking ties. No real outcome selected a checkpoint.
Feature scaling was inherited from the previous experiment and therefore uses all
300 cloning episodes; validation targets received no gradient updates.

The linear baseline used the previously selected ridge coefficient 1.0 and was fit
on those same 240 episodes. Neither controller was refit on the validation episodes.
Consequently, this baseline differs from the 300-episode refit evaluated in the
preceding 256-state gate. Those gate numbers should not be substituted in this comparison.

The selected MLP was seed 20261004 at epoch 82. The best validation MSEs of the three
seeds were 0.11537, 0.11826 and 0.11660. The comparison uses one selected controller,
not three independent real-evaluation replicates.

## Offline and development outcomes

| Quantity | Linear full block | MLP full block |
|---|---:|---:|
| Validation full-block MSE | 0.13330 | 0.11537 |
| Validation block-mean MSE | 0.00743 | 0.00457 |
| Validation within-block MSE | 0.12587 | 0.11080 |
| Predicted within-block variation RMS | 0.16783 | 0.22565 |
| Real violations on all development starts | 13/24 (54.2%) | 9/24 (37.5%) |
| Real violations on PPO-Lagrangian starts | 5/16 | 1/16 |
| Real violations on PPO starts | 8/8 | 8/8 |
| Mean progress per 0.8 s | 1.794 m | 1.684 m |
| Fraction of recorded progress | 85.0% | 79.8% |
| Imagined violations | 0/24 | 2/24 |

The target within-block variation RMS is 0.39363. Validation action MSE decreased
by 13.5%, and the MLP captured more temporal variation, though substantial error remains.
The exact recorded tapes had 0/24 violations and mean progress 2.110 m.

The MLP's overall violation rate has a Wilson interval of 21.2% to 57.3%. Its 1/16
PPO-Lagrangian rate has an interval of 1.1% to 28.3%; this small sample cannot establish
a reliable safe controller. The paired source-episode bootstrap difference in overall
violation rate is -16.7 percentage points (interval -33.3 to -4.2). Progress changed
by -0.110 m (interval -0.229 to +0.003). These describe this selected model on 24
reserved development episodes, not an untouched final result or a Gate 0 verdict.

[diagnostic.json](diagnostic.json) contains the summaries and intervals;
[development_rows.json](development_rows.json) contains every paired outcome;
[fit.json](fit.json) and [training_history.json](training_history.json) record selection.
The private run bundle also stores the actual action tapes, dense states and latents
for both real controllers, making trajectory-level follow-up possible without rerunning.

## Interpretation and next experiment

The result supports nonlinear imitation capacity as a useful change for the demonstrated
gait. It also leaves a sharp source-family mismatch: all PPO starts still failed,
and PPO demonstrations were absent from cloning. Broader demonstration coverage is
therefore the next controlled experiment to consider, keeping this architecture fixed
and measuring both families separately. The data association does not itself prove
causation, and mixing the two gaits may introduce a trade-off that must be measured.

The world model still underestimates this MLP's violations (2 imagined versus 9 real).
Controller viability must be established before a new Gate 0 run, a transfer check or
selection-pressure experiments. No such stages ran here.

## Verification and resources

All 614 tests and Ruff passed. Four new tests cover trainable-network versus stateless
policy agreement, saved checkpoint selection with no validation gradients, development
role restrictions, exact real physics, grouping invariance, and action feedback in
both real and imagined rollouts. See [validation.json](validation.json).

The experiment took 25 seconds. It used 14,100 optimiser updates, 7,200 new simulator
steps (2,400 each for linear, MLP and recorded tapes), 480 imagined predictor rows,
and 552 new history/block-end renders and encodes. It reused the cached 60,000 cloning
frames rather than rendering them again. Verification tests are separate from these
scientific-run costs. [manifest.json](manifest.json) carries the source run's earlier
costs as provenance, not additional new collection.

The cloning cache is pinned to private Hugging Face revision
`d70d6863c2bcda9834ffbd2b6dd24f9a7dd6e863`, with content hashes checked before fitting.
The new bundle is `runs/walker2d-evo-s2-mlp-20261004-1/`, with intended private destination
`Sachioster/safetydial-walker2d`, under `evo/walker2d-evo-s2-mlp-20261004-1`.
The upload receipt, when present, records the archived revision. Weights and dense run
arrays are excluded from git; small metrics and provenance are committed here.

Reproduce locally with a fresh run ID:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  uv run python experiments/scripts/evo_mlp_diagnostic.py --run-id NEW_RUN_ID --no-upload
```
