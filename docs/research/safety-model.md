# Working PushT learned safety checker

Updated 29 October 2026. The learned checker now controls all safety decisions in
three completed PushT demonstrations: island, bar and gate. Frozen LeWM still supplies
the task-progress score. SafeCEM and its fixed settings are unchanged. The new result
is a working replacement for handwritten predictive dynamics on this PushT setup,
not an evolutionary advantage or a broad task-success benchmark.

## What changed after the first prototype

The earlier latent-only clearance MLP remains preserved and unvalidated. Its training
MAE was 13.42 px and it wrongly accepted 429 of 3,751 candidate checks in a short
reference-controlled shadow run. Its implementation receipt is
`progress/receipts/20261029-pusht-safety-model-implementation.json`.

The working version separates two jobs:

1. A small supervised model predicts the pusher position at every future physics
   substep from native observed position/velocity and candidate actions.
2. Shared circle, capsule, box and workspace distance routines turn that predicted
   motion into full-pusher clearance and a violation score for SafeCEM.

This input choice was explicitly approved by Sachin: the working model uses PushT's native four-value
`proprio` observation (pusher x, y, vx, vy). It does not infer safety solely from LeWM
latents. It receives no T-block pose, goal state or counterfactual simulator state.
The T-block remains excluded from hazard safety, while reaching its goal remains
part of task completion. Geometry is supplied, not learned.

## Model and training

The motion predictor is a causal affine layer, fitted by ordinary least squares.
It predicts up to 15 actions at 10 physics substeps each. Time masks prevent future
actions from affecting earlier positions. No PD-controller gains, handwritten response
matrix, simulator preview or analytic motion equation enters inference.

A protocol was frozen before collecting native simulator trajectories:
`configs/safety_model/pusht-motion-v1.json`. It chose 64 training scenes, 12 calibration
scenes and 12 independent test scenes, with eight randomized 15-action motions per
scene. All motions from a source scene stay in the same partition. This was one
supervised fit, not a parameter sweep.

The 704 recorded trajectories cost 10,560 executed actions, 105,600 executed physics
steps and 1,408 reset physics steps. Collection and fitting took about 5.5 seconds
on the local CPU; no paid compute was rented. The simple kinematic pusher's motion
was recovered to floating-point precision on 96 independent test trajectories.
The maximum position error was about 5.75e-12 px. Calibration uses source-group worst
position error, with a 0.001 px numerical floor. This is empirical validation of this
pinned simulator's dynamics, not a universal safety guarantee.

## Learned-controlled demonstrations

A separate panel was frozen in `configs/safety_model/pusht-motion-e2e-v1/` before the
runs. It uses the existing known source scene, the same original joint pusher/block
goal, 6 px margin around the full 15 px-radius pusher, and the existing fixed planner
settings. Every physics step, including the initial state, is checked by a passive
outcome monitor. It never vetoes actions.

- Island: safe completion in 26 actions; minimum hazard clearance 6.905 px.
- Bar: safe completion in 43 actions; minimum hazard clearance 6.954 px.
- Gate: safe completion in 25 actions; minimum hazard clearance 6.405 px.

Safe completion was 3/3, observed violation episodes 0/3 and unfinished episodes 0/3.
All three also respected the arena margin and moved the T-block to its goal.
They share one task scene and seed, so these demonstrations do not establish general
safe-completion rates. Zero observed violations here does not establish zero risk.
There is no new optimizer, ablation or other-environment result.

A separate audit reloads the saved checkpoint and compares predictions with the
actual recorded physics of these executed plans. All three passed; maximum errors
were below 3.3e-12 px and within the 0.001 px allowance. It also verifies disjoint
training/calibration/test source groups and the original joint goal. This audit used
20 offline model queries and no new simulator interactions. The reusable command is
`experiments/scripts/audit_pusht_learned_motion.py`.

The actual runner installs guards that raise if the old analytic response or reference
checker is invoked. All three completed with those guards active. No simulator
previews were used. The first island launch failed before model loading because of
relative-path handling; it consumed no interactions, was preserved, and was retried
with the same scene, seed and settings after the path repair.

## Reproduce and continue

Reusable implementation: `experiments/helpers/learned_motion.py`. The runner is
`experiments/scripts/run_safety_pusht.py`, selected with protocol backend
`observed_motion`. The initial latent-only backend remains available for development.
Checkpoints, motion labels, source freezes, traces and videos stay outside git.
The verified 75 MB release is preserved in the private `Sachioster/SafetyDial-PushT`
dataset, under `campaigns/pusht-learned-safety-20261029-v1`. The working receipt
records the pinned revision and downloaded checkpoint/video hash verification.
The [private release](https://huggingface.co/datasets/Sachioster/SafetyDial-PushT/tree/34256873f32a27e90c16ee081d30660aecc6acb4/campaigns/pusht-learned-safety-20261029-v1)
contains the checkpoints, training evidence, source snapshots and all three replays.
Local artifacts are under `outputs/pusht-learned-motion-v1/` and
`outputs/pusht-learned-motion-e2e-v1/`.

Use the venv Python and fresh output directories. These recorded commands are
reproduction context, not instructions to launch another study automatically:

```bash
.venv/bin/python experiments/scripts/train_pusht_motion.py --config configs/safety_model/pusht-motion-v1.json
.venv/bin/python experiments/scripts/run_safety_pusht.py --protocol configs/safety_model/pusht-motion-e2e-v1/island.json --out OUTPUT/island
.venv/bin/python experiments/scripts/media_learned_pusht.py --root outputs/pusht-learned-motion-e2e-v1 --out OUTPUT/media
.venv/bin/python -m pytest -q experiments/tests/test_learned_motion.py experiments/tests/test_safety_model.py experiments/tests/test_pusht_smoke.py experiments/tests/test_lewm16.py
```

Next work stays on PushT: optimizer/configuration choices and broader development
validation can now be studied separately. Image-only safety, ablations and other
environments remain separate extensions. Do not present the current affine motion
model as evidence for latent-only safety or cross-environment weight transfer.
