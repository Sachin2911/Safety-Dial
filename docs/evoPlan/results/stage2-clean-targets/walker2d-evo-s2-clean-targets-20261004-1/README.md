# Broad clean source-teacher targets did not repair the controller

The clean-target controller failed 4/24 development segments, the same count as
the archived matched recorded-target control. Mean progress fell from 2.120 m to
2.061 m. Both imagined zero violations. PPO failures changed from three to two
of eight; PPO-Lagrangian failures changed from one to two of sixteen. The paired
failure-rate difference was zero, with bootstrap interval [-0.208, 0.208]; progress
difference was -0.059 m, interval [-0.155, 0.034]. The earlier mixed controller
remains the reference. Gate 0 was not rerun.

All 45,986 existing fit examples were relabelled. Each example's exact source
teacher generated ten actions with fresh feedback each simulator step and zero
action noise. Each example starts an independent branch from the cached state
with zero solver warm start; the ten steps proceed without intermediate restores.
Every example was retained, including the single teacher block with a health
violation. No validation episode was queried for clean targets or used for fitting.
The deployed controller receives visual features and previous actions only;
privileged teacher observations are training supervision, not policy inputs.

Both arms use the same 444-512-256-60 network, scalers, 45,986 fit inputs, 23,640
validation inputs, three fitting seeds, 100 epochs and 13,500 training updates.
The recorded-target control's training and outcomes were reused from the pinned
information continuation rather than recomputed. Common offline validation uses
recorded actions for both arms. Epochs 20, 50, 100 plus each seed's best offline
checkpoint enter the same real-validation selection rule on 64 setA roots.
Selection chose seed 20261006, epoch 68, with
5/64 validation failures versus the control's 7/64. No development
outcome was used for checkpoint selection.

The new run consumed 459,860 teacher simulator steps and queries, 79,200 evaluation
steps, 13,500 optimiser updates, 240 imagined rows, 8,248 renders and 8,184 encodes.
Total new simulator cost is 539,060 steps; the reused control's historical training
and evaluation costs remain in its own manifest. Wall time was 122.65 seconds.
All 640 tests passed with 49 warnings, and Ruff passed. Protocol commit: 5283873.

This small, repeatedly inspected development set does not prove clean targets are
useless. This declared replacement alone did not establish the reliable visual
starting controller needed for Gate 0. All labels, branch-end states, dense health
clearances, candidate weights, evaluation trajectories, source and logs are saved.
Private archival is pending.
