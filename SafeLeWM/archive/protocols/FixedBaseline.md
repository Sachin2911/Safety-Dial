# TwoRoom fixed baseline, 1 November 2026

Sachin authorised establishing the no-hazard baseline before proceeding to hazards.
Run on the existing L4 pod, which remains running until he requests shutdown.

## Frozen evaluation

Evaluate the unchanged `random05` implementation from the completed baseline
comparison on all 50 of its dataset cases, in their original order. Reuse the exact
saved inputs, source rows, goal offset 100, planner seed 42 and 150-action limit.
No selection, tuning, hazard, adapter, training or replacement of failed cases.
The preset is historical `random-05`, not a new random search. These are development
cases, not held-out confirmation of a selected method.

Use the identical checkpoint, preprocessing, action scaler, cached inference and
native environment as the completed comparison. Verify their frozen source hashes
and all input hashes before loading the model. Save a new source and panel freeze
before any reset. Reuse the prior passing model-parity and action-unit preflight
only if its frozen implementations and upstream environment/solver hashes match.

Count 50 scheduled episodes and at most 7,500 primitive actions, 50 resets and
13,500,000 candidate model steps. Stop the worker after 30 minutes. The overall
setup, evaluation and preservation allowance is one compute hour ($0.59 at the
current rate, excluding storage and unrelated idle time), with at most 2 GB of new
files. No additional simulator engineering episodes are planned. Preserve failures
and accounting; do not restart an outcome-bearing episode.

## Reporting and decision

Report native completion, unfinished episodes and engagement with denominators,
success by 100 actions, actions, model work, wall time and proposed versus executed
action clipping. Native rewards are zero; completion is the task outcome. There
are no added hazards, so no safety-violation claim is made. Audit every saved trace
against native termination, action bounds and resource counts. Preserve all traces
and inspect trajectory plots.

Compare paired outcomes with the previous paper-profile 5/50 on these exact inputs;
do not compare this panel as if it were the paper's original 87% experiment. Report
the exploratory Wilson interval, with the caveat that source episodes can repeat.
Separate same-room and cross-room outcomes to identify where competence is missing.
Do not tune a replacement if performance is weak. Baseline establishment can return
a negative result; hazards and adapter learning remain separate next steps.

Inspect the authors' unchanged evaluation entry point and its installed upstream
dependencies for differences from our runner. That source audit is not another
evaluation arm or a claim of published-baseline reproduction. Any further execution
requires its own prospective configuration and case freeze within user authority.

Outputs: `outputs/fixed-baseline-20261101/run-v1/`. Original baseline evidence stays
unchanged under `outputs/baseline-20261101/run-v2/` on the pod.
