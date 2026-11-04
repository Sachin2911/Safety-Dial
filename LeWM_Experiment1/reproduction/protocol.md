# Frozen LeWM checkpoint reproduction, 4 November 2026

This prospective protocol implements Sachin's explicit request to rent a GPU and
perform the reproduction described in `theOdessy/reproduction.pdf`. It does not
reopen any completed SafeLeWM or SafeBlock study. All simulation, model execution,
installation and experiment tests run on the rented host. Local work is editing,
orchestration, transfer and inspection.

## Question and scope

Can the released frozen LeWM checkpoints, with the original CEM planner, reproduce
nearby-goal task competence on TwoRoom, Reacher, PushT and OGBench single Cube?
No training, AP, retrieval, adapter, safety filter or outcome-driven tuning is used.
This is a released-checkpoint evaluation, not a replication of training seeds.

The reference is arXiv:2603.19312v3, Figure 6 and Appendices D and F. The paper PDF
SHA256 is `553a01501ac74ec56721cfdb936d3825ac941734133a1fd761f614c01b6887e9`.
Official source is `lucas-maes/le-wm` commit
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. Public model and dataset revisions and
compressed-file SHA256 values are recorded in `assets.json` before evaluation.
The pre-outcome runtime decisions are recorded in `compatibility.md`, including
the installed evaluator API and the exclusive-end slice correction required to
select a goal exactly 25 actions after the start.

## Fixed evaluation

Each environment uses 50 cases for each of seeds 42, 43 and 44. Seed 42 is the
primary paper comparison; 43 and 44 are separately reported robustness repeats.
Cases follow the released evaluator: eligible dataset rows have a goal 25 actions
later within the same episode; NumPy default_rng(seed) samples without replacement
from `len(valid_indices) - 1`, then rows are sorted. Preserve that released
off-by-one sampling convention. Save exact rows, episode IDs and start/goal steps
before the first episode. The same cases are reused in the sensitivity comparison.

Restore the recorded start and goal using the official task-specific callables.
Use the benchmark success monitor and at most 50 executed actions per case. Image
input is 224 by 224 with ImageNet normalization, and nonpixel input/action scaling
is fitted on the complete corresponding dataset as in the evaluator. Freeze weights
and use evaluation mode. Retain the released checkpoint's three positional history
slots and the original model's rollout behavior. The paper describes history one
for TwoRoom, while the released metadata specifies three; changing checkpoint
structure would no longer test the released checkpoint.

CEM uses 300 candidates, 30 elites, initial variance scale 1, batch size 1, five
planning blocks, five environment actions per block and five blocks executed before
replanning. The main condition uses Appendix D's 10 CEM iterations for TwoRoom,
Reacher and Cube, and 30 for PushT. A predeclared sensitivity condition uses the
released shared configuration's 30 iterations on the same seed-42 panel for the
other three environments. PushT already uses 30, so it is not duplicated.

Use the original model and stable-worldmodel CEM/evaluation behavior where the
published APIs permit it. Any necessary compatibility change is recorded with its
source and effect before scientific outcomes. Pin the actual installed runtime.
Rendering uses the image's validated OSMesa profile unless a pre-outcome engineering
check establishes that the original EGL profile is available; record the final
choice. This is a renderer deviation if OSMesa is used. Reacher's released dataset
filename differs from the paper's collection description; retain the public data
and report the discrepancy rather than silently substituting another dataset.

## Budgets and engineering

One Vast RTX 4090; six-hour work limit; maximum $1/hour including allocated disk;
$8 estimated session budget including 200 GB input-transfer allowance and 2 GB of
outputs plus verification readback. Provider billing continues until manual rental
destruction. The budget is an estimate, not a provider spending cap.

Scientific maximum: 600 main-condition episodes plus 150 sensitivity episodes,
37,500 executed environment actions. Engineering maximum: 20 episodes and 1,000
actions in total, counted separately and excluded from scientific estimates.
Engineering cases use seed 0 and must not be used to choose a better-performing
planner. Dataset reads, environment resets and rendering are recorded separately
from action steps. Count reset-internal simulation separately where observable.
Count failed/aborted action attempts and model queries, preserving their evidence.
No additional scientific panels or optimizer runs are authorized by this protocol.

Compatibility corrections before scientific evaluation do not spend a fresh
engineering allowance. A scientific result is never discarded because it scores
poorly. If an implementation defect invalidates a run, retain its actions and
evidence and document the defect; do not exceed these budgets by silently rerunning.
Stop at the finite deadline and preserve incomplete work if a gate prevents progress.

## Analysis fixed before outcomes

Paper success references are TwoRoom 87%, Reacher 86%, PushT 96%, Cube 74%.
For the seed-42 engineering competence check, a result no more than 10 percentage
points below its reference is considered numerically consistent enough to proceed
to interpretation. This is a practical tolerance, not an equivalence test. Report
all seed-specific counts and Wilson 95% intervals, plus pooled descriptive counts
with the caveat that repeated trajectories and a shared checkpoint are correlated.
Report the 30-iteration sensitivity separately, never substitute it for a failed
main comparison. Rounded paper scores and an unavailable training-seed distribution
do not support exact equality or formal equivalence claims.

For every condition preserve episode outcomes and denominators, exact cases,
configuration and code hashes, checkpoint/data provenance, simulator actions,
model cost calls, candidate rollouts, predicted latent steps and elapsed time.
Report failures, timeouts and unresolved deviations explicitly. A negative result
fulfills the evaluation objective when correctly executed and recorded; it does
not establish a competent foundation for that environment.

Publish immutable completed artifacts to private dataset
`Sachioster/lewm-experiment1`, under unique `runs/<run-id>/` prefixes. Keep download
caches outside the run directory. Verify final artifacts by independent readback,
write the local results report and notify Sachin when the rental is ready for
manual destruction. No automatic destruction is authorized.
