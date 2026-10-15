# UNISafe migrated baseline pilot

**15 October 2026.** Frozen pretrained controllers, RTX 3090, Isaac Sim 6.2 / Isaac Lab 3.

The controller runs end to end, but this pilot does not establish a benefit from the
safety filter. Both arms completed 2/20 tasks, encountered 2/20 physical failures, and
left 16/20 unfinished. All 32 timeouts showed very little middle-block movement in the
last three seconds while the arm continued moving. This is a useful task-progress problem
to investigate, but the evidence does not yet isolate filter-induced continuation failure.

## Primary comparison

The run used seeds 0 through 19, alternating filtered/unfiltered order. Each pair had
exactly matching saved initial rigid-body poses/velocities and robot joint states.
Both arms used the same policy seed and model-query schedule. Checkpoints were frozen.

| Measurement | Unfiltered | Filtered |
| --- | ---: | ---: |
| Episodes | 20 | 20 |
| Safe successes | 2 (10%) | 2 (10%) |
| Physical failures | 2 (10%) | 2 (10%) |
| Timeouts at 15 s | 16 (80%) | 16 (80%) |
| End-effector boundary exits | 0 | 0 |
| Executed control steps | 4,981 | 5,239 |
| Applied interventions | 0 | 1,207 (23.0% of steps) |
| Successful completion times | 3.55 s, 2.35 s | 6.85 s, 1.35 s |

The descriptive 95% Wilson interval for each 2/20 success or failure count is about
2.8% to 30.1%. This small pilot cannot establish equivalence or a general safety rate.
The sequential renderer history also limits an independent-binomial interpretation.

Failures use the original physical task predicates, not learned predictions. All four
observed failures crossed the middle-block height threshold of 0.13 m. The predicate
can fire on a small loss of height; it does not mean the block fell to the floor.
Success also uses the original compound geometry predicate, including moving the middle
block more than 0.07 m from the base in XY while leaving the top block supported.

Filtered successes occurred at seeds 2 and 14, both with zero interventions. Unfiltered
successes occurred at seeds 1 and 17. Failures occurred at seeds 9 and 11 with filtering,
and seeds 10 and 18 without it. The differences between these individual trajectories
must not be interpreted as causal filter effects.

### Matching and repeatability

Paired reset images differed despite exactly matching saved physical tensors: mean absolute
pixel differences were 1.70 to 2.67 for the front camera and 2.02 to 2.94 for the wrist camera
on the 0 to 255 scale. Task proposals diverged before the first intervention. In particular,
filtered seeds 2 and 14 succeeded without an intervention while their unfiltered partners
timed out. These pairs therefore demonstrate the importance of rendered inputs and execution
history rather than filter benefit.

The first four episodes reproduced the integration pilot bitwise in saved camera arrays,
end-effector positions and quaternions when rerun in the same order. This supports
repeatability of that procedure on this machine. It does not validate arbitrary simulator
snapshot restoration, hidden contact state, or results on the original simulator version.

## What the unfinished episodes show

All 32 primary timeouts had less than 2 mm of middle-block motion over their final 60
control steps. The maximum was 1.66 mm without filtering and 0.14 mm with filtering.
The arm still moved through position extents of 3.9 to 17.9 cm without filtering and
2.8 to 15.6 cm with filtering. The median fraction of those final steps with the gripper
open was 80% without filtering and 100% with filtering. The recordings show unsuccessful
continuation after the middle block has stopped progressing, often without reacquiring it.

This pattern occurs in both arms. It is not simply a robot frozen by repeated safety
rejections. Ordinary task competence, recovery after losing the grasp, visual differences,
filter actions and the learned safety estimates all remain relevant explanations.

The value condition triggered on 1,098 filtered steps, compared with 154 uncertainty
triggers, with overlap; their union produced 1,207 interventions. Those counts describe
filter behavior, not ground-truth danger or unnecessary interventions. Counterfactual
unexecuted proposals have not been physically evaluated.

On filtered seed 11, the final proposal passed the filter (uncertainty 3.970 below 4.3,
next minimum safety Q 0.469 above 0.1), and the following physics step crossed the failure
height threshold. On seed 9, the filter intervened on the final step and failure still
occurred. These are two different cases to inspect; neither establishes whether a safe
recovery was still possible at that point.

Median per-episode one-step end-effector prediction error was 6.4 mm without filtering and
5.2 mm with filtering. This is a diagnostic of executed-action predictions, not a contact
or safety calibration test. The decoder does not directly predict block geometry, and
these errors cannot explain the failure mechanism by themselves.

## Controlled follow-ups

The follow-up plan was fixed after inspecting the first pilot and is exploratory. Each
variant used a fresh process or separately reset episode; every attempted trajectory is
counted. None changes model weights.

| Diagnostic | Starts / changes | Result |
| --- | --- | --- |
| Longer allowance | Seed 0, released filter, 30 s | Timeout after 600 steps |
| Alternative continuation | Seed 0, reseed future draws at step 300, keep physical/recurrent state | Timeout after 600 steps |
| Longer allowance / continuation pair | Seed 1, same two protocols | Both failed at step 99, before the scheduled change |
| Six-dimensional arm clipping | Seeds 0, 1, 3, 15 s | Three timeouts |
| Alternative initial stochastic trajectories | Seeds 0, 1, 3; RNG seeds offset by 1000 | Seeds 0 and 3 timed out; seed 1 failed at step 79 |

For seed 0, the extended-horizon and reseeded-continuation runs have bitwise identical
camera prefixes and exactly matching recorded numerical traces for all 300 pre-change
steps. The reseed preserves physics and both recurrent states; it changes subsequent
sampled task posteriors and imagined filter predictions together. This is one valid,
unsuccessful alternative from the same observed stalled prefix, not an exhaustive recovery
search. The seed-1 pair is ineligible for that test because failure occurred before 15 s.

A further useful check compares fresh-process filtered seed 0 with primary unfiltered
seed 0. Their initial images and physical states match exactly, and camera inputs, task
proposals and physical observations remain identical through the first filter decision
at zero-based step 28. By step 25, the gripper was already open and the middle block had
stopped at about 4.93 cm from the base, short of the 7 cm success component. It remained
there during both runs. Thus this inspected failure of task progress began before the
first intervention.

The action-processing variant did not rescue the selected three starts. Its outcomes do
not establish which convention is correct or calibrate the filter. The independent-start
probes also retain possible rendering-history differences; physical matching alone cannot
attribute a difference solely to the RNG change. Failed finite alternatives do not establish
that a state is irrecoverable.

## Controller and migration contract

The primary run retains the released six-coordinate safety postprocessing, including its
last-coordinate binarization. In this environment that sixth coordinate is arm rotation;
the actual task gripper is the seventh coordinate. The `arm_clip` follow-up changes this
one convention and is explicitly a diagnostic variant.

The safety snapshot is step 200000. Uncertainty threshold 4.3 comes from the saved config;
reachability threshold 0.1 and action denominator 1 come from the released function defaults.
Missing evaluation configuration prevents verifying these as the paper's evaluation settings.
Both recurrent dynamics tracks receive the normalized, clipped action actually executed.
The task posterior and hypothetical proposal dynamics are sampled; the filter posterior
uses its mode. The unfiltered arm evaluates the shadow filter for matched queries and
random draws. This is a documented migrated controller, not exact historical reproduction.

The encoder uses the two RGB views and end-effector pose. Object geometry and outcome flags
are evaluation evidence, not privileged task-policy inputs. Terminal images and state are
cloned before automatic reset. The underlying legacy Franka asset, block dimensions/masses,
20 Hz control rate, 100 Hz physics, action scaling and outcome predicates are retained.
The supported tensor-to-renderer transform path replaces the stale native Fabric path.

Upstream revision: `ab0fdf87f38edefa48b54a291ec5b7156c17b6e0`.
Private checkpoint mirror revision: `39bd3c84a1c9f3e84891eb09b1122d252c84e9c9`.
Resolved settings, dependency versions, checkpoint hashes and compatibility hashes are
preserved with each run. The original upstream checkout and research archive are unchanged.

## Cost, verification and evidence

The primary run executed 10,220 control steps and 51,100 physics steps in 1,041.7 s
(17.4 minutes), including startup, diagnostics and recording. Median per-episode inference
latency was 22.5 ms. Peak PyTorch allocated memory was 1.33 GiB; this excludes renderer,
PhysX and other GPU allocations. A live device snapshot during the run used about 4.1 GiB,
which is not a measured peak.

| Baseline-stage work | Autonomous episodes | Control steps | Physics steps |
| --- | ---: | ---: | ---: |
| Integration pilot | 4 | 971 | 4,855 |
| Primary comparison | 40 | 10,220 | 51,100 |
| All exploratory follow-ups | 10 | 2,977 | 14,885 |
| Autonomous total | 54 | 14,168 | 70,840 |
| Separate injected-fixture acceptance check | 0 | 60 | 300 |
| Total executed control/physics steps | 54 | 14,228 | 71,140 |

The completed autonomous runs report 1,602.3 s cumulatively; the final engineering check
reports another 20.1 s. These per-run timers exclude process shutdown and some external
analysis. The complete follow-up launcher, including subprocess exit and analysis, took
455.0 s; this overlaps its constituent run times and must not be added again.

Autonomous model queries were 14,168 encoder calls, 28,336 posterior updates, 14,168 task
actor calls, 14,168 uncertainty-ensemble evaluations, 17,006 imagined dynamics steps,
28,336 safety-actor calls, 28,336 safety-critic calls, 14,168 failure-head calls and 5,676
decoder calls. These include shadow-filter evaluation and diagnostics. Training updates: 0.
The initial integration attempt failed in preprocessing before its first environment step;
its elapsed time was not instrumented. Earlier setup and migration debugging are outside
this baseline-stage ledger and remain in their separate run folders.

All 54 completed autonomous episodes passed an artifact audit of source snapshot hashes,
action transformations, gripper preservation, intervention and outcome counts, terminal
state/frame counts and physics-step accounting. Three regression tests, Ruff checks and
the shell syntax check passed. The final simulator acceptance check passed all five
injected outcomes and verified that terminal camera/state buffers survive automatic reset.

Primary evidence: `runs/unisafe/baseline-released-20261015/`. Each episode contains proposed,
backup, selected and executed actions, physical states, true terminal observations, model
predictions and intervention reasons, camera arrays, a GIF and a summary. The `analysis/`
folder contains episode CSV data, paired comparisons and progress plots. Follow-ups are in
`runs/unisafe/followups-20261015/`; their plan and every completed attempt are retained.

Commands and interpretation rules are in the
[experiment guide](../../../experiments/README.md#unisafe-autonomous-baseline-and-diagnosis).
The [compact evidence record](UNISafe%20baseline%20pilot.results.json) preserves the outcome
counts, diagnostic results, provenance and interaction accounting in git. Large raw run
folders remain local and ignored; they are not included in the repository push.

## What remains a research decision

The baseline is sufficient to discuss the next investigation. It is not sufficient to
choose an evolutionary algorithm or claim a filter-specific contribution. Before an
adaptation study, clarify why basic extraction and reacquisition perform poorly on the
migrated stack, settle the safety-action convention, and establish a successful alternative
under controlled observations and the same safety constraints. Ordinary task-policy learning
with the same filter remains an essential later comparison. No training, evolutionary
search, ROSARL integration or method selection was performed here.

## Subsequent diagnosis of the shown video

A later [matched seed-1 investigation](UNISafe%20release%20and%20switching%20diagnosis.md)
isolates premature task-policy gripper opening before the first safety intervention. A
brief gripper hold completes that case under the same filter. The earlier results and
unsuccessful follow-ups above remain unchanged; the new probes have a separate interaction
ledger. This establishes avoidable loss of progress in one episode, not a filter-induced
stall or recovery from the already-stalled state.
