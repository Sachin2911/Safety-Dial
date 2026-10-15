# UNISafe premature release and filter switching

**15 October 2026.** Diagnosis of the filtered seed-1 episode shown in the video, using
frozen pretrained components on the migrated RTX 3090 / Isaac Sim 6.2 / Isaac Lab 3 stack.

The task policy opens its gripper before completing extraction. The block loses its grasp,
settles near the stack and stops progressing. Rapid safety-filter switching starts afterward
and makes the arm commands more abrupt. The yellow block is not repeatedly falling off the
table. The moving wrist camera and looping presentation can give that impression.

A controlled replay that keeps the proposed gripper closed at the release point completes
this episode at **2.80 seconds**, with the original safety filter enabled. This establishes
an avoidable loss of task progress in this specific case. It does not establish that the
filter caused the initial stall, explain the learned policy's reason for opening, or
validate a general correction.

[Side-by-side recorded comparison](../../../runs/unisafe/causal-probes-20261015/video/gripper_comparison.mp4),
[annotated original recording](../../../runs/unisafe/causal-probes-20261015/video/explained_rollout.mp4),
and [timeline](../../../runs/unisafe/causal-probes-20261015/video/timeline.png) are local,
ignored artifacts. The comparison uses the fixed front camera. After the successful run
terminates, its terminal frame is held and explicitly labelled. No later stability was
simulated. These files are not included in the git push.

## What happens in the original episode

All step indices are zero based; times below indicate when an action is issued. The trace's
`simulation_time_s` field instead records the end of that 0.05-second control step.

| Time | Recorded event |
| --- | --- |
| 2.15 s, step 43 | The policy begins sustained gripper opening. The middle block is only 3.29 cm from the base in XY and 17.06 cm high. The fingers are still around the block. |
| 2.50 s, step 50 | The first safety replacement occurs. The released block has settled about 6.09 cm from the base, short of the greater-than-7 cm success component. |
| Remainder to 15 s | The block stays effectively stationary while the arm moves and filtering switches repeatedly. The episode times out without a physical-failure predicate. |

The middle block's position extents after step 50 are below one micrometre on each axis.
Its settled height is about 13.29 cm, above the task's 13 cm failure threshold. This is
recorded simulator geometry, separate from the model's safety estimates.

The filter replaces only six arm coordinates and preserves the seventh, the gripper
proposal. Thus the opening was a task-policy action, not a gripper command from the backup
controller. Negative gripper commands close the fingers; positive commands open them.

The original episode contains 63 intervention steps and 100 transitions between filter 0
and 1. There are 50 intervention bursts, 39 lasting just one control step. Here **1 means
that the backup arm action was applied**, not that a physical failure occurred. The value
condition triggers on 60 steps and uncertainty on three, producing the 63-step union.

The controller checks its thresholds independently at 20 Hz, with no hysteresis or minimum
dwell time. When a trigger changes, it abruptly switches between task and backup arm
commands. The median normalized arm-command change is 2.19 at switches versus 0.78 without
a switch. These are command-vector differences, not measured physical jerk.

## Separating policy behavior, sampling and switching

Offline replay of all 300 recorded inputs reproduces the original task actions, executed
actions, uncertainty, both safety values and intervention decisions with **maximum numerical
error 0.0**. Additional probes preserve the replay's random-number stream.

At each of 16 selected frames, 256 hypothetical next states are sampled while holding the
filter belief and proposed action fixed. At step 49, 148/256 samples trigger the value
condition; at step 50, 241/256 do. Sampling can therefore change the binary decision even
with identical inputs. These are conditional model outputs, not 256 physical trials or
empirical probabilities of a real failure.

A separate resampling of the task posterior at step 43 produces opening proposals in
219/256 samples (85.5%); the modal posterior also opens. The observed release is therefore
consistent with the policy's response to this belief, rather than depending on a single
rare latent draw. This does not reveal why the learned policy chooses that response.

Four physical diagnostic variants replay the two preceding seed-0 episodes in their
original order, then modify the seed-1 filtered target. The saved initial physical tensors,
recorded physical observations and camera inputs match the reference through the first
changed executed action. All preceding actions match exactly. Both warmup episodes match
in full for every variant. This validates the observed prefixes on this host; it is not
an arbitrary simulator snapshot/restore claim.

| Target variant | First changed action | Outcome | Intervention steps | Filter switches |
| --- | ---: | --- | ---: | ---: |
| Recorded reference | None | Timeout, 15 s | 63 | 100 |
| Suppress safety replacements from step 50, retain shadow queries | 50 | Timeout, 15 s | 0 | 0 |
| Clip all six backup arm coordinates without final-coordinate binarization | 50 | Timeout, 15 s | 77 | 130 |
| Use modal hypothetical next state, consume the original random draw | 49 | Timeout, 15 s | 63 | 106 |
| Hold proposed gripper closed from step 43 | 43 | **Success, 2.80 s** | 0 | 0 |

Bypassing replacements reduces the median normalized arm-command change from 0.95 to 0.63,
and median finite-difference end-effector acceleration from 3.04 to 1.46 m/s². The block
still remains stuck. The safety bypass is a simulator diagnostic control, not a proposed
safe controller.

The release's six-coordinate backup postprocessor binarizes its last coordinate as though
it were a gripper, although that coordinate is actually arm rotation. Removing this quirk
does not rescue this episode. Substituting modal next-state prediction also does not remove
switching. Prediction sampling contributes decision variability, but it is not the sole
cause: changing observations/proposals and abrupt threshold switching remain relevant.
Both of these variants change the controller and are not pooled with the frozen baseline.

The gripper probe instead changes the task proposal before assessing it with the unchanged
released filter. It plans to hold closed for steps 43 through 62, but success terminates
the run after 13 overridden commands, or 0.65 seconds. No safety replacement is needed.
The final middle/base XY distance is 7.014 cm, and the original compound success predicate
is true: the top is supported, the extracted middle block is no longer stacked, and its
height remains above the failure threshold. There is no observed failure or boundary exit.
This prevents loss of grasp before the stall; it does not demonstrate recovery after the
block has already been released.

## Scope, accounting and verification

These are post-hoc, hand-selected and hand-timed probes of one episode. No weights are
trained and the default controller remains the released baseline. The earlier 20-seed
results are unchanged. All four target probes have no physical failure, but this small,
dependent set does not estimate a general safety rate.

All 16 new physical episodes are counted: eight warmups, four target probes and four
trailing unfiltered episodes. The trailing episodes follow altered target histories and
are not matched causal comparators. One succeeds, one physically fails and two time out.
Across all 16 episodes, the totals are two successes, one physical failure and 13 timeouts.

| New work | Control steps | Physics steps | Measured elapsed time |
| --- | ---: | ---: | ---: |
| All four physical jobs, including warmups and trailing episodes | 4,099 | 20,495 | 495.89 s summed run timers |
| Offline replay and conditional sampling | 0 | 0 | 62.57 s |
| Video rendering and artifact analysis | 0 | 0 | Not instrumented |

The physical launcher takes 509.47 s including subprocess shutdown and analysis. It overlaps
the run timers and must not be added to them. Training updates: zero. These interactions
are additional to the baseline-stage ledger, not a replacement for its earlier counts.

Physical-run model queries are 4,099 encoder, 8,198 posterior, 4,099 task-actor, 4,099
uncertainty-ensemble, 4,920 imagined-dynamics, 8,198 safety-actor, 8,198 safety-critic,
4,099 failure-head and 1,642 decoder evaluations. Offline replay adds 300, 600, 300, 300,
360, 600, 600, 300 and 120 respectively. Its additional conditional probes use 4,096
imagined-dynamics, 4,112 safety-actor, 8,224 safety-critic and 4,112 task-actor evaluations,
counted per sample, not per batched kernel call.

All 16 episodes pass an artifact audit covering implementation hashes, action transforms,
probe scope, raw task-proposal preservation, filter gripper preservation, intervention
counts, frame counts, terminal observations, physical predicates recomputed from geometry,
and physics-step accounting. Three regression tests, Ruff and `git diff --check` pass.
The comparison MP4 contains 301 frames at 20 fps; its 15.05-second file duration includes
the reset frame. The displayed original simulation lasts 15 seconds.

Upstream source remains `ab0fdf87f38edefa48b54a291ec5b7156c17b6e0`, and the private checkpoint
mirror revision is `39bd3c84a1c9f3e84891eb09b1122d252c84e9c9`. Snapshot 200000 and thresholds
4.3 for uncertainty and 0.1 for safety value are unchanged. The
[compact evidence record](UNISafe%20release%20and%20switching%20diagnosis.results.json)
contains individual outcomes, matched-prefix checks, model provenance, query accounting,
conditional sampling results and raw-artifact SHA-256 hashes. Reproduction commands are in
[the experiment guide](../../../experiments/README.md#unisafe-release-and-switching-probes).

The next investigation should establish why extraction ends with premature opening on this
stack and how often the same mechanism occurs. Compare controlled observations and task
behavior before choosing an adaptation target. A more general grasp-maintenance rule would
need validation on untouched episodes under the same filter. Altering filter thresholds or
adding hysteresis is a separate controller change that needs its own physical evaluation.
The current result supports fixing basic task behavior before an evolutionary study; it
does not yet establish a filter-induced stall or select an evolutionary method.
