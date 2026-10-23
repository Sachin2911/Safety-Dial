# Todo 2 review (after commit `4ffa97a`, 23 October 2026)

Reviewer: the workstation session. Host: RunPod pod `f1o65q3iygv372` (A40, driver
595.91.07), 11:36 to 13:58 UTC, stopped by the agent with `scripts/todo2/stop_pod.py` and
terminated from the account at 15:24 UTC after verification. RunPod spend $1.49.

## Verification

Dataset revision `11b5a0f9`: 561 files; every file of the four runs of record is present
and unchanged against `9300cebf` (sizes and LFS hashes compared, 0 missing, 0 changed);
`analysis/02/` holds 75 files (41 frame sheets, 32 pair strips plus `pairs.json`, the
104-line timelines), none zero-size, 76 MB. Nothing was written outside `analysis/02/`.
Repository: receipts, scripts, `progress/panels.json` and the leverage note are on
`main`; HEAD equalled `origin/main` at the stop.

## Accepted

Items 1 to 6, 8 and 9. The panels were written first and match the rule in the
checklist. The timelines needed one story the checklist had not foreseen,
`closed_stall` (lid closed, jar never lifted, no violation), and the agent added it with
the rule written in the script rather than forcing those episodes into a wrong label;
that is the right call. The frame sheets, task notes and pair strips were looked at and
described in plain sentences, as asked. The sensitivity rule as written in the checklist
was too strict (no task came out `systematic` under the fine story labels), and the
receipt says so and reports the coarse-outcome reading alongside, which is the one to
use: 8 tasks systematic (7, 18, 21, 24, 25 always violate; 4, 19 always close and
retreat; 12 always succeeds), 18 knife-edge.

## Not achieved: item 7

The pinned stack installed on the pod (98 minutes, two deviations repaired, freezes
matching instance C on every pinned component), the policy server served, but Isaac Sim
4.5's RTX renderer segfaulted at startup four times under this pod's driver 595.91.07
(open kernel module, CUDA 13.2), inside the scene-database plugin while sizing its
acceleration-structure buffer, before any scene loaded. The gdb backtrace and the Kit log
comparison with instance C are in `02-host.md`. A driver cannot be changed inside a pod
and the checklist forbade trying, so the agent correctly stopped, validated the recorder
against mock objects only, and deferred the real episode. Constraint for the project:
simulator episodes run on hosts with driver 580.x (the Vast machine 54514 pattern);
RunPod is fine for desk work. If RunPod is used again for simulation, create the pod with
`allowed_cuda_versions` at or below 12.8 so that an older driver is more likely.

The recorder (`scripts/todo2/record_rollout.py`, an observer on the evaluator's callback
hook, with `check_recording.py` to prove scoring is unchanged) is therefore the first item
on the next Vast rental: one episode, about an hour including install.

## What the description says

- The approach is the same every time. Same-seed repeats make first contact within a
  median of 12 steps, and the pictures stay alike through the approach and usually
  through the lid push. The first proposition to differ is closure, median step 358; the
  pictures separate later, median 543, always after contact.
- Outcomes are decided in two places after contact: whether the lid push gets the lid
  over, and the regrasp descent after it (does the gripper leave the lid, descend, and
  land on the body rather than the rim). 7 of 10 same-seed pairs split in the regrasp.
- Idle is as common as crashing: 36 of 104 episodes end without violation and without
  lifting (18 stalls, 18 with the lid closed and the policy parked). 19 episodes tip the
  jar after closing it on the table, 9 tip it before closing, 5 fail during the carry.
- Three of the policy's "successes" are it giving up on the lid and grasping the open
  jar (tasks 10, 17, 23).

## On the ranked shortlist

The ranking (phase-gated execution dial, retry on stall, sampling-noise scale) follows
from the evidence, and the sizing is right: 200 episodes per condition, seeds interleaved
on one host. Two reservations before any of it is frozen into a stage 2 protocol:

- Experiment 1 gates on the monitor's `jar_closed` proposition, which is simulator
  state, not something the policy or a deployed wrapper can see. As a first lever test it
  is acceptable and should be said plainly; a deployable dial needs a proxy (gripper
  state, time since contact, or a learned detector), and that gap is part of the result.
- Experiments 1 and 3 share a native arm and should run as one package (about 1,000
  episodes, $44 at pod rates, less on a Vast 3090 pair). Experiment 2 needs a retreat
  primitive and a stall detector first; it is the most interesting for the Safety Dial
  story and the most engineering, so it comes after the package, not with it.

Next: a Vast 580.x rental that validates the recorder on one episode, then the stage 2
protocol frozen from that shortlist. The Odyssey has a task 2 entry.
