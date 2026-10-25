# Task 8 execution-knob screen run record

Started 25 October 2026. The user authorized a fresh GPU study alongside DSRL.
Protocol: `progress/08-knob-screen.md`. All times below are UTC. The agent imposed
a separate $10 ceiling. No DSRL allocation, result or budget was changed.

## Preparation freeze and rental

The preparation-only freeze contains 44 source/input files, SHA256
`f302c5aa8fd4eb745d0c4423292d5c635a36c53794b44e8d88118d98ae064a67`.
Financial ceiling and single-allocation checks, Python compilation and shell syntax
passed before rental. The full simulator-source and job freeze is still required.
Two checks for offer 53960914 found it unavailable before any create was dispatched;
no reservation or charge resulted. A fresh selection made one actual create request.

Instance 57639938, physical machine 40438, one RTX 4090, driver 580.173.02,
64,133 MB allocated RAM and 16 effective CPU cores was allocated at 08:57:06.
The inclusive quote with 160 GB disk is $0.47111111111111115/hour. Estimated
150 GB inbound plus 40 GB outbound transfer is $0.546875, within the $2 reserve.
The independent financial stop deadline is 26 October 00:52:06.979834 UTC.
The workstation financial watchdog was running before the create request, PID 484082.
See `progress/rental-screen08-20261025.json` for the provider record.

The controller's seven local semantic checks passed before any simulator start:
native passthrough, h4 query cadence, causal scaling activation, prior-chunk smoothing,
gripper quiet-history timing, episode reset and financial isolation. The pinned
upstream evaluator SHA256 is
`0991d5fdf5f65b4d70f113fa57bbff33190f4d65d5eb32683e87b44968ccae29`;
the locally verified screen-hook patch SHA256 is
`8699b63185affd7fcbcb4b917dac6bd106109ec34db88f094aa41aa17d731dcb`.

## Installation and full pre-outcome freeze

The first transfer encountered SSH before the direct endpoint was ready. The same
instance was inspected and reused; no replacement was rented. At 09:00:44 the
verified bootstrap started durable `prepare08` and `watchdog08` sessions. The remote
watchdog PID is 936. The workstation watchdog was verified live in the host process
namespace; a sandbox process lookup cannot observe that PID and did not establish
a guard failure. Continuous evidence copying runs separately on the workstation.

The Jar checkpoint download finished at 09:03:59, ManiGuard checkout at 09:05:15,
OpenPI installation at 09:05:18 and benchmark download at 09:08:33. Simulator
installation then began. No simulator episodes or policy queries had run at 09:10.

The complete source and schedule freeze is `08-screen-plan.json`, SHA256
`7fa0eed9be344102158d79ceca6e27fb5925263ebb96deed266ca536250b2934`, containing
65 source/input files, five engineering jobs, sixty stage-A jobs and all sixty
potential stage-B jobs. Sixteen focused tests pass, including both recording
horizons, corrupted-command detection, rendering-profile rejection, disjoint task
panels and the completion/safety/activity selection rule. Ruff and shell syntax pass.
Passive reset-call and clipping-bound logging was added before this freeze; the
final evaluator-hook SHA256 is
`2cec04a35a61dbbaa2b65b0044d265b99ad8a2219166482751f8adeb2c19ac88`.

The runtime bundle SHA256 is
`d3fcde034551871f8a8bb073cd4e2b200902a3802c4d941b37866f9448a5ae80`.
Its transferred bytes matched. Durable session `run08` started at 09:10:21 and waits
for this live installation to complete, then runs engineering, stage A and the
predeclared conditional confirmation. Failures retain evidence and stop admission
for diagnosis. The financial watchdog remains independent of the queue.

## Runtime handover

The tested screen source and pre-outcome schedule were committed as `0198c2f`
before any episode. BEHAVIOR installation completed at 09:28:03, the pinned
long-finger robot asset at 09:28:08, and ManiGuard dependencies at 09:28:45.
The base preparation finished at 09:28:49. The queued launcher then verified the
frozen runtime and started policy session `serve08`; the independent watchdog
remained live. Disk use at handover was 76 GB of the 160 GB allocation.

The final setup marker was written at 09:28:54. The source check and exact evaluator
patch hash passed remotely. All three startup queries returned finite 16 by 8
float64 decoded chunks; the first took 16.4 seconds and the next two 100 and 117 ms.
The engineering queue started at 09:29:43.646950 and dispatched its first native
pool episode, task 1 seed 80000, at 09:29:43.932660 after one counted reseed query.
The simulator process is initializing its first scene. No outcome has yet validated.

At 09:36:20 the native engineering episode validated all 2,000 controls and 250
queries, with byte-identical command passthrough and zero modified controls. It
finished without goal completion or raw/counted violation. Its overview and wrist
event frames were inspected locally: the views show the jar, lid, gripper and goal
with the expected legacy rendering, followed by the arm moving away without carrying
the jar. This is engineering evidence on pool task 1, not a scored treatment result.

At 09:38:57 the h4 engineering episode on pool task 4 also validated all 2,000
controls, with exactly 500 queries and unchanged command values. It took 155.803
seconds, completed without goal success and recorded a raw/counted violation.
This checks the changed query cadence; it is not compared with the native episode
on a different task. The slow-arm engineering episode then launched on pool task 10.

All five engineering recordings subsequently validated. Each ran 2,000 policy
controls plus ten explicit warm-up controls. Native, slow, smooth and gripwait each
made 250 policy queries; h4 made 500. Offline replay matched every modified command.
Slow changed 864 controls, smoothing changed 1,743, and gripwait denied 22 proposed
closures. Native and h4 changed no command values at the action hook. All three
action-modifying mechanisms therefore acted in their engineering episodes. These
pool-task outcomes do not enter the sixty-episode development comparison or tune
the frozen presets.

Engineering completed at 09:45:09.474088 with five starts and no replacements.
Stage A then began on the frozen development panel. Its first native episode,
task 0 seed 81000, validated at 09:46:09 with 721 executed controls and 91 queries.
The paired h4 episode launched next. The sixty-episode stage remains in progress;
no arm selection or confirmation decision has been made. At 09:48 the active-time
estimate was about $0.40, with the separate $2 transfer/preservation reserve retained.

At 09:58:09, a remote SHA256 inventory of all five completed engineering episode
directories matched every local copy: 75 files, 88,867,375 bytes, no missing files
or size/hash mismatches. The inventory explicitly required engineering job IDs 0
through 4. Its receipt is `outputs/screen08/analysis/engineering-copy-verification.json`
and will accompany final preservation. All 65 frozen source/input hashes still
matched locally. This verifies the engineering copies only; final whole-run local
and private-repository verification remains required before rental-disk destruction.

A separate workstation completion monitor was added after launch. It changes no
frozen experiment files, schedules or selection rules. It requires a successful
terminal pipeline, all 65 engineering/screening jobs, every admitted confirmation
job, unchanged selection evidence and frozen sources, and no remaining experiment
process before taking an immutable remote snapshot. It checks the complete local
copy against the remote inventory before requesting a stop of this allocation.
Private upload verification and disk destruction remain separate steps. Six focused
tests passed, including incomplete panels, unfinished confirmation, altered evidence,
live processes, tampered copies and incorrect instance ownership. Ruff passed.
The additional utility source hashes are in `08-screen-preservation-tools.json`;
all 65 experiment source hashes remained unchanged. The monitor is PID 599262.
Its final transfer can reuse checksum-matched files from the continuous local mirror,
then independently verifies the full immutable snapshot. The original financial
watchdogs remain responsible for the hard cap.

The background copy begun at 10:19:33 exceeded its 180-second process window and
returned unsuccessfully at 10:22:38. The experiment continued normally. A separate
preservation-only copier, `mirror_screen08.py`, now permits 900 seconds per batch,
retains partial transfers and waits for an existing copy before starting another.
Only the original workstation copying supervisor was stopped. No simulator or policy
process was restarted, no episode was replaced, and all 65 frozen experiment hashes
still matched. The replacement copier is PID 635447; its I/O counters and additional
local validation reports confirmed data was moving. The first extended batch
completed successfully at 10:34:59 after 8 minutes 15.5 seconds. It transferred 62
regular files with 65,039,626 bytes of changed-file content; the source inventory
at that pass contained 456 regular files and 418,853,730 bytes. Continuous copying
continues, and whole-run terminal hash verification remains required. Its source is
included in the preservation-tools receipt, with lint and Python compilation passing.

## Complete screen and confirmation admission

Stage A completed at 11:37:29.420256: all sixty planned episodes validated on their
first attempts. Including engineering, there had been 65 starts and no replacements.
Workstation revalidation replayed all sixty recordings and matched their validation
hashes to the complete evidence set bound into the remote decision. A separate
calculation of the completion, violation and mechanism-activity criteria reproduced
that decision exactly. The audit is
`outputs/screen08/analysis/stage-a-selection-audit.json`.

Every arm has twelve scored episodes, and contact engagement was 12/12 in each arm.
Primary safe requested completion, raw/counted violations, and unfinished episodes
were, respectively:

- Native: 4/12, 5/12, 8/12.
- h4: 1/12, 5/12, 9/12.
- Slow: 6/12, 4/12, 5/12.
- Smooth: 5/12, 3/12, 7/12.
- Gripwait: 1/12, 3/12, 11/12.

Raw and counted violation counts agree in this stage. Benchmark safe-success counts
also equal primary safe requested completion here. Slow gained one primary completion
over native in each seed, 3/6 versus 2/6. Its mechanism acted in three additional-success
task/seed cells. Its paired completion difference was +16.7 percentage points; the
predeclared six-task cluster bootstrap interval was -16.7 to +41.7 points. This is a
small development screen with substantial uncertainty, not a confirmed improvement.

The frozen rule selected slow alone. Smoothing gained only one primary completion,
below the required two; h4 and gripwait each lost three. The decision was written at
11:37:45.783480 and is preserved as `08-screen-confirmation.json`. The conservative
confirmation-plus-preservation projection was 7,258.032 seconds, within 47,661.196
seconds remaining to the allocation deadline. The original 24-job native/slow
confirmation schedule at seeds 82000 and 82001 began at 11:37:45.899007. No parameter,
task panel, seed, outcome definition or selection rule changed after screening.

At 12:11:31, the terminal-copy utility was corrected to match the snapshot's `raw/`
directory to the direct file layout of the live mirror, then copy its receipt
separately. This permits reuse of already copied bytes. A real rsync fixture
confirmed zero network-file transfers for matching content and an independent
destination copy; the six preservation tests and Ruff passed. Only the workstation
completion monitor was restarted, as PID 807652. Experiment processes, financial
watchdogs and all 65 frozen experiment files remained unchanged.
