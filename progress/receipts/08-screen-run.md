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
