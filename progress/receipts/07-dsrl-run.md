# Task 7 DSRL repair run record

Started 25 October 2026. Status at 08:41 UTC: numerical checks passed and the first
live engineering episode is running. No development learning or evaluation has run.
All times below are UTC.

The user authorized a four-GPU follow-up and persistence through recoverable
failures. The protocol is `progress/07-dsrl-repair.md`; cumulative DSRL spend remains
capped at $15, including task 6. Stage A froze 167 source/input files, SHA256
`aee893a0154bc1074508560354a2c74bc435719737707ddd6088ae17c6f0429c`.
All 177 local tests passed with zero skips. Ruff and shell syntax passed.
The bootstrap has 169 files, SHA256
`9dea86641c3e5a6e56aae8280ffbd92c27f9d76aabddc145ff9bfb755864bc67`.

## Allocation

The preferred offer was unavailable on the initial check, before any create request.
One actual create selected instance 57632188 on physical machine
102318, driver 580.173.02, four RTX 4090s,
72.0 effective CPU cores and 386687 MB RAM.
The quoted rate including 200 GB disk is $1.850370370/hour.
Reservation began at 2026-10-25T07:45:06.606764+00:00 and is charged from that timestamp.
The independent stop deadline is 2026-10-25T13:03:29.994783+00:00.
Task 6's active charges and full transfer/stopped-disk allowance are retained in
the new financial ledger. No old instance or experimental queue is restarted.
Initial SSH refusal occurred while the provider was still starting the image.

## Preparation

Bootstrap hashes and GPU/driver identity verified after SSH became available.
Independent remote watchdog and preparation sessions started at 07:48:07; local
financial monitoring and continuous evidence copying are also live. The pinned
Jar download verified at 07:49:25, ManiGuard checkout completed at 07:51:01,
OpenPI environment at 07:52:47, and benchmark download at 07:54:36. Simulator
installation began at 07:54:36. At 07:57 the original preparation process and its
PyTorch installer were confirmed live. No preparation restart or replacement has
been made. The isolated learner imports JAX 0.5.0 and enumerates all four GPUs;
this is environment validation, not a learner update or policy inference.

While setup runs, the separate stage-B freeze preparer was completed. Its six
focused local checks pass: it binds the unchanged 24/48 schedule to source hashes,
numerical and live validation, the same rental and a current time projection.
Source drift, a different host, failed actor parity, changed numerical evidence
and an expired projection are rejected. These two new source/test files were not
part of the stage-A bootstrap and do not alter it; they are included in the stage-B
source inventory when that later freeze is created. Stage B is not yet admitted.

## Fixed-batch learner diagnosis during installation

The isolated learner environment and pinned author source were ready before the
simulator. The original managed `learner-default` operation ran from 08:03:17 to
08:04:08 on GPU 3, with five reference/bridge pairs and ten charged engineering
updates. All five parity comparisons failed; maximum difference was
0.00019738743867492303. Repeated direct author updates also differed, by up to
0.00014509260654449463, after restoring identical initial state and RNG each time.
The discrepancy is therefore not unique to the bridge path in this check.

The predeclared `learner-deterministic` operation ran from 08:05:25 to 08:06:18
with `--xla_gpu_deterministic_ops=true` set before JAX initialization. All five
reference/bridge pairs and all repeated-reference comparisons were exactly equal.
The original 1e-6 tolerance is unchanged. This passes the fixed-batch numerical
check; source parity on a live batch and actor execution still remain to be tested.
The fallback with autotuning disabled was not needed. Total task-7 engineering
update attempts are twenty, with zero training updates or replay episodes.

Both original process receipts, launch records, before/after checkpoints and the
shared learner ledger are preserved. The original fixed-batch checks will be
adopted by their managed operation identities, not repeated by the later queue.
The result JSON SHA256 values in `diagnostics/` are:

- `learner-default/result.json`: `ac3becf51e36548e042149821fbf96c06c16ca50a6ffdf18bed0515af71e5ea9`.
- `learner-deterministic/result.json`: `a96df02844941b6b743faea77e5230a5b1b12c1816f3974b42b5b3aa519943e8`.

## Validation repair and decoder launch

Before a new VLA query or simulator episode, inspection found that the reused
validator still selected task 6's job schedule. The explicit
[repair](07-dsrl-validation-repair.md) passes the verified current schedule through
both queue validation paths. All 186 local DSRL tests passed with zero skips.
Stage A v2 SHA256 is
`5e6275008ccbae872959a82f227d4823f91b6099b497c2aea962cd18cba64b7f`;
the nine-file repair archive SHA256 is
`0aa6cb3a90289b3dd7b710c92c55f8f6f7a341fe3a0364ba55db2ec6a9f29c8d`.
All 172 frozen source/input files verified on the host after deployment. No earlier
diagnostic result or source archive was rewritten.

The model-only numerical coordinator started in durable session `numerical07` at
08:12:41, using the stable isolated learner interpreter while the simulator
continued installing. Policy and probe subprocesses use the pinned OpenPI
interpreter. This stage does not admit simulator episodes. The learner diagnostic
operations above already exited and are adopted without duplicate updates.

The numerical stage completed at 08:16:04 with live integration admitted. All 216
probe queries and 12 startup queries returned. All same-process repeats were exact
on the eight observations on each of four GPU servers. All eight explicit native-draw
passthroughs on server 0 were also exact. Bounded candidates showed nonzero decoded
influence; gripper sensitivity remained one of eight inputs, on task 10. This is a
descriptive finding, not a completion result or the old two-task authority gate.

Eight of 24 cross-process comparisons met the original tolerance; the largest
absolute difference was 0.013146638870239258 in decoded action coordinates. All
converted prefix gripper commands agreed. These cross-process differences remain
reproducibility evidence and are not relabelled as same-process failures. The frozen
evaluation schedule rotates arms across GPU lanes. GPU 3's numerical policy process
was stopped as planned; servers 0 to 2 remain available for live engineering.

At numerical completion there were zero simulator starts, zero policy controls,
zero training updates and twenty engineering update calls. No learner diagnostic
was repeated by the coordinator. The fresh-path offline analyzer and its four
focused checks were separately frozen before any live episode in
`07-dsrl-analysis-source.json`; it keeps numerical integration distinct from the
six required live engineering records and the final learning comparison.

## Environment completion and live engineering

The original simulator installation completed at 08:28:16, robot assets at 08:28:21,
ManiGuard dependencies at 08:29:25 and final environment checks at 08:29:33. The
task-7 preparation script finished its explicit pin repairs and provenance checks
at 08:29:41. There was no setup restart or replacement. Final versions are NumPy
1.26.4, Torch 2.6.0+cu124, dm-control 1.0.49, MuJoCo 3.16.0,
googleapis-common-protos 1.75.5 and vhacdx 0.1.0. All four source revisions, asset
provenance and the actual passive evaluator patch hash verified. Ten environment
files are hashed in `runtime-environment.json` before the first simulator episode.

Before live engineering, inspection found that the full-pilot projection could
omit the final block's existing admission-time reservation. The
[timing repair](07-dsrl-timing-repair.md) corrects that calculation while retaining
the full schedule, timeouts and money cap. All 195 local tests pass with zero skips.
Stage A v3 SHA256 is
`ac5fd5f27d28d9348a2282565f1933139438fff0d4fb98a7d9fa0ff95cb91440`;
the ten-file repair archive SHA256 is
`c88bde202533232bf53c7a84f9604a0438f7c16fd6e8740096296f84926c31c2`.
All 178 declared source/input files verified after deployment. The completed
numerical evidence and live policy processes on lanes 0 to 2 remain unchanged.

Durable session `engineering07` started at 08:37:55 using the BEHAVIOR interpreter.
GPU 3's new `live-native` policy generation completed its three charged startup
queries. Native E3, task 11 and root seed 70003, started at 08:38:31.779203 with
the frozen 40-minute first-episode timeout. Its original worker was confirmed live
at 08:41. The remaining native lanes await its first policy response. At that
check, all 231 VLA requests were startup or diagnostic calls, and engineering
learner update attempts remained twenty.

The private dataset and newly prepared private task-7 model repository are
accessible. Their pre-upload state is recorded in
`07-dsrl-preservation-access.json`. No task-7 artifact upload has yet occurred;
continuous local evidence copying and both financial stop guards remain active.
