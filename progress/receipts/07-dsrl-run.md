# Task 7 DSRL repair run record

Started 25 October 2026. Status at 07:59 UTC: environment installation is running.
No new model, learner-update or simulator calls have run. All times below are UTC.

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
