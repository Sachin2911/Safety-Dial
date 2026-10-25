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
