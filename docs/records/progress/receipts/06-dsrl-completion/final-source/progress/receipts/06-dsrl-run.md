# Task 6 DSRL run record

Status: closed, 25 October 2026. Four native engineering episodes completed;
numerical gates did not admit learning. Private preservation passed and instance
57575920 is destroyed. See the [result](06-dsrl-results.md) and
[completion proofs](06-dsrl-completion/README.md). All times below are UTC.

## Initial preparation entry

This entry was written during environment preparation, before any simulator episode.
Later sections record the source repairs and final accounting.

Sachin authorized implementation of `progress/06-dsrl-plan.md`, Vast rental and SSH
execution, with persistence through technical failures inside the frozen protocol
and hard $15 total cap. The source freeze is `06-dsrl-stage-a.json`, SHA256
`e52f4602da0fe4c07e2eeee3db38d140dbd2b5c975513e88ad36e0f1a89c7b8d`.
All 137 workstation tests passed with zero skips. CPU author-update parity and
64 stochastic actor cases matched exactly. Two workstation gradient attempts are
already charged; no workstation VLA query or simulator control was used.

## Allocation

- Instance: 57575920. Physical machine: 139769. Fresh contemporary controls only.
- Four RTX 4090 GPUs, driver 580.173.02, Xeon Platinum 8352V, 57.6 allocated CPU
  cores, 309,350 MB RAM, 200 GB disk.
- Quoted compute plus disk: $1.903703704/hour. Incoming/outgoing transfer:
  $0.009114583/$0.013020833 per GB. The 100 GB incoming plus 8 GB outgoing estimate
  is $1.015625, below the separate $2 reserve.
- Conservative charging begins at 2026-10-24T22:55:22.998435+00:00, before create.
- Nominal aggregate deadline: 2026-10-25T04:55:22.998435+00:00. Independent stop
  requests begin five minutes earlier, at 04:50:22.998435+00:00.
- Working-host engineering deadline: 2026-10-25T00:25:22.998435+00:00.

The initially requested offer disappeared before any create request was issued.
One actual create request selected the preferred source-compatible host from a
fresh quote. No duplicate allocation or replacement was created. Provider compute
rate matches the quote. The complete mutable rental/accounting records are
`progress/rental-dsrl06-20261025.json` and ignored
`outputs/dsrl06/control/finance.json`.

## Preparation

The provider first reported loading, then running with SSH exposed. Initial SSH
refusals occurred during loading, before any remote experiment was launched. SSH
succeeded at 23:00:07. The transferred 121-file bootstrap archive has SHA256
`74d83e995b213a8aff674ebf2dfc3f024d207e29c1036b26c39177280aa16877`.
Source hashes and four-GPU driver identity passed on the host. The remote watchdog
confirmed its own identity before setup. Independent `watchdog06` and `prepare06`
sessions started at 23:00:34. Continuous local evidence copying is running.

Read-only access to the established private dataset was verified before engineering.
Its observed pre-run revision was `f470abd948a13434c84a4cebc8fb9c82ceea84b4`.
No experiment evidence has been uploaded yet.

## Source repair before engineering

While the simulator installed, a source audit identified that default GPU actor
matrix arithmetic can use TF32, while CPU inference uses full float32. At 23:17:52,
before any VLA query or simulator start, the narrowly scoped
[precision repair](06-dsrl-precision-repair.md) was applied and all replacement
source hashes verified on the host. Only actor sampling requests highest precision;
the author SAC update, weights, RNG, noise bounds and 1e-5 parity gate are retained.

The operative freeze is now `06-dsrl-stage-a-v2.json`, SHA256
`5f28bc11edc52234a87331ba09102bb52cd83953c89f0e118ab9437435a879a6`.
The original freeze and bootstrap archive remain evidence. The explicit old/new
hash repair record is `06-dsrl-source-repair-01.json`; its 10-file source archive
has SHA256 `d13f7cd82d8638bf829224d63f3066088724ed608ccdea34958c735d3d7b3aa2`.
The full repaired-source suite passed 139 tests with zero skips. Two paired CPU
checks matched exactly and consumed four additional actor forwards, bringing
workstation controller calls to 134. Gradient attempts remain two, and workstation
VLA queries and simulator controls remain zero. GPU-versus-CPU parity is still a
runtime engineering gate.

## Installation observations

The pinned Jar download verified in 77 seconds. ManiGuard source checkout completed
at 23:03:33; the 792-file benchmark download verified at 23:06:37 with no missing
or mismatched files. OpenPI completed at 23:03:58. The isolated learner imported
JAX 0.5.0 and Flax 0.10.2 and enumerated all four GPUs.

BEHAVIOR installation started at 23:06:37. By the 23:16 check, cuRobo's CUDA wheel
had built and Isaac Sim was installing; by 23:20, robot and simulation assets were
downloading. The Joylo stage temporarily installed NumPy 2.2.6 and newer MuJoCo /
dm-control packages, producing dependency warnings. The existing pinned dependency
step restores NumPy below 2 and the task 6 preparation script reapplies the known
dm-control 1.0.49 / MuJoCo 3.16.0 repairs. Their final installed versions still need
verification before engineering. These warnings did not halt the installer.

Preparation completed at 23:30:07. Final versions verified were NumPy 1.26.4,
Torch 2.6.0+cu124, dm-control 1.0.49, MuJoCo 3.16.0,
googleapis-common-protos 1.75.5 and vhacdx 0.1.0. All four full source revisions
matched their pins. The installed passive evaluator patch hash was
`a670d9783d70de05b11fcf916d631bf8de11fe12fa5515da55fd6fcb77558423`.
Checkpoint, benchmark and robot provenance had zero missing files and mismatches.
A 10-file runtime environment hash receipt was written before any VLA query.

## Engineering

The frozen v2 queue launched at approximately 23:31. The GPU 3 policy server's
three startup queries all returned and were confirmed by the client by
23:32:13.509. The first inference compiled in approximately 22.1 seconds; the next
two took approximately 0.11 seconds each. These are startup queries, not task
completion evidence, and all three are charged to the query ledger.

E3, diagnostic-pool task 11 with policy seed 60003, started its first simulator
attempt at 23:32:13.614, with the frozen 40-minute wall timeout. E0 to E2 remain
unadmitted until its first task-policy response. Engineering has no result yet.

### Native enum correction during engineering

The first E3 policy response returned at 23:37:57.013547, admitting the other native
lanes. The scorer then exposed an enum validity bug: OmniGibson's known FALSE is -1,
UNKNOWN is 0 and TRUE is 1. The implementation had rejected negative values.
New admissions were paused at 23:40:21 by stopping only the coordinator. E0, E1
and E3 had already been admitted and continued their original attempts. All time
and interactions remained charged; no evaluator was paused or restarted.

Read-only source inspection proved that the already recorded native
`goal_detail['held']` uses the exact same `is_grasping == TRUE` predicate. The
[grasp repair](06-dsrl-grasp-repair.md) preserves raw enum values and original
validity flags. Derived validation accepts the former FALSE=-1 flags only after
checking every N+1 state for known enum, valid AP/goal checks and exact agreement
between the raw enum, saved native held column and journalled native predicate.
UNKNOWN, missing data and unexplained invalid flags remain rejected.

All 155 tests passed. The operative source freeze became `06-dsrl-stage-a-v3.json`,
SHA256 `f2a9fe14e249e5464098bbbbb82bc3a8fab7c3cfbe0098965a0ac08060947dec`.
The old/new hash receipt is `06-dsrl-source-repair-02.json`; source archive SHA256
is `6a39c6b883aafc4756d172bd937c87d659567e5fb4982e3d81fb4ed8d23577af`.
The coordinator alone was replaced at 23:46:19, adopting the original live workers.
E2 was admitted at 23:46:20.002084. There were no duplicate or replacement starts.

E3 finished its full 2,000 controls and 250 task queries at 23:44:27.845275. Its
2,001-state reconstruction passed with zero UNKNOWN values and return identity
error 0. Its recorded outcome remained unsafe and unfinished, with no closure or
grasp. The [separate completed audit](06-dsrl-grasp-audit-E3.md) preserves the raw
file hashes; original recording and journal bytes were unchanged. This correction
does not authorize a repeat of that unsuccessful outcome. Authority selection
still follows the original nonviolating snapshot rule.

### Numerical checks and stop

All four native episodes completed and validated, without replacements. The
original GPU profile returned HTTP 409 at 23:50:26; its error body was not retained
by the original HTTP worker. It attempted the two source-parity calls and never
entered the 128-update timing loop. The complete attempted interval overlapped
all three collectors. A separate captured diagnostic, with zero timed updates,
returned a 3.1837262213230133e-6 source-state difference and raised cumulative
engineering calls to six, including the original two CPU calls.

Source v4 corrected reference static metadata and retained exact HTTP errors and
request fingerprints. It explicitly recorded the
[measurement-association deviation](06-dsrl-measurement-amendment.md); the original
combined operation remained failed. Its source SHA256 is
`3ef7306af4645aefb9d7d1863758481adf0d11dc97ede90f44d73d8d5cca58fc`.
All 166 local tests passed. The checkpointed GPU retry charged calls seven/eight
and again returned HTTP 409, with a 2.1345913410186768e-6 difference at
`actor.opt_state[0].mu.network.Dense_0.bias`. No tolerance was relaxed, no timed
updates ran and no CPU/GPU actor comparison was reached.

The original frozen decoder probe was then completed independently of that failed
learner check. All 128 calls returned and were confirmed. At 00:19:36, native replay
identity failed and the gate refused admission. Four of sixteen factual comparisons
passed; twelve from other GPU lanes failed. Descriptive candidate sensitivity was
one of eight inputs on one task. All repeat pairs were exact and all converted
gripper commands matched the factual recordings. See the
[offline decoder audit](06-dsrl-decoder-audit.md) for the distinction between the
invalid formal gate and descriptive sensitivity. No additional probe, deterministic
kernel retry, E4/E5, training or evaluation was run after this stop.

The saved checkpoint contains all eight charged engineering calls. Final usage is
four starts, zero replacements, 8,000 policy controls, 40 warm-up controls and
1,140 VLA attempts/returns/client confirmations (1,000 task, 12 startup, 128 probe).
Workstation actor forwards total 134; remote actor inference requests are zero.
Internal scene/reset physics remains unmeasured. The completed local raw inventory
contains 1,411 files; the consistent SQLite backup contains 1,140 query rows and
has SHA256 `ec8e174e1b127545e6fcaf17df96d28ad6847739c0a1d65e27bac2bbc5a6b1aa`.

Compute stop was verified at 00:27:37.763250. Conservative active time, including
provisioning and preservation, is 1.537434671 hours, or $2.926820 at the quote.
The last metered transfer estimate is $0.538760. Their sum is $3.465580 before
retained-disk storage. The $15 hard cap remains in force. Private pinned revisions,
provider disk destruction and final billing are appended in the
[completion receipts](06-dsrl-completion/README.md). Executed source hashes were
verified before the subsequent result-status and PDF-log documentation updates;
original executed versions remain in the immutable bootstrap and repair archives.
