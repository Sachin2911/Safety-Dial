# Task 7 DSRL repair run record

Started 25 October 2026. Status at 10:18 UTC: all numerical and live engineering
checks passed. Eighteen training episodes are valid, 60,000 development updates
are complete, and the next three episodes are running with trained actor
version 4. Evaluation has not started.
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

E3 returned its first policy response before 08:44:21. The original E0, E1 and E2
workers were then admitted at 08:44:20.949, 08:44:21.064 and 08:44:21.094,
respectively. All four original worker identities were confirmed live at 08:46;
E3 had 41 returned policy queries. No episode had completed or been replaced at
that check. This establishes live native execution, not a learned-controller result.

E3 finished at 08:54:25.518 with a valid 2,000-control recording and 250 policy
queries. It was unfinished, with a raw and counted violation, contact but no grasp.
The original E0 to E2 workers continued without replacement. The learner started
on GPU 3 after those lanes had returned their first policy query. At 09:00:21 its
live-image/state check matched the author update exactly, and its 128-update
profile completed with a median of 0.015460644 seconds per update. The timed block
took 101.741 seconds including compilation; synthetic zero actions and targets
were used for profiling. All 150 learner update attempts to this point are
engineering charges, including the twenty prior diagnostics. Training replay
remains empty. These measurements do not establish a trained-controller benefit.

All four native engineering recordings were valid by 09:03:33. They yielded 0/4
safe requested completions, 3/4 raw and counted violations, 4/4 unfinished episodes
and 0/4 episodes with grasp. CPU/GPU actor agreement then passed on eight inputs:
maximum difference 2.384185791015625e-7 against the frozen 1e-5 tolerances, with
median warm CPU inference of 0.004996796 seconds. The bounded-random E4 and
initialized-actor E5 workers started at 09:03:50.027 and 09:03:50.127. Their live
execution and cached duration remain the last engineering admission checks.

## Engineering admission and learning launch

E4 finished at 09:10:11.288 and E5 at 09:10:55.941. Both recordings were valid;
both were unfinished with raw and counted violations and no grasp. The six
engineering episodes therefore produced 0/6 safe requested completions, 5/6
raw and counted violations, and 6/6 unfinished episodes. They remain engineering
evidence and do not enter training replay or the development comparison.

Stage A admitted the full learning comparison at 09:10:59.833. Its slowest cached
episode took 425.814 seconds. The projection is 12,196.915 seconds including the
existing preservation and final-start reserves, against 13,922.913 seconds remaining
at the stage-B freeze. At engineering completion, all six starts were valid without
replacement: 12,000 policy controls, 60 explicit warm-up controls, 1,731 VLA queries,
266 actor calls and 150 engineering updates. Development updates remained zero.

Stage B was frozen at 09:11:27.082 with SHA256
`3bac6f6e66cd51e726231b4208f896b9ba7dd4c2bf530467e350a98bfc2816be`.
It retains all 24 training and 48 evaluation jobs and the cumulative $15 cap. The
freeze was copied back locally and its hash verified. Durable session `pilot07`
launched the learning queue. At 09:12:08, original workers T00, T01 and T02 were
running the first warm-up collection round. No trained-controller improvement is
claimed. The rental's 178 frozen source/input files verified unchanged before this
launch; concurrent task-8 documentation edits on the workstation are separate from
the source snapshot running this experiment.

By 09:19:15, T00 to T02 had completed valid recordings and the first collection
round had appended 750 transitions to the fresh training replay. Checkpoint
`round00` was saved. T03 to T05 were running the second Gaussian warm-up round.
The learner remained at zero development gradient updates, with the 150 engineering
calls separately accounted and no engineering episodes in replay.

At 09:27:09, all six Gaussian warm-up episodes had valid recordings, replay held
1,500 transitions and `warmup_complete` was true. Checkpoint `round01` was saved.
The original T06, T07 and T08 workers were running actor-version-0 collection;
the frozen protocol updates after that three-episode collection barrier. No
training start had failed or been replaced. The first three warm-up recordings
had zero completions, three raw violations and exact return-identity reconciliation.
These untrained outcomes do not measure a learned-controller benefit.

The first actor-driven round, T06 to T08, completed valid recordings and appended
750 transitions at 09:34:21.688, bringing replay to 2,250. Its scheduled 15,000
updates began at 09:34:22.878 and completed at 09:40:08.317. The batch took
345.438 seconds including compilation and ended with finite metrics. Actor
version 1 was published at 09:40:09.338; checkpoint `round02` was saved at
09:40:14.666 with manifest SHA256
`87a585de4a2335551874191aeca2de5f6be8915225b98047c8695854eda45b84`.

A read-only weight audit at 09:43:51 verified every file hash in the warm-up and
trained checkpoints. Of 1,077,622 actor parameter scalars across 22 leaves,
1,075,701 changed from `round01`; the largest absolute change was 0.219228432.
All trained actor parameters were finite and exactly matched the published actor
parameters. The audit performed no forward passes or gradient calls and is saved
as `outputs/dsrl07/control/round02-weight-audit.json`. The actor ledger confirms
T09, T10 and T11 requests use version 1. At 09:44:33 those original workers were
roughly halfway through their episodes, without a failed or replaced start. This
verifies a functioning learning and deployment loop; safe-completion efficacy
still awaits the frozen final comparison.

By 09:48:39, T09 to T11 were valid and replay held 3,000 transitions. The next
15,000-update batch completed in 251.517 seconds; the episode-12 checkpoint was
written at 09:52:00.840 with actor version 2 and no pending transitions. Its
manifest SHA256 is
`e63f84386469f77b772bad10cc4af1272d3521ac3949b1ed9f533e50e810bf9c`.
All checkpoint files were copied locally and verified against that manifest.
At 09:52:42, the original T12, T13 and T14 workers were running the next round.

Across the first twelve training episodes there were 0/12 strict safe completions,
10/12 raw and counted violations, 12/12 unfinished episodes, 11/12 with contact
and 0/12 with grasp. These combine warm-up and changing training actors and are
not the final evaluation comparison. No start had failed or been replaced.
The original local budget watchdog and sync processes were confirmed live at
09:54; the most recent evidence copy completed successfully at 09:54:16.657.

At 09:59:27, T12 to T14 had valid recordings and replay held 3,750 transitions.
Their 15,000-update batch finished at 10:03:17.8. By 10:04:27, the coordinator had
published actor version 3, saved checkpoint `round04`, and started the original
T15, T16 and T17 workers. Training updates totalled 45,000, with the 150 engineering
updates separate and no interrupted training round, failed start or replacement.
The provider independently confirmed instance 57632188 running on machine 102318
at the frozen hourly rate. The full 24-training and 48-evaluation schedule remains
unchanged.

The continuous copy returned rsync exit 24 at 10:03:50.877; its following pass
returned exit 1 after about 182 seconds at 10:07:37.889. The original sync
supervisor remained live, and the next pass succeeded at 10:09:23.599. The complete
`round04` checkpoint then verified locally against manifest SHA256
`1722a280d3411bf0ea02d79dd298a6aca8e906bcdfa97db877a2a266b53c772d`.
A stale `round04.partial` copy also remained in the continuous mirror. Final
preservation must therefore use a fresh exact inventory after worker quiescence,
with the frozen SQLite backup, rather than treating the live mirror as final.
This operational requirement is recorded in
`outputs/dsrl07/control/preservation-readiness.json`. Scientific workers were not
restarted or interrupted by these copy failures.

T15 to T17 had valid recordings by 10:12:26, bringing replay to 4,500 transitions.
Their update batch finished at 10:14:56.5. At 10:15:31 actor version 4 and checkpoint
`round05` were published, all 60,000 development updates were complete, and the
original T18, T19 and T20 workers were running. Across the first eighteen training
episodes there were 0/18 strict safe completions, 11/18 raw and counted violations,
18/18 unfinished episodes, 14/18 with contact and 0/18 with grasp. These are training
outcomes under changing actors, not a final performance comparison.
