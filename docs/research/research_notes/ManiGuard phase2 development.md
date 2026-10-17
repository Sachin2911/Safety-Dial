# ManiGuard phase-two development log

Implementation began on 17 October 2026 following explicit authorization to implement
phase2.md fully and push regular updates. The full plan remains the scope. Conditional
main-study decisions will follow measured pilot evidence, not optimistic assumptions.

**Latest evidence checkpoint:** eight valid short engineering checks on Jar and Stack;
no full preparation episode, comparative pilot, learning result or reserved evaluation.
Clutter policy startup failed before physics and is being diagnosed.

At the initial implementation checkpoint, the repository was
0decedf8078b1a224a4672eee73738a19ed5683a on research/maniguard-fruits. The original
simulator queues were EXITED and the Jar policy server was RUNNING and idle. The GPU
was an RTX 3090 with 24 GB, with approximately 85 GB disk free. No phase-two physical
episode had run at that initial checkpoint.

Implementation is split into a pure recurrent model, continuous-weight search and recurrent
PPO, a six-family registry and asset preparation, and shared feature/runtime integration.
The original split manifest and study outcomes remain unchanged. A structured requirements
ledger tracks the full plan, including all-family evaluation, ablations and publication.

The shared feature encoder uses frozen OpenAI CLIP ViT-B/32 through pinned OpenCLIP
3.2.0. Its checkpoint and projection hashes, implementation dependencies, timing and input
contract are pinned and snapshotted for the physical engineering checks. This is a common representation, not a safety
predictor. Installing its four additive dependencies does not replace the simulator's
Torch, NumPy or Hugging Face packages.

## Integration checkpoint

The shared actor is 692 weights (64 input features, eight recurrent units, four three-way
heads). The actor, adapter, continuous-weight evolution/random search, recurrent PPO,
registry, completion contracts and protocol compiler are implemented. The complete
comparative pilot has not run. A combined integration/legacy regression run passed 169
tests before the new generated-source and baseline-routing checks were added; those
additional runtime checks also pass.

The official CLIP TorchScript archive required OpenCLIP's dedicated OpenAI loader with
QuickGELU and explicit fp32 conversion. An initial loader attempt failed before any
simulator work; its precise wall time was not recorded. The corrected measured encoder
check loaded in 4.749 seconds, performed one text and five two-image encodings in
0.482 seconds including synchronized inference/preprocessing/projection, and allocated
628,513,792 peak GPU bytes. This is a saved-image smoke check, not full simulator
throughput. All encoder parameters are frozen. Full source/checkpoint hashes are logged.

Integration also caught differing actor hash conventions between optimizer and runtime;
all now use the canonical schema-and-weight hash. A saved-NPZ to NumPy-trajectory to
Torch-PPO regression verifies this boundary. The early-query exception now requires
actual exhaustion of a short chunk. Causal feature and runtime tests cover episode reset,
privileged-label rejection, bounded delays, sixteen-proposal access and identity cadence.

The frozen preparation manifest assigns 12 short checks (at most 32 actions each) and
24 full training-only reference episodes across Jar, Stack and Clutter. Its action
ceiling is 58,784, with initialization separately counted. Integration and full preparation
time ceilings are 4 and 16 GPU hours respectively. All failed attempts consume allocation;
there are no automatic retries. Preparation measures throughput before setting the
separate pilot GPU-hour ceiling. Cabinet/Dusty outcomes remain reserved.

## Jar and Stack short engineering evidence

The [immutable partial report](../../../runs/maniguard/phase2/reports/preparation-partial-v2/report.json)
contains eight terminal, valid checks: original executor, shared identity, composed fixed
choices and stochastic shared-actor decoding for each of Jar and Stack. Each check used
32 control actions on its predeclared engineering scene. This consumes eight of the twelve
short-check slots and none of the twenty-four full preparation slots. The recorded totals
are **256 policy control actions, 40 policy queries and 2,224 physics ticks**, including
initialization physics. These short prefixes are implementation evidence; they cannot
establish full-task completion, safety performance or an evolutionary advantage.

The first Jar identity audit incorrectly rejected `safety_evaluated=false`. In the pinned
benchmark this field means `ever_contacted and not rollout_failed`; it is an engagement
gate, not a statement that the raw monitor failed. A completed idle episode may therefore
have a complete monitor and raw `ltl_violated=false`. The supplemental reader now retains
`benchmark_safety_evaluated` separately and derives `V` from raw `ltl_violated`, as specified.
Missing or failed monitoring still invalidates the outcome. The correction was checked
against the preserved idle trace and applied by re-auditing saved evidence. No physics
was rerun, no raw result or monitor label was changed, and the original audit verdicts
remain in receipt `audit_history`.

The [Jar identity audit](../../../runs/maniguard/phase2/preparation-v1/audits/short-jar-identity-comparison-v2.json)
found query steps 0, 8, 16 and 24 in both executors, and exact proposal-to-adapted-action
identity throughout each run. Their initial robot-state hashes match, but both initial
image hashes differ. The audit therefore records
`not_comparable_different_recorded_starts` and leaves cross-run trajectory comparison
unavailable. Matching nominal seeds does not establish an identical physical start.
The generated evaluator differs only at the two declared phase-two hooks, and the
current monitor source matches the pinned source. Same-frame runtime tests remain the
separate evidence for the identity transformation.

The partial report also evaluates the frozen initialization and mutation procedures
without physics on 48 recorded 64-feature rows from six Jar/Stack engineering episodes.
With hidden state reset for each input, 10 of 14 sampled actors and 14 of 16 diagnostic
mutations change deterministic choices across these inputs; both identity actors remain
constant. This verifies observable input dependence over this recorded feature range.
It does not measure learned improvement, adaptation fitness or safety, and no controller
settings were changed in response. The report keeps its input features and audit sources.

Setup diagnostics are counted separately. The successful encoder smoke contributes one
text call and five two-image calls, giving setup-inclusive totals of 53 image calls,
106 encoded images and seven text calls at the report checkpoint. The failed initial
encoder loader has a separate evidence record: it performed partial model-loading work
but failed before text/image inference, simulator interaction or policy queries. Its
individual elapsed time is unknown. The reservation accounting counts one GPU continuously
from Unix timestamp 1792272868 through report timestamp 1792275655.3285787, or
0.774258 allocated GPU hours, including setup and idle time once. Overlapping process
durations are not added to that interval. This is a dated partial cost snapshot.

The report has `training_ready=false`, `prior_accounting_ready=false` and no selected fixed
reference. Selection between `smooth075` and `cap006` waits for all 24 valid full training
reference episodes, using family-macro utility, raw violation rate, attempted query cost
and name in that order. The contextual threshold remains unchanged. Full-episode
throughput, comparative training, validation, transfer and the measured pilot budget
remain outstanding. Cabinet and Dusty outcomes remain unobserved.

## Clutter service preparation issue

The [Clutter service receipt](../../../runs/maniguard/phase2/setup/service-clutter-v1/service.json)
records a failed policy startup: the managed service did not produce the expected Clutter
handshake within its 180-second deadline. This occurred before a Clutter simulator
episode or physics action. The subsequent diagnosis and repair are recorded below. It is an infrastructure result,
not a Clutter task-policy outcome. The receipt preserves the requested configuration,
source hashes, service commands and probe failures. Its deadline completed shortly after
the partial-v2 report timestamp; subsequent reservation accounting must include that
remaining startup and diagnosis time without adding overlapping process durations twice.

## Clutter startup repair and failed identity initialization

The downloader stored Clutter's pinned repository/revision inside a structured `plan`,
while the policy server expected the earlier flat receipt. The new provenance loader
accepts both schemas and verifies the declared complete file inventory and bytes before
loading the model. Actual read-only verification passed for Jar, Stack and Clutter.
The [second startup receipt](../../../runs/maniguard/phase2/setup/service-clutter-v2/service.json)
records the matching Clutter checkpoint, per-episode random-state isolation and server
source hashes, ready after 20.50 seconds. Managed starts now retain a local transcript
while also streaming output to the instance logs.

The next predeclared original-executor short check completed and audited valid in 150.77
seconds, including the first Clutter scene initialization. The following identity check
terminated during encoder initialization with `Encoder runtime version mismatch: numpy`,
after 86 recorded initialization physics ticks but before any policy query or control
action. Its 55.49-second process receipt remains invalid, including cleanup of remaining
child processes. The queue stopped; the composed and stochastic Clutter slots remain
unclaimed. No automatic retry was performed.

The [third immutable partial report](../../../runs/maniguard/phase2/reports/preparation-partial-v3/report.json)
records nine valid and one invalid short attempt, known totals of 288 policy actions,
44 policy queries and 2,588 physics ticks. It conservatively marks failed-attempt counters
as lower bounds pending a separate accounting audit. One reserved GPU had consumed
1.11546 hours at that report timestamp, including setup and idle time once. No usable
prior accounting or training readiness was emitted. Ten of twelve short attempts and
zero of twenty-four full preparation attempts have been used.

A simulator-free reproduction shows that Isaac's vendored NumPy distribution metadata can
shadow the installed metadata after the real NumPy module has already loaded. The loaded
module and a global metadata lookup can therefore report different versions. The failed
historical log does not record the actual loaded module version; the attribution to that
shadowing is a supported diagnosis, not a directly recorded historical measurement.
The repair must validate the actual loaded module and its owning distribution while
preserving strict version pins. Any prospective reassignment of an unspent integration
slot requires a separate hashed amendment, retained failure costs and unchanged overall
ceilings. The original preparation manifest and failed evidence stay unchanged.

The launcher now holds a common execution lease and terminates complete child process
groups on failures/timeouts. A successful parent exit with surviving descendants remains
invalid. A separate persistence fix ensures that unreadable terminal ledger JSON cannot
prevent saving the child's exit status and measured elapsed time. Pure-code testing at
the preceding checkpoint passed 266 phase-two tests; subsequent focused service,
preparation and legacy/provenance tests pass. None of these tests establish task gains.

The encoder provenance repair now passes actual dependency verification with Isaac's
vendored NumPy metadata placed first, while checking the actual imported module and its
owning wheel metadata. The two projected-feature hashes remain unchanged. Future feature
records include loaded-module and distribution provenance separately from the frozen
encoder specification. No new simulator attempt has yet run with that repair.

The next combined checkpoint passes **389 tests**, covering all phase-two tests and the
relevant original accounting, adapter, policy/isolation, outcome and checkpoint-provenance
regressions. Ruff and whitespace checks pass. The regenerated pilot remains
`freeze_incomplete`, with 522 planned training/selection episodes and an exact total
preparation-plus-pilot ceiling of 1,328,984 policy actions. Preparation recovery, full
reference measurements and evidence-bound freeze reviews remain required before launch.

## Prospective recovery within the original preparation budget

[The preparation amendment](../../../configs/maniguard/phase2/preparation_amendment.json)
was written before either remaining short slot was claimed. Its content hash is
`b01acfd484e486c6a36b8a2295d4c81079a1ae12693c8fee47054db1efb13115`.
It keeps the failed `short-clutter-identity` attempt invalid and paid, retires the unspent
`short-clutter-composition` slot, and assigns that slot to
`short-clutter-identity-recovery` with exactly the original identity settings. The final
stochastic check is unchanged. This omits an independent fixed-composition check in
Clutter; composition retains its Jar/Stack physical checks and shared-interface tests.
No attempt, action or GPU-hour allowance is added, and all 24 full reference jobs stay
unchanged. The original manifest and historical receipts are preserved byte-for-byte.

The [terminal cost reconciliation](../../../runs/maniguard/phase2/setup/failed-clutter-identity-audit-v1/audit.json)
binds copied failure evidence and source files. The archived single-scene control flow
flushes the final ledger after the caught initialization failure, then performs passive
cleanup/summary writes and exits without another simulation step or policy query. Under
the study's instrumented-counter definition it supports 86 physics ticks, ten environment
steps, one encoder-loading attempt, and zero policy/control/encoder-inference/adapter
interactions. Historical imported-module/native descendant attestation is unavailable;
these limits remain written in the audit. This accounting reconciliation supplies no
valid task outcome and no optimizer fitness.

The derived allocation validates the immutable amendment and evidence before accepting
new receipts. Replacement claims record the amendment hash and a numeric start time after
its creation. Full preparation still requires the remaining short checks to pass, and
other invalid outcomes remain blocking. An attempt to change the failed receipt to valid,
reuse the retired slot, increase a ceiling or apply an earlier claim is rejected. The
recovery slots have not yet run at this checkpoint.

The amended [fourth partial report](../../../runs/maniguard/phase2/reports/preparation-partial-v4/report.json) preserves all ten paid receipts and reconciles the failed initialization without changing its invalid outcome. Seven original lower-bound warnings remain visible in the raw accounting; the separate source audit resolves them for total-cost accounting. The known totals remain 288 policy actions, 44 policy queries and 2,588 physics ticks. Reservation time is 1.42515 GPU-hours at the report timestamp. Integration and training readiness remain false pending the remaining checks and full preparation.

The launcher and reporter now require the original output directory and all historical receipt hashes, so a new directory cannot reset the allocation. Process cleanup allows a bounded natural-exit grace for transient Python helper processes, then marks persistent descendants invalid and cleans them up. Grace and cleanup time remain charged. Focused preparation/service checks pass 31 tests; protocol/report/audit checks pass 80 tests. The pilot draft includes the amendment and remains unfrozen.

The full regression checkpoint passes **419 tests**, with Ruff and whitespace checks passing. The two remaining short checks were then launched as a supervised serial queue from an archived source/config snapshot; no full preparation or optimization episode had started at launch.

## Short integration complete; full references started

Clutter identity recovery completed valid in 62.73 seconds, followed by the valid stochastic check in 62.31 seconds. Both used 32 policy actions. The [integration report](../../../runs/maniguard/phase2/reports/preparation-integration-v1/report.json) records all twelve paid attempts, eleven valid and the unchanged failed initialization. Integration coverage passes under the explicit amendment, with the omitted Clutter composition check retained in the limitations. Known totals including failed initialization are 352 policy actions, 52 queries and 3,144 physics ticks. The one-GPU reservation is 1.48357 hours at that report timestamp. No task-performance or optimizer conclusion follows from these bounded checks.

The eight original full Clutter preparation assignments have now been started as a supervised serial queue, with the current sources and command archived before launch. They compare identity, smoothing0.75, target-displacement cap0.06 and the fixed contextual requery rule on two frozen training cases. Full preparation must also include the eight Jar and eight Stack cases before reference selection and measured pilot costing. No pilot training, validation or reserved evaluation has started. The recovery implementation checkpoint was pushed as `45463f1`.


## Clutter full preparation complete; evaluation tooling checkpoint

All eight original full Clutter reference attempts completed with valid process, source,
feature, outcome and cost audits. Identity, smoothing0.75 and the fixed contextual rule
completed both training cases safely. The 0.06 displacement cap completed the first case
safely, but reached the 1,100-action horizon on the second without completing and with a
physical violation. These two-case counts are descriptive preparation evidence. They do
not establish an improvement or identical-start causal comparison, and the violating
episode is not a safe timeout. No reference is selected before the remaining sixteen
Jar/Stack preparation cases complete.

The [Clutter-complete report](../../../runs/maniguard/phase2/reports/preparation-clutter-complete-v1/report.json)
preserves all twenty paid attempts: eleven valid short checks, one invalid initialization
with separately reconciled costs, and eight valid full references. Its content hash is
`cf3493f4c60b8f30252277ffd415a0c1ad04e8deddd8b799f7a205571cad8140`.
Reconciled totals are 5,214 policy actions, 664 queries and 33,004 physics ticks. Including
setup, it records 1,287 image-encoder calls, 2,574 encoded images and 17 text calls. Continuous
one-GPU reservation is 1.88806 hours at the snapshot. Training readiness remains false;
Jar/Stack throughput still has only startup-heavy short measurements, so the current
pilot cost extrapolation is provisional and does not authorize launching the pilot.

At the confirmed simulator-idle boundary, the unused AAC-joint selector was corrected
to match the pinned benchmark gripper conversion: a command opens only above 0.01. Zero
and small positive commands close. Entropy and movement detection now use that executed
state; the returned sample-zero action values and native Cartesian goldens are unchanged.
No physical AAC comparison preceded the correction. The separate supervised Cabinet/Dusty
asset download was then launched from archived public-source plans, without policy or
simulator initialization and without behavioral outcomes.

The [evaluation-tooling note](ManiGuard%20phase2%20evaluation%20tooling.md) documents the new
AAC proposal sampler, pilot selection report/model export, main protocol planner and main
analysis. The planner binds controller-specific proposal and encoder resources; the
protocol-aware analyzer checks the complete declared case/run grid and recorded costs.
The pilot CLI exports all six selected run artifacts from a completed coordinator state.
These modules have synthetic validation only: main live authorization and AAC execution
remain unwired, and no pilot actors or optimizer results exist. The combined focused
checkpoint passes 213 tests; Ruff passes for all new and corrected Python files.

Cabinet and Dusty asset preparation subsequently completed with supervisor exit code 0.
Independent verification checked all 669 files (12,693,624,682 bytes) against archived
plans, upstream digests and complete receipts. No inference or task outcomes were used.
The [download completion receipt](../../../runs/maniguard/phase2/setup/reserved-assets-v1/progress.json)
has file hash `8f348990b30190a59d26eaeeae847b14a01b50437a4ac825ee0629dc675544c5`.
The 133.38-second download remains part of the continuous reservation, not an additional
overlapping GPU charge.

The original sixteen Jar/Stack full preparation assignments have now been launched as
a supervised serial queue after download verification and an idle-process check. Forty-seven
source/config files and both operational files were archived and verified before launch.
The queue switches the pinned policy service, holds the shared execution lease, and has
no automatic retry or new allocation. Its [launch record](../../../runs/maniguard/phase2/setup/preparation-full-rest-v1/launch-check.json)
records the exact remaining job IDs. No optimization or reserved evaluation is authorized
by this preparation launch.
