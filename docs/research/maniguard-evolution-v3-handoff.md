# ManiGuard v3 remote Codex handoff

Sachin requested this handoff on 30 October 2026 because the laptop must go offline.
He will authenticate Codex himself. The original execution goal remains authorized.
The experiment is unfinished. Continue it faithfully, then preserve and tear down.

## Start here

Read this file, `docs/research/maniguard-evolution-v3-authorized-goal.txt`,
the active runtime project's `AGENTS.md` and `firstPrinciples.md`,
`docs/research/maniguard-evolution-v3-experiment.md`,
`configs/maniguard_evolution/development-v3-protocol.json`, and
`progress/receipts/20261030-vast-maniguard-v3-preparation.json`.
The protocol's preparation-only fields record its earlier creation, not current
authorization. The explicit execution goal superseded preparation only.

GitHub contains source and handoff notes. It is public. Store all detailed episode
evidence, checkpoints, videos and private lifecycle records only in the existing
private Hugging Face dataset. Never publish credentials, process credential
environments, licence keys, model weights, simulator assets or unrelated studies.

## Exact rental and time limits

Only instance **58363639**, machine **141415**, label
`maniguard-v3-prepared-20261030T064632Z` is authorized. Verify all three against the
Vast API before further orchestration. Never operate on destroyed instance 58267560
or any other rental. Do not extend the deadline or rent another host.

- Absolute shutdown: **2026-10-30 16:46:32.937332 UTC**, 18:46:32 SAST.
- Finish experiment computation by **16:01:32.937332 UTC**, preserving 45 minutes.
- First scored episode requires seven hours remaining. If no valid episode has
  begun by **09:46:32.937332 UTC**, admission must close. Do not bypass the guard.
- Independent remote shutdown guard: PID **377**, directory
  `/workspace/setup-maniguard-v3-20261030`. Verify its live PID and command against
  `self-stop.pid` and `self-stop-ready.json`; deadline epoch **1793378792.937332**.
  It stops the instance and preserves paid disk, rather than destroying evidence.
- Laptop guard PID 1387486 and mirror PID 1483111 may disappear while the laptop
  is off. Neither is required for remote execution. Do not change the remote guard.
- Never inspect `/proc/*/environ`. Startup records, exact PID start ticks and
  command lines are the approved process identity checks.

## Actual execution state at handoff

Runtime project: **`/workspace/maniguard-evolution-v3-20261030`**.
Current run: **`run-v3-r1`**, service record **`service-v3-r1.json`**,
launched **08:13:35 UTC**, trainer PID **15853**, start ticks **49861303**.
Read `supervision/active-run.json` and inspect live process identity. These are
observations, not permission to restart. Never restart merely because SSH times out.

The first version, `run-v3`, stopped at 08:03:22 UTC because
`behavior-1k/datasets/omnigibson.key` was missing at scene load. Its trainer and
worker exited. It had **zero valid episodes, one failed start, two model queries
including one server preparation query, zero instrumented environment steps**.
Bootstrap physics before instrumentation is unknown. All original files remain.
The pinned upstream key installer succeeded at 08:09:41 UTC. The key is outside
the evidence project, mode 0600. Do not archive it or print it.

Recovery r1 has unchanged scientific settings and source, a new executable config
`configs/maniguard_evolution/development-v3-r1.json`, and an
`engineering_predecessor` pointing to the failed run. It charges previous starts
and queries. Server preparation is not charged twice. No separate robot screen ran.
Its runtime config SHA-256 is
`3c0161246d249d507de3525d6e11d06925af05c80e3929ef245a82616ad0c92f`.
Use saved r1 source/config freezes as authoritative; do not overwrite them.

Frozen policy server: PID **12980**, start ticks **49744748**, port **8000**.
Its exact launch record is `external-runtime/policy-server-process.json`, launcher
`external-runtime/serve-jar.sh`. One server preparation query completed in 37.5704s.
Do not prime again without charging an additional allowance.

## Interpreters, inputs and completed implementation

Experiment and software tests use
**`/workspace/miniforge3/envs/behavior/bin/python`**.
Frozen policy server uses
`/workspace/Safety-Dial/third_party/openpi/.venv/bin/python`.
Do not sync the repository's legacy root environment into either interpreter.
Actual Vulkan ICD: `/etc/vulkan/icd.d/nvidia_icd.json`.

Before the first attempt, **62 software tests passed in 15.52s**, without CMA
injection warnings, and Ruff **0.15.22** passed, on this rental with no simulator
episodes or model queries. Evidence is under `supervision/`. The tests exercise
all 108 synthetic scheduler jobs, selection and validation resume, native fallback,
deterministic ties, source binding, phase diagnostics, accounting and preservation.

Implemented: conservative mutation scales, exact native CMA anchor with correct
genotype/archive handling, corrected causal progress/contact fitness, nomination
of best evaluated non-anchor elite, native/elite/final-mean selection, strict-count
eligibility and native fallback, selection freezing, untouched validation, phase
diagnostics, checkpoint/resume, cumulative predecessor accounting, audits and
honest incomplete archival/sealing. Never revert to unconditional final-mean export.

Frozen protocol SHA-256:
`7fb435de56d367929d2e90f6b369f62450609785fe0bfe169b243b8262b7134b`.
Original public benchmark SHA-256:
`0991d5fdf5f65b4d70f113fa57bbff33190f4d65d5eb32683e87b44968ccae29`.
The generated instrumented evaluator is separate. The passive original monitor
must never veto actions. Preserve native, modified and executed commands separately.

## Frozen schedule and accounting

Run all 108 planned valid cases, without performance-based stopping:

- Training: 84 cases, six CMA-ES generations, population six including one native
  anchor, plus separate native references. Tasks 0 and 3; one policy seed per
  generation, 303100 through 303105. Optimizer seed 303021, initialization 303022.
- Selection: 12 cases, native, nominated non-anchor elite and final CMA phenotype
  mean; tasks 6 and 9, seeds 303200 and 303201. Task-major, rotate arm order.
  Adapter must exceed native strict safe-completion count. Otherwise use native.
  Elite nomination ties use earliest generation then lowest index. Eligible
  selection fitness ties prefer elite. Keep all three slots if vectors coincide.
- Validation: 12 cases, native and frozen selected arm; tasks 15 and 17, seeds
  303300 through 303302. Six cases per arm, alternating order. No selection,
  tuning, outcome retries or controller changes. Held-out tasks remain untouched.

Maximum 112 starts, including four failed engineering starts; 225120 instrumented
environment attempts; 28116 total model-query attempts, including at most four
server preparation queries. Every failed start, warmup, reset, discarded candidate,
query and elapsed time remains charged. Do not assume bootstrap physics is zero.

Repair only genuine technical issues within remaining time, retaining evidence
and versioning changed runs. A changed protocol requires a new prospective
protocol, not an outcome-driven edit. If limits prevent completion, report the
actual incomplete schedule, not a replacement study.

## Monitoring and preservation

Use the current service and each worker's `process.json` to check exact PIDs,
start ticks and commands. Read counters, then logs when necessary. Cold Isaac
startup can take several minutes. A stale counter or observation timeout alone
does not establish failure. Do not start a duplicate trainer.

HF dataset: **`Sachioster/maniguard-jar-replication-20261022`**, must remain private.
Campaign base prefix:
**`22_conservative_adapter_development/20261030T075407Z`**.
The launch snapshot was completely downloaded and byte-verified at revision
**`2ff13da95049278a7e7104aaa68b8b6a00869314`**, prefix `interim-launch`,
80 files, 652387 bytes. Later snapshots must use fresh names; never overwrite it.

`preserve_maniguard_evolution.py` stages immutable completed episodes and source
snapshots, uploads, pins a revision, downloads every file, and checks size/SHA-256.
Run its file orchestration with a suitable interpreter containing huggingface_hub.
On this host the behavior interpreter has the experiment dependencies. Source
the private account credential wrapper before service operations. Credentials stay
outside the runtime project and are never in configs, command arguments or archives.

Use snapshot and verification directories outside the live runtime project, e.g.
`/workspace/maniguard-v3-preservation/<unique-name>` and
`/workspace/maniguard-v3-verification/<unique-name>`. Keep receipts in the runtime
project's `progress/receipts/`. The existing `stage` function's interim selection
excludes top-level external-runtime/supervision. Preserve their immutable records
as well, and include the entire eligible project in the final terminal archive.
Do not seal or archive active, changing files as final evidence.

Continuously upload completed episode evidence while the laptop is offline. Save
complete checkpoints at stable generation boundaries; retain failed starts and all
versions. Include every imported helper in the source snapshot and hash checks.
Generation snapshots are under `generation-snapshots/`.

## Completion, honest failure and teardown

At completion or a genuine technical/time limit, stop new episodes, wait for exact
workers and trainer to retire, and close the policy server log safely. Audit the
active run with `analyze_maniguard_evolution.py`; audit the failed predecessor too.
For an incomplete terminal run use `--allow-incomplete`, retain missing cases and
invalid starts, and never mark the schedule complete.

Seal the full project with `seal_maniguard_evolution.py --run run-v3-r1` and its
`supervision/final-audit.json`. Use `--allow-incomplete` if required. Record the seal
SHA-256 independently. Verify an independent complete downloaded copy against
that exact seal. For final preservation use `--final` only for all 108 valid cases;
use `--terminal` for an honestly incomplete terminal result. Pin the private HF
revision and download/hash every file. A commit acknowledgment is insufficient.

Destroy only instance 58363639 after verified preservation. A process running on
the instance cannot reliably observe its own absence after destruction. Arrange
an external verifier, such as a tightly scoped GitHub Actions lifecycle job, or
an external authenticated controller, before self-teardown. Never claim confirmed
absence based only on a self-issued delete request. If no external verifier is
available, preserve and leave the independent deadline stop in force, explicitly
recording that destruction and absence confirmation remain outstanding.

After provider-confirmed absence, retire only its exact laptop guards when the
laptop is available. Save and privately verify a final lifecycle receipt after
deletion. Update GitHub implementation/status records without exposing private
episode data. Do not report the overall research goal complete until preservation,
destruction and provider absence confirmation have all been proved.

Report strict and benchmark safe completion, requested-sequence completion, raw
and counted violations, unfinished episodes, engagement and phase diagnostics
with denominators. Separate native anchors from mutations. If selection returns
native, call the selected validation arm a repeated native control and make no
evolved-improvement claim. One optimizer seed and six cases per validation arm
are development evidence only. The matched optimizer comparison remains future work.
