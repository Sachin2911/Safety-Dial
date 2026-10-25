# DSRL06 execution coordinator build

25 October 2026. Implementation only. No simulator start, VLA query, learner update
or rental was performed by these checks.

[run_dsrl06.py](../../experiments/scripts/run_dsrl06.py) owns one host's queue. Run
with the BEHAVIOR interpreter, using `--stage engineering|pilot --plan <freeze.json>
--rental progress/rental-dsrl06-20261025.json`. The default engineering freeze is
`progress/receipts/06-dsrl-stage-a.json`. It verifies launch authorization, exact
source hashes, frozen jobs/caps, the $15 cap and the current rental identifier.

The remote root is `/workspace/Safety-Dial`. The behavior, OpenPI and learner
interpreters are `/opt/miniforge3/envs/behavior/bin/python`,
`third_party/openpi/.venv/bin/python` and `third_party/dsrl-venv/bin/python`.
The frozen checkpoint is `third_party/checkpoints/jar/7400`. Preparation must apply
the separately hash-guarded evaluator patch before this coordinator runs.

## Engineering flow

E3 starts first. Its first returned policy query admits E0 to E2. E3's completed
evaluator and server release GPU 3. The learner is not initialized until all three
other native episodes have returned their first policy queries, avoiding overlap
with the four lanes' initial compilation. A fixed batch of recorded E3 RGB64 and
robot8 observations supplies 128 profile updates and two charged source-parity
updates. The two prior workstation engineering updates are imported exactly once.
A complete engineering learner/replay/RNG checkpoint is saved before reset/shutdown.

The published engineering actor is exported to the CPU-only service on port 9010.
Up to eight E3 inputs are sampled with identical actor version, job seed and query
index on GPU and CPU. Finite bounded outputs must agree at absolute/relative
tolerance 1e-5. CPU HTTP latency is recorded and included in the projection.
No extra VLA query is used for this comparison.

The eight authority snapshots follow the frozen history rule. Their query artifacts
and original image/state/prompt/noise/output bytes are hashed before the 128-query
probe. E4/E5 run only after a valid admitting authority gate. An insufficient panel
or valid negative gate is a preserved negative engineering outcome. Invalid episode
artifacts are infrastructure failures requiring diagnosis rather than task verdicts.

`stage-a-summary.json` records the gate, full learner checkpoint, source/CPU parity,
actual profile/rollout overlap, resources and the complete pilot projection. Learning
admission requires source parity at 1e-6, CPU parity, the entire profile interval
overlapping all three rollout workers, and the projected stage B plus preservation
fitting the remaining rental deadline. Resource samples preserve GPU memory/utilization
and host memory during execution. Stage A remains bounded by working-host creation
plus 90 minutes, independently of the aggregate rental guard.

## Pilot flow

Stage B requires a new freeze with `stage: "B"` (or `"pilot"`), the exact
`stage_a_summary_sha256`, and `actor_evaluation_mode: "cpu_published_actor"`, retaining
the original scientific controls. It discards engineering weights/replay through a
fresh training initialization. Eight rounds contain three frozen jobs each. The
first six episodes use tiled Gaussian warm-up, with one additional six-task cycle
only when the transition threshold is not met. Twelve insufficient warm-up episodes
stop the pilot. Warm-up receives no retroactive gradient credit.

Each later round retains its original actor version/mode across coordinator resume,
inserts the three validated episodes in frozen order, trains at 20 updates per new
transition, then publishes. Full checkpoints are saved after each round, including
the required episode-12 and episode-24 checkpoints. Evaluation uses only the final
published actor. Its actor-only export is restored on CPU port 9009 after the full
learner exits; GPU 3 then loads the fourth frozen VLA. Twelve four-arm blocks use the
frozen rotating lane assignment. Evaluation episodes never enter replay.

## Recovery, accounting and preservation

Every child has a durable command intention, wrapper/actual-child PID identity and
exit receipt. Linux boot identity/start ticks prevent adopting a reused PID. A
second coordinator holds an exclusive queue lock. Resume follows original live
workers and never duplicates a simulator start because an observation timed out.
An unknown launch or unresolved response remains pending. Completed source learner
events or immutable checkpoints can reconcile a lost HTTP response without issuing
another mutation.

A stopped learner requires an explicit `--repair-receipt` naming `resume_checkpoint`.
The checkpoint must belong to this run and contain every admitted replay round;
the source loader separately rejects uncheckpointed update attempts. Ambiguous
partial learning is preserved for diagnosis. Simulator replacements require the
same receipt's concrete `repair`, `job_id` and next `attempt`, an invalid recorded
attempt, and the remaining global replacement allowance. That lane's server is
restarted so the unchanged episode CRC seed resets its native sampler. Valid poor
outcomes are never repeated.

Before a start is charged, the effective allocation deadline must have at least
60 minutes remaining for a lane's first episode, or 45 minutes for later episodes.
These reserve the 40/25-minute episode limit plus 20-minute preservation window.
The same check precedes a repaired server's restart. The fixed stage/deadline guards
remain active during all waits. Policy-control and warm-up headroom also covers all
active attempts' remaining full horizons. The provider watchdog independently owns
financial stopping.

The protocol ledger imports every server SQLite reservation, including failed or
unconfirmed queries, attempted `before_action` and `warmup_attempt` events, and source
learner update reservations. Incremental source offsets retain partial last lines
until complete and reject truncation. Imported workstation updates are distinguished
from on-host updates and never charged twice. Actor requests are reported separately;
unexposed scene/reset internal physics is recorded as unknown.

Technical source repairs may use a new immutable freeze with `supersedes_sha256` and
`amendment.type: "engineering_repair"`. An explicit repair receipt must name old/new
freeze hashes and a concrete repair. Every field except source hashes, creation time
and amendment metadata must remain equal. Original freeze bytes remain preserved.
This does not change episode replacement rules.

On final stop, all managed process groups are terminated and verified dead before
`outputs/dsrl06/quiescence.json` records `instance_id`, `workers_stopped: true` and
all recorded wrapper/actual-child PIDs. Coordinator and financial watchdog PIDs are
excluded. Pending work and a positive engineering receipt awaiting stage B do not
emit a false quiescence marker. Query SQLite backup and private artifact upload are
owned by the separate preservation command.

## Checks

`./.venv/bin/python -m pytest -q experiments/tests/test_dsrl06_queue.py`: 17 passed.
The combined queue/protocol/validation suite passes 53 tests. Ruff check/format pass.
Tests exercise real short CPU child processes for live adoption and termination,
unknown launch intentions, actual failed-attempt/query/update accounting, prior-update
import, partial journal rows, repaired sampler restart, no poor-outcome repeats,
technical amendment constraints, stage/preservation clocks, 45/60-minute boundaries,
all eight training rounds and 48 evaluation lane assignments, insufficient warm-up,
fixed mode/version across mid-round resume, source-completed HTTP recovery, and the
engineering cold-response barrier and actual profile overlap.

Live GPU memory, decoder authority, cross-device parity, throughput and robotics
effectiveness remain engineering measurements, not results of these CPU tests.
