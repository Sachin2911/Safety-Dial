# DSRL06 CPU protocol implementation

25 October 2026. Implementation of the protocol logic in
[dsrl06_protocol.py](../../experiments/helpers/dsrl06_protocol.py), with behavioral
checks in [test_dsrl06_protocol.py](../../experiments/tests/test_dsrl06_protocol.py).
This build performs no simulator starts, neural inference, learner updates or rentals.

## Interfaces

- `EpisodeScorer.observe(step, ...)` consumes initial state 0 and every post-control
  state in order, with closure, support, grasp, native instantaneous goal, raw
  violation and optional counted-violation predicates. It latches first closure,
  handles closure/departure ties, preserves raw violations, and counts the native
  ten-control goal hold without counting the initial state.
- `potential(terminal=False)` implements the fixed priority table. `labels(success,
  valid=True, counted_violation=None)` applies the final benchmark contact-gated label
  if supplied. Invalid attempts receive no task verdict or reward. `state_dict()`
  and `from_state_dict()` support history-preserving resume.
- `reward_transition(phi_before, phi_after, controls, terminal=...,
  strict_safe_requested=...)` returns the actual duration, discount, bootstrap mask,
  terminal flag, raw/effective next potential, and separate base/shaping/total rewards.
  `check_return_identity(transitions)` rejects incorrect masks, discounts, reward
  values or query-boundary potential history and reconciles the complete return.
- `frozen_jobs()` returns `engineering_jobs`, `training_jobs` and `evaluation_jobs`.
  Jobs have `job_id`, `task`, `seed`, `lane`, `mode`, `stage` and frozen RNG metadata.
  Training jobs also have round/cycle and warm-up eligibility; evaluation has arm/block.
  E0 to E2 depend on E3's first response, and E4/E5 on the authority gate.
- `actor_query_seed(job_id, query_index)` is an unused CPU stream helper, not the
  runtime JAX actor sampler. The learner service derives its base key from the
  little-endian first four SHA256 bytes of
  `dsrl06-actor:<job actor_rng_seed>:<job_id>`, then calls JAX `fold_in(query_index)`.
  `decoder_vectors()` returns the fourteen names and float32 vectors whose byte hash
  matches the frozen probe receipt.
- `select_probe_snapshots(episodes)` applies the predeclared rule to E0 to E3 query
  records with `step`, `closed_once`, `raw_violation` and `order_failed` fields.
- `evaluate_authority(snapshots)` accepts eight dictionaries containing `task`,
  `snapshot_id`, decoded `reference` (16 by 8), two `native_replays` (2 by 16 by 8),
  and fourteen decoded `candidates` (14 by 16 by 8). It returns `valid`, `admit`,
  `reason`, and per-snapshot gripper and arm metrics. Candidate order is the frozen
  `decoder_vectors()` order. Nonfinite/incomplete responses, wrong task coverage,
  failed native identity or changed native converted commands invalidate admission.
- `ProtocolLedger(path)` exposes `reserve_start`, `finish_start`, `charge` and
  `summary`. Reservations precede work. Replacements require a recorded invalid
  attempt and `repair_receipt`. Charges accept zero and optional idempotent event IDs.
  Full SAC update calls use `phase="engineering"` or `phase="learning"`; `training`
  is an explicit alias for the latter. Limits are immutable and can be serialized
  using `dict(CAPS)`. Monetary guards are owned by the supervisor, not this ledger.

The append-only ledger flushes and fsyncs each reservation/charge. Resume replays
its sequence and SHA-256 chain; truncated or altered records fail rather than resetting
counts. Server query journals remain separate and are aggregated by the single owner.

## Seed limitation

The pinned evaluator derives its per-episode sampler seed as CRC32 of
`<base seed>:task_XXXX/base`, masks it to 31 bits, seeds client numpy and sends that
value to the native policy server. The generator preserves this exact rule. A known
task 5 example, task 1/base seed 390, produces episode seed 1523358492.

The evaluator does not expose a separate explicit simulator-reset seeding call.
Accordingly, every frozen job records `simulator_reset_seed=null`, rather than
claiming a reset seed was applied. Scene assets and the original setup remain pinned;
same-seed trials are not guaranteed identical trajectories.

## Validation

`./.venv/bin/python -m pytest -q experiments/tests/test_dsrl06_protocol.py`:
18 passed. `ruff check` and `ruff format --check` pass on the two owned Python files.
Tests include raw/count gate disagreement, equal-time closure/lift, irreversible
failure history, closure flicker, full-horizon and shortened-terminal return identities,
native threshold crossings within numerical tolerance, bounded-domain authority
coverage, exact frozen vector bytes, lane/seed balance, incomplete probe panels,
counter caps, repair-only replacements, idempotent resume and corrupted ledger refusal.

These are CPU implementation checks. They do not establish live decoder authority,
GPU throughput, robotics effectiveness or a safety bound.

## Offline episode admission

[dsrl06_validation.py](../../experiments/helpers/dsrl06_validation.py) adds
`build_episode(directory, output_path=None, *, recording_path=None, row_path=None,
sidecar_path=None, path_map=None)`. It returns a JSON-safe `validation` receipt, a
learner `descriptor` or null, and validated `snapshots`. The snapshots preserve the
query receipt and scorer history at each query, with permitted `pixels` uint8 RGB64
and `state` float32 robot8 arrays for authority and engineering checks. They contain
NumPy arrays and should not be dumped directly as JSON. Explicit absolute-prefix
`path_map` entries relocate preserved remote artifact references.

Admission first applies the existing complete tracking validation. It then replays
every native scorer state, including state zero, verifies raw and contact-gated flags,
checks the native goal hold and stopping step, and reconstructs all transition rewards,
duration discounts and terminal masks. Server artifacts must match frozen job/seed
metadata and executed proposals, retain finite float32 noise and uint32 policy RNG
keys, preserve inter-query RNG continuity, and match the exact image/state transform.
Zero, random and Gaussian modes are checked against their recorded numpy RNG stream;
actor, zero and random actions stay bounded. Gaussian warm-up remains unbounded.

Only tiled non-native rollouts produce learner NPZ files. Native full 16 by 32 draws
remain authority evidence and return a null descriptor. Learner arrays are `pixels`,
`states`, `noise`, `rewards`, `discounts`, `masks`, `durations`, `query_steps` and
`actor_versions`, with N-query inputs plus the final observation. Descriptor
`order_index` is the frozen T-job ordinal, or -1 outside training; the coordinator
assigns a contiguous admission index if invalid jobs are skipped. Existing different
learner files cannot be overwritten. This builder does not prove a saved JAX key's
split or regenerate neural actor samples; the pinned server and learner own those
operations and save their original keys and outputs.

Combined verification:
`./.venv/bin/python -m pytest -q experiments/tests/test_dsrl06_protocol.py
experiments/tests/test_dsrl06_validation.py`: 36 passed. Ruff checks pass. The 18 new
offline tests include a full 2,000-control timeout, a short successful terminal chunk,
closure flicker, raw/count disagreement, native untiled exclusion, bounded and
Gaussian modes, actual learner-loader compatibility, corrupt images/state/actions/
noise, missing scorer history, wrong final observation, broken RNG continuity,
explicit preserved-path remapping and existing-output refusal.
