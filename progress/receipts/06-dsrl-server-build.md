# Task 6 frozen-policy server and recorder build

25 October 2026. Engineering implementation only. No simulator steps or real neural
inference were used for these local checks. GPU replay and decoder-authority gates
remain required by `progress/06-dsrl-plan.md`.

`experiments/helpers/dsrl06_server.py` wraps the pinned JAX `Policy` without replacing
its sampler. Native requests call the original `Policy.infer(observation)` with no
noise argument. The wrapper records the input and output PRNG keys, then reconstructs
the full 16 by 32 Gaussian from the original sample key after inference. The artifact
labels this as same-key reconstruction, rather than a captured sampler tensor. It
does not consume or update the policy RNG. Episode reseeding uses the evaluator's
original `episode_seed`; the native evaluator derives this from its task seed. This
does not assert an explicit simulator reset seed.

Zero, bounded uniform random, Gaussian warm-up and actor modes pass explicit tiled
32-dimensional noise through the pinned `Policy.infer(..., noise=...)` interface.
The actor receives only the overview image resized from 256 by 256 to 64 by 64 with
Pillow RGB BILINEAR and the eight-dimensional state. There is no crop or flip. Its
localhost HTTP request also identifies the job, query, actor version and private actor
RNG seed. The server rejects a returned actor version mismatch and out-of-bounds
bounded noise. Probe-only routes support both original-native replay from a saved
policy key and explicit replay from a saved full Gaussian. Probe calls restore the
policy RNG and episode seed afterward.

Every reserved request counts toward the same SQLite limit of 21,000, including
startup requests, inference exceptions and responses the client does not receive.
The persisted limit cannot change when the server resumes. The ledger separates
model return, completed artifact and client confirmation. Each row includes the job,
query index, purpose, mode, actor version, packet identifier and attempt identifier.
A live ledger must be preserved with SQLite's backup API, including its WAL state.

Lossless query artifacts are `<record-root>/<job-id>/query_<global-id:08d>.npz`.
They contain both original 256 by 256 camera images, original state and prompt,
actor pixels and float32 state, 16 by 32 noise, sample and policy PRNG keys, original
decoded actions and the float32 16 by 8 chunk consumed by the evaluator. JSON `meta`
contains the request identity, simulator control step, provenance and status.

`experiments/helpers/dsrl06_recording.py` extends the existing passive recorder.
It confirms exact saved-query to proposed-chunk alignment and records the native
raw monitor result, contact-gated counted violation, instantaneous native goal check,
grasp predicate and Jar propositions at every state. `scorer_state` journal entries
cover state zero and every returned policy control. Warm-up attempted and returned
steps are journalled before recorder start, so a warm-up failure remains accountable.
The final raw cameras, state, prompt and actor observation are saved separately as
`<scene>.final_observation.npz`. Proposed, applied and executed action records retain
the existing recorder's distinctions.

`experiments/scripts/patch_dsrl06.py` checks the exact pinned evaluator and goal
checker hashes before adding optional observer callbacks. The native per-control
goal checker runs once; its existing returned result is copied. One additional
scorer-only initial goal read supplies state zero. The native consecutive-success
counter is untouched. The goal checker's `held` detail means its instantaneous
grasp predicate, not the ten-control success hold.

Source audit:

- OpenPI revision `215abfb217dbac7d5f1273282331b9b1866c0479`,
  `src/openpi/policies/policy.py`, SHA256
  `a2ac5236e23b11d0f5a840d4a8fef148d89dba921d5452ec5dfcd5a6ffa46f20`.
  Its original infer method splits `_rng` once and omits explicit noise when none is
  supplied. `src/openpi/models/pi0.py`, SHA256
  `24b32d7c6ed5e409459afa8ca5af96a4dae09c2dcdf48e828907e411f4f12342`,
  draws its native Gaussian from that sample key.
- ManiGuard revision `6bf4c61a1f168cc46e02eb22c41bcb51815e9726`,
  `maniguard/eval/benchmark.py`, SHA256
  `0991d5fdf5f65b4d70f113fa57bbff33190f4d65d5eb32683e87b44968ccae29`.
- The same ManiGuard revision's `maniguard/eval/goal_checker.py`, SHA256
  `ace47ce46cc0a540c759b0f5d0c89f050bdebbe83877e6052e770bf09acc74b0`.
  `GoalRegionChecker.check`, lines 71 to 104, reads the already-resolved target and
  region and returns holding/intersection predicates. It does not step physics,
  render, advance a counter or alter the benchmark's success hold.

Validation used `.venv/bin/python`, without syncing the legacy local environment
into a ManiGuard environment:

- `pytest -q experiments/tests/test_dsrl06_server.py`: 13 passed in 1.12 seconds.
  All tests ran locally, including pinned-source checks. The original pinned
  `Policy.infer` method was executed with real CPU JAX and a synthetic non-neural
  sampler for three sequential native requests. Outputs and key advancement matched
  exactly. Other cases cover explicit routing, episode reseeding, native replay,
  actor information boundaries, concurrent global reservations, failures and lost
  responses, warm-up accounting, pure initial goal recording and exact reversible
  evaluator patching.
- Ruff check and formatting passed for the five new Python files at line length 100.

Entry point: `experiments/scripts/serve_dsrl06.py`. Runtime imports of OpenPI/JAX
remain lazy and belong in the intended pinned OpenPI interpreter. The loaded model
must retain latent shape 16 by 32 and decoded chunk shape 16 by 8.

The additional model-only entry point `experiments/scripts/probe_dsrl06.py` provides
`startup` and `authority` subcommands. Startup accepts one to three charged synthetic
native queries per frozen server-start identity. Authority verifies the panel,
candidate array and gate-specification hashes before sending any request. It issues
exactly 128 requests for a complete gate, comprising two original native-path replays
and fourteen explicit tiled candidates for each of eight frozen snapshots. It calls
the shared protocol's `evaluate_authority`, and saves the decoded chunks, per-query
artifact references and hashes, SQLite reservation audit and gate result for offline
verification.

Packet identifiers and a persisted send intent are deterministic. A shared run lock
and state file beside the SQLite ledger prevent duplicate requests across resumes
or alternate output paths. After a transport timeout, the runner polls the recorded
packet. A completed artifact is recovered without inference; a failed request
invalidates the gate; an unknown or still-reserved request remains pending and is
never resent. The output distinguishes confirmed RPC responses from artifacts
recovered without an RPC response. Preserve the probe state files beside the query
ledger. Exit code 2 means pending, exit code 1 means invalid, and code 0 means the
measurement is valid; admission separately requires `gate.admit` to be true.

`pytest -q experiments/tests/test_dsrl06_probe.py`: 8 passed in 3.03 seconds using
CPU fake transports and a synthetic non-neural policy. This includes the complete
128-request sequence and resume without extra calls, lost completed responses,
unknown and reserved outcomes, delayed completion during polling, failed calls and
frozen-input tampering. Ruff checks passed for both additional Python files.

The final integration review also made rollout packet identifiers include the attempt
number, so an allowed replacement cannot reuse its original attempt's packet identity.
The global SQLite reservation identifier remains unique across all attempts. A CPU
test verifies distinct identifiers for both attempts of the same job and query.
