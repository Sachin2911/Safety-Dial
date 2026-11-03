# PushT AP-CEM reproduction and paired wrapper check

Authorised by Sachin on 2 November 2026 after the distinct-scene screen failed.
This is a fresh experiment on existing Vast instance 58676387. Rental shutdown
remains manual. No adapter training, hazard evaluation or new rental is included.

## Question

Does the released AP-CEM evaluator recover its published PushT standard-start
completion rate, and does our existing wrapper behave differently on identical
queries, retrieval memory and optimizer seeds?

## Frozen comparison

Use public AP revision `984327424ca7a2e52c75a8279197750d9d0b7c8e`, its released
PushT checkpoint and original expert dataset. Run the 128 PushT queries in
`code/ap/protocols/main.json`, preserving their order, initial states, goals,
environment seeds and query-index-derived optimizer seeds. Use all 16,688
specified memory episodes, excluding the query episodes exactly as recorded.
Do not exclude whole starting-scene groups or remove initially successful cases.

Two arms, 128 episodes each:

1. Authors: unmodified released `runtime.evaluate` with a derived protocol that
   selects only `clean` and `gaussian_observed`. Instrumentation records physical
   states, actions, images and resource counts without changing decisions.
2. Existing wrapper: the existing `manip_pipeline.run_episode`, `arm="ap"`,
   with the same queries and memory. A reset shim supplies the recorded environment
   seed while the existing seed argument supplies the original optimizer seed.
   Its fp32 retrieval encoding, one-neighbour query and pixel preprocessing remain
   unchanged. Source files remain unchanged; all shims are saved and hashed.

Both use the same isolated runtime, checkpoint, cache and eight-GPU host. Each
query ordinal maps to GPU `ordinal % 8` in both arms. Authors run first, followed
by the wrapper. No outcomes-driven tuning, replacement cases or early success stop.
The paper used H20 GPUs; our RTX 4090 hardware is a declared deviation. Install
the published dependency pins separately from all previous environments. Record
Python patch-level and driver differences. A dependency mismatch blocks evaluation
until resolved or explicitly documented in a prospective amendment.

Horizon 140; maximum 280 executed actions; five-action replanning; 300 CEM
candidates, 30 elites, 30 iterations, standard deviation 1/3, zero mean and
standard-deviation floor 1e-5, as implemented in the reference. Retrieval cache
uses the reference bfloat16 encoder and float16 storage, batch size 128. Cache
construction may split complete contiguous batches across eight GPUs. Original
row order and full coverage must pass checks before evaluation.

## Engineering, bounds and evidence

Before rollout, check source hashes, original/derived protocol equality except
arm/condition selection, query-memory disjointness, full cache coverage, dependency
pins, finite embeddings and normalized action bounds. Allow at most four simulator
engineering resets, five actions each, checking instrumented versus uninstrumented
replay of a fixed command sequence. These are not performance observations.
No model-based test episodes outside the 256 final episodes.

Cap: 262 attempted resets and 72,260 attempted actions, including engineering
and a two-reset/560-action engineering-failure reserve. Failed attempts remain
recorded; do not silently retry or replace performance episodes. Stop job execution
at four hours of active preparation/evaluation, $14.66 equivalent host time, or
less than 2 GiB free storage. The rental remains running. No unspent historical
allowance is used. Cache encoding and all model queries are counted separately.

Outputs: `outputs/pusht-ap-reproduction-20261102/`, new versioned paths only.
Preserve protocol, code, dependencies, checkpoint/data provenance, cache hashes,
all reset/action attempts, per-episode physical state/action traces, outcomes,
elapsed time and model/encoder/retrieval calls. Audit all 256 episodes against
the geometric PushT success rule, including initial success and terminal flags.
Report completion, unfinished and engagement with denominators; no safety rate
is claimed because there is no hazard. Compare paired starts and final outcomes,
first divergent retrieval/action when available, and the published 89/128.

This identifies an implementation discrepancy if one appears, but cannot assign
it uniquely to retrieval precision, preprocessing or another wrapper difference.
Do not launch corrective ablations without a new decision. The previously failed
whole-scene holdout remains a separate, harder generalization result.
