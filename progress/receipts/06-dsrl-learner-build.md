# Task 6 learner implementation and workstation checks

Recorded on 25 October 2026. This receipt covers the learner bridge and its local
checks. GPU throughput, real replay, decoder authority and the paid experiment
remain separate checks in the frozen plan.

## Reused implementation and environment

The bridge imports `nakamotoo/dsrl_pi0` at
`7f48937d4553e95244cd81c79236a3256df80597`, under its MIT license, from the ignored
`third_party/dsrl_pi0` checkout. It verifies the checkout revision and rejects
tracked source changes. No author file was edited. Actor sampling, networks,
augmentation, optimizers, temperature learning and the SAC update are the author's
implementations. The wrapper does not introduce another SAC implementation.

The separate Python 3.11 learner environment uses
`configs/dsrl06/learner-requirements.txt`, which pins all 82 resolved packages.
It uses JAX/JAXlib 0.5.0, Flax 0.10.2, Optax 0.2.4, Distrax 0.1.5 and
TensorFlow Probability 0.25.0. The GPU variant adds the JAX CUDA 12 plugin and
matching NVIDIA packages. It requires neither TensorFlow nor LIBERO. Wandb and
Matplotlib are installed because unused author encoder modules import them;
tracking is disabled and Matplotlib uses Agg. Do not install these dependencies
in the OpenPI or behavior environment.

```bash
python3.11 -m venv /venv/dsrl06-learner
/venv/dsrl06-learner/bin/python -m pip install -r configs/dsrl06/learner-requirements.txt
```

A dependency dry run resolved the complete GPU requirements without conflicts.
Its SHA256 is
`1c3b9e351d61d1e37c6430e97af016d350ca1fc6365ee9f2d2f108b55a206e2d`.
The workstation source check used `/tmp/dsrl06-cpu-venv/bin/python`, a separate
CPU environment with the same core pins, without CUDA plugins. The root
workstation `.venv` remains suitable for boundary tests and does not import JAX
when importing the bridge.

The resolved source settings include ten critics with mean reduction, actor
learning rate 0.0001, critic and temperature rates 0.0003, hidden dimensions
128/128/128, small encoder, bottleneck 50, spatial softmax, group normalization,
augmentation of next observations, action magnitude 1 and one camera. Inherited
constructor defaults are explicitly frozen as `color_jitter=True`,
`init_temperature=1.0` and `decay_steps=None`. Automatic target entropy resolves
to -16 for the 32-dimensional latent action. Trainable counts are 1,077,622 actor
parameters and 1,484,480 critic parameters.

## Learner service contract

The entry point is `experiments/scripts/dsrl06_learner_service.py`. It binds only
to `127.0.0.1`, using port 9009 by default. The remote start must import the two
workstation engineering updates already incurred:

```bash
CUDA_VISIBLE_DEVICES=3 /venv/dsrl06-learner/bin/python \
  experiments/scripts/dsrl06_learner_service.py \
  --source third_party/dsrl_pi0 \
  --work-dir outputs/dsrl06/learner --data-root outputs/dsrl06 \
  --phase engineering --prior-engineering-updates 2 \
  --prior-engineering-receipt sha256:669c5e16bf6a67c80350510a4d658323748ff9033682f148166c1a1a85efe079
```

`GET /status` exposes the resolved configuration, actor version, replay and
episode counts, update counts, activity and interrupted rounds. JSON POST routes
are:

- `/act`: `job_id`, `query_index`, `rng_seed`, `actor_version`, `pixels_b64` and
  `state`. Pixels are raw RGB64 uint8 bytes encoded as base64; state is exactly
  eight finite native robot coordinates. Response contains `noise` and
  `actor_version`. The sampler remains stochastic with independent per-job keys.
- `/append`: `round_id`, Boolean `warmup` and three episode descriptors in frozen
  job order. A descriptor contains `job_id`, `order_index`, `path`, `sha256` and
  `actor_version`. Files must be inside the declared data root.
- `/train`: `round_id` and optionally `expected_new_transitions`. It applies
  exactly 20 updates per newly eligible transition, with batch size 256. Warm-up
  transitions never receive retrospective update credit.
- `/publish`: `round_id`. Publishes the completed round and advances the actor
  version once. Sampling uses the preceding immutable actor snapshot until this
  request, even while learner weights change.
- `/profile`: recorded observation NPZ `path`, `sha256`, `updates` and optionally
  `parity` (default true). The profile uses recorded RGB64/state8 observations
  with explicitly synthetic zero actions, rewards and terminal targets. Source
  parity consumes two updates in addition to the requested timed updates.
- `/initialize`: `phase`, set to `training` after engineering. It creates a fresh
  seed-6060 learner and empty replay while preserving engineering charges. A
  learner already collecting training episodes cannot be reset this way.
- `/save`: immutable checkpoint `name`; `/load`: checkpoint directory `path`.
  `/shutdown` exits the service and releases its GPU after updates finish.
- `/export-actor`: immutable export `name`. Returns its directory `path`,
  `manifest_sha256`, `actor_version` and `actor_evaluation_mode`.

The frozen actor sampling rule is: derive the uint32 seed from the little-endian
first four SHA256 bytes of `dsrl06-actor:<rng_seed>:<job_id>`, create the JAX PRNG
key, then `fold_in(query_index)`. Query ordering in other jobs cannot change a
job's samples. This rule is included in the resolved configuration.

An episode NPZ contains these exact array names:

```text
pixels          uint8    (T+1,64,64,3)
states          float32  (T+1,8)
noise           float32  (T,32)
rewards         float32  (T,)
discounts       float32  (T,)    = 0.9998 ** durations
masks           float32  (T,)    = 1 except the final 0
durations       integer  (T,)    = 8 except a possible final partial chunk
query_steps     integer  (T,)    = cumulative executed controls, beginning at 0
actor_versions  integer  (T,)    = one admitted version throughout the episode
```

Here `1 <= T <= 250` and total duration cannot exceed 2,000 controls. Warm-up
Gaussian latent noise is deliberately not clipped. Learned noise must remain
within magnitude 1. The image helper uses PIL bilinear RGB64 resizing with no
flip or crop. The actor and critic receive only the image and eight native
coordinates; reward, monitor state and simulator state stay outside their input.

Replay holds 6,000 transitions without eviction and samples using a separate
checkpointed NumPy RNG. Readiness is checked after six warm-up episodes and, if
needed, twelve, requiring more than 500 transitions. Three-episode append,
completed training and publication requests are idempotent. Duplicate round
identities with changed descriptors are rejected.

All gradient attempts are charged durably before dispatch, including failed
attempts and both halves of source parity. Engineering is capped at 2,000,
training at 90,000 and all gradients at 120,000. Importing prior workstation
charges occurs once in a fresh ledger. Restarting or resetting cannot erase
them. A mid-round exception stops that round instead of granting its full
credit again. An HTTP timeout should be followed by `/status` polling while the
existing request finishes.

Checkpoints contain actor, critics, target critics, temperature, all optimizer
states, learner RNG, published actor, replay arrays and RNG, configuration,
round metadata and counters, with SHA256 manifests. Resume is exact at saved
boundaries. Uncheckpointed gradient attempts cause an explicit ambiguous-resume
failure. The caller must preserve and diagnose them rather than restarting
training with a reduced counter.

## CPU evidence and accounting

At `2026-10-24T22:23:47.295643+00:00`, a synthetic batch of 256 observations was
used for a real pinned-source CPU parity check. One direct author clone update
and one bridge update started from identical complete mutable state and RNG.
Maximum state difference was **0.0**. Serializing and restoring the complete
learner into a fresh author instance also reproduced every mutable state leaf
exactly. The operation took 10.609 seconds, including compilation and the
serialization check. This is not a GPU throughput estimate.

Exactly **two engineering gradient attempts** were incurred. No frozen VLA
query, simulator control or rental was used. These events must remain charged
when the remote service starts, leaving at most 1,998 engineering updates:

```json
[
  {"event":"engineering_update_reserved","origin":"workstation_cpu_preflight","label":"author_reference","update":1},
  {"event":"engineering_update_reserved","origin":"workstation_cpu_preflight","label":"bridge_parity","update":2}
]
```

The raw evidence is in ignored `outputs/dsrl06-preflight/learner/` and must be
preserved with the run. Relevant hashes are:

- `cpu-source-parity.json`: `669c5e16bf6a67c80350510a4d658323748ff9033682f148166c1a1a85efe079`.
- `cpu-update-events.json`: `ae61cfb2d5ec681bf4181f46451344590e2e50e28a9b0517b8ddeddac42f7fab`.
- Synthetic batch: `64cb9f693bc52e23496ad3c56eb01d9713afaee1854281d38427416d3c25a49f`.
- Serialized learner: `dabf4c94ec5b7f6bbe7253f757ed0bb2ff30ec831cef01865efccf301c484701`.
- Author `jaxrl2/agents/pixel_sac/pixel_sac_learner.py`:
  `6f06c5d555e6c1fee748eeec6f8eb10c6f09f34130e209e961b97c520b142c49`.

The final control-plane tests pass using
`.venv/bin/python -m pytest -q experiments/tests/test_dsrl06_learner.py`:
**14 passed** when localhost sockets are available. In the restricted sandbox,
12 pass and the two HTTP cases skip because sockets are blocked. The HTTP
checks passed outside that restriction using fake learner and actor backends. These tests cover
the input boundary, duration and mask validation, Gaussian versus learned
bounds, no-eviction replay, warm-up credit, immutable publication, checkpointed
replay sampling, idempotence, inherited engineering charges, interrupted updates
and HTTP initialization, inference errors and shutdown, plus published actor
export, identical actor request sampling, immutable exports, hash validation and
rejection of training routes in the actor-only service. They do not execute
additional real gradient updates. Ruff checks and formatting pass at line
length 100. The remote engineering pass must still verify final-file source
parity and measured GPU throughput on recorded observations.

## Published CPU actor for final evaluation

Final evaluation uses `actor_evaluation_mode="cpu_published_actor"`. After saving
the complete final learner checkpoint, `/export-actor` writes only the published
actor parameters and batch statistics, configuration, version and provenance.
It never exports the mutable unpublished actor by mistake. Critic weights,
temperature, optimizer state and replay are absent. The export is independently
hashed and immutable. Keep both the full learner checkpoint and lightweight
actor export in the private preservation artifact.

Stop the GPU learner with `/shutdown`, then start the CPU actor on the same
localhost port before admitting four VLA evaluation lanes:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  /venv/dsrl06-learner/bin/python experiments/scripts/dsrl06_actor_service.py \
  --source third_party/dsrl_pi0 --checkpoint <export-directory> \
  --work-dir outputs/dsrl06/actor-eval --host 127.0.0.1 --port 9009
```

The worker also forces CPU platform selection before importing JAX and rejects
a non-CPU backend. It constructs only the pinned author's `Encoder`,
`LearnedStdTanhNormalPolicy` and `PixelMultiplexer`, using the exact actor
constructor arguments from `PixelSACLearner`. It restores the exported actor
weights and calls the same author `sample_actions_jit` function as the learner.
There is no critic, temperature, optimizer or replay allocation. It serves the
same `/act`, `/status` and `/shutdown` contract. Training, initialization and
checkpoint mutation routes return 404. Its durable request ledger is separate
from the learner's gradient ledger.

An exact-source CPU check at `2026-10-24T22:40:38.827531+00:00` used synthetic
images, native-shape eight-dimensional states and a synthetic parameter offset
to distinguish published weights from the current learner's weights. It made
64 paired stochastic samples plus two repeat requests. Maximum restored
parameter difference and maximum sampled-action difference were both **0.0**.
Repeat requests were identical. The payload is 4,311,185 bytes, containing
1,077,622 actor parameters. No extra gradient updates, frozen VLA queries or
simulator controls were used. Controller forward calls were 64 for the author
reference and 66 for the CPU worker.

Local CPU timing, including input validation, sampling, Python result conversion
and the fsynced request ledger, was 3.061 ms median and 3.563 ms p95 after the
first request. First-request time was 139.377 ms; construction and restore into
an already imported CPU process took 198.361 ms. These figures exclude HTTP
transport and cold Python imports, and do not predict the rental host's timing.

The evidence is `outputs/dsrl06-preflight/actor/cpu-actor-parity.json`, SHA256
`7314730f26e63ae55caa15e4bee1baa2be28376a694e661a372b9badb0ec6c69`.
The synthetic input hash is
`66f22ec9c562206e57ea31652f82791a0ab2b5bdcbfb737109c71f289cc54cfa`;
the actor payload hash is
`76e868f0942584cc4a59909a8f07b34d6979b6053f42d0155fce24cf75a1cb4e`.
This is CPU versus CPU parity. GPU versus CPU numerical parity is still a
required engineering check on the preserved observation inputs. Run the CPU
worker on port 9010 beside the engineering learner on 9009, send identical
`/act` requests to both, check the frozen tolerance, and stop it afterwards.
This comparison requires no extra gradients, VLA calls or simulator controls.
