# Infrastructure: the 5090, Hugging Face checkpoints and file layout

## Starting again from scratch

The historical Vast instance and its disk no longer exist (confirmed 26 September 2026).
The committed results in [docs/safeDial](../safeDial/) and [docs/phase0](../phase0/) remain
valid records, but nothing numeric can be reproduced until the assets are rebuilt.

1. **Instance:** an RTX 5090 on Vast with `cuda_max_good >= 13.0` and at least 150 GB of
   disk. Note its CPU quota (`/sys/fs/cgroup/cpu.max`), which can be far below `nproc`.
2. **Secrets:** `.env` with `WANDB_API_KEY`, `HF_TOKEN` with **write** access (needed for
   uploads), `GITHUB_TOKEN`, `GIT_AUTHOR_NAME` and `GIT_AUTHOR_EMAIL`. Optionally
   `HF_NAMESPACE`, the Hugging Face user or organisation that owns the repositories below;
   it defaults to the owner of `HF_TOKEN`. Never print token values.
3. **Environment:** `bash scripts/setup.sh`, then `uv run python scripts/download_data.py`
   for the Push-T weights, expert data and upstream source. Record the Hugging Face and git
   revisions that were resolved. Check free disk before decompressing, and record the
   decompressed dataset size, which the repository does not document yet.
4. **Threads:** before importing torch, set `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
   `OPENBLAS_NUM_THREADS` to fit the CPU quota, then call `torch.set_num_threads`. A 96-thread
   pool on an 11.5-core quota once made CEM 15 times slower with the GPU idle
   ([notes/safeCEM.md](../../notes/safeCEM.md)).
5. **Rebuild the Push-T assets once:** fitted scalers, the pinned upstream commit and a
   small original-data replay subset. Push them to Hugging Face before running any
   experiment. The old pusher-probe cache is not rebuilt; the new study trains its own
   block-pose probe.
6. **Before an instance is recycled:** every checkpoint is on Hugging Face and every small
   result is committed. Treat the instance disk as scratch space.

## Hugging Face checkpoints

**Every model checkpoint goes to a private Hugging Face repository.** Nothing that took GPU
time lives only on the instance, and no checkpoint is ever committed to git.

| Repository (private) | Type | Contents |
|---|---|---|
| `<ns>/safetydial-pusht` | model | `assets/` (scalers, upstream pins, replay subset index), `probes/`, `adapted/`, `closedloop/` |
| `<ns>/safetydial-walker2d` | model | `policies/`, `lewm-a/`, `lewm-b/`, `probes/`, `adapted/` |
| `<ns>/safetydial-walker2d-data` | dataset | State-only HDF5 for sets A, B and probe; episode split lists |
| `<ns>/safetydial-pusht-banks` | dataset | Root and branch banks: seeds, reset options, action prefixes and tapes, dense state truth |

`<ns>` is `HF_NAMESPACE`. The released Push-T weights stay on the public
`quentinll/lewm-pusht`; record the exact revision used.

**One folder per run,** `<kind>/<run_id>/`, containing:

- the weights: `weights.pt`, or the LeWM `*_object.ckpt` and `*_weights.ckpt` pair so that
  `swm.policy.AutoCostModel` can load it once downloaded under `STABLEWM_HOME`;
- `config.yaml`, the fully resolved config;
- `scalers.npz` wherever normalisers are part of the model's interface;
- `manifest.json`: this repository's git commit and dirty flag, the upstream LeWM commit,
  package versions (`stable-worldmodel`, `torch`, CUDA), GPU, seeds, data split identifiers
  and hashes, charged simulator steps, optimiser steps, wall-clock time and headline
  metrics;
- a short `README.md` model card saying what the checkpoint is and what it is not.

**Run IDs** follow `<env>-<kind>-<detail>-<yyyymmdd>-<n>`, for example
`pusht-adapt-boundary-b256-s1-20261021-1`. Tag each upload with its run ID.

**Loading:** always pin `revision=` to a tag or commit hash, and record that revision in the
consuming run's manifest. Never load "latest".

**Size:** a full LeWM state dict is about 60 MB. The whole plan needs roughly 5 to 10 GB of
private storage; check the account's quota before E3. Long runs, such as LeWM training,
also push intermediate checkpoints so that a recycled instance loses at most one interval.

## Protecting earlier work

Nothing in the new study edits, moves or overwrites earlier experiments.

- **Read-only:** the notebooks in `experiments/`, every file that exists today in
  `experiments/helpers/` and `experiments/scripts/`, `notes/`, `docs/safeDial/`,
  `docs/phase0/`, `animations/`, `docs/research/`, `docs/researchDirection.md` and the
  submitted academic documents.
- **Import, do not edit.** If an old helper needs different behaviour, copy the function
  into a new module with a comment recording where it came from. Old results must stay
  reproducible from their recorded commands.
- **Never rerun an old script into its committed result path.** Reproductions write to
  `runs/`.
- **New code goes in new files only.** Proposed names, to adjust when implementing as long
  as they stay new:

| New helper in `experiments/helpers/` | Role |
|---|---|
| `hfStore.py` | Upload and download run folders with manifests; revision pinning |
| `runManifest.py` | Git, package, hardware and data fingerprints for every run |
| `pushtReplay.py` | Seeded reset and prefix replay, determinism checks, dense per-step logging |
| `pushtGeometry.py` | T polygons from the shapes, hazard shapes, signed clearance, overlays |
| `pushtLayouts.py` | The frozen hazard layout generator and the familiar and held-out families |
| `poseProbes.py` | Trajectory-split probes for Push-T pose and for Walker2d height, pitch and speed |
| `branchBank.py` | Roots, proposal generator, branch execution and storage, charged-cost ledger |
| `imagination.py` | Batched rollouts and the four-source decomposition |
| `predictorAdapt.py` | Frozen-encoder, predictor-side adaptation with cached targets and replay |
| `acquisition.py` | Random, boundary, learned and oracle selectors; acquisition rounds |
| `dialMetrics.py` | FSA, acceptance, dial curves, matched acceptance, cluster bootstrap |
| `walkerRules.py` | Walker2d speed and health rules, dense truth, exact snapshot branching |
| `walkerLewm.py` | Walker2d data config and training support for the upstream LeWM code |

- Small entry points go in `experiments/scripts/` under new names, for example
  `pusht_e0_replay.py` or `walker_s2_train_lewm.py`. Hydra configs go in new groups,
  `configs/pusht/` and `configs/walker2d/`.
- **Generated data** goes under `data/study/pusht/` and `data/study/walker2d/`, and run
  directories under `runs/`. Both are gitignored and mirrored to Hugging Face where needed.
- **Committed results** (small JSON, CSV or NPZ files and figures) go under
  `docs/mainPlan/results/<experiment>/`, each with a `manifest.json` that names the Hugging
  Face revisions used.

## Compute on the 5090

Estimates to replace with measurements as they come in.

| Job | Bottleneck | Estimate |
|---|---|---|
| Push-T encoding | End to end, including HDF5 reads | About 939 frames/s, measured historically |
| CEM solve, 300 samples and 30 iterations | GPU, with threads pinned | About 1 s |
| Push-T branch simulation | CPU (pymunk) | Cheap next to model compute; measure in E0 |
| Predictor-side adaptation | GPU, cached targets | Minutes per run; E3 needs about 45 runs |
| Walker2d collection | CPU (MuJoCo) | 6,500 to 8,200 steps/s per process, saturating near 24 workers |
| Walker2d rendering | CPU (EGL) | About 3,100 frames/s per process |
| LeWM training | GPU plus render workers | "A few hours" per the upstream README; measure in the smoke run |
| OmniSafe policies | CPU, isolated environment | Hours; start on day one |

## Pitfalls checklist

- Set `MUJOCO_GL=egl` before importing mujoco, fix the render size at `make()`, use `spawn`
  rather than an EGL context inherited through `fork()`, and check the render fingerprint.
- MuJoCo `set_state` leaves `qacc_warmstart` and `ctrl` dirty; zero them for reproducible
  branches.
- Import `hdf5plugin` before compressed HDF5 reads, and read contiguous chunks only.
- Load `AutoCostModel` once, outside episode loops, and assign `STABLEWM_HOME` rather than
  using `setdefault`.
- Seed numpy, torch and the solver, and audit the action sequence CEM actually returns.
- Keep frozen modules in `eval()` after any parent `.train()` call.
