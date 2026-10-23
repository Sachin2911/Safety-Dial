# Todo 1c, item 1: install replay on instance D (instance C's machine)

Host: Vast instance 57334420 (machine 54514, New Zealand, the machine that was instance C),
2x RTX 3090 (24 GB each), driver 580.173.02, CUDA toolkit 12.4 at `/usr/local/cuda`, Ubuntu
22.04, conda at `/opt/miniforge3`, `uv` at `/usr/local/bin/uv`, 250 GB RAM, 40 threads, 200 GB
disk, `workspace_is_volume: false`. Everything lives under `/workspace/Safety-Dial/third_party/`
(gitignored). Command logs are copied into the repository as
`01-runs/logs/01c-*.log.txt`. Times are UTC, 23 October 2026.

## Sources as checked out

Identical to [01-sources.json](01-sources.json) and to C ([01b-install.md](01b-install.md)):
ManiGuard `6bf4c61a1f168cc46e02eb22c41bcb51815e9726` (detached), submodule `behavior-1k` at
`88454bd04f75dc57c00ab1f1a00bcde1ff505950`, OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`
(detached), benchmark `af09438c`, long-finger robot `5f65ec0a`, Jar checkpoint `1d84eda0` step
7400, BEHAVIOR archives from `behavior-1k/zipped-datasets` (extracted `behavior-1k-assets/VERSION`
reads `3.7.2rc1`), `nvidia-curobo` at `cbaf7d32` (`direct_url.json`).

## Pre-check and display libraries

At handover `ldconfig -p` resolved only `libEGL.so.1`, `libEGL_mesa.so.0`, `libGL.so.1` and
`libvulkan.so.1` from the system; `libnvoptix.so.1`, `libEGL_nvidia.so.0` and `libGLX_nvidia.so.0`
were missing, `vast-capabilities` reported `render: {gl: false, optix: false, vulkan: false}`, and
`/etc/vulkan/icd.d/` was empty. As on C, `/opt/instance-tools/bin/install-display-drivers` was run
(tmux `display`, 06:53:32 to 06:53:45; it fetched the 580.173.02 `.run` and extracted the libraries
into `/opt/nvidia-drivers`). Afterwards all six resolve (the NVIDIA ones from
`/opt/nvidia-drivers/lib64`), `render: {gl: true, optix: true, vulkan: true}`, and
`/etc/vulkan/icd.d/nvidia_icd.json` exists.

The GLVND EGL vendor repair C found in its item 2 ([01b-smoke.md](01b-smoke.md)) was applied here
before any simulator start: `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` with
`library_path: libEGL_nvidia.so.0` (only `50_mesa.json` existed). `vulkaninfo --summary` with no
environment overrides then lists both RTX 3090s, Vulkan 1.4.312, driver 580.173.02. The lane
scripts also export `__EGL_VENDOR_LIBRARY_FILENAMES` and `VK_ICD_FILENAMES` as on C. Both GPUs idle,
0 MiB used.

## Commands, in order

Steps 1, 2, 3, 5 and the display-driver install ran concurrently in tmux sessions; the benchmark
download was chained after the clone because its destination is inside the clone.

| # | Step (C's number) | Command | Result | Start -> end |
|---|---|---|---|---|
| 1 | Clone ManiGuard, check out pin, init submodule (1) | `git clone https://github.com/NU-IDEAS-Lab/ManiGuard.git; git checkout 6bf4c61…; git submodule update --init --recursive` (tmux `clone`) | ok | 06:53:44 -> 06:57:12 |
| 2 | OpenPI clone + env (2) | `git clone …openpi.git; git checkout 215abfb…; git submodule update --init --recursive; GIT_LFS_SKIP_SMUDGE=1 uv sync; GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .` (tmux `openpi-env`) | ok: Python 3.11, jax 0.5.3, 240 packages, `jax.devices()` shows both CUDA devices | 06:53:44 -> 06:56:36 |
| 3 | X11/GL apt packages (3) | `apt-get update && apt-get install -y libsm6 libice6 libxt6 libglu1-mesa libxrender1 libxrandr2 libxcursor1 libxinerama1 libxi6 libxxf86vm1 libxfixes3 libxext6 libxkbcommon0 libxcomposite1 libxdamage1 libgl1 libegl1` (tmux `apt`) | ok: 7 new (libglu1-mesa, libice6, libopengl0, libsm6, libxcomposite1, libxdamage1, libxt6), the same set as C; the rest already present | 06:53:44 -> 06:53:52 |
| 4 | Benchmark scenes (4) | `python -I scripts/todo1/download_pinned.py benchmark third_party/ManiGuard/outputs/lerobot_datasets/maniguard-bench` (`/venv/main` python, huggingface_hub 1.30.0) | ok: 792 files, 0.05 GB, 0 mismatches, 0 missing, 26 base tasks | 06:57:12 -> 07:01:26 |
| 5 | Jar checkpoint (5) | `python -I scripts/todo1/download_pinned.py jar third_party/checkpoints/jar` (tmux `dl-jar`) | ok: 26 files, 6.34 GB, 0 mismatches, 0 missing | 06:53:44 -> 06:55:41 |
| 6 | BEHAVIOR install (6), `scripts/todo1b/install_behavior_replay.sh` unchanged (tmux `behavior-install`) | new-env block by hand (conda create, `numpy<2` + `setuptools<=79`, torch 2.6.0 cu124), curobo `--no-build-isolation` at `cbaf7d32`, then `./setup.sh --omnigibson --bddl --joylo --dataset --eval --primitives --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda` | ok: new env 135 s, curobo 425 s, installer 1135 s; OmniGibson 3.7.2 + Isaac Sim 4.5.0.0 (same wheel sha256s as B and C), BDDL 3.7.0, JoyLo 3.7.0, torch-cluster 1.6.3+pt26cu124, datasets `omnigibson-robot-assets`, `behavior-1k-assets` (3.7.2rc1), the encryption key and (unrequested, upstream's `--dataset`) `2025-challenge-task-instances` | 06:57:18 -> 07:25:33 (1695 s) |
| 7 | Long-finger robot asset (7), started by a watcher once step 6 had extracted `models/franka/{franka_panda,franka_mounted,franka_dexhand}` | `python -I scripts/todo1/download_pinned.py robot …/omnigibson-robot-assets/models/franka/franka_panda_longfinger` (tmux `dl-robot`) | ok: 8 files, 0.04 GB, 0 mismatches; `usd/franka_panda_longfinger.usda` present | 07:15:51 -> 07:16:25 |
| 8 | Evaluator deps + ManiGuard (8), `scripts/todo1b/install_maniguard_deps.sh` unchanged | `conda install -c conda-forge spot "numpy<2" -y; pip install -e openpi/packages/openpi-client; pip install -e "ManiGuard[serve]"` | ok: spot 2.16, openpi-client 0.1.0, maniguard 0.1.0.dev598+g6bf4c61a1; numpy 1.26.4 | 07:25:47 -> 07:27:12 (85 s) |
| 9 | **New: pin three drifted packages back to C's versions** | `pip install "dm_control==1.0.49" "mujoco==3.16.0" "googleapis-common-protos==1.75.5"` in the `behavior` env | ok; see difference 1 below | 07:27 -> 07:27:53 |
| 10 | Clone wiring, as on C | `configs/eval/jar_transport_joint_legacy256.yaml` copied from [01-jar-eval.yaml](01-jar-eval.yaml) (byte-identical, `cmp`); `scripts/eval_family_lane.sh` symlinked to `scripts/todo1/eval_family_lane.sh`, whose diff against upstream `scripts/eval_family.sh` at the pin equals [01b-lane-wrapper.diff](01b-lane-wrapper.diff) | ok | 06:58 |

Wall time for item 1, first command to last: 06:53 to 07:28 (about 35 min; C took 43, B 44).

## Verification

- `behavior` env: Python 3.10.22, numpy 1.26.4, torch 2.6.0+cu124 with `torch.cuda.device_count() == 2`,
  omnigibson 3.7.2, isaacsim 4.5.0.0, spot 2.16 (`import spot` ok), bddl 3.7.0, joylo 3.7.0,
  torch-cluster 1.6.3+pt26cu124, nvidia-curobo at `cbaf7d32`, maniguard 0.1.0.dev598+g6bf4c61a1,
  openpi-client 0.1.0.
- OpenPI venv: `jax.devices()` returns `[CudaDevice(id=0), CudaDevice(id=1)]`, jax 0.5.3.
- `pip freeze` of the behavior env ([01c-env-behavior-freeze.txt](01c-env-behavior-freeze.txt), 298
  lines) is identical to C's [01b-env-behavior-freeze.txt](01b-env-behavior-freeze.txt) and B's
  [01-env-behavior-freeze.txt](01-env-behavior-freeze.txt) after normalising the temporary directory
  the Isaac Sim wheels were staged in; the 25 Isaac Sim wheel sha256s are identical. This holds after
  step 9; before it the freeze differed in three lines (difference 1).
- `uv pip freeze --python .venv/bin/python` of the OpenPI venv
  ([01c-env-openpi-freeze.txt](01c-env-openpi-freeze.txt), 240 lines) is identical to B's and C's.
- Hashes: benchmark, checkpoint and robot asset verified file by file against the Hub manifests by
  `download_pinned.py` (`download-provenance.json` beside each download); zero mismatches, zero
  missing. The BEHAVIOR archives are fetched and deleted by the upstream installer, as on B and C;
  `hf_hub_download` verifies their LFS sha256 and the extracted `VERSION` matches.

## Disk use after install

| Item | Size |
|---|---|
| `behavior` conda env | 19 GB |
| OpenPI `.venv` | 7.7 GB |
| `behavior-1k-assets` | 33 GB |
| OmniGibson robot assets incl. long-finger | 2.4 GB |
| `2025-challenge-task-instances` (unrequested, upstream `--dataset`) | 0.4 GB |
| Jar checkpoint step 7400 | 6.0 GB |
| Benchmark `jar_transport` | 0.05 GB |
| Total used on `/` | 78 GB of 200 GB |

## Differences from instance C

1. **Upstream package drift, pinned back.** The first freeze differed from C's (and B's) in three
   unpinned transitive packages that received new PyPI releases after C installed (C finished
   2026-10-22 21:41): `dm_control` 1.0.49 -> 1.0.50 (released 2026-10-23 00:29), `mujoco` 3.16.0 ->
   3.17.0 (2026-10-22 23:34; required by `dm_control>=1.0.50`) and `googleapis-common-protos` 1.75.5 ->
   1.76.0 (2026-10-22 23:43). `dm_control` comes in through JoyLo (teleoperation) and
   `googleapis-common-protos` through the OpenTelemetry exporter; none is on the evaluator path. They
   were reinstalled at C's versions (step 9) so that the environment is identical to C's, which is
   the control this rerun depends on. `pip check` reports only the `opencv-contrib-python` numpy>=2
   note that the identical freeze implies C also had.
2. **EGL vendor file created during install** rather than after two crashed smoke attempts as on C.
   Same file, same content.
3. **`apt-get update`** ran before the apt install (C ran the install alone). The same seven
   packages were added.
4. **Benchmark download chained after the clone** instead of started 9 s after it. Same command.
5. The OpenPI freeze was first taken without `--python` and picked up `/venv/main` (49 lines); it
   was retaken against `third_party/openpi/.venv/bin/python`. Recorded because the file was overwritten.
6. Same driver (580.173.02), same GPUs, same machine as C. `install-display-drivers` was run by the
   agent here (C's host had it done before handover).
7. Git identity: the instance had none; the repository-local identity was set to the one on every
   earlier commit, and pushes authenticate through a repository-local credential helper that reads
   `GITHUB_TOKEN` from the environment at call time (the token is never written to disk or printed).
8. Bookkeeping: C's `01-runs/seed1/runner.log` and `seed2/runner.log`, named in
   [01b-results.md](01b-results.md), were never committed because `.gitignore` excludes `*.log`. This
   rerun stores runner and install logs as `*.log.txt`, B's convention, so they reach the repository.

No simulator episode has run yet; that is item 2.
