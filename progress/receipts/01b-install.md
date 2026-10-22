# Todo 1b, item 1: install replay on instance C

Host: Vast instance 57271835 (machine 54514, New Zealand), 2x RTX 3090 (24 GB each),
driver 580.173.02 (driver_max_cuda 13.0), CUDA toolkit 12.4 at `/usr/local/cuda`,
Ubuntu 22.04.5, conda 26.5.3 at `/opt/miniforge3`, `uv` 0.12.10, 250 GB RAM, 40 threads,
200 GB disk, `workspace_is_volume: false`. Everything lives under
`/workspace/Safety-Dial/third_party/` (gitignored); command logs in `third_party/logs/01b-*.log`
on the instance. Times are UTC, 22 October 2026.

## Sources as checked out

Identical to [01-sources.json](01-sources.json): ManiGuard
`6bf4c61a1f168cc46e02eb22c41bcb51815e9726` (detached), submodule `behavior-1k` at
`88454bd04f75dc57c00ab1f1a00bcde1ff505950`, OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`
(detached), benchmark `af09438c`, long-finger robot `5f65ec0a`, Jar checkpoint `1d84eda0`
step 7400, BEHAVIOR archives from `behavior-1k/zipped-datasets` (extracted
`behavior-1k-assets/VERSION` reads `3.7.2rc1`).

## Pre-check

`ldconfig -p` resolves `libEGL.so.1`, `libEGL_nvidia.so.0`, `libGL.so.1`, `libGLX_nvidia.so.0`,
`libnvoptix.so.1` and `libvulkan.so.1` (the first five from `/opt/nvidia-drivers/lib64`,
installed before handover with `install-display-drivers`); `vast-capabilities` reports
`render: {gl: true, optix: true, vulkan: true}`. The NVIDIA Vulkan ICD is at
`/etc/vulkan/icd.d/nvidia_icd.json` (not `/usr/share/vulkan/icd.d/`, which upstream's
`eval_family.sh` defaults to), so the lane scripts export `VK_ICD_FILENAMES` as B's did.
Both GPUs idle, 0 MiB used.

## Commands, in order

Steps 1, 2, 4, 5, 6 and the apt install ran concurrently in tmux sessions.

| # | Step (B's number) | Command | Result | Start -> end |
|---|---|---|---|---|
| 1 | Clone ManiGuard, check out pin, init submodule (1) | `git clone https://github.com/NU-IDEAS-Lab/ManiGuard.git; git checkout 6bf4c61…; git submodule update --init --recursive` (tmux `clone`) | ok | 20:58:47 -> 21:02:30 |
| 2 | OpenPI clone + env (2, 4) | `git clone …openpi.git; git checkout 215abfb…; git submodule update --init --recursive; GIT_LFS_SKIP_SMUDGE=1 uv sync; GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .` (tmux `openpi-env`) | ok: Python 3.11.16, jax 0.5.3, 240 packages, `jax.devices()` shows both CUDA devices | 20:58:47 -> 21:02:16 |
| 3 | X11/GL apt packages from [01-smoke.md](01-smoke.md) attempt 1, installed up front | `apt-get install -y libsm6 libice6 libxt6 libglu1-mesa libxrender1 libxrandr2 libxcursor1 libxinerama1 libxi6 libxxf86vm1 libxfixes3 libxext6 libxkbcommon0 libxcomposite1 libxdamage1 libgl1 libegl1` (tmux `apt`) | ok: 6 new (libsm6, libice6, libxt6, libglu1-mesa, libxcomposite1, libxdamage1, plus dependency libopengl0); the other 11 were already present | 20:58:47 -> 20:58:56 |
| 4 | Benchmark scenes (5) | `python -I scripts/todo1/download_pinned.py benchmark third_party/ManiGuard/outputs/lerobot_datasets/maniguard-bench` (tmux `dl-bench`, `/venv/main` python, huggingface_hub 1.30.0) | ok: 792 files, 0.05 GB, 0 mismatches, 0 missing, 26 base tasks | 20:58:56 -> 21:03:37 |
| 5 | Jar checkpoint (6) | `python -I scripts/todo1/download_pinned.py jar third_party/checkpoints/jar` (tmux `dl-jar`) | ok: 26 files, 6.34 GB, 0 mismatches, 0 missing | 20:58:56 -> 21:01:09 |
| 6 | BEHAVIOR install (3a-3c), `scripts/todo1b/install_behavior_replay.sh` (tmux `behavior-install`) | with `CONDA_ENVS_PATH=/opt/miniforge3/envs`: `conda create -n behavior python=3.10 -c conda-forge -y; conda activate behavior; pip install "numpy<2" "setuptools<=79"; pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url …/cu124` (116 s); `pip install ninja wheel; CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation "nvidia-curobo @ git+https://github.com/StanfordVL/curobo@cbaf7d32…"` (418 s); `./setup.sh --omnigibson --bddl --joylo --dataset --eval --primitives --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda` (1582 s) | ok: BDDL 3.7.0, OmniGibson 3.7.2 + Isaac Sim 4.5.0.0 (same wheel sha256s as B), cffi 1.17.1, JoyLo 3.7.0, torch-cluster 1.6.3+pt26cu124, conda `av` + `numpy<2`; datasets `omnigibson-robot-assets`, `behavior-1k-assets` (3.7.2rc1), the encryption key and (unrequested, upstream's `--dataset`) `2025-challenge-task-instances` | 21:03:53 -> 21:39:09 (2116 s) |
| 7 | Long-finger robot asset (7), after step 6 had extracted the stock robot bundle (`models/franka/{franka_panda,franka_mounted,franka_dexhand}` present) | `python -I scripts/todo1/download_pinned.py robot third_party/ManiGuard/behavior-1k/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger` (tmux `dl-robot`) | ok: 8 files, 0.04 GB, 0 mismatches; `usd/franka_panda_longfinger.usda` present | 21:30:09 -> 21:30:32 |
| 8 | Evaluator deps + ManiGuard (8), `scripts/todo1b/install_maniguard_deps.sh` (tmux `mg-deps`) | in the `behavior` env: `conda install -c conda-forge spot "numpy<2" -y; pip install -e openpi/packages/openpi-client; pip install -e "ManiGuard[serve]"` | ok: spot 2.16, openpi-client 0.1.0, maniguard 0.1.0.dev598+g6bf4c61a1; numpy 1.26.4 | 21:39:51 -> 21:41:16 (85 s) |

Wall time for item 1, first command to last: 20:58 to 21:41 (about 43 min; B took 44).

## Verification

- `behavior` env: Python 3.10.22, numpy 1.26.4, torch 2.6.0+cu124 with `torch.cuda.device_count() == 2`,
  omnigibson 3.7.2, isaacsim 4.5.0.0, spot 2.16, bddl 3.7.0, joylo 3.7.0, torch-cluster
  1.6.3+pt26cu124, nvidia-curobo at `cbaf7d32`, maniguard 0.1.0.dev598+g6bf4c61a1.
- `pip freeze` of the behavior env ([01b-env-behavior-freeze.txt](01b-env-behavior-freeze.txt),
  298 lines) is identical to B's [01-env-behavior-freeze.txt](01-env-behavior-freeze.txt)
  after normalising the temporary directory the Isaac Sim wheels were staged in (same
  wheel sha256s). `uv pip freeze` of the OpenPI venv
  ([01b-env-openpi-freeze.txt](01b-env-openpi-freeze.txt), 240 lines) is identical to B's.
- Hashes: benchmark, checkpoint and robot asset verified file by file against the Hub
  manifests by `download_pinned.py` (`download-provenance.json` beside each download);
  zero mismatches, zero missing. The BEHAVIOR archives are fetched and deleted by the
  upstream installer, as on B; `hf_hub_download` verifies their LFS sha256 on download and
  the extracted `VERSION` matches.
- `configs/eval/jar_transport_joint_legacy256.yaml` copied into the clone from
  [01-jar-eval.yaml](01-jar-eval.yaml) (byte-identical).

## Disk use after install

| Item | Size |
|---|---|
| `behavior` conda env | 19 GB |
| OpenPI `.venv` | 7.7 GB |
| `behavior-1k-assets` | 33 GB |
| OmniGibson robot assets incl. long-finger | 2.4 GB |
| `2025-challenge-task-instances` (unrequested, upstream `--dataset`) | 0.4 GB |
| Jar checkpoint step 7400 | 6.0 GB |
| Benchmark `jar_transport` | 0.06 GB |
| Total used on `/` | 77 GB of 200 GB |

## Differences from instance B

1. **Installer entry.** B ran `setup.sh --new-env …` twice: the first attempt failed on
   this image's conda layout, the second (with `CONDA_ENVS_PATH`) failed 472 s later when
   `nvidia-curobo` built in pip's isolated environment against a CUDA 13 torch. Here the
   `--new-env` block of `setup.sh` (conda create, activate, `numpy<2` + `setuptools<=79`,
   torch 2.6.0 cu124; the same four commands, with the same `CONDA_ENVS_PATH` workaround)
   was run by hand, curobo was pre-built as in B's step 3c, and the installer was then
   resumed with B's exact 3c flags. End state identical (freeze diff empty); the two
   known-failing attempts were not repeated.
2. **X11/GL packages installed before the first simulator start** rather than after a
   crashed smoke attempt. Same package list as B's fix.
3. **Driver** 580.173.02 here (B: 580.178.04); both are CUDA 13.0 drivers, and every wheel
   is a CUDA 12.x build, so minor-version compatibility applies. Recorded, not a change.
4. **Two GPUs.** torch and jax both see two devices. Each lane will pin
   `CUDA_VISIBLE_DEVICES` (item 2).
5. The same pip warning about JoyLo transiently raising numpy to 2.2.6 appeared; the
   installer's `--eval` step returned it to 1.26.4 before anything ran, as on B.
6. OptiX/EGL libraries were installed on this host before handover
   (`install-display-drivers`), noted in [01b-lanes.md](../01b-lanes.md); verified, not redone.

No simulator episode has run yet; that is item 2.
