# Todo 1, item 2: install receipt

Host: Vast instance 57261620, RTX 3090 (24 GB), driver 580.178.04, CUDA toolkit 12.4,
Ubuntu 22.04, conda 26.5.3 at `/opt/miniforge3`, `uv` 0.x at `/usr/local/bin/uv`.
Everything lives under `/workspace/Safety-Dial/third_party/` (gitignored). Logs of every
command below are in `third_party/logs/` on the instance. Times are UTC, 22 October 2026.

## Sources as checked out

See [01-sources.json](01-sources.json). ManiGuard `6bf4c61` (detached), submodule
`behavior-1k` at `88454bd`, OpenPI `215abfb` (detached), benchmark `af09438c`,
robot asset `5f65ec0a`, Jar checkpoint `1d84eda0` step 7400.

## Commands, in order

| # | Step | Command | Result | Duration |
|---|---|---|---|---|
| 1 | Clone ManiGuard, check out pin, init submodule | `git clone https://github.com/NU-IDEAS-Lab/ManiGuard.git; git checkout 6bf4c61…; git submodule update --init --recursive` (tmux `submod`) | ok | ~3 min |
| 2 | Clone OpenPI, check out pin | `git clone https://github.com/Physical-Intelligence/openpi.git; git checkout 215abfb…` | ok | ~1 min |
| 3a | BEHAVIOR installer, attempt 1 | `cd behavior-1k && ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos` (19:45:18) | **failed** after `conda create`: "Failed to activate environment". This image's conda puts envs under `/venv`, so `conda activate behavior` sets `CONDA_DEFAULT_ENV=/venv/behavior`, which the script's `!= "behavior"` check rejects. Env removed. | 16 s |
| 3b | Attempt 1b, same command with `CONDA_ENVS_PATH=/opt/miniforge3/envs` (19:46:50) | numpy 1.26.4, setuptools 79, torch 2.6.0+cu124 installed, BDDL installed; **failed** in `pip install -e OmniGibson[primitives,eval]`: `nvidia-curobo` builds a CUDA extension in pip's isolated build env, which resolved a current PyPI torch (CUDA 13.0) against nvcc 12.4 (`RuntimeError: The detected CUDA version (12.4) mismatches the version that was used to compile PyTorch (13.0)`). | 472 s |
| 3c | curobo against the env's own torch, then resume the installer (20:10) | `pip install ninja wheel; CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation "nvidia-curobo @ git+https://github.com/StanfordVL/curobo@cbaf7d32…"` (the exact pin from OmniGibson's `setup.py`), then `./setup.sh --omnigibson --bddl --joylo --dataset --eval --primitives --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda` inside the activated env | **ok**: curobo built in 417 s; OmniGibson 3.7.2 + Isaac Sim 4.5.0.0 wheels (pypi.nvidia.com), cffi 1.17.1, JoyLo 3.7.0, torch-cluster 1.6.3+pt26cu124, conda `av` + `numpy<2`; datasets: `omnigibson-robot-assets.zip`, `behavior-1k-assets-3.7.2rc1.zip`, the BEHAVIOR encryption key, and (unrequested, part of upstream's `--dataset`) `2025-challenge-task-instances.zip`. | 1831 s total |
| 4 | OpenPI env | `cd openpi && git submodule update --init --recursive && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .` (tmux `openpi-env`, 19:45:18) | ok: Python 3.11.16, jax 0.5.3 with CUDA device visible, 240 packages | 269 s |
| 5 | Benchmark scenes | `python -I scripts/todo1/download_pinned.py benchmark third_party/ManiGuard/outputs/lerobot_datasets/maniguard-bench` (jar_transport only, 19:45:18) | ok: 792 files, 0.05 GB, 0 hash mismatches, 26 base tasks | 158 s |
| 6 | Jar checkpoint | `python -I scripts/todo1/download_pinned.py jar third_party/checkpoints/jar` (step 7400 + README, 19:45:18) | ok: 26 files, 6.34 GB, 0 mismatches | 159 s |
| 7 | Long-finger robot asset (after step 3c's robot-assets extraction, so the stock bundle was not skipped) | `python -I scripts/todo1/download_pinned.py robot third_party/ManiGuard/behavior-1k/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger` (20:19) | ok: 8 files, 0.04 GB, 0 mismatches; `usd/` present where `maniguard._omnigibson_patches` looks | 7 s |
| 8 | Evaluator deps the upstream guide leaves implicit, then ManiGuard (20:27) | `conda install -c conda-forge spot "numpy<2"; pip install -e openpi/packages/openpi-client; pip install -e "ManiGuard[serve]"` in the `behavior` env | ok: spot 2.16, openpi-client, maniguard 0.1.0.dev598+g6bf4c61a1; numpy stays 1.26.4 | 100 s |

Wall time for item 2, first command to last: 19:45 to 20:29 (about 44 min, with the
installer, OpenPI sync and the three downloads running in parallel tmux sessions).

## Hash verification

- Benchmark, checkpoint and robot asset: every file checked against the Hub manifest
  (LFS sha256, or git blob sha1 for small files) by `scripts/todo1/download_pinned.py`;
  the per-file records are in `download-provenance.json` beside each download. Zero
  mismatches, zero missing.
- BEHAVIOR archives: upstream's installer downloads them to a temp dir and deletes the
  zip after extraction, so the local archive hash could not be taken. Hub manifest
  values at revision `9f0d57d4`: `behavior-1k-assets-3.7.2rc1.zip` 31,280,662,307 bytes,
  sha256 `3bd7cdc1a8c5621768c2aaaf57c12a024560ab4f49d0162e540e769d8b165e8b`;
  `omnigibson-robot-assets.zip` 501,401,227 bytes, sha256
  `5e9fe726172c8b4beb4ba4e12dbf5cf6c5791553a05c52be661eb2d6a3d241b8`. The extracted
  `behavior-1k-assets/VERSION` reads `3.7.2rc1`, matching the archive OmniGibson 3.7.2
  requests. `hf_hub_download` verifies LFS sha256 on download.

## Disk use after install

| Item | Size |
|---|---|
| `behavior` conda env (Isaac Sim, torch, OmniGibson) | 19 GB |
| OpenPI `.venv` | 7.7 GB |
| BEHAVIOR assets (`behavior-1k-assets`) | 33 GB |
| OmniGibson robot assets incl. long-finger | 2.4 GB |
| Jar checkpoint step 7400 | 6.0 GB |
| Benchmark `jar_transport` | 0.05 GB |
| Total used on `/` | 79 GB of 200 GB |

Environment freezes: [01-env-behavior-freeze.txt](01-env-behavior-freeze.txt) (298
lines) and [01-env-openpi-freeze.txt](01-env-openpi-freeze.txt) (240 lines).

## Policy server check (not yet an episode)

`scripts/todo1/serve_jar.sh` runs upstream's `python -m maniguard.serve.openpi_native
--config pi05-base_datagen_v1_jar_joint_2cam_lora --checkpoint …/7400` from the OpenPI
venv with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `MEM_FRACTION=0.45` so the simulator
can share the single GPU. It restored the checkpoint in 9.6 s, loaded the norm stats
from the checkpoint's `assets/`, and advertises `serve_config` and `checkpoint` in its
handshake. Resident: 8.5 GB.

Finding: the first query after a server start blocks on the JAX JIT compile for well over
the openpi client's 20 s keepalive, so that connection is dropped with
`keepalive ping timeout` while the compile completes server-side. Later queries answer
in about 220 ms with a `(16, 8)` float chunk. `scripts/todo1/probe_server.py` absorbs
that compile with synthetic 256 px observations before the official runner connects.
Model queries spent on this so far: 1 compile query (connection dropped) + 3 probe
queries = 4, none in an episode.

## Deviations and notes

- `CONDA_ENVS_PATH=/opt/miniforge3/envs` for the installer (image-specific conda layout).
- `nvidia-curobo` pre-built with `--no-build-isolation` at the exact commit OmniGibson
  pins; otherwise the upstream installer and flags are unchanged.
- Spot and `openpi-client` are required by `maniguard.eval.benchmark` (LTL monitor is
  mandatory for safety scoring; `use_openpi_client: true` is the default) but are not in
  the upstream install page; installed as above.
- JoyLo's install transiently raised numpy to 2.2.6; upstream's `--eval` step
  (`conda install av "numpy<2"`) returned it to 1.26.4 before anything ran.
- No simulator episode has run yet; that is item 4.
