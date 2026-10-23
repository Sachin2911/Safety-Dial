# Todo 3, item 1: host ready (instance I)

Vast instance 57411902 (machine 27657, Oman; `progress/rental-20261022.json` role I), template
"ManiGuard latent evolution" (`62414e8fca6befba085051de7ed36190`), 2x RTX 3090 24 GB, driver
**580.126.09** (older than the 580.173.02 of instance D and the 580.178.04 of B/C; the
prepare script's host check accepted it), CUDA 12.4 image, 96 threads, 944 GB RAM, 120 GB disk
(44 GB free after the install), `workspace_is_volume: false`. Repository at
`/workspace/Safety-Dial`, everything else under `third_party/` (gitignored). Times UTC,
23 October 2026.

## Preparation: `scripts/stage2/prepare_vast_host.sh` in tmux `prepare`

Started from the workstation at 19:00:03, **PREPARE DONE at 19:43:06 (43 min)**. The agent
found it running and waited; nothing was started twice. Phase log in
`03-logs/prepare-phases.txt`, per-phase logs `03-logs/prepare-<phase>.log.txt`.

| phase | result | time |
|---|---|---|
| host_check (driver, OptiX and EGL libraries resolve, Vulkan ICD present) | ok, no display-driver repair needed: `libnvoptix.so.1`, `libnvidia-rtcore.so.580.126.09`, `libGLX_nvidia.so.0`, `libEGL_nvidia.so.0` already in `/lib/x86_64-linux-gnu`; `/etc/vulkan/icd.d/nvidia_icd.json` present; `render: {gl, optix, vulkan} = true` | 1 s |
| clone_maniguard (`6bf4c61`, submodule `88454bd`) | ok | 148 s |
| download_benchmark (`af09438c`) | ok: 792 files, 50.4 MB, 0 mismatches, 0 missing | 211 s |
| openpi_env (`215abfb`, `uv sync`, jax sees both GPUs) | ok | 244 s |
| apt_gl (the X11/GL packages of `01-smoke.md`) | ok: 7 newly installed (libice6, libxdamage1, libopengl0, libglu1-mesa, libxcomposite1, libsm6, libxt6), the same set as C and D | 5 s |
| download_jar (`1d84eda0`, step 7400) | ok: 26 files, 6.34 GB, 0 mismatches, 0 missing | 83 s |
| behavior_install (`scripts/todo1b/install_behavior_replay.sh` unchanged) | ok: OmniGibson 3.7.2, Isaac Sim 4.5.0.0, BDDL 3.7.0, JoyLo 3.7.0, curobo `cbaf7d32`, torch-cluster 1.6.3+pt26cu124, assets `3.7.2rc1` (36 GB) | 2097 s |
| download_robot (`5f65ec0a`) | ok: 8 files, 36.9 MB, 0 mismatches; `usd/franka_panda_longfinger.usda` present | 5 s |
| maniguard_deps (`scripts/todo1b/install_maniguard_deps.sh` unchanged) | ok: spot 2.16, openpi-client 0.1.0, maniguard 0.1.0.dev598+g6bf4c61a1 | 109 s |
| finish (eval config copied, lane wrapper linked, env check, freezes) | ok | 12 s |

## Pins verified

- ManiGuard `6bf4c61a1f168cc46e02eb22c41bcb51815e9726`, `behavior-1k` `88454bd04f75dc57c00ab1f1a00bcde1ff505950`, OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479` (`git rev-parse HEAD`).
- Benchmark, checkpoint and robot asset verified file by file by `download_pinned.py` (`download-provenance.json` beside each: zero mismatches, zero missing).
- `configs/eval/jar_transport_joint_legacy256.yaml` byte-identical to `01-jar-eval.yaml` (`cmp`); `scripts/eval_family_lane.sh` symlinked to `scripts/todo1/eval_family_lane.sh`.
- `behavior` env: Python 3.10.22, numpy 1.26.4, torch 2.6.0+cu124 with 2 CUDA devices, omnigibson 3.7.2, isaacsim 4.5.0.0, spot 2.16, maniguard 0.1.0.dev598+g6bf4c61a1.
- After the install, `pip freeze` of the behavior env differed from D's `01c-env-behavior-freeze.txt` in five packages upstream leaves unpinned (certifi, dm_control 1.0.50, googleapis-common-protos 1.76.0, mujoco 3.17.0, vhacdx 0.1.1). As D did in its step 9, the four non-certificate ones were pinned back (`pip install dm_control==1.0.49 mujoco==3.16.0 googleapis-common-protos==1.75.5 vhacdx==0.1.0`, 19:45). `03-env-behavior-freeze.txt` (298 lines) now differs from D's only in `certifi` (2026.10.23 vs 2026.10.19). `03-env-openpi-freeze.txt` (240 lines) is identical to D's. (The prepare script's own openpi freeze had been taken from the activated behavior env by mistake; it was regenerated with `uv pip freeze --python .venv/bin/python`.)

## EGL vendor on this host

The lane scripts of todo 1b/1c default `__EGL_VENDOR_LIBRARY_FILENAMES` to
`scripts/todo1b/nvidia-egl-vendor.json`, which points at `/opt/nvidia-drivers/lib64/libEGL_nvidia.so.0`;
that directory does not exist here (the libraries are in the system path and
`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` already selects NVIDIA). The todo 3 wrappers
(`scripts/todo3/run_episode_lane.sh`, `run_gate_lane.py`) select
`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` when `/opt/nvidia-drivers` is absent. No other
host-specific change.

## Policy servers

`scripts/todo1b/serve_jar_lane.sh` unchanged, started 19:43:45 in tmux `serve0` (GPU 0, port 8000)
and `serve1` (GPU 1, port 8001), logs `third_party/logs/03-serve0.log`, `03-serve1.log`; GPU sampler
(`scripts/todo1b/gpu_sampler.sh`, 5 s) in tmux `sampler` to `third_party/logs/03-gpu-samples.csv`.
Warm-up and query accounting are in `03-identity.md`.
