# Todo 2, item 7: host check, install replay and the renderer crash

Pod `f1o65q3iygv372`, RunPod secure cloud, one NVIDIA A40 48 GB, driver **595.91.07 (open
kernel module, CUDA 13.2)**, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
Ubuntu 22.04.5, kernel 6.8.0-138, 48 cores (quota 7.65), 503 GB RAM, `/workspace` a 150 GB
network volume (`mfs`). Times UTC, 23 October 2026.

## Pre-check (passed)

`ldconfig -p` resolves `libnvoptix.so.1`, `libnvidia-rtcore.so.595.91.07`,
`libGLX_nvidia.so.0`, `libEGL_nvidia.so.0` (all in `/lib/x86_64-linux-gnu`). The NVIDIA
Vulkan ICD is `/etc/vulkan/icd.d/nvidia_icd.json` (`libGLX_nvidia.so.0`, API 1.4.329);
`/usr/share/vulkan/icd.d` does not exist. The GLVND EGL vendor file
`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` is present, so the Mesa dispatch fix from
`scripts/todo1b/run_smoke_lane.sh` is not needed. `vulkaninfo --summary` enumerates the A40
as a discrete GPU. OptiX and EGL exist, so by the checklist the install went ahead.

## Install replay (`scripts/todo2/install_runpod.sh`, then `install_runpod_datasets.sh`)

Same pins as `01-sources.json` and the same commands as `01b-install.md`, with Miniforge at
`/workspace/miniforge3` (`CONDA_ENVS_PATH=/workspace/miniforge3/envs`), all caches under
`/workspace/.cache`, and `uv` installed with `pip`. Logs under `progress/receipts/02-logs/`.

| stage | what | result | time |
|---|---|---|---|
| A1 | Miniforge (latest installer, conda 26.7.2) | ok | 182 s |
| A2 | ManiGuard at `6bf4c61` (cloned for item 6), submodule `behavior-1k` at `88454bd` | ok | 256 s |
| A3 | OpenPI `215abfb`, `uv sync`, `uv pip install -e .` | ok, Python 3.11, jax 0.5.3 sees the A40 | 197 s |
| A4 | the X11/GL apt packages of `01-smoke.md` plus `libvulkan1 vulkan-tools` | ok | 7 s |
| A5 | benchmark `af09438c` via `download_pinned.py` | 792 files, 0 mismatches | 171 s |
| A6 | Jar checkpoint `1d84eda0` step 7400 | 26 files, 6.34 GB, 0 mismatches | 39 s |
| B | behavior env (conda create, numpy<2, torch 2.6.0 cu124), curobo `cbaf7d32` prebuilt, upstream `setup.sh --omnigibson --bddl --joylo --dataset --eval --primitives` | OmniGibson, Isaac Sim 4.5.0, BDDL, JoyLo, torch-cluster installed; **the `--dataset` step failed** (below) | 352 + 410 + 1,904 s |
| C | long-finger robot `5f65ec0a`, `conda install spot numpy<2`, `openpi-client`, `ManiGuard[serve]`, eval config copied | ok (spot 2.16, maniguard 0.1.0.dev598+g6bf4c61a1) | 660 s |
| D | numpy repair, robot-assets bundle, BEHAVIOR-1K assets 3.7.2rc1, 2025 task instances | ok | 100 + 97 + 1,776 + 149 s |

Wall time 12:02 to 13:40, about 98 minutes against the 43 to 45 minutes of instances B
and C; the difference is the network volume (Isaac Sim wheels and the 31 GB asset
archive unpack onto it) and the extra stage D.

**Deviation 1, numpy.** On this conda (26.7.2) the installer's `--eval` step
(`conda install av "numpy<2"`) left pip's `numpy-2.2.6.dist-info` (installed by the
JoyLo step) beside conda's `numpy-1.26.4`, and `import omnigibson` then failed in scipy
with `All ufuncs must have type numpy.ufunc`, so the `--dataset` step aborted before
downloading anything. On instance C (conda 26.5.3) the same sequence ended with a clean
1.26.4. Repair: `pip uninstall numpy` twice, remove the stale `numpy*` directories, `conda
install numpy=1.26.4 --force-reinstall`. After it, `import scipy.spatial, omnigibson`
works and the final check reads numpy 1.26.4, torch 2.6.0+cu124 (1 device), omnigibson
3.7.2, isaacsim 4.5.0.0, spot 2.16.

**Deviation 2, robot assets order.** Stage C had put the long-finger folder under
`datasets/omnigibson-robot-assets/` before the bundle existed, which would have made the
upstream downloader skip the bundle; stage D moved it aside, downloaded the bundle and
put it back (`franka_panda`, `franka_mounted`, `franka_dexhand`, `franka_panda_longfinger`
all present).

**Freezes.** `02-env-openpi-freeze.txt` is identical to `01b-env-openpi-freeze.txt` (240
lines, after normalising the local path). `02-env-behavior-freeze.txt` differs from
`01b-env-behavior-freeze.txt` in packages upstream leaves unpinned and that moved in the
day between installs: certifi, dash, deepdiff, dm_control 1.0.49 to 1.0.50, mujoco 3.16 to
3.17, googleapis-common-protos, scikit-image, typing_extensions; ImageIO's capitalisation;
PyWavelets dropped. Every pinned component (torch, OmniGibson, Isaac Sim wheels, BDDL,
JoyLo, curobo, spot, maniguard, openpi-client, numpy) matches.

## The episode did not run: renderer crash under driver 595

Policy server: `scripts/todo1/serve_jar.sh` (tmux `serve`, 13:04), model loaded, listening
on 8000. `probe_server.py`: the compile query dropped the connection after 50 s and the
reconnect timed out on the handshake while the server was still compiling (instance C's
reconnect at 5 s had succeeded); a second probe run 3 minutes later answered the warm-up
in 0.3 s and three queries in 259 to 310 ms (instance C: 209 to 225 ms on an RTX 3090).
Model queries: 5 (1 compile, 1 warm-up, 3 probe). The server never received an evaluator
query.

`scripts/todo2/run_smoke_pod.sh task_0000/base 0` (the todo 1 smoke arguments plus the
recorder flags) was started four times and segfaulted each time during Isaac Sim
startup, in `SimulationApp.__init__` at `_wait_for_viewport`, 82 to 117 s in, before any
scene was discovered or loaded:

| attempt | start | change | result |
|---|---|---|---|
| 1 | 13:41:34 | none | segfault, 117 s |
| 2 | 13:44:35 | plain retry | segfault, 82 s |
| 3 | 13:47:33 | under `gdb` for the backtrace | segfault; see below |
| 4 | 13:50:20 | `[settings.rtx.sceneDb] maxInstances = 9574251` appended to `omnigibson_4_5_0.kit` | setting not picked up (`MaxInstances from setting -1`), segfault, 111 s; kit file reverted |

The Kit log ends at the RTX scene database: `SceneDbContext : TLAS limit buffer size
7512601600`, `TLAS limit : valid true, within: false`, `TLAS instance limit 16777215`, and
nothing after. Instance C's log at the same point reads `valid false, within: false`, then
`decrement 167690`, `New limit 9574251`, a 4.29 GB buffer, `valid true, within: true`, and
startup completes. The `gdb` backtrace (`02-logs/gdb_smoke.log.txt`) puts the fault in
`librtx.scenedb.plugin.so` called from `libcarb.scenerenderer-rtx.plugin.so` and
`libomni.hydra.rtx.plugin.so` on a `carb.tasking` thread, that is inside Isaac Sim 4.5's
RTX renderer while it sizes its acceleration-structure buffer for this device. Isaac Sim
4.5 lists 535.129.03 as its tested driver; this pod has 595.91.07 with the open kernel
module, ECC on and IOMMU on. The X11/GL libraries that caused instance B's first crash
are present (`libSM.so.6`, `libXt.so.6`, `libGLU.so.1`), and Vulkan enumerates the GPU, so
this is not the todo 1 failure mode.

The checklist forbids installing display drivers in the container, and a driver cannot be
changed inside a pod anyway. Nothing else about the stack is in doubt: the same pins and
commands produced the same environments as instance C, and the server serves. The
recorder (`scripts/todo2/record_rollout.py`) therefore stays validated only against mock
objects (section below in `02-logging.md`) and is to be validated on the next Vast rental
on machine 54514 or another RTX 3090 host with driver 580.x, as the checklist's fallback
says. Run directories left: four `outputs/eval_logs/jar_transport_joint_legacy256/*_SMOKE/`
each holding only `eval_config.json`; no row, no video, no `.npz`.

## Accounting

Simulator interactions: 0 environment steps; 4 failed simulator starts. Model queries: 5
(probe). Elapsed on the GPU for item 7: 13:04 to 13:53. The policy server was stopped at
13:53; `nvidia-smi` shows no compute process and 0 MiB used.
