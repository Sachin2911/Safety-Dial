#!/usr/bin/env bash
# Lane smoke test: scripts/todo1/run_smoke.sh with the GPU and server port chosen per lane.
# Exactly the per-instance arguments of scripts/eval_family.sh, tagged smoke.
#
#   GPU=0 PORT=8000 bash scripts/todo1b/run_smoke_lane.sh task_0000/base 1
#   GPU=1 PORT=8001 bash scripts/todo1b/run_smoke_lane.sh task_0000/base 2
set -uo pipefail
TASK="${1:-task_0000/base}"
SEED="${2:-0}"
ROOT=/workspace/Safety-Dial/third_party
PY=/opt/miniforge3/envs/behavior/bin/python
export OMNIGIBSON_HEADLESS=1 NVIDIA_DRIVER_CAPABILITIES=all OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
# Host-specific (instance C): the NVIDIA display libs live in /opt/nvidia-drivers and the
# image ships only the Mesa EGL vendor JSON, so GLVND dispatched EGL to Mesa and the
# NVIDIA Vulkan ICD failed with ERROR_INCOMPATIBLE_DRIVER. Select the NVIDIA EGL vendor
# explicitly (same repair as archive/20261022 platform-v2 on this machine).
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/workspace/Safety-Dial/scripts/todo1b/nvidia-egl-vendor.json}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
cd "${ROOT}/ManiGuard"
echo "START smoke task=${TASK} seed=${SEED} gpu=${CUDA_VISIBLE_DEVICES} port=${PORT:-8000} $(date -u +%FT%TZ)"; SECONDS=0
"${PY}" -m maniguard.eval.benchmark --config configs/eval/jar_transport_joint_legacy256.yaml \
    --host 127.0.0.1 --port "${PORT:-8000}" \
    --benchmark-root outputs/lerobot_datasets/maniguard-bench/jar_transport \
    --scenes "${TASK}" --max-scenes 1 --tag smoke \
    --metrics success safety --headless --seed "${SEED}"
rc=$?
echo "EXIT=${rc} ELAPSED=${SECONDS}s END $(date -u +%FT%TZ)"
exit "${rc}"
