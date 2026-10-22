#!/usr/bin/env bash
# One GPU lane of the Jar replication: the lane wrapper (scripts/todo1/eval_family_lane.sh,
# upstream eval_family.sh plus PORT and TASKS) over the 26 base tasks at one policy seed.
#
#   GPU=0 PORT=8000 bash scripts/todo1b/run_jar_seed_lane.sh 1
#   GPU=1 PORT=8001 bash scripts/todo1b/run_jar_seed_lane.sh 2
set -euo pipefail
SEED="${1:?usage: run_jar_seed_lane.sh <seed> [output_leaf]}"
LEAF="${2:-jar_transport_joint_legacy256/seed${SEED}}"
ROOT=/workspace/Safety-Dial/third_party
export PYTHON_CMD="${PYTHON_CMD:-/opt/miniforge3/envs/behavior/bin/python}"
export LEVELS=base
export SEED PORT="${PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
# Host-specific (instance C): the NVIDIA display libs live in /opt/nvidia-drivers and the
# image ships only the Mesa EGL vendor JSON, so GLVND dispatched EGL to Mesa and the
# NVIDIA Vulkan ICD failed with ERROR_INCOMPATIBLE_DRIVER. Select the NVIDIA EGL vendor
# explicitly (same repair as archive/20261022 platform-v2 on this machine).
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/workspace/Safety-Dial/scripts/todo1b/nvidia-egl-vendor.json}"
cd "${ROOT}/ManiGuard"
echo "START seed=${SEED} leaf=${LEAF} gpu=${CUDA_VISIBLE_DEVICES} port=${PORT} $(date -u +%FT%TZ)"; SECONDS=0
bash scripts/eval_family_lane.sh jar_transport "${LEAF}" configs/eval/jar_transport_joint_legacy256.yaml
rc=$?
echo "EXIT=${rc} ELAPSED=${SECONDS}s END $(date -u +%FT%TZ)"
exit "${rc}"
