#!/usr/bin/env bash
# One Jar base episode on one GPU lane through the official evaluator, with the server port
# chosen per lane (direct 8000/8001, or a supervisor proxy 9000/9001 off, 9100/9101 retry).
# The evaluator arguments are those of scripts/todo1b/run_smoke_lane.sh; an optional third
# argument attaches the todo 2 recorder (item 3 only).
#
#   GPU=0 PORT=8000 TAG=identity_direct bash scripts/todo3/run_episode_lane.sh task_0000/base 0 /workspace/recordings/direct
#   GPU=0 PORT=9100 TAG=smoke bash scripts/todo3/run_episode_lane.sh task_0003/base 0
set -uo pipefail
TASK="${1:-task_0000/base}"
SEED="${2:-0}"
REC="${3:-}"
ROOT=/workspace/Safety-Dial/third_party
PY=/opt/miniforge3/envs/behavior/bin/python
export OMNIGIBSON_HEADLESS=1 NVIDIA_DRIVER_CAPABILITIES=all OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
# EGL vendor: instance C/D kept the NVIDIA libs under /opt/nvidia-drivers; on a host where the
# image already ships them in the system path, the GLVND vendor file 10_nvidia.json is used.
if [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]; then
  if [ -f /opt/nvidia-drivers/lib64/libEGL_nvidia.so.0 ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES=/workspace/Safety-Dial/scripts/todo1b/nvidia-egl-vendor.json
  else
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  fi
fi
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
REC_ARGS=()
if [ -n "$REC" ]; then
  export PYTHONPATH="/workspace/Safety-Dial/scripts/todo2${PYTHONPATH:+:$PYTHONPATH}"
  REC_ARGS=(--recording-factory record_rollout:create_observer --recording-output-dir "$REC")
fi
cd "${ROOT}/ManiGuard"
echo "START task=${TASK} seed=${SEED} gpu=${CUDA_VISIBLE_DEVICES} port=${PORT:-8000} tag=${TAG:-smoke} recorder=${REC:-off} $(date -u +%FT%TZ)"; SECONDS=0
"${PY}" -m maniguard.eval.benchmark --config configs/eval/jar_transport_joint_legacy256.yaml \
    --host 127.0.0.1 --port "${PORT:-8000}" \
    --benchmark-root outputs/lerobot_datasets/maniguard-bench/jar_transport \
    --scenes "${TASK}" --max-scenes 1 --tag "${TAG:-smoke}" \
    --metrics success safety --headless --seed "${SEED}" "${REC_ARGS[@]}"
rc=$?
echo "EXIT=${rc} ELAPSED=${SECONDS}s END $(date -u +%FT%TZ)"
exit "${rc}"
