#!/usr/bin/env bash
# Todo 2 item 7: one Jar base episode through the official evaluator with the per-step
# recorder attached (scripts/todo2/record_rollout.py on the observer hook), tagged smoke.
# Same evaluator arguments as scripts/todo1/run_smoke.sh; RunPod paths. Requires the policy
# server on 127.0.0.1:8000 warmed by scripts/todo1/probe_server.py.
#
#   bash scripts/todo2/run_smoke_pod.sh [task_dir] [seed] [recording_dir]
set -uo pipefail
TASK="${1:-task_0000/base}"
SEED="${2:-0}"
REC="${3:-/workspace/recordings/smoke}"
ROOT=/workspace/Safety-Dial/third_party
PY=/workspace/miniforge3/envs/behavior/bin/python
export OMNIGIBSON_HEADLESS=1 NVIDIA_DRIVER_CAPABILITIES=all OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/workspace/Safety-Dial/scripts/todo2${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}/ManiGuard"
echo "START smoke task=${TASK} seed=${SEED} recorder=${REC} $(date -u +%FT%TZ)"; SECONDS=0
"${PY}" -m maniguard.eval.benchmark --config configs/eval/jar_transport_joint_legacy256.yaml \
    --host 127.0.0.1 --port 8000 \
    --benchmark-root outputs/lerobot_datasets/maniguard-bench/jar_transport \
    --scenes "${TASK}" --max-scenes 1 --tag smoke \
    --metrics success safety --headless --seed "${SEED}" \
    --recording-factory record_rollout:create_observer --recording-output-dir "${REC}"
rc=$?
echo "EXIT=${rc} ELAPSED=${SECONDS}s END $(date -u +%FT%TZ)"
exit "${rc}"
