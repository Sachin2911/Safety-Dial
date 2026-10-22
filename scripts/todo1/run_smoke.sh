#!/usr/bin/env bash
# Smoke test: one Jar base episode through the official evaluator with exactly the
# arguments scripts/eval_family.sh passes per instance, tagged SMOKE so it lands in its
# own run directory. Requires the policy server on 127.0.0.1:8000 (warmed by probe_server.py).
#
#   bash scripts/todo1/run_smoke.sh [task_dir] [seed]
set -uo pipefail
TASK="${1:-task_0000/base}"
SEED="${2:-0}"
ROOT=/workspace/Safety-Dial/third_party
PY=/opt/miniforge3/envs/behavior/bin/python
export OMNIGIBSON_HEADLESS=1 NVIDIA_DRIVER_CAPABILITIES=all OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json   # image puts the NVIDIA ICD here
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${ROOT}/ManiGuard"
echo "START smoke task=${TASK} seed=${SEED} $(date -u +%FT%TZ)"; SECONDS=0
"${PY}" -m maniguard.eval.benchmark --config configs/eval/jar_transport_joint_legacy256.yaml \
    --host 127.0.0.1 --port 8000 \
    --benchmark-root outputs/lerobot_datasets/maniguard-bench/jar_transport \
    --scenes "${TASK}" --max-scenes 1 --tag smoke \
    --metrics success safety --headless --seed "${SEED}"
rc=$?
echo "EXIT=${rc} ELAPSED=${SECONDS}s END $(date -u +%FT%TZ)"
exit "${rc}"
