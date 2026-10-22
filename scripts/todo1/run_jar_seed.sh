#!/usr/bin/env bash
# Run the official family runner for the 26 Jar base tasks at one policy seed,
# with the paper-protocol legacy 256 px config. Nothing here changes the runner.
#
#   bash scripts/todo1/run_jar_seed.sh <seed> [output_leaf]
#
# Logs: third_party/ManiGuard/outputs/eval_logs/<leaf>/ (runner) and
#       third_party/logs/eval-<leaf>.log (this wrapper, with timing).
set -euo pipefail
SEED="${1:?usage: run_jar_seed.sh <seed> [output_leaf]}"
LEAF="${2:-jar_transport_joint_legacy256/seed${SEED}}"
ROOT=/workspace/Safety-Dial/third_party
export PYTHON_CMD="${PYTHON_CMD:-/opt/miniforge3/envs/behavior/bin/python}"
export LEVELS=base
export SEED
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${ROOT}/ManiGuard"
echo "START seed=${SEED} leaf=${LEAF} $(date -u +%FT%TZ)"; SECONDS=0
bash scripts/eval_family.sh jar_transport "${LEAF}" configs/eval/jar_transport_joint_legacy256.yaml
rc=$?
echo "EXIT=${rc} ELAPSED=${SECONDS}s END $(date -u +%FT%TZ)"
exit "${rc}"
