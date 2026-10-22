#!/usr/bin/env bash
# One policy server per GPU lane: same command as scripts/todo1/serve_jar.sh (upstream's
# python -m maniguard.serve.openpi_native from the OpenPI venv), parameterised by lane.
#
#   GPU=0 PORT=8000 bash scripts/todo1b/serve_jar_lane.sh
#   GPU=1 PORT=8001 bash scripts/todo1b/serve_jar_lane.sh
set -euo pipefail
ROOT=/workspace/Safety-Dial/third_party
CKPT="${CKPT:-${ROOT}/checkpoints/jar/7400}"
[[ -d "${CKPT}/params" ]] || { echo "missing checkpoint params at ${CKPT}" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
export MANIGUARD_SKIP_OMNIGIBSON_PATCH=1      # no simulator in the policy process
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="${ROOT}/ManiGuard${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}/ManiGuard"
echo "SERVER lane gpu=${CUDA_VISIBLE_DEVICES} port=${PORT:-8000} start $(date -u +%FT%TZ)"
exec "${ROOT}/openpi/.venv/bin/python" -u -m maniguard.serve.openpi_native \
    --config pi05-base_datagen_v1_jar_joint_2cam_lora \
    --checkpoint "${CKPT}" --host 127.0.0.1 --port "${PORT:-8000}" "$@"
