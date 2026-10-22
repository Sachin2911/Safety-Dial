#!/usr/bin/env bash
# Serve the pinned Jar pi0.5 checkpoint with ManiGuard's official native server
# (python -m maniguard.serve.openpi_native) from the OpenPI uv environment.
# The simulator shares the single RTX 3090, so JAX must not preallocate.
set -euo pipefail
ROOT=/workspace/Safety-Dial/third_party
CKPT="${CKPT:-${ROOT}/checkpoints/jar/7400}"
[[ -d "${CKPT}/params" ]] || { echo "missing checkpoint params at ${CKPT}" >&2; exit 1; }
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
export MANIGUARD_SKIP_OMNIGIBSON_PATCH=1      # no simulator in the policy process
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="${ROOT}/ManiGuard${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}/ManiGuard"
exec "${ROOT}/openpi/.venv/bin/python" -u -m maniguard.serve.openpi_native \
    --config pi05-base_datagen_v1_jar_joint_2cam_lora \
    --checkpoint "${CKPT}" --host 127.0.0.1 --port 8000 "$@"
