#!/usr/bin/env bash
# Run task 15 stage 0 on the rental with the OpenPI interpreter, alone on the GPU.
#   bash experiments/scripts/stage0_regrasp15.sh <records-sha256> [steer args...]
set -euo pipefail
ROOT=/workspace/Safety-Dial
SHA="$1"; shift
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"
export MANIGUARD_SKIP_OMNIGIBSON_PATCH=1      # no simulator in the policy process
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="${ROOT}/third_party/ManiGuard"
cd "${ROOT}/third_party/ManiGuard"
exec "${ROOT}/third_party/openpi/.venv/bin/python" -u "${ROOT}/experiments/scripts/steer_regrasp15.py" \
    --sha256 "${SHA}" "$@"
