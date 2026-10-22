#!/usr/bin/env bash
# Local policy service; no public listener or portal entry is needed.
set -euo pipefail
STUDY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MANIGUARD_SKIP_OMNIGIBSON_PATCH=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec "${STUDY_ROOT}/third_party/openpi/.venv/bin/python" -u \
    "${STUDY_ROOT}/experiments/scripts/maniguard_serve.py" "$@"
