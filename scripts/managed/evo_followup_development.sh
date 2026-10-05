#!/bin/bash
set -euo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/Safety-Dial
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
pty timeout --signal=TERM --kill-after=5s 45m uv run python -u experiments/scripts/evo_followup.py --protocol configs/evo/stage6_development_20261005.yaml --execute --run-id walker2d-evo-s6-development-20261005-1 2>&1
