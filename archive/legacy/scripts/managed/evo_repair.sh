#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
set -eo pipefail
cd /workspace/Safety-Dial
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
pty timeout --signal=TERM --kill-after=5s 65m uv run python -u experiments/scripts/evo_repair.py --execute --run-id walker2d-evo-s6-repair-20261005-1 2>&1
