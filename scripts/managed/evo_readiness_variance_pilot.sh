#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/Safety-Dial
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
pty uv run python -u experiments/scripts/evo_readiness_variance_pilot.py --execute --run-id walker2d-evo-s5-variance-pilot-20261004-1 2>&1
