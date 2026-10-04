#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/Safety-Dial || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
pty uv run python -u experiments/scripts/evo_readiness_main.py --protocol configs/evo/stage5_main_locked_20261004.yaml --execute --run-id walker2d-evo-s5-main-20261004-1 2>&1
