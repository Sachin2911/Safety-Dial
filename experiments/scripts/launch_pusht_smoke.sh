#!/usr/bin/env bash
# New smoke test only. The independent existing pod lease guard stays in force.
set -euo pipefail
source /workspace/activate-lewm.sh
cd /workspace/Safety-Dial
mkdir -p outputs/pusht-hazards-20261029
trap 'task_exit=$?; printf "%s\n" "$task_exit" > outputs/pusht-hazards-20261029/launch-exit.txt' EXIT

python -u experiments/scripts/prepare_pusht_smoke.py \
  --data /workspace/Safety-Dial/data/pusht-smoke-20261029 \
  --scratch /tmp/pusht-smoke-20261029 \
  > outputs/pusht-hazards-20261029/preparation-split.log 2>&1

timeout 3600 python -u experiments/scripts/run_pusht_smoke.py \
  --data /workspace/Safety-Dial/data/pusht-smoke-20261029 \
  --out /workspace/Safety-Dial/outputs/pusht-hazards-20261029 \
  > outputs/pusht-hazards-20261029/run.log 2>&1
