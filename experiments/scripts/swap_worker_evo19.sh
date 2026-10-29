#!/usr/bin/env bash
# Restart the task 19 host worker between episodes so it loads updated helpers
# (amendment 3). STOP makes the worker exit after its current episode, and the
# orchestrator assigns nothing while STOP exists; queued jobs stay for the new worker.
set -u
ROOT=/workspace/Safety-Dial
OUT="$ROOT/outputs/evo19"
BEHAVIOR=/opt/miniforge3/envs/behavior/bin/python
touch "$OUT/STOP"
while pgrep -f "experiments/scripts/worker_evo19.py" > /dev/null; do sleep 2; done
rm -f "$OUT/STOP"
tmux kill-session -t worker19 2>/dev/null || true
tmux new-session -d -s worker19 \
  "cd $ROOT && $BEHAVIOR -u experiments/scripts/worker_evo19.py >> $OUT/logs/worker.log 2>&1"
echo "$(date -u +%FT%TZ) worker restarted with $(sha256sum $ROOT/experiments/helpers/evo19_runtime.py | cut -c1-12)" >> "$OUT/logs/swap.log"
