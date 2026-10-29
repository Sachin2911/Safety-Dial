#!/usr/bin/env bash
# Task 19 host pipeline: wait for preparation, apply the evo19 evaluator hook, start the
# pi0.5 Jar policy server, run the startup probe, then start the queue worker.
#   tmux new -d -s launch19 'bash experiments/scripts/launch_evo19.sh'
set -euo pipefail
ROOT=/workspace/Safety-Dial
cd "$ROOT"
OUT="$ROOT/outputs/evo19"
BEHAVIOR=/opt/miniforge3/envs/behavior/bin/python
OPENPI="$ROOT/third_party/openpi/.venv/bin/python"
mkdir -p "$OUT/logs" "$OUT/server" "$OUT/source"
while [ ! -f "$OUT/prepare-complete.txt" ]; do
    if [ -f "$OUT/prepare-failed.txt" ] || ! tmux has-session -t prepare19 2>/dev/null; then
        echo "Preparation stopped without successful completion"; exit 1
    fi
    sleep 10
done
python3 - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('outputs/evo19/environment/checkpoint-provenance.json').read_text())
assert '1d84eda070313a202a595e449fcf41a1a1e8a546' in json.dumps(p) and not p['mismatches']
print('checkpoint provenance verified', flush=True)
PY
"$BEHAVIOR" experiments/scripts/patch_evo19.py third_party/ManiGuard --diff "$OUT/source/evaluator-hook.diff"
cmp progress/receipts/01-jar-eval.yaml configs/regrasp15/jar_eval.yaml
tmux kill-session -t serve19 2>/dev/null || true
tmux new-session -d -s serve19 "cd $ROOT/third_party/ManiGuard && \
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 MANIGUARD_SKIP_OMNIGIBSON_PATCH=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    PYTHONPATH=$ROOT/third_party/ManiGuard \
    $OPENPI -u $ROOT/experiments/scripts/serve_regrasp15.py \
    --checkpoint $ROOT/third_party/checkpoints/jar/7400 \
    --ledger $OUT/server/candidates.jsonl > $OUT/logs/server.log 2>&1"
python3 - <<'PY'
import socket, time
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with socket.create_connection(('127.0.0.1', 8000), timeout=2):
            break
    except OSError:
        time.sleep(2)
else:
    raise RuntimeError('Policy server never opened its socket')
PY
"$OPENPI" -I experiments/scripts/probe_regrasp15.py --warmup --k 16 > "$OUT/logs/startup-probe.log" 2>&1
date -u +%FT%TZ > "$OUT/server-ready.txt"
tmux kill-session -t worker19 2>/dev/null || true
tmux new-session -d -s worker19 "cd $ROOT && $BEHAVIOR -u experiments/scripts/worker_evo19.py > $OUT/logs/worker.log 2>&1"
echo "worker started"
