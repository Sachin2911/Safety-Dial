#!/usr/bin/env bash
# One sequential task-8 policy/evaluator lane on its own recorded rental.
set -euo pipefail
ROOT=/workspace/Safety-Dial
cd "$ROOT"
OUT="$ROOT/outputs/screen08"
BEHAVIOR=/opt/miniforge3/envs/behavior/bin/python
OPENPI="$ROOT/third_party/openpi/.venv/bin/python"
mkdir -p "$OUT/logs" "$OUT/source"
test ! -e "$OUT/pipeline-started.txt"
date -u +%FT%TZ > "$OUT/pipeline-started.txt"
cleanup() {
    rc=$?
    trap - EXIT
    tmux kill-session -t serve08 2>/dev/null || true
    echo "$rc" > "$OUT/pipeline-exit-code.txt"
    date -u +%FT%TZ > "$OUT/pipeline-ended.txt"
    exit "$rc"
}
trap cleanup EXIT

while [ ! -f "$OUT/prepare-complete.txt" ]; do
    if ! tmux has-session -t prepare08 2>/dev/null; then
        echo "Preparation stopped without successful completion"
        exit 1
    fi
    sleep 15
done
python3 - <<'PY'
import hashlib,json,os
from pathlib import Path
plan=json.loads(Path('progress/receipts/08-screen-plan.json').read_text())
rental=json.loads(Path('progress/rental-screen08-20261025.json').read_text())
assert os.environ.get('CONTAINER_ID') == str(rental['instance_id'])
for name,expected in plan['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
print('Frozen source verified before any simulator start',flush=True)
PY
python3 experiments/scripts/patch_screen08.py third_party/ManiGuard \
    --diff "$OUT/source/evaluator-hook.diff"
cmp progress/receipts/01-jar-eval.yaml configs/screen08/native.yaml
tmux new-session -d -s serve08 \
    "cd $ROOT && GPU=0 PORT=8000 bash scripts/todo1b/serve_jar_lane.sh > $OUT/logs/server.log 2>&1"
python3 - <<'PY'
import socket,time
deadline=time.time()+900
while time.time()<deadline:
    try:
        with socket.create_connection(('127.0.0.1',8000),timeout=2):
            break
    except OSError:
        time.sleep(2)
else:
    raise RuntimeError('Policy server never opened its socket')
PY
"$OPENPI" -I experiments/scripts/probe_screen08.py --queries 2 \
    > "$OUT/logs/startup-probe.log" 2>&1
"$BEHAVIOR" -u experiments/scripts/run_screen08.py --stage engineering \
    > "$OUT/logs/engineering-queue.log" 2>&1
"$BEHAVIOR" -u experiments/scripts/run_screen08.py --stage stage_a \
    > "$OUT/logs/stage-a-queue.log" 2>&1
"$BEHAVIOR" -u experiments/scripts/decide_screen08.py \
    > "$OUT/logs/confirmation-decision.log" 2>&1
if python3 - <<'PY'
import json
raise SystemExit(0 if json.load(open('outputs/screen08/stage-b-decision.json'))['admit'] else 1)
PY
then
    "$BEHAVIOR" -u experiments/scripts/run_screen08.py --stage stage_b \
        > "$OUT/logs/stage-b-queue.log" 2>&1
fi
date -u +%FT%TZ > "$OUT/study-complete.txt"
