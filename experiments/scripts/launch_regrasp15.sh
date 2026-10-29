#!/usr/bin/env bash
# Task 15 live pipeline on its recorded rental: one policy server and one sequential
# evaluator lane. Stage A engineering, stage B dial screen, the stage C decision and
# stage C confirmation, each admitted only after its predecessor.
#   bash experiments/scripts/launch_regrasp15.sh progress/receipts/15-regrasp-live-v1.json
set -euo pipefail
ROOT=/workspace/Safety-Dial
cd "$ROOT"
PLAN="${1:?live freeze}"
OUT="$ROOT/outputs/regrasp15"
BEHAVIOR=/opt/miniforge3/envs/behavior/bin/python
OPENPI="$ROOT/third_party/openpi/.venv/bin/python"
mkdir -p "$OUT/logs" "$OUT/source" "$OUT/server"
test ! -e "$OUT/pipeline-started.txt"
date -u +%FT%TZ > "$OUT/pipeline-started.txt"
cleanup() {
    rc=$?
    trap - EXIT
    tmux kill-session -t serve15 2>/dev/null || true
    echo "$rc" > "$OUT/pipeline-exit-code.txt"
    date -u +%FT%TZ > "$OUT/pipeline-ended.txt"
    exit "$rc"
}
trap cleanup EXIT

while [ ! -f "$OUT/prepare-complete.txt" ]; do
    if [ -f "$OUT/prepare-failed.txt" ] || ! tmux has-session -t prepare15 2>/dev/null; then
        echo "Preparation stopped without successful completion"
        exit 1
    fi
    sleep 15
done
python3 - "$PLAN" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
plan=json.loads(Path(sys.argv[1]).read_text())
rental=json.loads(Path('progress/rental-regrasp15-20261025.json').read_text())
assert os.environ.get('CONTAINER_ID') == str(rental['instance_id'])
assert plan['stage'] == 'live' and plan['authorization']['user_launch']
for name,expected in plan['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
print('Frozen source verified before any simulator start',flush=True)
PY
K=$(python3 -c "import json;print(json.load(open('configs/regrasp15/design.json'))['k'])")
python3 experiments/scripts/patch_regrasp15.py third_party/ManiGuard \
    --diff "$OUT/source/evaluator-hook.diff"
cmp progress/receipts/01-jar-eval.yaml configs/regrasp15/jar_eval.yaml
tmux new-session -d -s serve15 "cd $ROOT/third_party/ManiGuard && \
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 MANIGUARD_SKIP_OMNIGIBSON_PATCH=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    PYTHONPATH=$ROOT/third_party/ManiGuard \
    $OPENPI -u $ROOT/experiments/scripts/serve_regrasp15.py \
    --checkpoint $ROOT/third_party/checkpoints/jar/7400 \
    --ledger $OUT/server/candidates.jsonl > $OUT/logs/server.log 2>&1"
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
"$OPENPI" -I experiments/scripts/probe_regrasp15.py --warmup --k "$K" \
    > "$OUT/logs/startup-probe.log" 2>&1
"$BEHAVIOR" -u experiments/scripts/run_regrasp15.py --plan "$PLAN" --stage stage_a \
    > "$OUT/logs/stage-a-queue.log" 2>&1
"$BEHAVIOR" -u experiments/scripts/run_regrasp15.py --plan "$PLAN" --stage stage_b \
    > "$OUT/logs/stage-b-queue.log" 2>&1
"$BEHAVIOR" -u experiments/scripts/analyze_regrasp15.py --decide \
    > "$OUT/logs/stage-c-decision.log" 2>&1
"$BEHAVIOR" -u experiments/scripts/run_regrasp15.py --plan "$PLAN" --stage stage_c \
    > "$OUT/logs/stage-c-queue.log" 2>&1
"$BEHAVIOR" -u experiments/scripts/analyze_regrasp15.py \
    > "$OUT/logs/analysis.log" 2>&1
date -u +%FT%TZ > "$OUT/study-complete.txt"
