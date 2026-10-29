#!/usr/bin/env bash
# Task 15 recovery pipeline on the recovery rental: one policy server and one sequential
# evaluator lane. Stage A engineering; then, once the workstation writes the measured
# re-admission, stage C; then stage D after a semantically complete stage C.
#   bash experiments/scripts/launch_regrasp15r1.sh progress/receipts/15-regrasp-recovery-live-v1.json
set -euo pipefail
ROOT=/workspace/Safety-Dial
cd "$ROOT"
PLAN="${1:?recovery live freeze}"
OUT="$ROOT/outputs/regrasp15r1"
BEHAVIOR=/opt/miniforge3/envs/behavior/bin/python
OPENPI="$ROOT/third_party/openpi/.venv/bin/python"
mkdir -p "$OUT/logs" "$OUT/source" "$OUT/server"
test ! -e "$OUT/pipeline-started.txt"
date -u +%FT%TZ > "$OUT/pipeline-started.txt"
cleanup() {
    rc=$?
    trap - EXIT
    tmux kill-session -t serve15r1 2>/dev/null || true
    echo "$rc" > "$OUT/pipeline-exit-code.txt"
    date -u +%FT%TZ > "$OUT/pipeline-ended.txt"
    exit "$rc"
}
trap cleanup EXIT

while [ ! -f "$OUT/prepare-complete.txt" ]; do
    if [ -f "$OUT/prepare-failed.txt" ] || ! tmux has-session -t prepare15r1 2>/dev/null; then
        echo "Preparation stopped without successful completion"
        exit 1
    fi
    sleep 15
done
python3 - "$PLAN" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
plan=json.loads(Path(sys.argv[1]).read_text())
rental=json.loads(Path('progress/rental-regrasp15r1-20261026.json').read_text())
assert os.environ.get('CONTAINER_ID') == str(rental['instance_id'])
assert plan['stage'] == 'recovery_live' and plan['authorization']['user_launch']
for name,expected in plan['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
provenance=json.loads(Path('outputs/regrasp15r1/environment/checkpoint-provenance.json').read_text())
text=json.dumps(provenance)
assert '1d84eda070313a202a595e449fcf41a1a1e8a546' in text and not provenance['mismatches'], 'checkpoint'
print('Frozen recovery source and checkpoint provenance verified before any simulator start',
      flush=True)
PY
K=$(python3 -c "import json;print(json.load(open('configs/regrasp15/design.json'))['k'])")
python3 experiments/scripts/patch_regrasp15.py third_party/ManiGuard \
    --diff "$OUT/source/evaluator-hook.diff"
cmp progress/receipts/01-jar-eval.yaml configs/regrasp15/jar_eval.yaml
tmux new-session -d -s serve15r1 "cd $ROOT/third_party/ManiGuard && \
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
"$BEHAVIOR" -u experiments/scripts/run_regrasp15r1.py --plan "$PLAN" --stage stage_a \
    > "$OUT/logs/stage-a-queue.log" 2>&1
date -u +%FT%TZ > "$OUT/engineering-complete.txt"
# The workstation measures the new host's speed from stage A and writes admission-c.json.
while [ ! -f "$OUT/admission-c.json" ]; do
    if [ -f "$OUT/STOP" ]; then
        echo "Stopped before stage C"
        exit 4
    fi
    sleep 15
done
"$BEHAVIOR" -u experiments/scripts/run_regrasp15r1.py --plan "$PLAN" --stage stage_c \
    > "$OUT/logs/stage-c-queue.log" 2>&1
date -u +%FT%TZ > "$OUT/stage-c-ended.txt"
set +e
"$BEHAVIOR" -u experiments/scripts/run_regrasp15r1.py --plan "$PLAN" --stage stage_d \
    > "$OUT/logs/stage-d-queue.log" 2>&1
rc=$?
set -e
date -u +%FT%TZ > "$OUT/stage-d-ended.txt"
# Exit 3 is an orderly incomplete stage D (the declared admission rule); anything else fails.
if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then
    exit "$rc"
fi
date -u +%FT%TZ > "$OUT/study-complete.txt"
