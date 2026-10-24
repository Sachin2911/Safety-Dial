#!/usr/bin/env bash
# Finish the pinned environment, warm both native servers, then start both lanes.
set -euo pipefail
ROOT=/workspace/Safety-Dial
cd "$ROOT"
OUT="$ROOT/outputs/tracking04"
LOGS="$OUT/logs"
mkdir -p "$OUT/source" "$OUT/environment" "$LOGS"
BEHAVIOR=/opt/miniforge3/envs/behavior/bin/python
OPENPI="$ROOT/third_party/openpi/.venv/bin/python"

while [ ! -f third_party/.prepare/finish ]; do
    if ! tmux has-session -t prepare04 2>/dev/null; then
        echo "Preparation ended without its finish marker; no experiment admitted."
        exit 1
    fi
    sleep 20
done

# Replay the same repairs made on the previous results hosts.
"$BEHAVIOR" -m pip install dm_control==1.0.49 mujoco==3.16.0 \
    googleapis-common-protos==1.75.5 vhacdx==0.1.0 > "$LOGS/pin-repair.log" 2>&1
cmp progress/receipts/01-jar-eval.yaml \
    third_party/ManiGuard/configs/eval/jar_transport_joint_legacy256.yaml
test "$(git -C third_party/ManiGuard rev-parse HEAD)" = \
    6bf4c61a1f168cc46e02eb22c41bcb51815e9726
test "$(git -C third_party/ManiGuard/behavior-1k rev-parse HEAD)" = \
    88454bd04f75dc57c00ab1f1a00bcde1ff505950
test "$(git -C third_party/openpi rev-parse HEAD)" = \
    215abfb217dbac7d5f1273282331b9b1866c0479
"$BEHAVIOR" -m pip freeze > "$OUT/environment/behavior-freeze.txt"
uv pip freeze --python "$OPENPI" > "$OUT/environment/openpi-freeze.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    > "$OUT/environment/gpus.txt"
cp third_party/logs/prepare-*.log "$LOGS/"
cp third_party/ManiGuard/outputs/lerobot_datasets/maniguard-bench/download-provenance.json \
    "$OUT/environment/benchmark-provenance.json"
cp third_party/checkpoints/jar/download-provenance.json "$OUT/environment/checkpoint-provenance.json"
cp third_party/ManiGuard/behavior-1k/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger/download-provenance.json \
    "$OUT/environment/robot-provenance.json"
python3 - <<'PY'
import json
from pathlib import Path
for path in Path('outputs/tracking04/environment').glob('*-provenance.json'):
    source = json.loads(path.read_text())
    assert not source['mismatches'] and not source['missing_from_pattern'], path.name
    print(path.name, source['revision'], source['n_files'])
PY
tar -czf "$OUT/source/executed-source.tar.gz" experiments scripts \
    progress/04-tracking-check.md progress/receipts/04-tracking-plan.json \
    progress/receipts/01-jar-eval.yaml progress/receipts/01-sources.json \
    progress/rental-tracking04-20261024.json

for lane in 0 1; do
    if ! tmux has-session -t "serve04_$lane" 2>/dev/null; then
        tmux new-session -d -s "serve04_$lane" \
            "cd $ROOT && GPU=$lane PORT=$((8000+lane)) bash scripts/todo1b/serve_jar_lane.sh >> $LOGS/server$lane.log 2>&1"
    fi
done
cleanup() {
    for lane in 0 1; do tmux kill-session -t "serve04_$lane" 2>/dev/null || true; done
}
trap cleanup EXIT
python3 - <<'PY'
import socket, time
deadline = time.time() + 900
for port in (8000, 8001):
    while True:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=2):
                break
        except OSError:
            if time.time() > deadline:
                raise SystemExit('Policy server did not become ready')
            time.sleep(5)
PY
while tmux has-session -t prewarm04 2>/dev/null; do sleep 5; done
if [ ! -f "$OUT/prewarm-complete.txt" ]; then
    timeout 900 "$OPENPI" -u -I scripts/todo1/probe_server.py --port 8000 --queries 1 \
        > "$LOGS/probe0.log" 2>&1 & p0=$!
    timeout 900 "$OPENPI" -u -I scripts/todo1/probe_server.py --port 8001 --queries 1 \
        > "$LOGS/probe1.log" 2>&1 & p1=$!
    wait "$p0"
    wait "$p1"
fi
"$BEHAVIOR" -u experiments/scripts/run_tracking_check.py > "$LOGS/queue.log" 2>&1
date -u +%FT%TZ > "$OUT/queue-complete.txt"
