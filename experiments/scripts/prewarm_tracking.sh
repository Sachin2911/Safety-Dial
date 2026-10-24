#!/usr/bin/env bash
# Overlap model loading and the planned model-only probes with simulator setup.
set -euo pipefail
ROOT=/workspace/Safety-Dial
cd "$ROOT"
OUT="$ROOT/outputs/tracking04"
LOGS="$OUT/logs"
for lane in 0 1; do
    tmux new-session -d -s "serve04_$lane" \
        "cd $ROOT && GPU=$lane PORT=$((8000+lane)) bash scripts/todo1b/serve_jar_lane.sh > $LOGS/server$lane.log 2>&1"
done
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
OPENPI="$ROOT/third_party/openpi/.venv/bin/python"
timeout 900 "$OPENPI" -u -I scripts/todo1/probe_server.py --port 8000 --queries 1 \
    > "$LOGS/probe0-initial.log" 2>&1 & p0=$!
timeout 900 "$OPENPI" -u -I scripts/todo1/probe_server.py --port 8001 --queries 1 \
    > "$LOGS/probe1-initial.log" 2>&1 & p1=$!
wait "$p0"
wait "$p1"
date -u +%FT%TZ > "$OUT/prewarm-complete.txt"
