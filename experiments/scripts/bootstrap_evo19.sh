#!/usr/bin/env bash
# Task 19 host bootstrap: unpack the source bundle, check the GPU and driver, start the
# instance-scoped deadline guard, then run the pinned Jar preparation in tmux.
set -euo pipefail
ROOT=/workspace/Safety-Dial
mkdir -p "$ROOT"
tar -xzf /tmp/evo19-source.tar.gz -C "$ROOT"
cd "$ROOT"
mkdir -p outputs/evo19/{environment,source,logs,queue,results,episodes}
echo "${EVO19_HOST:-unknown}" > outputs/evo19/host-name.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    | tee outputs/evo19/environment/gpus-before-setup.txt
grep -E 'RTX (3090|4090)' outputs/evo19/environment/gpus-before-setup.txt | grep -q ' 580\.'
cp /tmp/evo19-source.tar.gz outputs/evo19/source/bootstrap-source.tar.gz
tmux kill-session -t guard19 2>/dev/null || true
tmux new-session -d -s guard19 \
    "cd $ROOT && python3 -u experiments/scripts/evo19_remote_guard.py > outputs/evo19/logs/remote-guard.log 2>&1"
sleep 3
test -f outputs/evo19/guard-ready.json
if [ ! -f outputs/evo19/prepare-complete.txt ]; then
    tmux kill-session -t prepare19 2>/dev/null || true
    tmux new-session -d -s prepare19 \
        "cd $ROOT && bash experiments/scripts/prepare_evo19.sh > outputs/evo19/logs/preparation.log 2>&1; date -u +%FT%TZ > outputs/evo19/prepare-ended.txt"
fi
tmux list-sessions
