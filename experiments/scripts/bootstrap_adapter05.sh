#!/usr/bin/env bash
# This script runs only on the separately recorded adapter05 instance.
set -euo pipefail
ROOT=/workspace/Safety-Dial
mkdir -p "$ROOT"
tar -xzf /tmp/adapter05-source.tar.gz -C "$ROOT"
cd "$ROOT"
mkdir -p outputs/adapter05/logs outputs/adapter05/environment outputs/adapter05/source
python3 - <<'PY'
import json, os, subprocess
from pathlib import Path
r = json.loads(Path('progress/rental-adapter05-20261024.json').read_text())
assert os.environ.get('CONTAINER_ID') == str(r['instance_id']), 'Wrong rental'
p = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                    '--format=csv,noheader'], capture_output=True, text=True, check=True)
lines = p.stdout.strip().splitlines()
assert len(lines) == 2 and all('RTX 3090' in s and '580.' in s for s in lines), lines
Path('outputs/adapter05/environment/gpus.txt').write_text(p.stdout)
print(p.stdout)
PY
cp /tmp/adapter05-source.tar.gz outputs/adapter05/source/bootstrap-source.tar.gz
tmux new-session -d -s watchdog05 \
    "cd $ROOT && python3 -u experiments/scripts/adapter05_watchdog.py progress/rental-adapter05-20261024.json > outputs/adapter05/logs/watchdog.log 2>&1"
tmux new-session -d -s prepare05 \
    "cd $ROOT && export MAX_JOBS=8 && bash scripts/stage2/prepare_vast_host.sh > outputs/adapter05/logs/prepare-phases.log 2>&1; date -u +%FT%TZ > outputs/adapter05/prepare-ended.txt"
tmux list-sessions
