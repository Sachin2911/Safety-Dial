#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/Safety-Dial
mkdir -p "$ROOT"
tar -xzf /tmp/regrasp15-source.tar.gz -C "$ROOT"
cd "$ROOT"
mkdir -p outputs/regrasp15/{control,environment,source,logs}
python3 - <<'PY'
from pathlib import Path
import hashlib,json,os,subprocess
r=json.loads(Path('progress/rental-regrasp15-20261025.json').read_text())
assert str(r['instance_id'])==os.environ.get('CONTAINER_ID'), 'Wrong rental'
plan=json.loads(Path('progress/receipts/15-regrasp-preparation-v1.json').read_text())
for path,sha in plan['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
p=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                  '--format=csv,noheader'],capture_output=True,text=True,check=True)
rows=p.stdout.strip().splitlines()
assert len(rows)==1 and all(('RTX 4090' in x or 'RTX 3090' in x) and '580.' in x for x in rows),rows
Path('outputs/regrasp15/environment/gpus-before-setup.txt').write_text(p.stdout)
PY
cp /tmp/regrasp15-source.tar.gz outputs/regrasp15/source/bootstrap-source.tar.gz
tmux new-session -d -s watchdog15 \
    "cd $ROOT && python3 -u experiments/scripts/regrasp15_watchdog.py --remote --rental progress/rental-regrasp15-20261025.json > outputs/regrasp15/logs/remote-watchdog.log 2>&1"
python3 - <<'PY'
from pathlib import Path
import json,os,time
path=Path('outputs/regrasp15/watchdog-ready.json')
for _ in range(100):
    if path.exists():
        ready=json.loads(path.read_text())
        assert str(ready['instance_id'])==os.environ['CONTAINER_ID']
        os.kill(ready['pid'],0)
        assert b'regrasp15_watchdog.py' in Path(f"/proc/{ready['pid']}/cmdline").read_bytes()
        break
    time.sleep(.1)
else:
    raise RuntimeError('Remote financial watchdog did not confirm readiness')
PY
tmux new-session -d -s prepare15 \
    "cd $ROOT && bash experiments/scripts/prepare_regrasp15.sh > outputs/regrasp15/logs/preparation.log 2>&1; date -u +%FT%TZ > outputs/regrasp15/prepare-ended.txt"
tmux list-sessions
