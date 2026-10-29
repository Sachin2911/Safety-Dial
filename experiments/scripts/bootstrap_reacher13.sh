#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/Safety-Dial
mkdir -p "$ROOT"
tar -xzf /tmp/reacher13-source.tar.gz -C "$ROOT"
cd "$ROOT"
mkdir -p outputs/reacher13/{control,environment,source,logs}
python3 - <<'PY'
from pathlib import Path
import hashlib,json,os,subprocess
r=json.loads(Path('progress/rental-reacher13-20261025.json').read_text())
assert str(r['instance_id'])==os.environ.get('CONTAINER_ID'), 'Wrong rental'
freeze_path=json.loads(Path('configs/reacher13/bootstrap.json').read_text())['freeze']
plan=json.loads(Path(freeze_path).read_text())
for path,sha in plan['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
p=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                  '--format=csv,noheader'],capture_output=True,text=True,check=True)
rows=p.stdout.strip().splitlines()
assert len(rows) in (1,2) and all(('RTX 3090' in x or 'RTX 4090' in x) for x in rows),rows
Path('outputs/reacher13/environment/gpus-before-setup.txt').write_text(p.stdout)
PY
bundle_sha="$(sha256sum /tmp/reacher13-source.tar.gz | cut -d ' ' -f 1)"
cp /tmp/reacher13-source.tar.gz "outputs/reacher13/source/source-$bundle_sha.tar.gz"
tmux new-session -d -s watchdog13 \
    "cd $ROOT && python3 -u experiments/scripts/reacher13_watchdog.py --remote --rental progress/rental-reacher13-20261025.json > outputs/reacher13/logs/remote-watchdog.log 2>&1"
python3 - <<'PY'
from pathlib import Path
import json,os,time
path=Path('outputs/reacher13/watchdog-ready.json')
for _ in range(100):
    if path.exists():
        ready=json.loads(path.read_text())
        assert str(ready['instance_id'])==os.environ['CONTAINER_ID']
        try:
            os.kill(ready['pid'],0)
            command=Path(f"/proc/{ready['pid']}/cmdline").read_bytes()
        except (ProcessLookupError, FileNotFoundError):
            command=b''
        if b'reacher13_watchdog.py' in command:
            break
    time.sleep(.1)
else:
    raise RuntimeError('Remote financial watchdog did not confirm readiness')
PY
tmux new-session -d -s prepare13 \
    "cd $ROOT && bash experiments/scripts/setup_reacher13.sh > outputs/reacher13/logs/preparation.log 2>&1; date -u +%FT%TZ > outputs/reacher13/prepare-ended.txt"
tmux list-sessions
