#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/Safety-Dial
mkdir -p "$ROOT"
tar -xzf /tmp/safecem10-source.tar.gz -C "$ROOT"
cd "$ROOT"
mkdir -p outputs/safecem10/{control,environment,source,logs}
python3 - <<'PY'
from pathlib import Path
import hashlib,json,os,subprocess
r=json.loads(Path('progress/rental-safecem10-20261025.json').read_text())
assert str(r['instance_id'])==os.environ.get('CONTAINER_ID'), 'Wrong rental'
freeze_path=json.loads(Path('configs/safecem10/bootstrap.json').read_text())['freeze']
plan=json.loads(Path(freeze_path).read_text())
for path,sha in plan['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
p=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                  '--format=csv,noheader'],capture_output=True,text=True,check=True)
rows=p.stdout.strip().splitlines()
assert len(rows)==1 and ('RTX 3090' in rows[0] or 'RTX 4090' in rows[0]),rows
Path('outputs/safecem10/environment/gpus-before-setup.txt').write_text(p.stdout)
PY
bundle_sha="$(sha256sum /tmp/safecem10-source.tar.gz | cut -d ' ' -f 1)"
cp /tmp/safecem10-source.tar.gz "outputs/safecem10/source/source-$bundle_sha.tar.gz"
tmux new-session -d -s watchdog10 \
    "cd $ROOT && python3 -u experiments/scripts/safecem10_watchdog.py --remote --rental progress/rental-safecem10-20261025.json > outputs/safecem10/logs/remote-watchdog.log 2>&1"
python3 - <<'PY'
from pathlib import Path
import json,os,time
path=Path('outputs/safecem10/watchdog-ready.json')
for _ in range(100):
    if path.exists():
        ready=json.loads(path.read_text())
        assert str(ready['instance_id'])==os.environ['CONTAINER_ID']
        try:
            os.kill(ready['pid'],0)
            command=Path(f"/proc/{ready['pid']}/cmdline").read_bytes()
        except (ProcessLookupError, FileNotFoundError):
            command=b''
        if b'safecem10_watchdog.py' in command:
            break
    time.sleep(.1)
else:
    raise RuntimeError('Remote financial watchdog did not confirm readiness')
PY
tmux new-session -d -s prepare10 \
    "cd $ROOT && bash experiments/scripts/setup_safecem10.sh > outputs/safecem10/logs/preparation.log 2>&1; date -u +%FT%TZ > outputs/safecem10/prepare-ended.txt"
tmux list-sessions
