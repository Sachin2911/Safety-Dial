#!/usr/bin/env bash
# setup: unpack the setup bundle, verify host and hashes, start the guard and preparation.
# run:   stage the frozen bundle, verify it and its unchanged preparation inputs, then
#        install changed files atomically (running preparation keeps its inodes) and
#        start the queue.
set -euo pipefail
MODE=${1:?usage: bootstrap_dsrl11.sh setup|run}
ROOT=/workspace/Safety-Dial
BUNDLE=/tmp/dsrl11-source.tar.gz
mkdir -p "$ROOT"
case "$MODE" in
setup)
    if [ -e "$ROOT/outputs/dsrl11/watchdog-ready.json" ]; then
        echo "Setup already ran on this rental" >&2
        exit 1
    fi
    tar -xzf "$BUNDLE" -C "$ROOT"
    PLAN=progress/receipts/11-dsrl-setup.json
    ;;
run)
    STAGE=$(mktemp -d /tmp/dsrl11-run.XXXXXX)
    tar -xzf "$BUNDLE" -C "$STAGE"
    python3 - "$STAGE" "$ROOT" <<'PY'
from pathlib import Path
import hashlib, json, os, sys
stage, root = Path(sys.argv[1]), Path(sys.argv[2])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
plan = json.loads((stage / 'progress/receipts/11-dsrl-stage-a.json').read_text())
setup = json.loads((root / 'progress/receipts/11-dsrl-setup.json').read_text())
frozen = {**plan['source_sha256'], **plan['input_sha256']}
for name, value in frozen.items():
    assert sha(stage / name) == value, name
# Files the live preparation and guard already loaded must be the frozen ones.
held = ('experiments/scripts/prepare_dsrl11.sh', 'experiments/scripts/patch_dsrl06.py',
        'experiments/scripts/dsrl11_watchdog.py', 'experiments/helpers/dsrl11_infra.py',
        'experiments/helpers/dsrl06_infra.py')
prefixes = ('configs/dsrl06/', 'progress/receipts/01', 'scripts/stage2/', 'scripts/todo1/',
            'scripts/todo1b/', 'scripts/todo2/')
for name, value in setup['source_sha256'].items():
    if name in held or name.startswith(prefixes):
        assert frozen.get(name) == value, f'preparation input changed after setup: {name}'
installed = 0
for path in sorted(p for p in stage.rglob('*') if p.is_file()):
    name = path.relative_to(stage)
    target = root / name
    if target.is_file() and sha(target) == sha(path):
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.dsrl11-new')
    temporary.write_bytes(path.read_bytes())
    os.chmod(temporary, path.stat().st_mode)
    os.replace(temporary, target)
    installed += 1
print(json.dumps({'run_bundle_installed_files': installed}))
PY
    PLAN=progress/receipts/11-dsrl-stage-a.json
    ;;
*)
    echo "Unknown bootstrap mode: $MODE" >&2
    exit 2
    ;;
esac
cd "$ROOT"
mkdir -p outputs/dsrl11/{control,environment,source,logs,ready}
python3 - "$PLAN" "$MODE" <<'PY'
from pathlib import Path
import hashlib, json, os, subprocess, sys
plan_path, mode = sys.argv[1], sys.argv[2]
rental = json.loads(Path('progress/rental-dsrl11-20261025.json').read_text())
assert str(rental['instance_id']) == os.environ.get('CONTAINER_ID'), 'Wrong rental'
plan = json.loads(Path(plan_path).read_text())
assert plan['schema'] == {'setup': 'dsrl11-setup-v1', 'run': 'dsrl11-freeze-v1'}[mode]
for path, sha in {**plan['source_sha256'], **plan['input_sha256']}.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                         '--format=csv,noheader'], capture_output=True, text=True, check=True)
rows = result.stdout.strip().splitlines()
assert len(rows) == 4 and all('RTX 4090' in x and '580.' in x for x in rows), rows
Path(f'outputs/dsrl11/environment/gpus-before-{mode}.txt').write_text(result.stdout)
print(json.dumps({'mode': mode, 'plan': plan_path, 'verified_files': len(plan['source_sha256'])
                  + len(plan['input_sha256'])}))
PY
cp "$BUNDLE" "outputs/dsrl11/source/$MODE-source.tar.gz"
if [ "$MODE" = setup ]; then
    tmux new-session -d -s watchdog11 \
        "cd $ROOT && python3 -u experiments/scripts/dsrl11_watchdog.py --remote --rental progress/rental-dsrl11-20261025.json > outputs/dsrl11/logs/remote-watchdog.log 2>&1"
fi
python3 - <<'PY'
from pathlib import Path
import json, os, time
path = Path('outputs/dsrl11/watchdog-ready.json')
for _ in range(100):
    if path.exists():
        ready = json.loads(path.read_text())
        assert str(ready['instance_id']) == os.environ['CONTAINER_ID']
        os.kill(ready['pid'], 0)
        assert b'dsrl11_watchdog.py' in Path(f"/proc/{ready['pid']}/cmdline").read_bytes()
        break
    time.sleep(.1)
else:
    raise RuntimeError('Remote financial watchdog did not confirm readiness')
PY
if [ "$MODE" = setup ]; then
    tmux new-session -d -s prepare11 \
        "cd $ROOT && bash experiments/scripts/prepare_dsrl11.sh > outputs/dsrl11/logs/preparation.log 2>&1; date -u +%FT%TZ > outputs/dsrl11/prepare-ended.txt"
else
    if tmux has-session -t queue11 2>/dev/null; then
        echo "The task-11 queue is already running" >&2
        exit 1
    fi
    # The queue needs only the learner interpreter; it waits for later readiness markers.
    tmux new-session -d -s queue11 \
        "cd $ROOT && while [ ! -f outputs/dsrl11/ready/learner.json ]; do [ -f outputs/dsrl11/prepare-failed.txt ] && exit 1; sleep 5; done; third_party/dsrl-venv/bin/python -u experiments/scripts/run_dsrl11.py --stage all > outputs/dsrl11/logs/queue.log 2>&1; echo \$? > outputs/dsrl11/queue-exit.txt"
fi
tmux list-sessions
