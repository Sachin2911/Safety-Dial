#!/usr/bin/env bash
# Replay the pinned environment in its intended interpreters, targeting Ada SM 8.9.
# The learner interpreter is ready first so A1 overlaps the simulator installation.
set -euo pipefail
ROOT=/workspace/Safety-Dial
cd "$ROOT"
OUT="$ROOT/outputs/dsrl11"
mkdir -p "$OUT/source" "$OUT/logs" "$OUT/environment" "$OUT/ready"
export MAX_JOBS=16

python3 - <<'PY'
from pathlib import Path
root = Path('/workspace/Safety-Dial')
out = root / 'outputs/dsrl11/source'
original = (root/'scripts/todo1b/install_behavior_replay.sh').read_text()
assert original.count('TORCH_CUDA_ARCH_LIST=8.6') == 1
(out/'install-behavior89.sh').write_text(original.replace(
    'TORCH_CUDA_ARCH_LIST=8.6', 'TORCH_CUDA_ARCH_LIST=8.9'))
source = (root/'scripts/stage2/prepare_vast_host.sh').read_text()
old = 'phase behavior_install bash scripts/todo1b/install_behavior_replay.sh'
assert source.count(old) == 1
(out/'prepare-pinned-runtime.sh').write_text(source.replace(
    old, 'phase behavior_install bash outputs/dsrl11/source/install-behavior89.sh'))
PY
bash -n "$OUT/source/prepare-pinned-runtime.sh"
setsid bash "$OUT/source/prepare-pinned-runtime.sh" > "$OUT/logs/prepare-pinned.log" 2>&1 &
prepare_pid=$!
cleanup() {
    result=$?
    trap - EXIT
    if [ "$result" -ne 0 ]; then
        # All preparation descendants share this deliberately separate process group.
        kill -- -"$prepare_pid" 2>/dev/null || true
        wait "$prepare_pid" 2>/dev/null || true
        date -u +%FT%TZ > "$OUT/prepare-failed.txt"
    fi
    exit "$result"
}
trap cleanup EXIT

# Author source is isolated; the bundled root pyproject/lock are never synced.
if [ ! -d third_party/dsrl_pi0/.git ]; then
    git clone --quiet https://github.com/nakamotoo/dsrl_pi0.git third_party/dsrl_pi0
fi
git -C third_party/dsrl_pi0 checkout -q 7f48937d4553e95244cd81c79236a3256df80597
git -C third_party/dsrl_pi0 diff --exit-code
if [ ! -x third_party/dsrl-venv/bin/python ]; then
    uv venv --python 3.11 third_party/dsrl-venv
fi
uv pip install --python third_party/dsrl-venv/bin/python \
    -r configs/dsrl06/learner-requirements.txt > "$OUT/logs/prepare-learner.log" 2>&1
third_party/dsrl-venv/bin/python -c \
    'import jax,flax,distrax; print(jax.__version__,flax.__version__,jax.devices())' \
    > "$OUT/environment/learner-import.txt" 2>&1
python3 - <<'PY'
from pathlib import Path
import datetime as dt, json
out = Path('/workspace/Safety-Dial/outputs/dsrl11')
(out / 'ready/learner.json').write_text(json.dumps({
    'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'import': (out / 'environment/learner-import.txt').read_text().strip(),
}) + '\n')
PY
wait "$prepare_pid"

BEHAVIOR=/opt/miniforge3/envs/behavior/bin/python
"$BEHAVIOR" -m pip install dm_control==1.0.49 mujoco==3.16.0 \
    googleapis-common-protos==1.75.5 vhacdx==0.1.0 > "$OUT/logs/pin-repairs.log" 2>&1
python3 experiments/scripts/patch_dsrl06.py third_party/ManiGuard \
    --diff "$OUT/source/maniguard-dsrl11.patch" > "$OUT/source/patch-application.txt"
python3 - <<'PY'
from pathlib import Path
import hashlib, json, subprocess
pins = {
    'third_party/ManiGuard': '6bf4c61a1f168cc46e02eb22c41bcb51815e9726',
    'third_party/ManiGuard/behavior-1k': '88454bd04f75dc57c00ab1f1a00bcde1ff505950',
    'third_party/openpi': '215abfb217dbac7d5f1273282331b9b1866c0479',
    'third_party/dsrl_pi0': '7f48937d4553e95244cd81c79236a3256df80597',
}
resolved = {path: subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
            for path in pins}
assert resolved == pins, resolved
evaluator = hashlib.sha256(
    Path('third_party/ManiGuard/maniguard/eval/benchmark.py').read_bytes()).hexdigest()
# The same passive hooks as task 7 must produce the identical patched evaluator.
assert evaluator == 'a670d9783d70de05b11fcf916d631bf8de11fe12fa5515da55fd6fcb77558423', evaluator
resolved['patched_evaluator_sha256'] = evaluator
Path('outputs/dsrl11/environment/source-pins.json').write_text(json.dumps(resolved, indent=2) + '\n')
PY
"$BEHAVIOR" -m pip freeze > "$OUT/environment/behavior-freeze.txt"
uv pip freeze --python third_party/openpi/.venv/bin/python > "$OUT/environment/openpi-freeze.txt"
uv pip freeze --python third_party/dsrl-venv/bin/python > "$OUT/environment/learner-freeze.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    > "$OUT/environment/gpus.txt"
cp third_party/logs/prepare-*.log "$OUT/logs/"
cp third_party/checkpoints/jar/download-provenance.json "$OUT/environment/checkpoint-provenance.json"
cp third_party/ManiGuard/outputs/lerobot_datasets/maniguard-bench/download-provenance.json \
    "$OUT/environment/benchmark-provenance.json"
cp third_party/ManiGuard/behavior-1k/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger/download-provenance.json \
    "$OUT/environment/robot-provenance.json"
python3 - <<'PY'
from pathlib import Path
import datetime as dt, json
out = Path('/workspace/Safety-Dial/outputs/dsrl11')
for path in (out / 'environment').glob('*-provenance.json'):
    data = json.loads(path.read_text())
    assert not data['mismatches'] and not data['missing_from_pattern'], path
(out / 'ready/runtime.json').write_text(json.dumps({
    'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'source_pins': json.loads((out / 'environment/source-pins.json').read_text()),
}) + '\n')
PY
date -u +%FT%TZ > "$OUT/prepare-complete.txt"
