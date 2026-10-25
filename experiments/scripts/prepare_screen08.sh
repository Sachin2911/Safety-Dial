#!/usr/bin/env bash
# Replay the pinned environment in its intended interpreters, targeting Ada SM 8.9.
set -euo pipefail
ROOT=/workspace/Safety-Dial
cd "$ROOT"
OUT="$ROOT/outputs/screen08"
mkdir -p "$OUT/source" "$OUT/logs" "$OUT/environment"
export MAX_JOBS=16

python3 - <<'PY'
from pathlib import Path
root = Path('/workspace/Safety-Dial')
out = root / 'outputs/screen08/source'
original = (root/'scripts/todo1b/install_behavior_replay.sh').read_text()
assert original.count('TORCH_CUDA_ARCH_LIST=8.6') == 1
(out/'install-behavior89.sh').write_text(original.replace(
    'TORCH_CUDA_ARCH_LIST=8.6', 'TORCH_CUDA_ARCH_LIST=8.9'))
source = (root/'scripts/stage2/prepare_vast_host.sh').read_text()
old = 'phase behavior_install bash scripts/todo1b/install_behavior_replay.sh'
assert source.count(old) == 1
(out/'prepare-pinned-runtime.sh').write_text(source.replace(
    old, 'phase behavior_install bash outputs/screen08/source/install-behavior89.sh'))
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

wait "$prepare_pid"

BEHAVIOR=/opt/miniforge3/envs/behavior/bin/python
"$BEHAVIOR" -m pip install dm_control==1.0.49 mujoco==3.16.0 \
    googleapis-common-protos==1.75.5 vhacdx==0.1.0 > "$OUT/logs/pin-repairs.log" 2>&1
python3 - <<'PY'
from pathlib import Path
import hashlib,json,subprocess
pins = {
    'third_party/ManiGuard': '6bf4c61a1f168cc46e02eb22c41bcb51815e9726',
    'third_party/ManiGuard/behavior-1k': '88454bd04f75dc57c00ab1f1a00bcde1ff505950',
    'third_party/openpi': '215abfb217dbac7d5f1273282331b9b1866c0479',
}
resolved = {path: subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip()
            for path in pins}
assert resolved == pins, resolved
resolved['native_evaluator_sha256'] = hashlib.sha256(
    Path('third_party/ManiGuard/maniguard/eval/benchmark.py').read_bytes()).hexdigest()
Path('outputs/screen08/environment/source-pins.json').write_text(json.dumps(resolved,indent=2)+'\n')
PY
"$BEHAVIOR" -m pip freeze > "$OUT/environment/behavior-freeze.txt"
uv pip freeze --python third_party/openpi/.venv/bin/python > "$OUT/environment/openpi-freeze.txt"
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
import json
for path in Path('outputs/screen08/environment').glob('*-provenance.json'):
    data=json.loads(path.read_text())
    assert not data['mismatches'] and not data['missing_from_pattern'], path
PY
date -u +%FT%TZ > "$OUT/prepare-complete.txt"
