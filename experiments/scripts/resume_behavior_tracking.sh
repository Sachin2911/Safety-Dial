#!/usr/bin/env bash
# Resume after slow public wheel downloads, using the same selected wheel bytes.
set -euo pipefail
ROOT=/workspace/Safety-Dial
cd "$ROOT"
export CONDA_ENVS_PATH=/opt/miniforge3/envs CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
export MAX_JOBS=8 CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate behavior
test "$(git -C third_party/curobo-pinned rev-parse HEAD)" = \
    cbaf7d32436160956dad190a9465360fad6aba73
test -z "$(git -C third_party/curobo-pinned diff --name-only)"
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('/workspace/Safety-Dial/third_party/tracking04-wheels')
for item in json.loads((root / 'wheel-manifest.json').read_text()):
    path = root / item['filename']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path.name
print('All selected dependency wheel hashes verified on host', flush=True)
PY
# Build directly from the verified checkout. Its partial-clone object store cannot
# serve an ordinary second clone; keep the local source path explicit in provenance.
python -m pip install --no-index --find-links third_party/tracking04-wheels \
    --no-build-isolation ./third_party/curobo-pinned
cd third_party/ManiGuard/behavior-1k
./setup.sh --omnigibson --bddl --joylo --dataset --eval --primitives \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda
cd "$ROOT"
touch third_party/.prepare/behavior_install
bash scripts/stage2/prepare_vast_host.sh
