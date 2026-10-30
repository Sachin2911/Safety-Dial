#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export XDG_CACHE_HOME=/workspace/.cache UV_CACHE_DIR=/workspace/.cache/uv
export PIP_CACHE_DIR=/workspace/.cache/pip HF_HOME=/workspace/.cache/huggingface
export TMPDIR=/workspace/tmp OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
SETUP=/workspace/setup-maniguard-v3-20261030
TP=/workspace/Safety-Dial/third_party
SIM=/workspace/miniforge3/envs/behavior/bin/python
UV=/workspace/bin/uv
trap 'printf "stage-c failed at line %s\n" "$LINENO" > "$SETUP/stage-c.failed"' ERR
while [[ ! -f "$SETUP/sim-core.ready" ]]; do
  [[ ! -f "$SETUP/stage-b.failed" ]] || exit 1
  sleep 5
done
date -u +%FT%TZ > "$SETUP/stage-c.started"
cat > "$SETUP/sim-constraints.txt" <<'EOF'
numpy==1.26.4
setuptools==79.0.1
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
cffi==1.17.1
EOF
"$UV" pip install --python "$SIM" -c "$SETUP/sim-constraints.txt" \
  -e "$TP/ManiGuard/behavior-1k/bddl3" \
  -e "$TP/ManiGuard/behavior-1k/OmniGibson" \
  -e "$TP/ManiGuard[serve]" -e "$TP/openpi/packages/openpi-client"
"$UV" pip install --python "$SIM" -c "$SETUP/sim-constraints.txt" \
  'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match
"$UV" pip install --python "$SIM" --no-deps 'torch-cluster==1.6.3+pt26cu124' \
  --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html
"$UV" pip install --python "$SIM" -c "$SETUP/sim-constraints.txt" requests 'gymnasium==0.29.1'
"$SIM" - <<'PY'
from pathlib import Path
import shutil, sys
for candidate in Path(sys.prefix).glob('lib/python*/site-packages/isaacsim/extscache/*/pip_prebundle/websockets'):
    shutil.rmtree(candidate)
    print('Removed upstream-vendored websocket override', candidate.parent.parent.name)
PY
"$SIM" - <<'PY'
import importlib.metadata as m
import numpy,scipy.spatial,spot,torch
assert numpy.__version__=='1.26.4'
assert hasattr(spot,'translate')
assert torch.cuda.is_available()
for name in ('omnigibson','bddl','isaacsim','maniguard','openpi-client','numpy','scipy','torch'):
    print(name,m.version(name))
PY
"$UV" pip install --python "$SIM" "cma==4.4.4" pytest ruff
touch "$SETUP/sim-packages.ready"
date -u +%FT%TZ > "$SETUP/stage-c.complete"
