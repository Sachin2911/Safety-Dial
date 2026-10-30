#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export XDG_CACHE_HOME=/workspace/.cache PIP_CACHE_DIR=/workspace/.cache/pip
export UV_CACHE_DIR=/workspace/.cache/uv HF_HOME=/workspace/.cache/huggingface
export TMPDIR=/workspace/tmp GIT_LFS_SKIP_SMUDGE=1
export CONDA_ENVS_PATH=/workspace/miniforge3/envs CONDA_PKGS_DIRS=/workspace/miniforge3/pkgs
SETUP=/workspace/setup-maniguard-v3-20261030
TP=/workspace/Safety-Dial/third_party
UV=/workspace/bin/uv
trap 'printf "stage-b failed at line %s\n" "$LINENO" > "$SETUP/stage-b.failed"' ERR
while [[ ! -f "$SETUP/stage-a.complete" ]]; do
  [[ ! -f "$SETUP/stage-a.failed" ]] || exit 1
  sleep 5
done
date -u +%FT%TZ > "$SETUP/stage-b.started"

(
  cd "$TP/openpi"
  "$UV" sync --frozen --python 3.11 --no-dev
  "$UV" pip install --python .venv/bin/python --no-deps -e .
  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python - <<'PY'
import jax
print('JAX',jax.__version__,jax.devices())
PY
  touch "$SETUP/openpi-env.ready"
) > "$SETUP/logs/openpi-env.log" 2>&1 &
OPENPI_JOB=$!

(
  /workspace/miniforge3/bin/conda create -y --override-channels -c conda-forge \
    -n behavior python=3.10 spot=2.16 'numpy=1.26.4' av
  SIM=/workspace/miniforge3/envs/behavior/bin/python
  "$UV" pip install --python "$SIM" 'setuptools==79.0.1' wheel ninja 'scipy==1.15.3'
  "$UV" pip install --python "$SIM" \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
  "$SIM" - <<'PY'
import numpy,scipy.spatial,spot,torch
assert numpy.__version__=='1.26.4'
assert torch.cuda.is_available()
assert hasattr(spot,'translate')
print('SIM CORE',numpy.__version__,torch.__version__,torch.cuda.get_device_name(0))
PY
  touch "$SETUP/sim-core.ready"
) > "$SETUP/logs/sim-core.log" 2>&1 &
SIM_JOB=$!

(
  "$UV" venv --python 3.11 /workspace/envs/downloads
  "$UV" pip install --python /workspace/envs/downloads/bin/python huggingface_hub==1.33.0
  touch "$SETUP/download-env.ready"
) > "$SETUP/logs/download-env.log" 2>&1 &
DOWNLOAD_JOB=$!

wait "$OPENPI_JOB"
wait "$SIM_JOB"
wait "$DOWNLOAD_JOB"
date -u +%FT%TZ > "$SETUP/stage-b.complete"
