#!/usr/bin/env bash
set +x
set -euo pipefail
umask 077
cd /workspace/Safety-Dial
mkdir -p outputs/lewm16/{environment,logs,control,attempts,source}
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libegl1 libosmesa6 rsync
uv --no-config pip install --python /venv/main/bin/python --torch-backend auto -r configs/lewm16/requirements.txt
uv --no-config pip freeze --python /venv/main/bin/python > outputs/lewm16/environment/installed-freeze.txt
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
/venv/main/bin/python -u experiments/scripts/prepare_lewm16.py
nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv > outputs/lewm16/environment/gpus.csv
touch outputs/lewm16/environment/ready
