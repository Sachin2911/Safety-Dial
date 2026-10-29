#!/usr/bin/env bash
set +x
set -euo pipefail
umask 077
cd /workspace/Safety-Dial
mkdir -p outputs/lewm17/{environment,control/workers,logs,attempts,source}
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libegl1 libosmesa6 \
  rsync tmux curl git ripgrep ca-certificates texlive-latex-base texlive-latex-recommended
uv --no-config pip install --python /venv/main/bin/python --torch-backend auto \
  -r configs/lewm17/requirements.txt
uv --no-config pip install --python /venv/main/bin/python pytest ruff
uv --no-config pip freeze --python /venv/main/bin/python > outputs/lewm17/environment/installed-freeze.txt
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/workspace/Safety-Dial/outputs/lewm17/hf-cache
export MPLCONFIGDIR=/workspace/Safety-Dial/outputs/lewm17/cache/matplotlib
/venv/main/bin/python -u experiments/scripts/prepare_lewm17.py
nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv > outputs/lewm17/environment/gpus.csv
/venv/main/bin/python -m pytest -q experiments/tests/test_lewm16.py experiments/tests/test_lewm16_analysis.py experiments/tests/test_lewm17_remote.py > outputs/lewm17/environment/tests.txt
touch outputs/lewm17/environment/ready
