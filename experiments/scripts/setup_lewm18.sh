#!/usr/bin/env bash
set +x
set -euo pipefail
umask 077
cd /workspace/Safety-Dial
mkdir -p outputs/lewm18/{environment,logs,attempts,source}
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libegl1 libosmesa6 rsync zstd
uv --no-config pip install --python /venv/main/bin/python --torch-backend auto -r configs/lewm18/requirements.txt
uv --no-config pip freeze --python /venv/main/bin/python > outputs/lewm18/environment/installed-freeze.txt
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_FIXED_DOWNLOAD_CONCURRENCY=8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
prepare_pids=()
for task_env in ${LEWM18_PREPARE_ENVS:-cube pusht}; do
    case "$task_env" in cube|pusht) ;; *) exit 2 ;; esac
    /venv/main/bin/python -u experiments/scripts/prepare_lewm18.py --env "$task_env" > "outputs/lewm18/logs/$task_env-prepare.log" 2>&1 &
    prepare_pids+=("$!")
done
for prepare_pid in "${prepare_pids[@]}"; do
    wait "$prepare_pid"
done
nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv > outputs/lewm18/environment/gpus.csv
