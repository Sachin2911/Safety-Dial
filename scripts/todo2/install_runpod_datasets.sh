#!/usr/bin/env bash
# Stage D of the RunPod install replay: the upstream installer's --dataset step, which failed
# in install_runpod.sh stage B because `import omnigibson` hit a scipy/numpy ufunc mismatch
# after JoyLo's pip step raised numpy to 2.2.6 and conda put 1.26.4 back over it. This script
# (1) checks and repairs numpy in the behavior env, (2) downloads the stock robot-assets bundle
# with the long-finger folder (stage C put it there first, which would have made the upstream
# downloader skip the bundle) moved aside and restored afterwards, (3) downloads the
# BEHAVIOR-1K assets and the 2025 challenge task instances exactly as setup.sh --dataset does.
set -uxo pipefail
export XDG_CACHE_HOME=/workspace/.cache PIP_CACHE_DIR=/workspace/.cache/pip HF_HOME=/workspace/.cache/huggingface
MF=/workspace/miniforge3
export CONDA_ENVS_PATH=$MF/envs CONDA_PKGS_DIRS=$MF/pkgs CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
source "$MF/etc/profile.d/conda.sh"
conda activate behavior
B1K=/workspace/Safety-Dial/third_party/ManiGuard/behavior-1k
cd "$B1K"
SECONDS=0
stamp() { echo "== $1 $(date -u +%FT%TZ) elapsed=${SECONDS}s"; }
stamp "D start"
python -c "import numpy; print('numpy', numpy.__version__, numpy.__file__)"
pip list 2>/dev/null | grep -i '^numpy '
ls -d "$CONDA_PREFIX"/lib/python3.10/site-packages/numpy* || true
if ! python -c "import scipy.spatial, omnigibson" 2>/dev/null; then
  echo "repairing numpy: removing every numpy copy, reinstalling numpy 1.26.4 from conda-forge"
  pip uninstall -y numpy || true
  pip uninstall -y numpy || true
  rm -rf "$CONDA_PREFIX"/lib/python3.10/site-packages/numpy "$CONDA_PREFIX"/lib/python3.10/site-packages/numpy-*.dist-info
  conda install -c conda-forge "numpy=1.26.4" -y --force-reinstall
  python -c "import numpy; print('numpy', numpy.__version__, numpy.__file__)"
  python -c "import scipy.spatial, omnigibson; print('omnigibson import ok')" || { echo "FAIL: omnigibson still does not import"; exit 1; }
else
  echo "omnigibson imports; no numpy repair needed"
fi
stamp "D numpy ok"
LF="$B1K/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger"
if [ -d "$LF" ] && [ ! -d "$B1K/datasets/omnigibson-robot-assets/models/franka/franka_panda" ]; then
  mv "$B1K/datasets/omnigibson-robot-assets" /workspace/longfinger-aside
fi
python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()" || { echo "FAIL robot assets"; exit 1; }
if [ -d /workspace/longfinger-aside ]; then
  mv /workspace/longfinger-aside/models/franka/franka_panda_longfinger "$B1K/datasets/omnigibson-robot-assets/models/franka/" && rm -rf /workspace/longfinger-aside
fi
ls "$B1K/datasets/omnigibson-robot-assets/models/franka/"
stamp "D robot assets done"
python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)" || { echo "FAIL behavior-1k assets"; exit 1; }
cat "$B1K/datasets/behavior-1k-assets/VERSION" || true
stamp "D behavior-1k assets done"
python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()" || { echo "FAIL 2025 task instances"; exit 1; }
stamp "D task instances done"
du -sh "$B1K"/datasets/* 2>/dev/null
python -c "import numpy, torch, omnigibson, isaacsim, spot, maniguard; print('numpy', numpy.__version__, 'torch', torch.__version__, torch.cuda.device_count(), 'omnigibson', omnigibson.__version__)"
pip freeze > /workspace/Safety-Dial/progress/receipts/02-env-behavior-freeze.txt
stamp "D DONE"
