#!/usr/bin/env bash
# Replay of the pinned ManiGuard install (progress/receipts/01b-install.md, 01-sources.json)
# on a RunPod pod: Miniforge under /workspace/miniforge3, every clone, env, cache and asset on
# the /workspace volume (the container disk is 40 GB). Same commands and pins as instance C;
# the Vast-only steps (/opt/instance-tools, /venv/main, /opt/miniforge3) are replaced by the
# paths below. Stages: A (parallel: miniforge, submodule, openpi env, apt, benchmark and
# checkpoint downloads), B (behavior env, curobo, upstream setup.sh), C (long-finger robot,
# evaluator deps + ManiGuard[serve]). Logs in /workspace/install-logs/.
#
#   tmux new -d -s install 'bash scripts/todo2/install_runpod.sh > /workspace/install-logs/install.log 2>&1'
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
export XDG_CACHE_HOME=/workspace/.cache PIP_CACHE_DIR=/workspace/.cache/pip UV_CACHE_DIR=/workspace/.cache/uv
export HF_HOME=/workspace/.cache/huggingface
ROOT=/workspace/Safety-Dial
TP=$ROOT/third_party
LOG=/workspace/install-logs
MF=/workspace/miniforge3
export CONDA_ENVS_PATH=$MF/envs CONDA_PKGS_DIRS=$MF/pkgs CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
mkdir -p "$TP" "$LOG"
stamp() { echo "== $1 $(date -u +%FT%TZ) elapsed=${SECONDS}s"; }
SECONDS=0
stamp "START"

# ---------------- stage A (parallel) ----------------
(
  stamp "A1 miniforge start"
  if [ ! -x "$MF/bin/conda" ]; then
    curl -fsSL -o /workspace/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /workspace/Miniforge3.sh -b -p "$MF"
  fi
  "$MF/bin/conda" --version
  stamp "A1 miniforge done"
) > "$LOG/A1-miniforge.log" 2>&1 &
(
  stamp "A2 submodule start"
  cd "$TP/ManiGuard" && git checkout -q 6bf4c61a1f168cc46e02eb22c41bcb51815e9726 && git submodule update --init --recursive
  git -C behavior-1k rev-parse HEAD
  stamp "A2 submodule done"
) > "$LOG/A2-submodule.log" 2>&1 &
(
  stamp "A3 openpi start"
  cd "$TP"
  [ -d openpi/.git ] || git clone https://github.com/Physical-Intelligence/openpi.git
  cd openpi && git checkout -q 215abfb217dbac7d5f1273282331b9b1866c0479 && git submodule update --init --recursive
  GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
  .venv/bin/python -c "import jax, sys; print(sys.version); print(jax.__version__, jax.devices())"
  stamp "A3 openpi done"
) > "$LOG/A3-openpi.log" 2>&1 &
(
  stamp "A4 apt start"
  apt-get update -qq && apt-get install -y -qq libsm6 libice6 libxt6 libglu1-mesa libxrender1 libxrandr2 libxcursor1 libxinerama1 libxi6 libxxf86vm1 libxfixes3 libxext6 libxkbcommon0 libxcomposite1 libxdamage1 libgl1 libegl1 libvulkan1 vulkan-tools
  stamp "A4 apt done"
) > "$LOG/A4-apt.log" 2>&1 &
(
  stamp "A5 benchmark download start"
  cd "$ROOT" && python3 -I scripts/todo1/download_pinned.py benchmark "$TP/ManiGuard/outputs/lerobot_datasets/maniguard-bench"
  stamp "A5 benchmark download done"
) > "$LOG/A5-dl-bench.log" 2>&1 &
(
  stamp "A6 checkpoint download start"
  cd "$ROOT" && python3 -I scripts/todo1/download_pinned.py jar "$TP/checkpoints/jar"
  stamp "A6 checkpoint download done"
) > "$LOG/A6-dl-jar.log" 2>&1 &
wait
stamp "stage A done"
for f in A1-miniforge A2-submodule A3-openpi A4-apt A5-dl-bench A6-dl-jar; do tail -2 "$LOG/$f.log"; done

# ---------------- stage B: behavior env (01b install_behavior_replay.sh with RunPod paths) ----------------
(
  set -x
  stamp "B behavior start"
  source "$MF/etc/profile.d/conda.sh"
  cd "$TP/ManiGuard/behavior-1k"
  conda create -n behavior python=3.10 -c conda-forge -y
  conda activate behavior
  [[ "$CONDA_DEFAULT_ENV" == "behavior" ]] || { echo "ERROR: Failed to activate environment"; exit 1; }
  pip install "numpy<2" "setuptools<=79"
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  stamp "B new-env done"
  pip install ninja wheel
  CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation \
    "nvidia-curobo @ git+https://github.com/StanfordVL/curobo@cbaf7d32436160956dad190a9465360fad6aba73"
  stamp "B curobo done"
  ./setup.sh --omnigibson --bddl --joylo --dataset --eval --primitives \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda
  stamp "B setup.sh done rc=$?"
) > "$LOG/B-behavior.log" 2>&1
tail -3 "$LOG/B-behavior.log"

# ---------------- stage C: robot asset, evaluator deps ----------------
(
  set -x
  stamp "C start"
  cd "$ROOT" && python3 -I scripts/todo1/download_pinned.py robot "$TP/ManiGuard/behavior-1k/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger"
  ls "$TP/ManiGuard/behavior-1k/datasets/omnigibson-robot-assets/models/franka/"
  source "$MF/etc/profile.d/conda.sh"
  conda activate behavior
  cd "$TP"
  conda install -c conda-forge spot "numpy<2" -y
  pip install -e openpi/packages/openpi-client
  pip install -e "ManiGuard[serve]"
  cp "$ROOT/progress/receipts/01-jar-eval.yaml" "$TP/ManiGuard/configs/eval/jar_transport_joint_legacy256.yaml"
  python -c "import numpy, torch, omnigibson, isaacsim, spot, maniguard; print('numpy', numpy.__version__, 'torch', torch.__version__, torch.cuda.device_count()); print('omnigibson', omnigibson.__version__)"
  pip freeze > "$ROOT/progress/receipts/02-env-behavior-freeze.txt"
  (cd openpi && uv pip freeze > "$ROOT/progress/receipts/02-env-openpi-freeze.txt")
  stamp "C done"
) > "$LOG/C-deps.log" 2>&1
tail -3 "$LOG/C-deps.log"
stamp "INSTALL DONE"
