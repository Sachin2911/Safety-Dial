#!/usr/bin/env bash
# Replay of instance B's BEHAVIOR install (progress/receipts/01-install.md, steps 3a-3c)
# on instance C. The --new-env block of upstream setup.sh is run by hand (identical
# commands) so the known nvidia-curobo build-isolation failure is not repeated; then
# curobo is pre-built against the env's torch exactly as in B's step 3c, and the upstream
# installer is resumed with B's flags.
set -euxo pipefail
export CONDA_ENVS_PATH=/opt/miniforge3/envs        # image-specific conda layout workaround (B)
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
source /opt/miniforge3/etc/profile.d/conda.sh
cd /workspace/Safety-Dial/third_party/ManiGuard/behavior-1k
date -u; SECONDS=0
# ---- setup.sh --new-env block, verbatim commands ----
conda create -n behavior python=3.10 -c conda-forge -y
conda activate behavior
[[ "$CONDA_DEFAULT_ENV" == "behavior" ]] || { echo "ERROR: Failed to activate environment"; exit 1; }
pip install "numpy<2" "setuptools<=79"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
echo "NEW_ENV_DONE elapsed=${SECONDS}s"; date -u
# ---- B step 3c: curobo against the env's own torch ----
pip install ninja wheel
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation \
  "nvidia-curobo @ git+https://github.com/StanfordVL/curobo@cbaf7d32436160956dad190a9465360fad6aba73"
echo "CUROBO_DONE elapsed=${SECONDS}s"; date -u
# ---- resume upstream installer with B's flags ----
./setup.sh --omnigibson --bddl --joylo --dataset --eval --primitives \
  --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda
echo "SETUP_DONE elapsed=${SECONDS}s"; date -u
