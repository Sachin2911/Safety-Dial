#!/usr/bin/env bash
# Replay of instance B's step 8 (01-install.md): evaluator deps the upstream guide leaves
# implicit (Spot LTL monitor, openpi-client), then ManiGuard[serve], all in the behavior env.
set -euxo pipefail
export CONDA_ENVS_PATH=/opt/miniforge3/envs
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate behavior
cd /workspace/Safety-Dial/third_party
date -u; SECONDS=0
conda install -c conda-forge spot "numpy<2" -y
pip install -e openpi/packages/openpi-client
pip install -e "ManiGuard[serve]"
echo "DEPS_DONE elapsed=${SECONDS}s"; date -u
