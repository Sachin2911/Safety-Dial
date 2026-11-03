#!/usr/bin/env bash
# Fresh authorized Vast host only. No simulator, training, or lifecycle commands.
set -euo pipefail
cd /workspace/SafeLeWM
mkdir -p assets data outputs/adapter-pilot-20261101
if ! command -v git >/dev/null; then
    apt-get update
    apt-get install -y --no-install-recommends git ca-certificates
fi
python -m venv --system-site-packages venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r configs/adapter-pilot/requirements.txt
venv/bin/python -m pip install --no-deps stable-pretraining==0.1.2
if [ ! -d assets/ap-reference/.git ]; then
    git clone https://github.com/daybraeklaxry/Aim-Short-to-Reach-Far assets/ap-reference
fi
git -C assets/ap-reference checkout 984327424ca7a2e52c75a8279197750d9d0b7c8e
venv/bin/python experiments/scripts/fetch_adapter_assets.py
venv/bin/python -m pip freeze > outputs/adapter-pilot-20261101/packages.txt
touch outputs/adapter-pilot-20261101/setup-complete
