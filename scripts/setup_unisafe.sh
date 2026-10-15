#!/usr/bin/env bash
# Add the pinned UNISafe dependencies to the verified Vast platform.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
uv --no-config pip install --python /venv/main/bin/python \
    --constraint configs/vast/requirements-linux-py312.txt \
    --overrides configs/vast/uv-overrides.txt \
    --torch-backend cu130 --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --requirement configs/unisafe/requirements-lock.txt
/venv/main/bin/python scripts/download_unisafe.py
/venv/main/bin/python -c 'import sys; sys.path.insert(0,"experiments"); from helpers.unisafe_source import prepare_source; print(prepare_source())'
