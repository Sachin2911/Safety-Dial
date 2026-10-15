#!/usr/bin/env bash
# Manual platform setup after connecting to an "evo safety filter" instance.
# Run from a repository clone: bash scripts/vast/setup.sh
# This prepares the base image's existing environment. Nothing runs at instance boot.
set +x
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: bash scripts/vast/setup.sh\n'
    printf 'Install the pinned Isaac and PyTorch platform using uv after connecting to Vast.\n'
    exit 0
fi
if (($#)); then
    printf 'Unknown argument. Usage: bash scripts/vast/setup.sh\n' >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
state_dir=/workspace/.evo-safety-filter
runtime_python=/venv/main/bin/python
lab_revision=6bfde876089661e116dc5a5d9c6805f18127fe32
if [[ ! -x "$runtime_python" ]]; then
    printf 'Run this script on the Vast base image with /venv/main/bin/python.\n' >&2
    exit 1
fi
if [[ "${EUID}" -ne 0 ]]; then
    printf 'Run as root on the Vast instance to install the required system libraries.\n' >&2
    exit 1
fi
mkdir -p "$state_dir"
exec > >(tee -a "$state_dir/setup.log") 2>&1

trap 'printf "Setup failed. See /workspace/.evo-safety-filter/setup.log\n" >&2' ERR
exec 9>"$state_dir/setup.lock"
flock 9

if [[ -f "$state_dir/ready" ]]; then
    printf 'evo safety filter platform was already installed.\n'
    exit 0
fi

printf 'Installing evo safety filter platform using uv.\n'
"$runtime_python" - <<'PY'
import platform
import sys

assert sys.version_info[:2] == (3, 12), sys.version
assert platform.machine() == "x86_64", platform.machine()
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    libgl1 libegl1 libglvnd0 libglu1-mesa libvulkan1 vulkan-tools \
    libx11-6 libxcursor1 libxi6 libxinerama1 libxrandr2 libxrender1 \
    libxext6 libsm6 libice6 libnss3 libglib2.0-0 libgomp1 libxt6 \
    libasound2t64 ffmpeg

command -v uv
cp -- "$repo_root/configs/vast/uv-overrides.txt" "$state_dir/uv-overrides.txt"
cp -- "$repo_root/configs/vast/requirements-linux-py312.txt" \
    "$state_dir/requirements-linux-py312.txt"
test -s "$state_dir/uv-overrides.txt"
test -s "$state_dir/requirements-linux-py312.txt"

# Use Lab's matching overrides and the CUDA 13.0 wheels. The public rc1 wheel
# still requests Sim 6.1, so install the pinned 3.0 branch's aggregate package.
uv pip install --python "$runtime_python" --no-cache \
    --torch-backend cu130 \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease allow \
    --overrides "$state_dir/uv-overrides.txt" \
    --requirement "$state_dir/requirements-linux-py312.txt"

"$runtime_python" - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

import torch

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.split("+")[0] == "2.11.0", torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
assert metadata.version("torchvision").split("+")[0] == "0.26.0"
assert metadata.version("isaacsim") == "6.2.0.0", metadata.version("isaacsim")
assert torch.cuda.is_available(), "CUDA is unavailable in the container"
torch.ones((16, 16), device="cuda").square().sum().item()
torch.cuda.synchronize()
versions = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torchvision": metadata.version("torchvision"),
    "pytorch_cuda": torch.version.cuda,
    "isaaclab": metadata.version("isaaclab"),
    "isaacsim": metadata.version("isaacsim"),
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": torch.cuda.get_device_capability(0),
    "validation": "Package versions and a CUDA tensor operation; renderer untested",
}
Path("/workspace/.evo-safety-filter/versions.json").write_text(
    json.dumps(versions, indent=2) + "\n"
)
print(json.dumps(versions, indent=2))
PY

"$runtime_python" -m ipykernel install --prefix /usr/local \
    --name evo-safety-filter --display-name 'Evo safety filter (Python 3.12)'
uv pip freeze --python "$runtime_python" > "$state_dir/installed.txt"
printf '%s\n' "$lab_revision" > "$state_dir/isaaclab-revision.txt"
touch "$state_dir/ready"
printf 'Platform ready. Choose the Evo safety filter kernel in Jupyter.\n'
printf 'Isaac Sim rendering and the UNISafe port still need GPU smoke tests.\n'
