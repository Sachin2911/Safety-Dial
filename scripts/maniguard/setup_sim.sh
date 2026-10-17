#!/usr/bin/env bash
# Install research packages only in the dedicated ManiGuard simulator environment.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
SIM_ENV="${ROOT}/data/maniguard/envs/sim"
UPSTREAM="${ROOT}/third_party/ManiGuard"
CONSTRAINTS="${ROOT}/configs/maniguard/sim-constraints.txt"
cd "${ROOT}"
# These userspace rendering libraries are not supplied by the NVIDIA driver.
if ! ldconfig -p | grep 'libGLU.so.1' >/dev/null || ! ldconfig -p | grep 'libXt.so.6' >/dev/null; then
    apt-get update -qq
    apt-get install -y --no-install-recommends libglu1-mesa libxt6
fi
if [[ ! -x "${SIM_ENV}/bin/python" ]]; then
    conda create --prefix "${SIM_ENV}" --override-channels -c conda-forge python=3.10 spot -y
fi
"${SIM_ENV}/bin/python" -c 'import spot, buddy; assert hasattr(spot, "translate")'
uv pip install --python "${SIM_ENV}/bin/python" \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "${SIM_ENV}/bin/python" -c "${CONSTRAINTS}" \
    'numpy==1.26.4' 'setuptools==79.0.1' \
    -e "${UPSTREAM}/behavior-1k/bddl3" \
    -e "${UPSTREAM}/behavior-1k/OmniGibson" \
    -e "${UPSTREAM}[serve]" \
    -e "${ROOT}/third_party/openpi/packages/openpi-client" \
    pytest ruff
# Isaac Sim's packaged runtime and local extension cache avoid startup downloads.
OMNI_KIT_ACCEPT_EULA=YES uv pip install --python "${SIM_ENV}/bin/python" \
    -c "${CONSTRAINTS}" 'isaacsim[all,extscache]==4.5.0' \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
uv pip freeze --python "${SIM_ENV}/bin/python" > "${ROOT}/data/maniguard/sim-freeze.txt"
