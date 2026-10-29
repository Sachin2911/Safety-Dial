#!/usr/bin/env bash
# Explicit platform installation in the Vast image's existing Python 3.12 environment.
set +x
set -euo pipefail
umask 077

usage() {
    cat <<'HELP'
Usage: bash experiments/scripts/setup_unisafe09.sh [--dry-run | --check]
  (no flag)  Install/reconcile the pinned platform, validate CUDA, prepare UNISafe.
  --dry-run  Check host prerequisites and print the plan; no installs or state writes.
  --check    Validate installed package pins and CUDA; no installs or state writes.
Requires Ubuntu 24.04 x86-64, /venv/main Python 3.12, uv and an NVIDIA GPU.
Research repositories, data and checkpoints are separate preparation steps.
HELP
}
mode=install
case "${1:-}" in
    '') ;;
    --dry-run) mode=plan ;;
    --check) mode=check ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
if (($# > 1)); then usage >&2; exit 2; fi

if [[ "${SAFETY_DIAL_PLATFORM:-}" == maniguard ]]; then
    printf '%s\n' 'This installer targets UNISafe with Python 3.12 and CUDA 13.' >&2
    printf '%s\n' 'For the ManiGuard base, see configs/vast/README.md for manual preparation.' >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
state_dir="$repo_root/outputs/unisafe09/environment"
runtime_python=/venv/main/bin/python
requirements="$repo_root/configs/unisafe09/requirements-platform.txt"
overrides="$repo_root/configs/unisafe09/uv-overrides.txt"
checker="$script_dir/check_unisafe09_platform.py"

for command in uv nvidia-smi flock sha256sum dpkg-query apt-get git; do
    command -v "$command" >/dev/null || { printf 'Missing command: %s\n' "$command" >&2; exit 1; }
done
[[ -x "$runtime_python" ]] || { printf 'Missing /venv/main/bin/python\n' >&2; exit 1; }
[[ -s "$requirements" && -s "$overrides" ]] || { printf 'Missing platform pins\n' >&2; exit 1; }
"$runtime_python" "$checker" --host-only

if [[ "$mode" == check ]]; then
    exec "$runtime_python" "$checker"
fi
if [[ "$mode" == plan ]]; then
    printf 'Target: %s\nRequirements: %s\nOverrides: %s\n' "$runtime_python" "$requirements" "$overrides"
    printf 'Install Ubuntu libraries, pinned CUDA 13.0 PyTorch and Isaac packages using uv.\n'
    printf 'Validate all package pins and CUDA; verify frozen models.\n'
    printf 'Logs and validation records: %s\n' "$state_dir"
    df -h /venv/main /workspace
    exit 0
fi
[[ "$EUID" -eq 0 ]] || { printf 'Installation requires root for Ubuntu libraries.\n' >&2; exit 1; }
mkdir -p "$state_dir"
exec 9>"$state_dir/setup.lock"
flock -n 9 || { printf 'Another platform setup is running.\n' >&2; exit 1; }
exec > >(tee -a "$state_dir/setup.log") 2>&1
trap 'rm -f "$state_dir/ready" "$state_dir/versions.json.tmp"; printf "Setup failed; rerun after checking %s/setup.log\n" "$state_dir" >&2' ERR
trap 'rm -f "$state_dir/ready" "$state_dir/versions.json.tmp"; exit 130' INT TERM

fingerprint="$(cat "$requirements" "$overrides" "$checker" "${BASH_SOURCE[0]}" | sha256sum | cut -d ' ' -f 1)"
packages=(
    libgl1 libegl1 libglvnd0 libglu1-mesa libvulkan1 vulkan-tools
    libx11-6 libxcursor1 libxi6 libxinerama1 libxrandr2 libxrender1
    libxext6 libsm6 libice6 libnss3 libglib2.0-0 libgomp1 libxt6
    libasound2t64 ffmpeg build-essential cmake pkg-config
)
missing=()
for package in "${packages[@]}"; do
    if [[ "$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)" != 'install ok installed' ]]; then
        missing+=("$package")
    fi
done

validated=false
if [[ -f "$state_dir/ready" && "$(cat "$state_dir/ready")" == "$fingerprint" && ${#missing[@]} -eq 0 ]]; then
    printf 'Checking the previously installed platform.\n'
    if "$runtime_python" "$checker" > "$state_dir/versions.json.tmp"; then
        validated=true
    else
        printf 'Environment drift detected; reconciling the pinned packages.\n'
    fi
fi
# No previous success marker survives a partial or failed reconciliation.
rm -f "$state_dir/ready"
if [[ "$validated" != true ]]; then
    if ((${#missing[@]})); then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y --no-install-recommends "${missing[@]}"
    fi
    # Install rather than sync: retain the base image's Jupyter/management packages.
    # --no-config isolates this from the older research project's uv configuration.
    uv --no-config pip install --python "$runtime_python" \
        --torch-backend cu130 --extra-index-url https://pypi.nvidia.com \
        --index-strategy unsafe-best-match --prerelease allow \
        --overrides "$overrides" --requirement "$requirements"
    "$runtime_python" "$checker" > "$state_dir/versions.json.tmp"
fi
uv --no-config pip install --python "$runtime_python" \
    --constraint "$requirements" --overrides "$overrides" --torch-backend cu130 \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match \
    --prerelease allow --requirement "$repo_root/configs/unisafe09/requirements-policy.txt"
# Checkpoints arrive from the workstation. No HF credential is sent to this host.
for ((attempt=0; attempt<360; attempt++)); do
    [[ -f "$repo_root/data/unisafe09/pretrained/transfer-ready.json" ]] && break
    sleep 5
done
"$runtime_python" "$repo_root/experiments/scripts/download_unisafe09.py" --verify-only
"$runtime_python" -c 'import sys; sys.path.insert(0,"experiments"); from helpers.unisafe09_source import prepare_source; print(prepare_source())'
uv --no-config pip freeze --python "$runtime_python" > "$state_dir/installed-freeze.txt"
"$runtime_python" "$checker" > "$state_dir/versions.json.tmp"
cp -- "$requirements" "$state_dir/requirements-linux-py312.txt"
cp -- "$overrides" "$state_dir/uv-overrides.txt"
mv -- "$state_dir/versions.json.tmp" "$state_dir/versions.json"
printf '%s\n' "$fingerprint" > "$state_dir/ready.tmp"
mv -- "$state_dir/ready.tmp" "$state_dir/ready"
printf 'Platform packages and CUDA verified. UNISafe task-9 environment is ready.\n'
printf 'Run Python with /venv/main/bin/python. Simulator and cameras still require smoke tests.\n'
