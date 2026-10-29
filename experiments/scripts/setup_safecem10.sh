#!/usr/bin/env bash
# Install the pinned Push-T planning stack into the Vast image's Python 3.12 environment,
# then run the pinned data preparation. Idempotent: reruns reconcile and skip finished work.
set +x
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
state_dir="$repo_root/outputs/safecem10/environment"
runtime_python=/venv/main/bin/python
requirements="$repo_root/configs/safecem10/requirements.txt"

for command in uv nvidia-smi flock sha256sum dpkg-query apt-get; do
    command -v "$command" >/dev/null || { printf 'Missing command: %s\n' "$command" >&2; exit 1; }
done
[[ -x "$runtime_python" ]] || { printf 'Missing /venv/main/bin/python\n' >&2; exit 1; }
[[ -s "$requirements" ]] || { printf 'Missing requirements\n' >&2; exit 1; }
[[ "$EUID" -eq 0 ]] || { printf 'Installation requires root for Ubuntu libraries.\n' >&2; exit 1; }
mkdir -p "$state_dir"
exec 9>"$state_dir/setup.lock"
flock -n 9 || { printf 'Another setup is running.\n' >&2; exit 1; }
exec > >(tee -a "$state_dir/setup.log") 2>&1
trap 'rm -f "$state_dir/ready"; printf "Setup failed; see %s/setup.log\n" "$state_dir" >&2' ERR

fingerprint="$(cat "$requirements" "${BASH_SOURCE[0]}" | sha256sum | cut -d ' ' -f 1)"
packages=(libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libegl1 zstd)
missing=()
for package in "${packages[@]}"; do
    if [[ "$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)" != 'install ok installed' ]]; then
        missing+=("$package")
    fi
done
if [[ -f "$state_dir/ready" && "$(cat "$state_dir/ready")" == "$fingerprint" && ${#missing[@]} -eq 0 ]]; then
    printf 'Platform already installed.\n'
else
    rm -f "$state_dir/ready"
    if ((${#missing[@]})); then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y --no-install-recommends "${missing[@]}"
    fi
    uv --no-config pip install --python "$runtime_python" --torch-backend auto \
        --requirement "$requirements"
    uv --no-config pip freeze --python "$runtime_python" > "$state_dir/installed-freeze.txt"
    "$runtime_python" - <<'PY'
import json, torch, stable_worldmodel, transformers, importlib.metadata as m
assert torch.cuda.is_available(), "CUDA unavailable"
info = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(),
    "stable_worldmodel": m.version("stable-worldmodel"),
    "stable_pretraining": m.version("stable-pretraining"),
    "transformers": transformers.__version__,
}
print(json.dumps(info))
open("outputs/safecem10/environment/versions.json", "w").write(json.dumps(info, indent=2) + "\n")
PY
    printf '%s\n' "$fingerprint" > "$state_dir/ready"
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$state_dir/gpus.txt"
cd "$repo_root"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export STABLEWM_HOME="$repo_root/data/safecem10/stablewm"
"$runtime_python" -u experiments/scripts/prepare_safecem10.py
printf 'Preparation complete.\n'
