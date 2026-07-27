# Shared helpers for local_setup.sh and vast_setup.sh.
# shellcheck shell=bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

load_dotenv() {
  local env_file="${REPO_ROOT}/.env"
  if [[ -f "${env_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
    echo "[setup] loaded ${env_file}"
  fi
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    echo "[setup] uv: $(uv --version)"
    return 0
  fi
  echo "[setup] installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] ERROR: uv not found after install" >&2
    return 1
  fi
  echo "[setup] uv: $(uv --version)"
}

configure_git_identity() {
  # Prefer repo-local config so we do not rewrite the user's global ~/.gitconfig.
  cd "${REPO_ROOT}"
  if [[ -n "${GIT_AUTHOR_NAME:-}" ]]; then
    git config user.name "${GIT_AUTHOR_NAME}"
    echo "[setup] git user.name=${GIT_AUTHOR_NAME} (repo-local)"
  fi
  if [[ -n "${GIT_AUTHOR_EMAIL:-}" ]]; then
    git config user.email "${GIT_AUTHOR_EMAIL}"
    echo "[setup] git user.email=${GIT_AUTHOR_EMAIL} (repo-local)"
  fi
}

configure_github_https() {
  if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "[setup] GITHUB_TOKEN unset; clone/pull of public repos still works; push from this machine will need auth"
    return 0
  fi
  cd "${REPO_ROOT}"
  # Repo-local helper; credentials still land in ~/.git-credentials when using "store".
  git config credential.helper store
  if printf "protocol=https\nhost=github.com\nusername=x-access-token\npassword=%s\n\n" \
    "${GITHUB_TOKEN}" | git credential approve 2>/dev/null; then
    echo "[setup] GitHub HTTPS credentials stored for push"
  else
    echo "[setup] WARN: could not write git credentials (check home dir permissions); GITHUB_TOKEN remains in env"
  fi
}

configure_huggingface() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[setup] HF_TOKEN unset; skipping huggingface-cli login"
    return 0
  fi
  if uv run huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null; then
    echo "[setup] Hugging Face CLI logged in"
  else
    # huggingface_hub may not be installed yet; env var alone is enough for most downloads
    echo "[setup] HF_TOKEN present (CLI login skipped; token will be used via env)"
  fi
}

sync_python_env() {
  cd "${REPO_ROOT}"
  # After a folder rename, a previously activated venv can leave VIRTUAL_ENV
  # pointing at the old absolute path; uv then warns and ignores it.
  local expected_venv="${REPO_ROOT}/.venv"
  if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${expected_venv}" ]]; then
    echo "[setup] clearing stale VIRTUAL_ENV=${VIRTUAL_ENV} (expected ${expected_venv})"
    unset VIRTUAL_ENV
  fi
  if [[ -f "${REPO_ROOT}/uv.lock" ]]; then
    echo "[setup] uv sync --frozen"
    uv sync --frozen --extra dev
  else
    echo "[setup] uv sync (no lockfile yet)"
    uv sync --extra dev
  fi
}

configure_jupyter_kernel() {
  cd "${REPO_ROOT}"
  local kernel_name="safetydial-venv"
  local display_name="Python (safetydial .venv)"

  # Keep kernel registration tied to the repo venv used by uv.
  if ! uv run python -c "import ipykernel" >/dev/null 2>&1; then
    echo "[setup] installing ipykernel into repo venv"
    uv pip install ipykernel
  fi

  if uv run python -m ipykernel install --user --name "${kernel_name}" --display-name "${display_name}" >/dev/null 2>&1; then
    echo "[setup] registered Jupyter kernel: ${display_name}"
  else
    echo "[setup] WARN: failed to register Jupyter kernel (${kernel_name})"
  fi
}

sanity_check() {
  cd "${REPO_ROOT}"
  echo "[setup] sanity check..."
  uv run python - <<'PY'
import importlib
import os

mods = ["torch", "gymnasium", "hydra", "wandb", "pymoo", "numpy"]
missing = []
for name in mods:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{name}: {exc}")

import torch

print(f"  python ok | torch {torch.__version__} | cuda={torch.cuda.is_available()}")
print(f"  WANDB_API_KEY set: {bool(os.environ.get('WANDB_API_KEY'))}")
print(f"  GITHUB_TOKEN set: {bool(os.environ.get('GITHUB_TOKEN'))}")
print(f"  HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
if missing:
    print("  missing imports:")
    for line in missing:
        print(f"    - {line}")
    raise SystemExit(1)
print("[setup] all core imports ok")
PY
}

export_env_for_ssh_sessions() {
  # Vast SSH/Jupyter often hide custom env vars unless written here.
  if [[ "${EUID:-$(id -u)}" -eq 0 ]] && [[ -w /etc/environment ]]; then
    env >> /etc/environment
    echo "[setup] appended current env to /etc/environment (SSH sessions)"
  else
    echo "[setup] skip /etc/environment (not root or not writable)"
  fi
}
