#!/usr/bin/env bash
# SafetyDial environment setup (local laptop or Vast.ai).
#
# In a clone:
#   bash scripts/setup.sh
#   bash scripts/setup.sh --pull    # git pull, then setup
#
# Vast On-start (paste this, or curl | bash). Clones/pulls then re-runs from the repo:
#   curl -fsSL https://raw.githubusercontent.com/Sachin2911/Safety-Dial/main/scripts/setup.sh | bash
#
# Env (Vast account vars and/or repo .env):
#   WANDB_API_KEY, GITHUB_TOKEN, HF_TOKEN
# Optional: REPO_URL, REPO_DIR, GIT_AUTHOR_NAME, GIT_AUTHOR_EMAIL

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Sachin2911/Safety-Dial.git}"

is_vast() {
  [[ -f /etc/vast_capabilities.json ]] \
    || [[ -n "${CONTAINER_ID:-}" ]] \
    || [[ -n "${VAST_CONTAINERLABEL:-}" ]]
}

want_pull=0
for arg in "$@"; do
  case "${arg}" in
    --pull) want_pull=1 ;;
    -h|--help)
      sed -n '2,16p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "[setup] unknown argument: ${arg}" >&2
      echo "[setup] usage: bash scripts/setup.sh [--pull]" >&2
      exit 1
      ;;
  esac
done

script_path="${BASH_SOURCE[0]:-}"
script_dir=""
if [[ -n "${script_path}" && -f "${script_path}" ]]; then
  script_dir="$(cd "$(dirname "${script_path}")" && pwd)"
fi

# Piped (curl | bash) or pasted On-start: helpers/_common.sh is not next to this file.
if [[ -z "${script_dir}" || ! -f "${script_dir}/helpers/_common.sh" ]]; then
  if is_vast; then
    REPO_DIR="${REPO_DIR:-/workspace/Safety-Dial}"
  else
    REPO_DIR="${REPO_DIR:-}"
    if [[ -z "${REPO_DIR}" ]]; then
      echo "[setup] ERROR: not running from a clone. Set REPO_DIR or run: bash scripts/setup.sh" >&2
      exit 1
    fi
  fi

  if [[ "${EUID:-$(id -u)}" -eq 0 ]] && [[ -w /etc/environment ]]; then
    env >> /etc/environment
    echo "[setup] appended current env to /etc/environment (SSH sessions)"
  fi

  mkdir -p "$(dirname "${REPO_DIR}")"
  if [[ -d "${REPO_DIR}/.git" ]]; then
    echo "[setup] pulling ${REPO_DIR}"
    git -C "${REPO_DIR}" pull --ff-only || git -C "${REPO_DIR}" pull
  else
    echo "[setup] cloning ${REPO_URL} -> ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
  fi
  exec bash "${REPO_DIR}/scripts/setup.sh"
fi

# shellcheck disable=SC1091
source "${script_dir}/helpers/_common.sh"

if ((want_pull)); then
  echo "[setup] git pull --ff-only"
  git -C "${REPO_ROOT}" pull --ff-only || git -C "${REPO_ROOT}" pull
fi

echo "[setup] repo: ${REPO_ROOT}"
if is_vast; then
  echo "[setup] environment: vast"
  export_env_for_ssh_sessions
else
  echo "[setup] environment: local"
fi

load_dotenv
ensure_uv
if is_vast; then
  ensure_system_deps
fi
configure_git_identity
configure_github_https
sync_python_env
configure_jupyter_kernel
configure_huggingface
sanity_check
echo "[setup] done"
echo "[setup] tip: uv run python ...   |   git pull && bash scripts/setup.sh"
echo "[setup] tip: LeWM source + Hub weights/data: uv run python scripts/download_data.py"
