#!/usr/bin/env bash
# Vast.ai instance setup for SafetyDial.
# Run after cloning the repo (or from vast_onstart.sh):
#   bash scripts/vast_setup.sh
#
# Expects secrets from Vast Account Environment Variables and/or repo .env:
#   WANDB_API_KEY, GITHUB_TOKEN, HF_TOKEN
# Optional overrides (defaults already set for commits):
#   GIT_AUTHOR_NAME, GIT_AUTHOR_EMAIL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

echo "[vast_setup] repo: ${REPO_ROOT}"
export_env_for_ssh_sessions
load_dotenv
ensure_uv
configure_git_identity
configure_github_https
sync_python_env
configure_jupyter_kernel
configure_huggingface
sanity_check
echo "[vast_setup] done"
echo "[vast_setup] tip: uv run python ...   |   git pull && bash scripts/vast_setup.sh"
