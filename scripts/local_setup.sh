#!/usr/bin/env bash
# Local laptop / workstation setup for SafetyDial.
# Usage (from anywhere):
#   bash scripts/local_setup.sh
#   # or after chmod +x:
#   ./scripts/local_setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

echo "[local_setup] repo: ${REPO_ROOT}"
load_dotenv
ensure_uv
configure_git_identity
configure_github_https
sync_python_env
configure_jupyter_kernel
# HF login after sync so huggingface_hub is available if added later
configure_huggingface
sanity_check
echo "[local_setup] done"
echo "[local_setup] tip: LeWM substrate (clone + HF ckpt): bash scripts/lewm/setup_lewm.sh"
