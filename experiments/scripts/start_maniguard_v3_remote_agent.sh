#!/usr/bin/env bash
set -euo pipefail
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Account credentials are deliberately outside every evidence and Git directory.
credential_file=/root/.config/maniguard-v3/handoff.env
if [[ -r "$credential_file" ]]; then
    source "$credential_file"
fi
export GIT_ASKPASS=/root/.config/maniguard-v3/git-askpass.py
export GIT_TERMINAL_PROMPT=0
export GIT_COMMITTER_NAME="${GIT_AUTHOR_NAME:-}"
export GIT_COMMITTER_EMAIL="${GIT_AUTHOR_EMAIL:-}"

cd /workspace/maniguard-evolution-v3-20261030
exec codex --full-auto \
    --add-dir /workspace/maniguard-v3-agent \
    --add-dir /workspace/maniguard-v3-preservation \
    --add-dir /workspace/maniguard-v3-verification \
    -c sandbox_workspace_write.network_access=true "$@"
