#!/usr/bin/env bash
# Prepare a fresh RunPod pod (runpod/pytorch 2.4.0, CUDA 12.4.1, Ubuntu 22.04) for a Claude
# Code agent working on this repository: Node and Claude Code, the repository clone under
# /workspace, tmux, and a `jar` tmux session with `claude` started in it. Run as root on the
# pod. Idempotent. Secrets come from the pod environment (set at creation); nothing is printed.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq tmux git curl ca-certificates >/dev/null
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
  apt-get install -y -qq nodejs >/dev/null
fi
npm install -g @anthropic-ai/claude-code >/dev/null 2>&1 || npm install -g @anthropic-ai/claude-code
mkdir -p /workspace && cd /workspace
if [ ! -d Safety-Dial/.git ]; then
  git -c credential.helper= clone "https://x-access-token:${GITHUB_TOKEN}@github.com/Sachin2911/Safety-Dial.git" Safety-Dial 2>/dev/null \
    || git clone https://github.com/Sachin2911/Safety-Dial.git Safety-Dial
  git -C Safety-Dial remote set-url origin https://github.com/Sachin2911/Safety-Dial.git
fi
cd Safety-Dial && git pull -q --ff-only origin main || true
git config --global user.name "${GIT_AUTHOR_NAME:-Sachin}"
git config --global user.email "${GIT_AUTHOR_EMAIL:-2699183@students.wits.ac.za}"
git config --global credential.helper '!f() { echo "username=x-access-token"; echo "password=${GITHUB_TOKEN}"; }; f'
echo "--- host check ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "render libraries (need libnvoptix and libEGL_nvidia): $(ldconfig -p | grep -c -E 'libnvoptix|libEGL_nvidia')"
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null | tr '\n' ' '; echo
df -h /workspace | tail -1
tmux has-session -t jar 2>/dev/null || tmux new-session -d -s jar -c /workspace/Safety-Dial 'claude; exec bash'
echo "ready: tmux attach -t jar, sign in, paste progress/goal-prompt-02.md"
