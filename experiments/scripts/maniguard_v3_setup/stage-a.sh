#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8 DEBIAN_FRONTEND=noninteractive
export XDG_CACHE_HOME=/workspace/.cache PIP_CACHE_DIR=/workspace/.cache/pip
export UV_CACHE_DIR=/workspace/.cache/uv HF_HOME=/workspace/.cache/huggingface
export TMPDIR=/workspace/tmp GIT_LFS_SKIP_SMUDGE=1
export CONDA_ENVS_PATH=/workspace/miniforge3/envs
export CONDA_PKGS_DIRS=/workspace/miniforge3/pkgs
SETUP=/workspace/setup-maniguard-v3-20261030
TP=/workspace/Safety-Dial/third_party
mkdir -p "$SETUP/logs" "$TP" "$TMPDIR" /workspace/bin
trap 'printf "stage-a failed at line %s\n" "$LINENO" > "$SETUP/stage-a.failed"' ERR
date -u +%FT%TZ > "$SETUP/stage-a.started"

apt-get update -qq > "$SETUP/logs/apt.log" 2>&1
apt-get install -y -qq --no-install-recommends \
  tmux git git-lfs curl wget ca-certificates build-essential pkg-config ninja-build \
  bzip2 ffmpeg unzip ripgrep libsm6 libice6 libxt6 libglu1-mesa libxrender1 \
  libxrandr2 libxcursor1 libxinerama1 libxi6 libxxf86vm1 libxfixes3 libxext6 \
  libxkbcommon0 libxcomposite1 libxdamage1 libgl1 libegl1 libvulkan1 vulkan-tools \
  libnss3 libnspr4 libglib2.0-0 libgomp1 >> "$SETUP/logs/apt.log" 2>&1

(
  if [[ ! -d "$TP/ManiGuard/.git" ]]; then
    git clone --filter=blob:none --no-checkout \
      https://github.com/NU-IDEAS-Lab/ManiGuard.git "$TP/ManiGuard"
  fi
  git -C "$TP/ManiGuard" checkout --detach 6bf4c61a1f168cc46e02eb22c41bcb51815e9726
  git -C "$TP/ManiGuard" submodule update --init --recursive --depth 1
  git -C "$TP/ManiGuard/behavior-1k" rev-parse HEAD
  touch "$SETUP/maniguard-source.ready"
) > "$SETUP/logs/maniguard-source.log" 2>&1 &
MANIGUARD_JOB=$!

(
  if [[ ! -d "$TP/openpi/.git" ]]; then
    git clone --filter=blob:none --no-checkout \
      https://github.com/Physical-Intelligence/openpi.git "$TP/openpi"
  fi
  git -C "$TP/openpi" checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479
  git -C "$TP/openpi" submodule update --init --recursive --depth 1
  touch "$SETUP/openpi-source.ready"
) > "$SETUP/logs/openpi-source.log" 2>&1 &
OPENPI_JOB=$!

(
  if [[ ! -x /workspace/miniforge3/bin/conda ]]; then
    curl -fsSL -o "$SETUP/Miniforge3.sh" \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    sha256sum "$SETUP/Miniforge3.sh" > "$SETUP/miniforge-installer.sha256"
    bash "$SETUP/Miniforge3.sh" -b -p /workspace/miniforge3
  fi
  /workspace/miniforge3/bin/conda --version
  touch "$SETUP/miniforge.ready"
) > "$SETUP/logs/miniforge.log" 2>&1 &
MINIFORGE_JOB=$!

curl -LsSf https://astral.sh/uv/install.sh -o "$SETUP/install-uv.sh"
UV_INSTALL_DIR=/workspace/bin bash "$SETUP/install-uv.sh" > "$SETUP/logs/uv.log" 2>&1
/workspace/bin/uv --version
wait "$MANIGUARD_JOB"
wait "$OPENPI_JOB"
wait "$MINIFORGE_JOB"
date -u +%FT%TZ > "$SETUP/stage-a.complete"
