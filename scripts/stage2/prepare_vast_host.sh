#!/usr/bin/env bash
# Prepare a fresh Vast "ManiGuard latent evolution" template host for stage 2 by replaying
# instance C's install (progress/receipts/01b-install.md) step for step at the pins in
# progress/receipts/01-sources.json. Run as root on the host, from the repository root,
# inside tmux:
#     tmux new -d -s prepare 'bash scripts/stage2/prepare_vast_host.sh; exec bash'
# Logs go to third_party/logs/prepare-*.log; the last lines of this script print the
# verification. Safe to re-run: finished phases are skipped by their marker files.
set -uo pipefail
ROOT=/workspace/Safety-Dial
TP="$ROOT/third_party"; LOGS="$TP/logs"; MARK="$TP/.prepare"
mkdir -p "$TP" "$LOGS" "$MARK"
cd "$ROOT"
PY=/venv/main/bin/python
export CONDA_ENVS_PATH=/opt/miniforge3/envs CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
stamp() { date -u +%FT%TZ; }
phase() {  # phase <name> <command...>: run once, log, mark on success
  local name="$1"; shift
  if [ -f "$MARK/$name" ]; then echo "$(stamp) [$name] already done"; return 0; fi
  echo "$(stamp) [$name] start"; local t0=$SECONDS
  if ( "$@" ) > "$LOGS/prepare-$name.log" 2>&1; then
    touch "$MARK/$name"; echo "$(stamp) [$name] ok ($((SECONDS-t0)) s)"; return 0
  else
    echo "$(stamp) [$name] FAILED ($((SECONDS-t0)) s), see $LOGS/prepare-$name.log"; tail -5 "$LOGS/prepare-$name.log"; return 1
  fi
}

# ---- phase 0: host and render libraries -------------------------------------------------
host_check() {
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  if [ "$(ldconfig -p | grep -c -E 'libnvoptix|libEGL_nvidia')" -lt 2 ]; then
    echo "OptiX or EGL missing; running the image's display-driver repair"
    /opt/instance-tools/bin/install-display-drivers
    ldconfig
  fi
  ldconfig -p | grep -E 'libnvoptix|libEGL_nvidia|libnvidia-rtcore|libGLX_nvidia'
  [ "$(ldconfig -p | grep -c -E 'libnvoptix|libEGL_nvidia')" -ge 2 ]
  ls /etc/vulkan/icd.d/nvidia_icd.json
  "$PY" -m pip install -q huggingface_hub
}
phase host_check host_check || exit 1

# ---- phase 1: concurrent clones, env, apt, downloads --------------------------------------
clone_maniguard() {
  [ -d "$TP/ManiGuard/.git" ] || git clone https://github.com/NU-IDEAS-Lab/ManiGuard.git "$TP/ManiGuard"
  cd "$TP/ManiGuard" && git checkout -q 6bf4c61a1f168cc46e02eb22c41bcb51815e9726 && git submodule update --init --recursive
  git -C "$TP/ManiGuard/behavior-1k" rev-parse HEAD | grep -q 88454bd04f75dc57c00ab1f1a00bcde1ff505950
}
download_benchmark() {
  "$PY" -I scripts/todo1/download_pinned.py benchmark "$TP/ManiGuard/outputs/lerobot_datasets/maniguard-bench"
}
openpi_env() {
  [ -d "$TP/openpi/.git" ] || git clone https://github.com/Physical-Intelligence/openpi.git "$TP/openpi"
  cd "$TP/openpi" && git checkout -q 215abfb217dbac7d5f1273282331b9b1866c0479 && git submodule update --init --recursive
  GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
  .venv/bin/python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"
}
apt_gl() {
  DEBIAN_FRONTEND=noninteractive apt-get install -y libsm6 libice6 libxt6 libglu1-mesa libxrender1 libxrandr2 libxcursor1 \
    libxinerama1 libxi6 libxxf86vm1 libxfixes3 libxext6 libxkbcommon0 libxcomposite1 libxdamage1 libgl1 libegl1
}
download_jar() { "$PY" -I scripts/todo1/download_pinned.py jar "$TP/checkpoints/jar"; }

( phase clone_maniguard clone_maniguard && phase download_benchmark download_benchmark ) & p_clone=$!
phase openpi_env openpi_env & p_openpi=$!
phase apt_gl apt_gl & p_apt=$!
phase download_jar download_jar & p_jar=$!
fail=0; for p in $p_clone $p_openpi $p_apt $p_jar; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "$(stamp) phase 1 had a failure; stopping"; exit 1; }

# ---- phase 2: BEHAVIOR install (serial, about 35 min), then robot asset, then deps --------
phase behavior_install bash scripts/todo1b/install_behavior_replay.sh || exit 1
download_robot() {
  "$PY" -I scripts/todo1/download_pinned.py robot "$TP/ManiGuard/behavior-1k/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger"
  ls "$TP/ManiGuard/behavior-1k/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger/usd/franka_panda_longfinger.usda"
}
phase download_robot download_robot || exit 1
phase maniguard_deps bash scripts/todo1b/install_maniguard_deps.sh || exit 1

# ---- phase 3: config, lane wrapper, verification -------------------------------------------
finish() {
  cp progress/receipts/01-jar-eval.yaml "$TP/ManiGuard/configs/eval/jar_transport_joint_legacy256.yaml"
  ln -sf "$ROOT/scripts/todo1/eval_family_lane.sh" "$TP/ManiGuard/scripts/eval_family_lane.sh"
  source /opt/miniforge3/etc/profile.d/conda.sh && conda activate behavior
  python - <<'PYEOF'
import numpy, torch, omnigibson, isaacsim, spot, maniguard
print("behavior env: numpy", numpy.__version__, "torch", torch.__version__, "cuda devices", torch.cuda.device_count(),
      "omnigibson", omnigibson.__version__, "isaacsim", getattr(isaacsim, "__version__", "?"), "spot", spot.version(), "maniguard", maniguard.__version__)
PYEOF
  pip freeze > "$LOGS/env-behavior-freeze.txt"
  (cd "$TP/openpi" && uv pip freeze > "$LOGS/env-openpi-freeze.txt")
  echo "behavior freeze: $(wc -l < "$LOGS/env-behavior-freeze.txt") lines; openpi freeze: $(wc -l < "$LOGS/env-openpi-freeze.txt") lines"
  diff <(sed 's#/tmp/[^ ]*##' "$LOGS/env-behavior-freeze.txt" | sort) <(sed 's#/tmp/[^ ]*##' progress/receipts/01b-env-behavior-freeze.txt | sort) | grep '^[<>]' | head -20 || true
  df -h / | tail -1
}
phase finish finish || exit 1
echo "$(stamp) PREPARE DONE: host ready for the stage 2 smoke (scripts/todo1b/serve_jar_lane.sh, run_smoke_lane.sh)"
