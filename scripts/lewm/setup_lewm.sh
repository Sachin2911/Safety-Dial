#!/usr/bin/env bash
# Clone le-wm, download the Push-T HF checkpoint, convert to _object.ckpt.
# Idempotent — safe to re-run.
#
# Usage (from anywhere):
#   bash scripts/lewm/setup_lewm.sh
#
# Optional env:
#   STABLEWM_HOME   checkpoint/data root (default: <repo>/data/stablewm)
#   LEWM_GIT_REF    git ref to checkout after clone (default: tip of default branch)
#   LEWM_REVISION   Hugging Face revision for quentinll/lewm-pusht (default: Hub tip)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../_common.sh"

LEWM_DIR="${REPO_ROOT}/third_party/le-wm"
LEWM_REPO_URL="${LEWM_REPO_URL:-https://github.com/lucas-maes/le-wm.git}"
HF_MODEL_ID="${HF_MODEL_ID:-quentinll/lewm-pusht}"

ensure_stablewm_home() {
  local default_home="${REPO_ROOT}/data/stablewm"
  if [[ -z "${STABLEWM_HOME:-}" ]]; then
    export STABLEWM_HOME="${default_home}"
  fi
  mkdir -p "${STABLEWM_HOME}"

  local env_file="${REPO_ROOT}/.env"
  if [[ -f "${env_file}" ]] && grep -qE '^[[:space:]]*STABLEWM_HOME=' "${env_file}"; then
    echo "[lewm] STABLEWM_HOME already in .env"
  else
    echo "STABLEWM_HOME=\"${STABLEWM_HOME}\"" >> "${env_file}"
    echo "[lewm] wrote STABLEWM_HOME=${STABLEWM_HOME} to ${env_file}"
  fi
}

clone_lewm() {
  if [[ -d "${LEWM_DIR}/.git" ]]; then
    echo "[lewm] clone already present: ${LEWM_DIR}"
  else
    mkdir -p "$(dirname "${LEWM_DIR}")"
    echo "[lewm] cloning ${LEWM_REPO_URL} -> ${LEWM_DIR}"
    git clone --depth 1 "${LEWM_REPO_URL}" "${LEWM_DIR}"
  fi
  if [[ -n "${LEWM_GIT_REF:-}" ]]; then
    echo "[lewm] checking out LEWM_GIT_REF=${LEWM_GIT_REF}"
    git -C "${LEWM_DIR}" fetch --depth 1 origin "${LEWM_GIT_REF}"
    git -C "${LEWM_DIR}" checkout "${LEWM_GIT_REF}"
  fi
}

download_hf_weights() {
  local dest="${STABLEWM_HOME}/hf_pusht"
  mkdir -p "${dest}"
  if [[ -f "${dest}/weights.pt" && -f "${dest}/config.json" ]]; then
    echo "[lewm] HF weights already present: ${dest}"
    return 0
  fi

  echo "[lewm] downloading ${HF_MODEL_ID} -> ${dest}"
  if [[ -n "${LEWM_REVISION:-}" ]]; then
    echo "[lewm] using LEWM_REVISION=${LEWM_REVISION}"
  fi
  STABLEWM_HOME="${STABLEWM_HOME}" \
  HF_MODEL_ID="${HF_MODEL_ID}" \
  LEWM_REVISION="${LEWM_REVISION:-}" \
    uv run python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

dest = Path(os.environ["STABLEWM_HOME"]) / "hf_pusht"
kwargs = {
    "repo_id": os.environ["HF_MODEL_ID"],
    "repo_type": "model",
    "local_dir": str(dest),
}
rev = os.environ.get("LEWM_REVISION") or None
if rev:
    kwargs["revision"] = rev
path = snapshot_download(**kwargs)
print(f"[lewm] downloaded to {path}")
PY
}

convert_object_ckpt() {
  local out="${STABLEWM_HOME}/checkpoints/pusht/lewm_object.ckpt"
  if [[ -f "${out}" ]]; then
    echo "[lewm] object checkpoint already present: ${out}"
    return 0
  fi

  echo "[lewm] converting HF weights -> ${out}"
  cd "${REPO_ROOT}"
  STABLEWM_HOME="${STABLEWM_HOME}" \
    uv run python - <<'PY'
import json
import os
from pathlib import Path

import torch
from hydra.utils import instantiate


def remap_vit_keys(sd: dict) -> dict:
    """Map pre-transformers-v5 ViT keys to the current vit_hf layout."""
    out = {}
    for key, value in sd.items():
        nk = key.replace("encoder.encoder.layer.", "encoder.layers.")
        nk = nk.replace("attention.attention.query", "attention.q_proj")
        nk = nk.replace("attention.attention.key", "attention.k_proj")
        nk = nk.replace("attention.attention.value", "attention.v_proj")
        nk = nk.replace("attention.output.dense", "attention.o_proj")
        nk = nk.replace("intermediate.dense", "mlp.fc1")
        nk = nk.replace(".output.dense", ".mlp.fc2")
        out[nk] = value
    return out


cache = Path(os.environ["STABLEWM_HOME"])
src = cache / "hf_pusht"
out = cache / "checkpoints" / "pusht" / "lewm_object.ckpt"

cfg = json.loads((src / "config.json").read_text())
model = instantiate(cfg)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(remap_vit_keys(sd), strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
print(f"[lewm] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
PY
}

smoke_tip() {
  cat <<EOF
[lewm] done
[lewm] STABLEWM_HOME=${STABLEWM_HOME}
[lewm] smoke load:
  export STABLEWM_HOME=${STABLEWM_HOME}
  uv run python -c "import stable_worldmodel as swm; m = swm.policy.AutoCostModel('pusht/lewm'); print(type(m))"
EOF
}

echo "[lewm] repo: ${REPO_ROOT}"
cd "${REPO_ROOT}"
load_dotenv
ensure_uv

# Parent shells on Vast often have VIRTUAL_ENV=/venv/main; uv then ignores the project .venv.
expected_venv="${REPO_ROOT}/.venv"
if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${expected_venv}" ]]; then
  echo "[lewm] clearing stale VIRTUAL_ENV=${VIRTUAL_ENV} (expected ${expected_venv})"
  unset VIRTUAL_ENV
fi

ensure_stablewm_home
clone_lewm
download_hf_weights
convert_object_ckpt
smoke_tip
