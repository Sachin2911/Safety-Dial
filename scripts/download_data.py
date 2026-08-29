"""Clone LeWM source, then download Hub checkpoints and expert datasets.

Usage (from repo root):
  uv run python scripts/download_data.py
  uv run python scripts/download_data.py --config-name pusht
  uv run python scripts/download_data.py --config-name cube
  uv run python scripts/download_data.py weights_only=true
  uv run python scripts/download_data.py clone_source=false
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import hydra
import torch
import zstandard as zstd
from huggingface_hub import hf_hub_download, snapshot_download
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_dotenv(env_file: Path) -> None:
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def ensure_stablewm_home() -> Path:
    default = REPO_ROOT / "data" / "stablewm"
    home = Path(os.environ.get("STABLEWM_HOME", default)).expanduser()
    if not home.is_absolute():
        home = (REPO_ROOT / home).resolve()
    home.mkdir(parents=True, exist_ok=True)
    os.environ["STABLEWM_HOME"] = str(home)

    env_file = REPO_ROOT / ".env"
    existing = env_file.read_text() if env_file.is_file() else ""
    if not re.search(r"^\s*STABLEWM_HOME=", existing, flags=re.MULTILINE):
        with env_file.open("a") as fh:
            fh.write(f'STABLEWM_HOME="{home}"\n')
        print(f"[download] wrote STABLEWM_HOME={home} to {env_file}")
    else:
        print("[download] STABLEWM_HOME already in .env")
    return home


def clone_lewm_source(cfg: DictConfig) -> Path:
    dest = REPO_ROOT / str(cfg.get("source_dir", "third_party/le-wm"))
    url = os.environ.get("LEWM_REPO_URL") or str(
        cfg.get("source_repo", "https://github.com/lucas-maes/le-wm.git")
    )
    git_ref = os.environ.get("LEWM_GIT_REF") or str(cfg.get("source_ref", "") or "")

    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").is_dir():
        print(f"[download] le-wm clone already present: {dest}")
    else:
        print(f"[download] cloning {url} -> {dest}")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
        )

    if git_ref:
        print(f"[download] checking out LEWM_GIT_REF={git_ref}")
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", git_ref],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", git_ref],
            check=True,
        )
    return dest


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


def download_weights(home: Path, task: DictConfig) -> Path:
    dest = home / str(task.hf_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "weights.pt").is_file() and (dest / "config.json").is_file():
        print(f"[download] HF weights already present: {dest}")
        return dest

    print(f"[download] snapshot {task.model_repo} -> {dest}")
    snapshot_download(
        repo_id=str(task.model_repo),
        repo_type="model",
        local_dir=str(dest),
    )
    return dest


def convert_object_ckpt(home: Path, task: DictConfig, src: Path) -> Path:
    out = home / str(task.ckpt_relpath)
    if out.is_file():
        print(f"[download] object checkpoint already present: {out}")
        return out

    print(f"[download] converting HF weights -> {out}")
    cfg = json.loads((src / "config.json").read_text())
    model = instantiate(cfg)
    sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(remap_vit_keys(sd), strict=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)
    print(f"[download] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def download_dataset_file(task: DictConfig) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / str(task.dataset_file)
    if dest.is_file():
        print(f"[download] dataset archive already present: {dest}")
        return dest

    print(f"[download] fetching {task.dataset_repo}/{task.dataset_file} -> {RAW_DIR}")
    path = Path(
        hf_hub_download(
            repo_id=str(task.dataset_repo),
            filename=str(task.dataset_file),
            repo_type="dataset",
            local_dir=str(RAW_DIR),
        )
    )
    print(f"[download] wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def processed_ready(dest: Path, kind: str) -> bool:
    if kind == "zstd_h5":
        return dest.is_file()
    if kind == "zstd_tar":
        return dest.is_dir() and any(dest.iterdir())
    raise ValueError(f"unknown dataset_kind: {kind}")


def decompress_dataset(src: Path, task: DictConfig) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED_DIR / str(task.processed_relpath)
    kind = str(task.dataset_kind)
    if processed_ready(dest, kind):
        print(f"[download] decompressed data already present: {dest}")
        return dest

    print(f"[download] decompressing {src} -> {dest}")
    if kind == "zstd_h5":
        dctx = zstd.ZstdDecompressor()
        with open(src, "rb") as fin, open(dest, "wb") as fout:
            dctx.copy_stream(fin, fout)
        print(f"[download] wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    if kind == "zstd_tar":
        dest.mkdir(parents=True, exist_ok=True)
        dctx = zstd.ZstdDecompressor()
        with open(src, "rb") as fin, dctx.stream_reader(fin) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(path=dest)
        print(f"[download] extracted archive into {dest}")
        return dest

    raise ValueError(f"unknown dataset_kind: {kind}")


def run_task(home: Path, task: DictConfig, weights_only: bool) -> None:
    print(f"[download] === {task.name} ===")
    src = download_weights(home, task)
    convert_object_ckpt(home, task, src)
    if weights_only:
        print(f"[download] skipping dataset for {task.name} (weights_only)")
        return
    archive = download_dataset_file(task)
    decompress_dataset(archive, task)


@hydra.main(
    version_base=None,
    config_path="../configs/download",
    config_name="all",
)
def main(cfg: DictConfig) -> None:
    os.chdir(REPO_ROOT)
    load_dotenv(REPO_ROOT / ".env")
    home = ensure_stablewm_home()
    weights_only = bool(cfg.get("weights_only", False))
    tasks = cfg.get("tasks")
    if not tasks:
        print("[download] ERROR: config has no tasks", file=sys.stderr)
        raise SystemExit(1)

    print(f"[download] repo: {REPO_ROOT}")
    print(f"[download] STABLEWM_HOME={home}")
    print(f"[download] weights_only={weights_only}")
    print(OmegaConf.to_yaml(cfg.tasks))

    clone_source = bool(cfg.get("clone_source", True))
    if clone_source:
        clone_lewm_source(cfg)
    else:
        print("[download] skipping le-wm source clone (clone_source=false)")

    for task in tasks:
        run_task(home, task, weights_only)

    print("[download] done")
    print("[download] smoke load:")
    print(f"  export STABLEWM_HOME={home}")
    for task in tasks:
        print(
            "  uv run python -c "
            f"\"import stable_worldmodel as swm; "
            f"m = swm.policy.AutoCostModel('{task.swm_id}'); print(type(m))\""
        )


if __name__ == "__main__":
    main()
