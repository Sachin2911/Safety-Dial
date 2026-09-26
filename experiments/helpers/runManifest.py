"""Run manifests: git, upstream, package, hardware and data fingerprints.

Every run folder that goes to Hugging Face carries a `manifest.json` built here, so a
number in the thesis can be traced to a repository commit, an upstream LeWM commit, a
package set, a GPU and the exact data it consumed ([infrastructure.md]).

    from helpers.runManifest import build_manifest, write_manifest, file_sha256
    m = build_manifest(run_id="pusht-assets-20260926-1", seeds={"numpy": 0}, extra={...})
    write_manifest(run_dir, m)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEWM_DIR = REPO_ROOT / "third_party" / "le-wm"

_PACKAGES = (
    "stable-worldmodel",
    "stable-pretraining",
    "torch",
    "torchvision",
    "numpy",
    "gymnasium",
    "pymunk",
    "mujoco",
    "h5py",
    "huggingface_hub",
    "scikit-learn",
)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_info(repo: Path = REPO_ROOT) -> dict:
    commit = _git(["rev-parse", "HEAD"], repo)
    status = _git(["status", "--porcelain"], repo)
    return {
        "path": str(repo),
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo),
        "remote": _git(["config", "--get", "remote.origin.url"], repo),
    }


def package_versions(names=_PACKAGES) -> dict:
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def hardware_info() -> dict:
    info: dict = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_visible": os.cpu_count(),
    }
    try:
        from helpers.threads import describe

        info["threads"] = describe()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
    except Exception:  # noqa: BLE001
        pass
    for key in ("CONTAINER_ID", "VAST_CONTAINERLABEL", "PUBLIC_IPADDR"):
        if os.environ.get(key):
            info[key.lower()] = os.environ[key]
    return info


def file_sha256(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def array_sha256(arr) -> str:
    import numpy as np

    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def build_manifest(
    *,
    run_id: str,
    kind: str | None = None,
    seeds: dict | None = None,
    data: dict | None = None,
    upstream_revisions: dict | None = None,
    costs: dict | None = None,
    metrics: dict | None = None,
    extra: dict | None = None,
    started_at: float | None = None,
) -> dict:
    now = time.time()
    manifest = {
        "run_id": run_id,
        "kind": kind,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_clock_s": (now - started_at) if started_at else None,
        "repo": git_info(REPO_ROOT),
        "upstream_lewm": git_info(LEWM_DIR) if (LEWM_DIR / ".git").exists() else None,
        "packages": package_versions(),
        "hardware": hardware_info(),
        "seeds": seeds or {},
        "data": data or {},
        "upstream_revisions": upstream_revisions or {},
        "costs": costs or {},
        "metrics": metrics or {},
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(run_dir: str | Path, manifest: dict, name: str = "manifest.json") -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return path


def read_manifest(run_dir: str | Path, name: str = "manifest.json") -> dict:
    return json.loads((Path(run_dir) / name).read_text())


def make_run_id(env: str, kind: str, detail: str = "", n: int = 1) -> str:
    """`<env>-<kind>-<detail>-<yyyymmdd>-<n>` per infrastructure.md."""
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    parts = [env, kind] + ([detail] if detail else []) + [day, str(n)]
    return "-".join(parts)
