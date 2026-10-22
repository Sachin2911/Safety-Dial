"""Validate the pinned platform without importing Isaac Sim or starting its renderer."""

import argparse
import json
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "vast"


def check_host():
    if sys.version_info[:2] != (3, 12) or platform.machine() != "x86_64":
        raise RuntimeError("The platform requires Python 3.12 on Linux x86-64.")
    release = platform.freedesktop_os_release()
    if (release.get("ID"), release.get("VERSION_ID")) != ("ubuntu", "24.04"):
        raise RuntimeError("This installer targets the Ubuntu 24.04 Vast image.")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.split(",", 1) for row in result.stdout.splitlines() if row.strip()]
    if not rows:
        raise RuntimeError("No NVIDIA GPU is visible.")
    unsupported = []
    for name, driver in rows:
        if tuple(int(part) for part in driver.strip().split(".")) < (580, 95, 5):
            raise RuntimeError("Host driver must be at least 580.95.05; do not replace it here.")
        print(f"GPU: {name.strip()}, driver: {driver.strip()}", file=sys.stderr)
        if re.search(r"\b(A100|H100|H200|A30|V100|P100)\b", name):
            unsupported.append(name.strip())
    if len(unsupported) == len(rows):
        raise RuntimeError(
            "Visible GPUs lack RT cores: "
            + ", ".join(unsupported)
            + ". Use an RTX-capable host for Isaac Sim and camera experiments."
        )


def check_packages():
    errors = []
    checked = {}
    for line in (CONFIG / "requirements-linux-py312.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " @ git+" in line:
            name, url = line.split(" @ git+", 1)
            repository, revision = url.split("#", 1)[0].rsplit("@", 1)
            try:
                dist = metadata.distribution(name)
                origin = json.loads(dist.read_text("direct_url.json") or "{}")
                if (
                    origin.get("url") != repository
                    or origin.get("vcs_info", {}).get("commit_id") != revision
                    or origin.get("subdirectory") != "tools/wheel_builder"
                ):
                    errors.append(f"{name}: installed source does not match the pinned revision")
                checked[name] = dist.version
            except metadata.PackageNotFoundError:
                errors.append(f"{name}: missing")
        else:
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
            if not match:
                raise RuntimeError("Unsupported requirement in the platform snapshot")
            name, expected = match.groups()
            try:
                actual = metadata.version(name)
                checked[name] = actual
                if actual != expected:
                    errors.append(f"{name}: expected {expected}, found {actual}")
            except metadata.PackageNotFoundError:
                errors.append(f"{name}: missing")
    if errors:
        raise RuntimeError("Platform package mismatch:\n" + "\n".join(errors))
    return checked


def check_runtime():
    packages = check_packages()
    import torch
    import torchaudio
    import torchvision

    if torch.version.cuda != "13.0" or not torch.cuda.is_available():
        raise RuntimeError("The CUDA 13.0 PyTorch runtime is unavailable.")
    value = torch.ones((16, 16), device="cuda")
    if (value @ value).sum().item() != 4096:
        raise RuntimeError("CUDA matrix multiplication produced an unexpected result.")
    torch.cuda.synchronize()
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchaudio": torchaudio.__version__,
        "pytorch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "packages": packages,
        "validation": "Pinned package metadata, PyTorch imports and CUDA matrix multiplication",
        "simulator_validation": "not run",
        "camera_validation": "not run",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-only", action="store_true")
    args = parser.parse_args()
    try:
        check_host()
        if not args.host_only:
            print(json.dumps(check_runtime(), indent=2))
    except (RuntimeError, OSError, ValueError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"Platform check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
