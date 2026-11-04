"""Check the image on rented compute; no benchmark episodes or asset downloads."""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


AP_REVISION = "984327424ca7a2e52c75a8279197750d9d0b7c8e"
ROOT = Path("/opt/lewm-runtime")


def pins(path):
    return {
        name: version
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith(("#", "-r"))
        for name, version in [line.strip().split("==")]
    }


def verify_packages():
    if platform.python_version_tuple()[:2] != ("3", "11"):
        raise RuntimeError("Expected Python 3.11")
    revision = subprocess.check_output(
        ["git", "-C", "/opt/ap-reference", "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != AP_REVISION:
        raise RuntimeError("AP source revision differs from the validated reference")
    expected = pins(ROOT / "requirements.ap.txt")
    upstream = Path("/opt/ap-reference/code/ap/requirements")
    authors = pins(upstream / "native-common.txt") | pins(upstream / "native.txt")
    if expected != authors:
        raise RuntimeError("Direct package pins differ from the pinned AP source")
    installed = {name: importlib.metadata.version(name) for name in expected}
    wrong = {name: version for name, version in installed.items()
             if version.split("+")[0] != expected[name]}
    if wrong:
        raise RuntimeError(f"Dependency versions differ: {wrong}")
    for name, expected_cuda in pins(ROOT / "constraints.cuda.txt").items():
        if importlib.metadata.version(name) != expected_cuda:
            raise RuntimeError(f"CUDA wheel mismatch: {name}")
    for name in (
        "hdf5plugin", "h5py", "tokenizers", "torchvision", "stable_worldmodel",
        "anchored_planning.model", "anchored_planning.runtime",
        "anchored_planning.environments", "anchored_planning.observation_bank",
        "anchored_planning.planners",
    ):
        importlib.import_module(name)
    import torch

    if torch.version.cuda != "12.6":
        raise RuntimeError("Expected PyTorch CUDA 12.6")
    return {"python": platform.python_version(), "ap_revision": revision,
            "packages": installed, "cuda_runtime": torch.version.cuda}


def verify_gpu():
    import torch
    from torchvision.ops import nms

    if not torch.cuda.is_available():
        raise RuntimeError("GPU validation requires an NVIDIA GPU exposed to the container")
    for dtype in (torch.float32, torch.bfloat16):
        array = torch.ones((16, 16), device="cuda", dtype=dtype)
        result = array @ array
        if not torch.all(result == 16).item():
            raise RuntimeError(f"CUDA matrix multiplication failed for {dtype}")
    boxes = torch.tensor([[0, 0, 2, 2], [0, 0, 2, 2]], device="cuda", dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], device="cuda")
    if nms(boxes, scores, 0.5).tolist() != [0]:
        raise RuntimeError("torchvision CUDA operator failed")
    torch.cuda.synchronize()
    return {"name": torch.cuda.get_device_name(0), "cuda_and_bfloat16": "passed"}


def verify_rendering():
    for key, expected in {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
                          "GALLIUM_DRIVER": "llvmpipe", "SDL_VIDEODRIVER": "dummy"}.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f"Renderer setting differs: {key}")
    import mujoco
    import numpy as np
    import pygame

    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <light pos="0 0 3"/>
          <camera name="overview" pos="1 -1 1" xyaxes="1 1 0 -1 1 2"/>
          <geom type="plane" size="2 2 .1" rgba=".2 .3 .4 1"/>
          <geom type="sphere" size=".15" pos="0 0 .2" rgba="1 0 0 1"/>
        </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=64, width=64) as renderer:
        renderer.update_scene(data, camera="overview")
        frame = renderer.render()
        if frame.shape != (64, 64, 3) or frame.dtype != np.uint8 or frame.std() == 0:
            raise RuntimeError("OSMesa did not produce a nonempty RGB frame")
    pygame.display.init()
    try:
        pygame.display.set_mode((32, 32))
        if pygame.display.get_driver() != "dummy":
            raise RuntimeError("SDL did not select the dummy display")
    finally:
        pygame.display.quit()
    return {"osmesa_frame": "passed", "sdl_dummy": "passed",
            "synthetic_frames": 1, "physics_steps": 0, "benchmark_episodes": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    report = verify_packages()
    if args.gpu:
        report["gpu"] = verify_gpu()
    if args.render:
        report["rendering"] = verify_rendering()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
