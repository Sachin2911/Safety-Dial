"""Explicit CUDA/renderer routing for the additional four-GPU UNISafe host."""

import os
import subprocess
from pathlib import Path


def configure_renderer(root):
    manifest = Path(root) / "configs/unisafe09p/vulkan-icd.json"
    if not manifest.is_file():
        raise ValueError("The additional host's frozen EGL manifest is missing")
    os.environ["VK_DRIVER_FILES"] = str(manifest.resolve())
    os.environ["VK_ICD_FILENAMES"] = str(manifest.resolve())


def verify_renderer(root):
    configure_renderer(root)
    result = subprocess.run(
        ["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=30
    )
    if result.returncode or result.stdout.count("NVIDIA GeForce RTX 4090") != 4:
        raise RuntimeError("The EGL driver must enumerate all four RTX 4090 GPUs")
    return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}


def gpu_routing(index):
    import carb
    import torch

    settings = carb.settings.get_settings()
    selected = settings.get("/renderer/multiGpu/activeCudaGpus")
    if str(selected).strip(", ") != str(index):
        raise RuntimeError("Renderer CUDA selection differs from the assigned GPU")
    if torch.cuda.current_device() != index:
        raise RuntimeError("PyTorch current device differs from the assigned GPU")
    properties = torch.cuda.get_device_properties(index)
    if "RTX 4090" not in properties.name:
        raise RuntimeError("The assigned CUDA device is not an RTX 4090")
    return {
        "gpu_index": index, "cuda_device": f"cuda:{index}",
        "torch_current_device": torch.cuda.current_device(),
        "renderer_active_cuda_gpus": selected,
        "physics_cuda_device_setting": settings.get("/physics/cudaDevice"),
        "gpu_name": properties.name, "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "cuda_visible_devices_environment": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "multi_gpu_requested": False, "kit_cpu_thread_limit": 8,
    }
