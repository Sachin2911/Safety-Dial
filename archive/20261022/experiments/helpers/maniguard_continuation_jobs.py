"""Prepare a fixed compound job and exact source bytes, without starting physics."""

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from helpers.maniguard_adaptation import AdapterConfig
from helpers.maniguard_continuation_admission import ROOT, policy_lease_path
from helpers.maniguard_continuation_guardian import atomic_record
from helpers.maniguard_continuation_protocol import validate_job
from helpers.maniguard_continuation_service import service_baseline
from helpers.maniguard_continuation_source import REQUIRED_SOURCES, prepare_runtime
from helpers.maniguard_latent import LatentController


def prepare_job(protocol_path, attempt, controller_path, clearance):
    raw = Path(protocol_path).read_bytes()
    protocol = json.loads(raw)
    if protocol["status"] != "prospectively_frozen":
        raise ValueError("The development draft cannot prepare a dispatchable physical job")
    if not set(REQUIRED_SOURCES) <= set(protocol["critical_source_hashes"]):
        raise ValueError("The source freeze omits an execution or validation component")
    with np.load(controller_path, allow_pickle=False) as data:
        controller = LatentController(data["weights"])
    if controller.sha256() != protocol["closure_controller_sha256"]:
        raise ValueError("The closure controller differs from the predeclared training choice")
    output = Path(attempt["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "controller.npz", weights=controller.weights)
    (output / "protocol.json").write_bytes(raw)
    native_config = ROOT / "third_party/ManiGuard/configs/eval/jar_transport_joint.yaml"
    native_bytes = native_config.read_bytes()
    if hashlib.sha256(native_bytes).hexdigest() != protocol["native_evaluator_config_sha256"]:
        raise ValueError("Native Jar evaluator configuration differs from its frozen source")
    (output / "native-config.yaml").write_bytes(native_bytes)
    scene_file = ROOT / "data/maniguard/benchmark/jar_transport" / protocol["scene"] / "scene_ep1.json"
    if hashlib.sha256(scene_file.read_bytes()).hexdigest() != protocol["scene_asset_sha256"]:
        raise ValueError("The selected native scene differs from its prospective digest")
    worker = protocol["workers"][str(attempt["worker"])]
    mode = attempt["condition"]
    job = {
        "study": "continuation-feasibility-v1", "family": "jar", "scene": protocol["scene"],
        "seed": protocol["seed"], "protocol_episode_id": attempt["protocol_episode_id"],
        "condition": mode, "worker": attempt["worker"], "output": str(output),
        "episode_id": str(output), "controller": str(output / "controller.npz"),
        "controller_sha256": controller.sha256(), "encoder_device": "cuda:0",
        "action_horizon": 2000, "max_steps_override": None,
        "adapter": asdict(AdapterConfig(kind="identity", horizon=8)),
        "random_policy": False, "camera_smoke": False, "policy": protocol["policy"],
        "source_profile": "high_fidelity", "renderer_vulkan_index": worker["renderer_vulkan_index"],
        "simulation_gpu_uuid": worker["simulation_gpu_uuid"],
        "policy_gpu_uuid": worker["policy_gpu_uuid"], "policy_port": worker["port"],
        "simulation_cuda_visible_devices": str(worker["simulation_physical_device"]),
        "continuation": protocol["selector_parameters"][mode],
        "continuation_contract_sha256": protocol["selector_contract_sha256"][mode],
        "runtime_limits": protocol["runtime_limits"], "admitted_epoch": attempt["started_epoch"],
        "native_evaluator_config_sha256": protocol["native_evaluator_config_sha256"],
        "scene_asset_sha256": protocol["scene_asset_sha256"],
        "service_ledger_baseline": service_baseline(protocol, attempt["worker"]),
        "deadline_epoch": attempt["deadline_epoch"], "original_clearance": clearance,
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "service_lock_path": str(policy_lease_path(protocol, attempt["worker"])),
        "code_hashes": {},
    }
    validate_job(protocol, job)
    for name, expected in protocol["critical_source_hashes"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid frozen source path")
        path = ROOT / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Current source differs from frozen screen: " + name)
        target = output / "study_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        job["code_hashes"][name] = expected
    runtime = prepare_runtime(output)
    if hashlib.sha256(runtime.read_bytes()).hexdigest() != protocol["instrumented_runtime_sha256"]:
        raise ValueError("Prepared native evaluator differs from the prospective source check")
    atomic_record(output / "job.json", job)
    return output / "job.json"


def physical_environment(protocol, worker):
    spec = protocol["workers"][str(worker)]
    return {**os.environ,
            "CUDA_VISIBLE_DEVICES": str(spec["simulation_physical_device"]),
            "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
            "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
            "__EGL_VENDOR_LIBRARY_FILENAMES": str(
                ROOT / "configs/maniguard/latent-evolution/nvidia-egl-vendor.json"),
            "OMNI_KIT_ACCEPT_EULA": "YES", "OMNIGIBSON_HEADLESS": "1"}
