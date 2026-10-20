"""One reserved compound episode, supervised independently of its coordinator."""

import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_continuation_admission import (
    assert_service_lease,
    file_hash,
    policy_lease_path,
    require_original_clearance,
)
from helpers.maniguard_continuation_guardian import atomic_record, run_guarded
from helpers.maniguard_continuation_limits import epoch, validate_limits
from helpers.maniguard_continuation_protocol import validate_job
from helpers.maniguard_latent_queue import process_identity


def verify_dispatch(job_path, lease_fd):
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text())
    output = job_path.parent
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if (output != Path(job["output"]).resolve() or job["episode_id"] != str(output)
            or protocol["status"] != "prospectively_frozen"
            or job["protocol_sha256"] != file_hash(protocol_path)
            or job["runtime_limits"] != protocol["runtime_limits"]
            or int(os.environ.get("CONTAINER_ID", "0")) != protocol["instance_id"]):
        raise ValueError("Only a matching frozen job on the authorized instance can dispatch")
    validate_limits(job["runtime_limits"])
    validate_job(protocol, job)
    if job["code_hashes"] != protocol["critical_source_hashes"]:
        raise ValueError("Job source snapshot differs from the complete protocol source freeze")
    for name, expected in job["code_hashes"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid frozen source path")
        if file_hash(ROOT / path) != expected or file_hash(output / "study_source" / path) != expected:
            raise ValueError("Current or archived dispatch source differs: " + name)
    if file_hash(output / "benchmark_instrumented.py") != protocol["instrumented_runtime_sha256"]:
        raise ValueError("The native evaluator copy differs from the preflight")
    if (file_hash(ROOT / "third_party/ManiGuard/configs/eval/jar_transport_joint.yaml")
            != protocol["native_evaluator_config_sha256"]
            or file_hash(output / "native-config.yaml") != protocol["native_evaluator_config_sha256"]
            or job["native_evaluator_config_sha256"] != protocol["native_evaluator_config_sha256"]):
        raise ValueError("The actual native evaluator configuration changed")
    scene_root = ROOT / "data/maniguard/benchmark/jar_transport"
    scene = (scene_root / job["scene"]).resolve()
    scene.relative_to(scene_root.resolve())
    if file_hash(scene / "scene_ep1.json") != job["scene_asset_sha256"]:
        raise ValueError("The actual native scene differs from the prospective asset digest")
    if require_original_clearance(protocol) != job["original_clearance"]:
        raise ValueError("Original comparison clearance changed after reservation")
    state = json.loads((output.parent.parent / "screen.json").read_text())
    matching = [a for a in state["attempts"] if a["protocol_episode_id"] == job["protocol_episode_id"]]
    if (state["protocol_sha256"] != job["protocol_sha256"] or len(matching) != 1
            or len(state["attempts"]) > 4 or len({a["id"] for a in state["attempts"]}) != len(state["attempts"])):
        raise ValueError("A unique charged reservation is required before child dispatch")
    attempt = matching[0]
    if (attempt["status"] not in ("reserved", "running") or Path(attempt["output"]).resolve() != output
            or attempt["started_epoch"] != job["admitted_epoch"]
            or attempt["deadline_epoch"] != job["deadline_epoch"]
            or time.time() >= min(job["deadline_epoch"], epoch(protocol["physical_cutoff_utc"]))):
        raise ValueError("The reservation is terminal, changed or past its physical deadline")
    expected_lease = policy_lease_path(protocol, job["worker"])
    if Path(job["service_lock_path"]) != expected_lease:
        raise ValueError("Only the declared global policy-service lease is accepted")
    assert_service_lease(lease_fd, expected_lease)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != job["simulation_cuda_visible_devices"]:
        raise ValueError("Actual simulator environment differs from the frozen physical device")
    return job, protocol


def physical_child(job_path, lease_fd):
    job, _ = verify_dispatch(job_path, lease_fd)
    output = Path(job["output"])
    guardian = json.loads((output / "guardian.json").read_text())["guardian"]
    if guardian["pid"] != os.getppid() or process_identity(guardian["pid"]) != guardian:
        raise ValueError("A physical child needs its live independently supervised guardian")
    atomic_record(output / "scene-asset-readback.json", {
        "scene": job["scene"], "sha256": job["scene_asset_sha256"], "checked_epoch": time.time(),
        "scope": "Native scene bytes checked before simulator startup; licensed asset not copied.",
    })
    os.environ["SAFETY_DIAL_MANIGUARD_JOB"] = str(Path(job_path).resolve())
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ["OMNIGIBSON_DATA_PATH"] = str(ROOT / "data/maniguard/datasets")
    sys.path.insert(0, str(ROOT / "third_party/ManiGuard"))
    runtime = output / "benchmark_instrumented.py"
    scene_root = ROOT / "data/maniguard/benchmark/jar_transport"
    scene = (scene_root / job["scene"]).resolve()
    scene.relative_to(scene_root.resolve())
    if not (scene / "scene_ep1.json").is_file():
        raise ValueError("The pinned native Jar scene is unavailable")
    sys.argv = [str(runtime), "--config",
                str(ROOT / "third_party/ManiGuard/configs/eval/jar_transport_joint.yaml"),
                "--benchmark-root", str(scene_root), "--scenes", job["scene"],
                "--seed", str(job["seed"]), "--host", "127.0.0.1", "--port", str(job["policy_port"]),
                "--source-profile", "high_fidelity", "--camera-resolution", "640",
                "--execute-horizon", "8", "--output-dir", str(output), "--run-name", "episode"]
    runpy.run_path(str(runtime), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--lease-fd", type=int, required=True)
    parser.add_argument("--physical-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.physical_child:
        physical_child(args.job, args.lease_fd)
        return
    job, _ = verify_dispatch(args.job, args.lease_fd)
    claim = Path(job["output"]) / "guardian-claimed.json"
    with claim.open("x") as stream:
        json.dump({"guardian": process_identity(os.getpid()),
                   "job_sha256": hashlib.sha256(args.job.read_bytes()).hexdigest()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    command = [str(ROOT / "data/maniguard/envs/sim/bin/python"), "-u", str(Path(__file__).resolve()),
               "--job", str(args.job.resolve()), "--lease-fd", str(args.lease_fd), "--physical-child"]
    code = run_guarded(command, output=job["output"], limits=job["runtime_limits"],
                       started_epoch=job["admitted_epoch"], cutoff_epoch=job["deadline_epoch"],
                       lease_fd=args.lease_fd)
    atomic_record(Path(job["output"]) / "guardian-exit.json", {"exit_code": code})
    raise SystemExit(0 if code == 0 else 1)


if __name__ == "__main__":
    main()
