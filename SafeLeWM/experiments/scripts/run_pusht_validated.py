"""Bounded, staged PushT screen, with one persistent worker per GPU per stage."""

import argparse
import collections
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import hdf5plugin  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import manip as M  # noqa: E402
from helpers.pilot_common import save, digest  # noqa: E402
import manip_pipeline as P  # noqa: E402

BASE = M.ROOT / "outputs/pusht-validated-20261102"
OUT = BASE / "run-v1"
M.OUT = BASE
CONFIG = M.ROOT / "configs/pusht-validated/run.json"


REFERENCE = M.ROOT / "outputs/pusht-ap-reproduction-20261102"


def configure_runtime():
    import shutil
    sys.path.insert(0, str(M.REF))
    from anchored_planning.runtime import configure_runtime as official_runtime
    assert shutil.disk_usage(M.ROOT).free >= 2 * 1024**3
    official_runtime()


def load(path):
    return json.loads(Path(path).read_text())


def source_freeze():
    import torch
    from stable_worldmodel.envs.pusht.env import PushT

    files = [
        M.ROOT / "PushTValidatedQualification.md",
        CONFIG,
        M.ROOT / "configs/manip-screen/pusht-panel.json",
        *[
            M.ROOT / "experiments/scripts" / n
            for n in [
                "prepare_pusht_validated.py",
                "run_pusht_validated.py",
                "pusht_authority_worker_v2.py",
                "audit_pusht_validated.py",
                "launch_pusht_validated.py",
                "preserve_pusht_validated.py",
                "manip_pipeline.py",
                "fetch_manip_data.py",
                "audit_manip.py",
            ]
        ],
        *[
            M.ROOT / "experiments/helpers" / n
            for n in [
                "pusht_authority.py",
                "manip.py",
                "qualification.py",
                "recurrent_adapter.py",
                "pilot_common.py",
            ]
        ],
        M.ROOT / "experiments/tests/test_pusht_authority.py",
        M.ROOT / "experiments/tests/test_manip.py",
        *sorted((M.REF / "anchored_planning").glob("*.py")),
        M.REF / "protocols/scalers.json",
    ]
    bindings = {}
    for p in files:
        relative = p.relative_to(M.ROOT)
        dest = OUT / "sources" / relative
        dest.parent.mkdir(exist_ok=True, parents=True)
        dest.write_bytes(p.read_bytes())
        bindings[str(relative)] = digest(dest)
    import importlib.metadata
    assert torch.__version__ == "2.7.1+cu126"
    for package, expected in load(REFERENCE / "freeze.json")["dependencies"].items():
        assert importlib.metadata.version(package) == expected, package
    assert digest(REFERENCE / "latents.npy") == load(BASE / "data-receipt.json")["cache_sha256"]
    native = Path(inspect.getfile(PushT))
    (OUT / "sources/native-env.py").write_bytes(native.read_bytes())
    for name, path in [
        ("panel.json", BASE / "pusht/panel.json"),
        ("memory.json", BASE / "pusht/memory-v1/panel.json"),
        ("memory-complete.json", BASE / "pusht/memory-v1/complete.json"),
        ("data-receipt.json", BASE / "data-receipt.json"),
    ]:
        (OUT / name).write_bytes(path.read_bytes())
    freeze = dict(
        created=time.time(),
        hostname=socket.gethostname(),
        sources=bindings,
        native_environment_sha256=digest(native),
        torch=torch.__version__,
        panel_sha256=digest(OUT / "panel.json"),
        checkpoint_sha256=digest(M.ROOT / M.TASKS["pusht"]["weights"]),
        deadline=load(BASE / "data-receipt.json")["started"] + load(CONFIG)["active_seconds"],
    )
    save(OUT / "freeze.json", freeze)
    return freeze


def worker(lane, jobs_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(lane)
    configure_runtime()
    import torch
    from pusht_authority_worker_v2 import run_episode

    jobs = [j for j in load(jobs_path) if j["device"] == lane]
    deadline = load(OUT / "freeze.json")["deadline"]
    with P.open_data("pusht") as data, torch.inference_mode():
        from anchored_planning.observation_bank import ObservationBank
        panel = load(OUT / "panel.json")
        bank = ObservationBank(dict(lengths=np.asarray(data["ep_len"],np.int64),
                                    offsets=np.asarray(data["ep_offset"],np.int64)),
            np.load(REFERENCE / "latents.npy",mmap_mode="r"),
            np.asarray(panel["memory_episodes"],np.int64),5,np.ones(3,np.float32),torch.device("cuda"))
        reset = M.reset_case
        forbidden = {c["id"] for c in panel["reserved_final_cases"]}
        _, native = M.native_model("pusht")
        for job in jobs:
            assert job["case"]["id"] not in forbidden
            environment_seed = job["case"].get("environment_seed",job["seed"])
            M.reset_case = lambda env,task,arrays,seed: reset(env,task,arrays,environment_seed)
            if (OUT / "abort.json").exists():
                raise RuntimeError("Another worker failed; stop before next episode")
            if time.time() >= deadline:
                raise TimeoutError("Active-work deadline")
            folder = OUT / "episodes" / job["id"]
            folder.mkdir(parents=True, exist_ok=False)
            save(folder / "job.json", job)
            parameters = None
            if job["parameters"]:
                parameters = np.load(OUT / job["parameters"])["parameters"].copy()
            totals = collections.Counter()
            try:
                result = run_episode(
                    "pusht",
                    folder,
                    job["case"],
                    job["arm"],
                    job["seed"],
                    data,
                    native,
                    bank,
                    totals,
                    time.monotonic() + max(0, deadline - time.time()),
                    hazard=job["case"],
                    parameters=parameters,
                )
                result.update(
                    variant=job["variant"], device=lane, job=job["id"], group=job["case"]["group"]
                )
                save(folder / "result.json", result)
            except BaseException:
                save(
                    OUT / f"failure-{lane}.json", dict(job=job["id"], error=traceback.format_exc())
                )
                raise


def stage(name, jobs):
    path = OUT / f"jobs-{name}.json"
    assert not path.exists()
    save(path, jobs)
    handles = []
    processes = []
    try:
        for lane in range(8):
            handle = (BASE / f"{name}-gpu{lane}.log").open("x")
            handles.append(handle)
            env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
            processes.append(
                subprocess.Popen(
                    [sys.executable, __file__, "--lane", str(lane), "--jobs", str(path)],
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        while any(p.poll() is None for p in processes):
            failures = [p.returncode for p in processes if p.poll() is not None and p.returncode]
            if failures and not (OUT / "abort.json").exists():
                save(OUT / "abort.json", dict(stage=name, returncodes=failures, time=time.time()))
            completed = sum((OUT / "episodes" / j["id"] / "result.json").exists() for j in jobs)
            save(
                OUT / "progress.json",
                dict(stage=name, completed=completed, total=len(jobs), time=time.time()),
            )
            time.sleep(5)
        assert all(p.returncode == 0 for p in processes), [p.returncode for p in processes]
    finally:
        for handle in handles:
            handle.close()
    return [load(OUT / "episodes" / j["id"] / "result.json") for j in jobs]


def job(case, variant, seed, device, arm="monitored", parameters=None):
    return dict(
        id=f"{case['id']}-{variant}-{seed}-gpu{device}",
        case=case,
        variant=variant,
        arm=arm,
        seed=seed,
        device=device,
        parameters=parameters,
    )


def trace(row):
    parent = OUT / "episodes" / row["job"]
    return np.load(next(parent.glob("*/trace.npz")))


def parity(a, b):
    with trace(a) as x, trace(b) as y:
        return all(np.array_equal(x[k], y[k]) for k in ["states", "actions", "frames", "rewards"])


def run():
    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter

    assert socket.gethostname() == "c45db82630e6"
    assert (BASE / "software-checks-passed").exists()
    OUT.mkdir(parents=True, exist_ok=False)
    freeze = source_freeze()
    assert time.time() < freeze["deadline"]
    panel = load(OUT / "panel.json")
    assert panel["status"] == "ready" and len(panel["cases"]) == 8
    cfg = load(CONFIG)
    cases = panel["cases"]
    actor = RecurrentActionAdapter(
        AdapterSpec(402, 2, initialization_seed=cfg["initialization_seed"])
    )
    neutral = actor.parameters_vector()
    scales = actor.coordinate_scales()
    (OUT / "parameters").mkdir()
    np.savez(OUT / "parameters/neutral.npz", parameters=neutral, scales=scales)
    generator = np.random.default_rng(cfg["probe_seed"])
    for i in range(cfg["probes"]):
        genotype = generator.normal(0, cfg["sigma"], 132).astype(np.float32)
        parameters = neutral.copy()
        parameters[-132:] += scales[-132:] * genotype
        np.savez(OUT / f"parameters/probe-{i:02d}.npz", parameters=parameters, genotype=genotype)
    save(
        OUT / "parameter-lock.json",
        dict(
            created=time.time(),
            files={p.name: digest(p) for p in sorted((OUT / "parameters").glob("*.npz"))},
        ),
    )
    rows = []
    summary = dict(status="running", results=rows, gates={}, new_training=False)
    try:
        reference = load(REFERENCE / "freeze.json")["queries"][0]
        bridge_case = dict(id="baseline-bridge", group=-1, episode=reference["episode"],
                           start_row=reference["global_start"], goal_row=reference["global_goal"],
                           horizon=140, environment_seed=reference["environment_seed"], hazard_centre=[0,0])
        bridge = stage("bridge",[job(bridge_case,"bridge",26091331,0,arm="clean")])
        rows.extend(bridge)
        saved_path = REFERENCE / "run-v1/wrapper/ap-fresh-pusht-000_ap_26091331/trace.npz"
        with trace(bridge[0]) as a, np.load(saved_path) as b:
            same = all(np.array_equal(a[k],b[k]) for k in ["states","actions","frames","rewards"])
        summary["gates"]["baseline_bridge_parity"] = same
        save(OUT / "summary.json",summary)
        if not same:
            summary["status"] = "gate_failed"
            return
        cross = stage(
            "cross-gpu", [job(cases[0], "cross", 18201, i, arm="clean") for i in range(8)]
        )
        rows.extend(cross)
        summary["gates"]["cross_gpu_parity"] = all(parity(cross[0], r) for r in cross[1:])
        save(OUT / "summary.json", summary)
        if not summary["gates"]["cross_gpu_parity"]:
            summary["status"] = "gate_failed"
            return
        jobs = []
        for i, c in enumerate(cases):
            jobs += [
                job(c, "risky", 18201, i, arm="replay"),
                job(c, "safe", 18201, i, arm="scripted"),
            ]
            for seed in cfg["seeds"]:
                jobs += [
                    job(c, "clean", seed, i, arm="clean"),
                    job(c, "native", seed, i),
                    job(c, "neutral", seed, i, parameters="parameters/neutral.npz"),
                ]
        qualified = stage("qualification", jobs)
        rows.extend(qualified)
        admitted = []
        checks = []
        for c in cases:
            per = [r for r in qualified if r["case"] == c["id"]]
            safe = next(r for r in per if r["variant"] == "safe")
            risky = next(r for r in per if r["variant"] == "risky")
            natives = [r for r in per if r["variant"] == "native"]
            valid = (
                safe["safe_completion"]
                and not safe["witness_invalid"]
                and risky["success"]
                and risky["violation"]
                and any(r["success"] for r in natives)
            )
            if valid:
                admitted.append(c["id"])
            for seed in cfg["seeds"]:
                paired = {r["variant"]: r for r in per if r["seed"] == seed}
                checks += [
                    parity(paired["clean"], paired["native"]),
                    parity(paired["native"], paired["neutral"]),
                ]
        natives = [r for r in qualified if r["variant"] == "native"]
        violating = {r["case"] for r in natives if r["violation"] and r["case"] in admitted}
        summary["gates"].update(
            parity=all(checks),
            admitted_scenes=admitted,
            native_completions=sum(r["success"] for r in natives),
            native_n=len(natives),
            violating_admitted_scenes=sorted(violating),
        )
        ready = (
            all(checks)
            and len(admitted) >= 4
            and sum(r["success"] for r in natives) >= 8
            and len(violating) >= 2
        )
        summary["gates"]["authority_screen_admitted"] = ready
        save(OUT / "summary.json", summary)
        if not ready:
            summary["status"] = "gate_failed"
            return
        jobs = [
            job(c, f"probe-{i:02d}", seed, lane, parameters=f"parameters/probe-{i:02d}.npz")
            for i in range(cfg["probes"])
            for lane, c in enumerate(cases)
            for seed in cfg["seeds"]
        ]
        rows.extend(stage("authority", jobs))
        summary["status"] = "complete"
    except BaseException:
        summary["status"] = "failed"
        summary["error"] = traceback.format_exc()
        raise
    finally:
        counts = collections.Counter()
        for p in (OUT / "episodes").glob("*/*/execution.json"):
            counts.update(load(p)["resources"])
        summary.update(resources=dict(counts), finished=time.time(), results=rows)
        save(OUT / "summary.json", summary)
        assert counts["environment_actions"] <= cfg["action_cap"]
        assert counts["environment_resets"] <= cfg["reset_cap"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", type=int)
    parser.add_argument("--jobs")
    args = parser.parse_args()
    if args.lane is not None:
        worker(args.lane, args.jobs)
    else:
        configure_runtime()
        with __import__("torch").inference_mode():
            run()


if __name__ == "__main__":
    main()
