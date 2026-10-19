"""Supervised serial policy lifecycle, guarded against interrupting a live rollout."""

import json
import subprocess
import time
from pathlib import Path

from helpers.maniguard_client import StudyClient
from helpers.maniguard_latent_hardware import GPU_UUIDS
from helpers.maniguard_latent_queue import ROOT, atomic, process_identity


def supervisor(*arguments):
    result = subprocess.run(["supervisorctl", *arguments], capture_output=True, text=True,
                            check=False)
    if result.returncode:
        raise RuntimeError("supervisorctl failed: " + result.stdout + result.stderr)
    return result.stdout.strip()


def assert_no_live_worker(run):
    path = Path(run) / "queue.json"
    if not path.exists():
        return
    queue = json.loads(path.read_text())
    for attempt in queue["attempts"]:
        identity = attempt.get("process_identity")
        if identity is not None and process_identity(identity["pid"]) == identity:
            raise RuntimeError("Refuse policy replacement while a physical worker is live: "
                               + attempt["id"])


def policy_matches(metadata, family, audit):
    sources = json.loads((ROOT / "configs/maniguard/phase2/sources.json").read_text())
    spec = sources["policies"][family]
    return (metadata.get("audit_service") is audit
            and metadata.get("serve_config") == spec["config"]
            and metadata.get("checkpoint_revision") == spec["revision"]
            and metadata.get("model_shape") == [16, 32]
            and metadata.get("native_noise_steering") is True)


def metadata(port=18000):
    client = StudyClient(host="127.0.0.1", port=port)
    try:
        return client.get_server_metadata()
    finally:
        client._ws.close()


def service_status(name):
    result = subprocess.run(["supervisorctl", "status", name], capture_output=True,
                            text=True, check=False)
    if result.returncode not in [0, 3] or "ERROR" in result.stdout:
        raise RuntimeError("Cannot observe supervisor service: " + result.stdout + result.stderr)
    return result.stdout.strip()


def stop_if_running(name):
    state = subprocess.run(["supervisorctl", "status", name], capture_output=True,
                           text=True, check=False).stdout
    if "RUNNING" in state or "STARTING" in state:
        supervisor("stop", name)


def ensure_worker_policies(run, family):
    if family not in ["jar", "clutter"]:
        raise ValueError("Undeclared family")
    ready = []
    for worker in [0, 1]:
        try:
            observed = metadata(18000 + worker)
            ready.append(policy_matches(observed, family, False)
                         and observed.get("physical_gpu_uuid") == GPU_UUIDS[worker])
        except (ConnectionError, OSError, TimeoutError):
            ready.append(False)
    if all(ready):
        return
    assert_no_live_worker(run)
    stop_if_running("maniguard_latent_policy")
    run = Path(run).resolve()
    for worker in [0, 1]:
        if ready[worker]:
            continue
        name = f"maniguard_latent_policy_worker{worker}"
        stop_if_running(name)
        config = f"""[program:{name}]
command={ROOT}/third_party/openpi/.venv/bin/python -u {ROOT}/experiments/scripts/maniguard_latent_serve.py --family {family} --port {18000 + worker} --ledger {run}/server-{family}-worker{worker}-physical.jsonl
directory={ROOT}
environment=CUDA_VISIBLE_DEVICES="{worker}",XLA_PYTHON_CLIENT_PREALLOCATE="false",MANIGUARD_SKIP_OMNIGIBSON_PATCH="1",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",JAX_COMPILATION_CACHE_DIR="{ROOT}/data/maniguard/jax-cache",JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1"
autostart=true
autorestart=false
stdout_logfile={run}/policy-{family}-worker{worker}.log
redirect_stderr=true
stdout_logfile_maxbytes=0
"""
        Path(f"/etc/supervisor/conf.d/{name}.conf").write_text(config)
        supervisor("reread")
        supervisor("update", name)
        if not any(word in service_status(name) for word in ["RUNNING", "STARTING"]):
            supervisor("start", name)
    deadline = time.monotonic() + 600
    pending = {0, 1}
    verified = {}
    while pending and time.monotonic() < deadline:
        for worker in sorted(pending):
            try:
                observed = metadata(18000 + worker)
            except (ConnectionError, OSError, TimeoutError):
                state = service_status(f"maniguard_latent_policy_worker{worker}")
                if any(word in state for word in ["EXITED", "FATAL", "STOPPED"]):
                    raise RuntimeError("Policy worker is terminal; inspect before repair: " + state) from None
                continue
            if (not policy_matches(observed, family, False)
                    or observed.get("physical_gpu_uuid") != GPU_UUIDS[worker]):
                raise RuntimeError("Wrong family, serving mode or physical GPU on worker endpoint")
            verified[str(worker)] = observed
            pending.remove(worker)
        if pending:
            time.sleep(5)
    if pending:
        raise RuntimeError("Worker policy observation timed out; inspect saved supervisor handles")
    atomic(run / "active-worker-policies.json", {"family": family, "workers": verified,
                                               "verified_epoch": time.time()})


def ensure_policy(run, family, audit=False):
    if family not in ["jar", "clutter"] or type(audit) is not bool:
        raise ValueError("Undeclared family or serving mode")
    try:
        if policy_matches(metadata(), family, audit):
            return
    except (ConnectionError, OSError, TimeoutError):
        pass
    assert_no_live_worker(run)
    for worker in [0, 1]:
        stop_if_running(f"maniguard_latent_policy_worker{worker}")
    state = service_status("maniguard_latent_policy")
    if "RUNNING" in state or "STARTING" in state:
        supervisor("stop", "maniguard_latent_policy")
    mode = "audit" if audit else "physical"
    run = Path(run).resolve()
    extra = " --audit" if audit else ""
    config = f"""[program:maniguard_latent_policy]
command={ROOT}/third_party/openpi/.venv/bin/python -u {ROOT}/experiments/scripts/maniguard_latent_serve.py --family {family} --port 18000 --ledger {run}/server-{family}-{mode}.jsonl{extra}
directory={ROOT}
environment=CUDA_VISIBLE_DEVICES="1",XLA_PYTHON_CLIENT_PREALLOCATE="false",MANIGUARD_SKIP_OMNIGIBSON_PATCH="1",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",JAX_COMPILATION_CACHE_DIR="{ROOT}/data/maniguard/jax-cache",JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1"
autostart=true
autorestart=false
stdout_logfile={run}/policy-{family}-{mode}.log
redirect_stderr=true
stdout_logfile_maxbytes=0
"""
    Path("/etc/supervisor/conf.d/maniguard_latent_policy.conf").write_text(config)
    supervisor("reread")
    supervisor("update", "maniguard_latent_policy")
    # If this exact config already existed in EXITED state, update does not start it.
    state = service_status("maniguard_latent_policy")
    if "RUNNING" not in state and "STARTING" not in state:
        supervisor("start", "maniguard_latent_policy")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            observed = metadata()
        except (ConnectionError, OSError, TimeoutError):
            state = service_status("maniguard_latent_policy")
            if any(word in state for word in ["EXITED", "FATAL", "STOPPED"]):
                raise RuntimeError("Policy is terminal; diagnose before restarting: " + state) from None
            time.sleep(5)
            continue
        if not policy_matches(observed, family, audit):
            raise RuntimeError("Replacement policy returned unexpected provenance or serving mode")
        atomic(run / "active-policy.json", {"family": family, "audit": audit,
                                            "verified_epoch": time.time(), "metadata": observed})
        return
    raise RuntimeError("Policy startup observation timed out; inspect same supervisor handle")


def start_runner(run, protocol, jobs, name, *, paired=False):
    if any(c not in "abcdefghijklmnopqrstuvwxyz_" for c in name):
        raise ValueError("Invalid supervisor coordinator name")
    run, protocol, jobs = Path(run).resolve(), Path(protocol).resolve(), Path(jobs).resolve()
    path = Path("/etc/supervisor/conf.d") / (name + ".conf")
    script = "maniguard_latent_parallel_run.py" if paired else "maniguard_latent_run.py"
    config = f"""[program:{name}]
command={ROOT}/data/maniguard/envs/tools/bin/python -u {ROOT}/experiments/scripts/{script} --run {run} --protocol {protocol} --jobs {jobs}
directory={ROOT}
autostart=true
autorestart=false
stdout_logfile={run}/{name}.log
redirect_stderr=true
stdout_logfile_maxbytes=0
stopasgroup=false
"""
    if not path.exists():
        path.write_text(config)
        supervisor("reread")
        supervisor("update", name)
        return
    # Existing coordinators may have a differently named log from the manual
    # launch. Their immutable command remains valid; never rewrite a live service.
    state = service_status(name)
    if "RUNNING" not in state and "STARTING" not in state:
        supervisor("start", name)
