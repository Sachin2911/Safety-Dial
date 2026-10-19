"""Supervised serial policy lifecycle, guarded against interrupting a live rollout."""

import json
import subprocess
import time
from pathlib import Path

from helpers.maniguard_client import StudyClient
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


def metadata():
    client = StudyClient(host="127.0.0.1", port=18000)
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


def ensure_policy(run, family, audit=False):
    if family not in ["jar", "clutter"] or type(audit) is not bool:
        raise ValueError("Undeclared family or serving mode")
    try:
        if policy_matches(metadata(), family, audit):
            return
    except (ConnectionError, OSError, TimeoutError):
        pass
    assert_no_live_worker(run)
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


def start_runner(run, protocol, jobs, name):
    if any(c not in "abcdefghijklmnopqrstuvwxyz_" for c in name):
        raise ValueError("Invalid supervisor coordinator name")
    run, protocol, jobs = Path(run).resolve(), Path(protocol).resolve(), Path(jobs).resolve()
    path = Path("/etc/supervisor/conf.d") / (name + ".conf")
    config = f"""[program:{name}]
command={ROOT}/data/maniguard/envs/tools/bin/python -u {ROOT}/experiments/scripts/maniguard_latent_run.py --run {run} --protocol {protocol} --jobs {jobs}
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
