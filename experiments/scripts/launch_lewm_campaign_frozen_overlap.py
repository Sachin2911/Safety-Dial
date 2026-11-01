"""Strict recovery-only overlap of one frozen stage with one exact live peer.

The scientific runner/protocol remains unchanged. A SHA-bound prospective record
names one live peer and one fresh stage; a host-wide flock prevents racing launches.
This tool never rents hosts, resumes jobs, modifies scientific sources or stops peers.
"""

import argparse
import base64
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
RECOVERY_POD = "nj2fqkp2prgk8x"
SCHEMA = "lewm-recovery-two-worker-authorization-v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def identity(pid):
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text()
        fields = raw[raw.rfind(")") + 2 :].split()
        return {
            "pid": int(pid),
            "start_ticks": fields[19],
            "state": fields[0],
            "parent_pid": int(fields[1]),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(owner):
    current = identity(owner["pid"])
    return bool(
        current and current["state"] != "Z" and current["start_ticks"] == str(owner["start_ticks"])
    )


def relative(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe project-relative path")
    require(str(path) == name and name not in ("", "."), "Noncanonical project path")
    require((root / path).resolve().is_relative_to(root), "Project path escapes root")
    return root / path


def descends_from(pid, ancestor):
    visited = set()
    while pid > 1 and pid not in visited:
        visited.add(pid)
        current = identity(pid)
        if not current:
            return False
        if pid == ancestor["pid"]:
            return current["start_ticks"] == str(ancestor["start_ticks"])
        pid = current["parent_pid"]
    return False


def verify_no_unknown_processes(root, peer):
    """Allow only this launcher, exact peer descendants and the exact live guard."""
    guard_path = root / "environment/deadline-process.json"
    guard = json.loads(guard_path.read_text())
    require(guard["pod_id"] == RECOVERY_POD and live(guard), "Independent guard is not live")
    own = identity(os.getpid())
    approved = [own, peer, guard]
    observed = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        current = identity(pid)
        if not current or current["state"] == "Z":
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            argv = raw.split(b"\0")
            cwd = Path(os.readlink(entry / "cwd"))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        executable = Path(os.fsdecode(argv[0])).name if argv else ""
        python = executable.startswith(("python", "pytest", "torchrun"))
        related = str(root).encode() in raw or cwd.is_relative_to(root)
        if not (python and related):
            continue
        require(
            any(descends_from(pid, owner) for owner in approved),
            "Unknown live project Python process: " + str(pid),
        )
        observed.append({"identity": current, "cmdline_sha256": hashlib.sha256(raw).hexdigest()})
    return {"guard": guard, "approved_project_processes": observed}


def check_gate(root, stage):
    gate = relative(root, "environment/" + stage["gate"])
    suite = ET.parse(gate / "tests.xml").getroot()
    require(
        len(list(suite.iter("testcase"))) == stage["tests"]
        and not any(list(suite.iter(k)) for k in ("failure", "error", "skipped")),
        "Existing software gate did not pass",
    )
    require("All checks passed!" in (gate / "ruff.log").read_text(), "Existing lint gate missing")
    return {name: sha(gate / name) for name in ("tests.xml", "ruff.log")}


def remote_launch(metadata):
    root = Path("/workspace/Safety-Dial")
    require(sha(__file__) == metadata["launcher_sha256"], "Remote orchestration source differs")
    authorization = relative(root, metadata["authorization"])
    require(sha(authorization) == metadata["authorization_sha256"], "Authorization differs")
    plan = json.loads(authorization.read_text())
    require(
        plan["schema"] == SCHEMA and plan["execution_pod_id"] == RECOVERY_POD,
        "Only the declared recovery host may overlap",
    )
    require(plan["launcher_sha256"] == sha(__file__), "Prospective launcher source changed")
    require(
        plan["max_concurrent_experiment_workers"] == 2
        and plan["scientific_settings_unchanged"] is True
        and plan["all_actual_work_charged"] is True,
        "Missing overlap safeguards",
    )
    require(
        plan["purpose"] in ("engineering_qualification", "qualified_optimizer_overlap"),
        "Unknown overlap stage",
    )
    peer, stage = plan["peer"], plan["stage"]
    require(
        peer["name"] != stage["name"] and len({peer["name"], stage["name"]}) == 2,
        "Exactly two different worker names required",
    )
    for name in (peer["name"], stage["name"]):
        require(name.replace("-", "").replace("_", "").isalnum(), "Unsafe worker name")
    require(stage["name"] == metadata["name"], "Wrong declared fresh stage")
    require(plan["deadline_utc"] == "2026-10-31T07:23:14+00:00", "Global cutoff changed")
    ready = json.loads((root / "environment/ready.json").read_text())
    require(ready["pod_id"] == RECOVERY_POD, "Wrong ready pod identity")
    # The activated runner must inherit the actual same pod; never synthesize its identity.
    process_env = Path("/proc/1/environ").read_bytes().split(b"\0")
    require(("RUNPOD_POD_ID=" + RECOVERY_POD).encode() in process_env, "Wrong actual pod")
    workers = root / "environment/workers"
    lock_path = root / "environment/recovery-two-worker-launch.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        peer_path = workers / (peer["name"] + ".json")
        require(sha(peer_path) == peer["record_sha256"], "Exact peer worker record changed")
        peer_record = json.loads(peer_path.read_text())
        require(
            all(str(peer_record[k]) == str(peer[k]) for k in ("pid", "start_ticks")),
            "Peer PID/start identity differs",
        )
        require(peer_record["pod_id"] == RECOVERY_POD and live(peer), "Exact peer is not live")
        active = []
        for path in workers.glob("*.json"):
            record = json.loads(path.read_text())
            require("pid" in record and "start_ticks" in record, "Worker record lacks identity")
            if live(record):
                require(path == peer_path, "Unexpected live registered worker: " + path.name)
                active.append(path.name)
        require(active == [peer_path.name], "Exactly the declared one live peer is required")
        process_review = verify_no_unknown_processes(root, peer)
        gate_hashes = check_gate(root, stage)
        protocol_path = relative(root, stage["protocol"])
        require(sha(protocol_path) == stage["protocol_sha256"], "Frozen protocol differs")
        design = json.loads(protocol_path.read_text())
        evaluation = design.get("evaluation", design)
        require(
            stage["entrypoint"] == "experiments/scripts/run_lewm_navigation_campaign_v2.py"
            and stage["entry_args"] == []
            and evaluation["maximum_episode_actions"] == 30,
            "This bounded exception permits only the frozen recovery NAV v2 cap30 runner",
        )
        require(stage["out"] == design["output_namespace"], "Frozen output namespace differs")
        require(
            evaluation["execution_pod_id"] == RECOVERY_POD
            and RECOVERY_POD in design.get("allowed_pod_ids", [RECOVERY_POD]),
            "Protocol belongs to another pod",
        )
        deadline = design.get("deadline_utc", evaluation.get("deadline_utc"))
        require(deadline == plan["deadline_utc"], "Protocol deadline differs")
        cutoff = dt.datetime.fromisoformat(deadline)
        require(
            dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=20) < cutoff,
            "Preservation reserve",
        )
        for declaration in (design, evaluation):
            for field in ("source_sha256", "asset_sha256"):
                for name, expected in declaration.get(field, {}).items():
                    require(
                        sha(relative(root, name)) == expected, "Frozen dependency changed: " + name
                    )
        entrypoint = relative(root, stage["entrypoint"])
        require(
            design.get("source_sha256", {}).get(stage["entrypoint"]) == sha(entrypoint),
            "Entrypoint is absent from the scientific source freeze",
        )
        if plan["purpose"] == "engineering_qualification":
            require(
                design.get("engineering_overlap_qualification", {}).get("no_final_access") is True
                and len(design["jobs"]) == 1
                and design["maximum_episode_starts"] == 1
                and design["maximum_environment_actions"] == 30,
                "Engineering qualification must be exactly one bounded development job",
            )
        else:
            ref = plan["qualification"]
            path = relative(root, ref["path"])
            require(sha(path) == ref["sha256"], "Overlap qualification changed")
            report = json.loads(path.read_text())
            require(
                report["qualified"] is True and report["pod_id"] == RECOVERY_POD,
                "Exact recovery overlap is not qualified",
            )
        record_path = workers / (stage["name"] + ".json")
        out = relative(root, stage["out"])
        require(
            not record_path.exists() and not out.exists(),
            "Never duplicate a stage or replay outcomes",
        )
        remaining = int((cutoff - dt.datetime.now(dt.timezone.utc)).total_seconds()) - 600
        observation_path = relative(root, plan["peer_observations_path"])
        require(not observation_path.exists(), "Never replay an overlap observation namespace")
        scientific_command = [
            "python",
            "-u",
            stage["entrypoint"],
            *stage["entry_args"],
            "--protocol",
            stage["protocol"],
            "--out",
            stage["out"],
        ]
        monitor = {
            "peer": peer,
            "observation_path": str(observation_path),
            "command": scientific_command,
            "authorization": metadata,
            "pod_id": RECOVERY_POD,
        }
        monitor_encoded = base64.b64encode(json.dumps(monitor).encode()).decode()
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            str(remaining) + "s",
            "python",
            "-u",
            str(Path(__file__).resolve()),
            "_monitor",
            "--metadata-base64",
            monitor_encoded,
        ]
        script = root / "environment" / (stage["name"] + ".sh")
        script.write_text(
            "\n".join(
                [
                    "set -eu",
                    "source /workspace/activate-lewm-campaign.sh",
                    "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                    "PYTHONPATH=/workspace/Safety-Dial/experiments",
                    "cd /workspace/Safety-Dial",
                    shlex.join(command),
                    "",
                ]
            )
        )
        with (root / "environment" / (stage["name"] + ".log")).open("ab", buffering=0) as log:
            worker = subprocess.Popen(
                ["bash", str(script)],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launched = identity(worker.pid)
        value = {
            **stage,
            **launched,
            "pod_id": RECOVERY_POD,
            "launched_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "overlap_authorization": metadata,
            "exact_live_peer": peer,
            "gate_sha256": gate_hashes,
            "process_review_before_launch": process_review,
            "maximum_declared_concurrency": 2,
        }
        write(record_path, value)
        print(json.dumps(value, indent=2), flush=True)


def monitor_worker(metadata):
    """Observe the exact peer around unchanged scientific child execution."""
    require(
        os.environ.get("RUNPOD_POD_ID") == metadata["pod_id"] == RECOVERY_POD,
        "Monitor belongs only to the actual recovery pod",
    )
    path = Path(metadata["observation_path"])
    require(not path.exists(), "Overlap observations already exist")
    peer = metadata["peer"]

    def observation():
        return {
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "monotonic_seconds": time.monotonic(),
            "expected_identity": {k: peer[k] for k in ("pid", "start_ticks")},
            "observed_identity": identity(peer["pid"]),
            "live": live(peer),
        }

    record = {
        "schema": "lewm-recovery-overlap-peer-observations-v1",
        "pod_id": RECOVERY_POD,
        "authorization": metadata["authorization"],
        "scientific_command": metadata["command"],
        "before": observation(),
        "samples": [],
        "new_model_simulator_calls_by_monitor": 0,
    }
    require(record["before"]["live"], "Exact peer retired before execution")
    write(path, record)
    child = subprocess.Popen(metadata["command"], stdin=subprocess.DEVNULL)
    record["child_identity"] = identity(child.pid)
    write(path, record)
    while child.poll() is None:
        record["samples"].append(observation())
        write(path, record)
        time.sleep(1)
    record["after"] = observation()
    record["returncode"] = child.returncode
    record["all_peer_observations_live"] = all(
        item["live"] for item in [record["before"], *record["samples"], record["after"]]
    )
    write(path, record)
    raise SystemExit(child.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=("recovery", "_remote", "_monitor"))
    parser.add_argument("--authorization")
    parser.add_argument("--name")
    parser.add_argument("--metadata-base64")
    args = parser.parse_args()
    if args.lane == "_monitor":
        monitor_worker(json.loads(base64.b64decode(args.metadata_base64)))
        return
    if args.lane == "_remote":
        remote_launch(json.loads(base64.b64decode(args.metadata_base64)))
        return
    require(args.authorization and args.name, "Prospective authorization and stage name required")
    path = relative(ROOT, args.authorization)
    metadata = {
        "authorization": args.authorization,
        "authorization_sha256": sha(path),
        "name": args.name,
        "launcher_sha256": sha(__file__),
    }
    spec = importlib.util.spec_from_file_location(
        "control", ROOT / "experiments/scripts/lewm_campaign_pods.py"
    )
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    ssh, host = control.ssh_args("recovery")
    encoded = base64.b64encode(json.dumps(metadata).encode()).decode()
    command = [
        "python3",
        "/workspace/Safety-Dial/experiments/scripts/" + Path(__file__).name,
        "_remote",
        "--metadata-base64",
        encoded,
    ]
    subprocess.run(["ssh", *ssh, host, shlex.join(command)], check=True)


if __name__ == "__main__":
    main()
