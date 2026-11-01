"""Launch one unchanged frozen final profile under SHA-bound new time authority."""

import argparse
import hashlib
import importlib.util
import inspect
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def remote_launch(metadata):
    """Stdlib-only remote checks, retaining the original software/source/host gates."""
    import datetime as dt
    import fcntl
    import hashlib
    import json
    import os
    from pathlib import Path
    import re
    import shlex
    import subprocess
    import xml.etree.ElementTree as ET

    root = Path("/workspace/Safety-Dial")
    wrapper = "experiments/scripts/run_lewm_navigation_continuation.py"
    launcher = "experiments/scripts/launch_lewm_campaign_continuation.py"
    original_deadline = "2026-10-31T07:23:14+00:00"
    hosts = {
        "pusht": "wq4ar2qx2ff6r4",
        "navigation": "vukovc3t4o1dui",
        "recovery": "nj2fqkp2prgk8x",
    }

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def relative(name):
        path = Path(name)
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and "\\" not in name
            and str(path) == name,
            "A canonical relative project path is required",
        )
        target = (root / path).resolve()
        require(target.is_relative_to(root), "Project path escapes canonical root")
        return target

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def identity(pid):
        try:
            raw = (Path("/proc") / str(pid) / "stat").read_text()
        except FileNotFoundError:
            return None
        fields = raw[raw.rfind(")") + 2 :].split()
        return {"pid": pid, "state": fields[0], "start_ticks": fields[19]}

    require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,120}", metadata["name"]), "Unsafe stage name")
    authority_path = relative(metadata["authority"])
    require(digest(authority_path) == metadata["authority_sha256"], "Authority SHA differs")
    authority = json.loads(authority_path.read_text())
    require(
        authority["schema"] == "lewm-frozen-final-continuation-authority-v1"
        and authority["original_deadline_utc"] == original_deadline
        and authority["maximum_duration_seconds"] == 5400
        and authority["total_campaign_cap_usd"] == 13
        and authority["no_new_training"] is True
        and authority["no_outcome_based_changes"] is True,
        "Continuation is outside its explicit frozen scope",
    )
    first = dt.datetime.fromisoformat(authority["first_restart_utc"])
    deadline = dt.datetime.fromisoformat(authority["new_deadline_utc"])
    now = dt.datetime.now(dt.timezone.utc)
    require(first.tzinfo is not None and deadline.tzinfo is not None, "Explicit timezone required")
    require(0 < (deadline - first).total_seconds() <= 5400, "Continuation exceeds90minutes")
    require(first <= now and now + dt.timedelta(minutes=20) < deadline, "Preservation reserve")
    pod = os.environ.get("RUNPOD_POD_ID")
    require(pod == hosts[metadata["lane"]], "Wrong actual continuation pod")
    ready = json.loads((root / "environment/ready.json").read_text())
    require(ready["pod_id"] == pod, "Ready record belongs to another pod")
    protocol_path = relative(metadata["protocol"])
    require(
        digest(protocol_path) == metadata["expected_protocol_sha256"],
        "Original frozen protocol differs",
    )
    protocol = json.loads(protocol_path.read_text())
    evaluation = protocol["evaluation"]
    require(
        protocol["schema"] in ("lewm-navigation-comparison-v1", "lewm-navigation-comparison-v2")
        and evaluation["deadline_utc"] == original_deadline,
        "Only original final comparison profiles may continue",
    )
    matches = [
        row for row in authority["profiles"] if row["protocol"]["path"] == metadata["protocol"]
    ]
    require(len(matches) == 1, "Exactly one authorized frozen profile is required")
    profile = matches[0]
    original_entrypoint = metadata["original_entrypoint"]
    require(
        original_entrypoint
        in {
            "experiments/scripts/run_lewm_navigation_campaign_v2.py",
            "experiments/scripts/run_lewm_navigation_campaign_v3.py",
        }
        and profile["entrypoint"]["path"] == original_entrypoint
        and profile["entrypoint"]["sha256"] == digest(relative(original_entrypoint))
        and profile["protocol"]["sha256"] == metadata["expected_protocol_sha256"]
        and profile["pod_id"] == pod == evaluation["execution_pod_id"]
        and pod in protocol.get("allowed_pod_ids", [pod])
        and profile["output_namespace"] == protocol["output_namespace"] == metadata["out"],
        "Profile/host/output/evaluator authority differs",
    )
    pins = authority["source_sha256"]
    require(
        pins.get(launcher) == metadata["orchestration_source_sha256"] == digest(relative(launcher))
        and pins.get(wrapper) == digest(relative(wrapper))
        and pins.get(original_entrypoint) == profile["entrypoint"]["sha256"],
        "Authority must pin the launcher, wrapper and unchanged evaluator",
    )
    for name, expected in pins.items():
        require(digest(relative(name)) == expected, "Continuation source differs: " + name)
    for declaration in (protocol, evaluation):
        for field in ("source_sha256", "asset_sha256"):
            for name, expected in declaration.get(field, {}).items():
                require(digest(relative(name)) == expected, "Frozen dependency differs: " + name)
    gate = relative("environment/" + metadata["gate"])
    suite = ET.parse(gate / "tests.xml").getroot()
    require(
        len(list(suite.iter("testcase"))) == metadata["tests"]
        and not any(list(suite.iter(kind)) for kind in ("failure", "error", "skipped")),
        "Existing software gate did not pass",
    )
    require("All checks passed!" in (gate / "ruff.log").read_text(), "Existing lint gate missing")
    require(Path("/workspace/activate-lewm-campaign.sh").is_file(), "Runtime activation missing")
    workers = root / "environment/workers"
    workers.mkdir(exist_ok=True)
    with (root / "environment/continuation-launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in workers.glob("*.json"):
            previous = json.loads(path.read_text())
            process = identity(previous["pid"])
            require(
                not process
                or process["state"] == "Z"
                or process["start_ticks"] != str(previous["start_ticks"]),
                "A registered stage worker remains live: " + path.name,
            )
        record = workers / (metadata["name"] + ".json")
        output = relative(metadata["out"])
        script = root / "environment" / (metadata["name"] + ".sh")
        log_path = root / "environment" / (metadata["name"] + ".log")
        sidecar = (
            root / "environment/continuation" / (metadata["expected_protocol_sha256"] + ".json")
        )
        require(
            not any(path.exists() for path in (record, output, script, log_path, sidecar)),
            "Never duplicate a stage, resume a partial attempt or overwrite evidence",
        )
        remaining = int((deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()) - 600
        require(remaining > 600, "Insufficient preservation reserve")
        argv = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            str(remaining) + "s",
            "python",
            "-u",
            wrapper,
            "--authority",
            metadata["authority"],
            "--authority-sha256",
            metadata["authority_sha256"],
            "--entrypoint",
            original_entrypoint,
            "--protocol",
            metadata["protocol"],
            "--out",
            metadata["out"],
        ]
        command = "\n".join(
            [
                "set -eu",
                "source /workspace/activate-lewm-campaign.sh",
                "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/Safety-Dial/experiments",
                "cd /workspace/Safety-Dial",
                shlex.join(argv),
                "",
            ]
        )
        with script.open("x") as stream:
            stream.write(command)
        with log_path.open("xb", buffering=0) as log:
            worker = subprocess.Popen(
                ["bash", str(script)],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        process = identity(worker.pid)
        require(process is not None, "New worker retired before identity registration")
        value = {
            **metadata,
            "entrypoint": wrapper,
            "pid": worker.pid,
            "start_ticks": process["start_ticks"],
            "pod_id": pod,
            "launched_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "original_deadline_utc": original_deadline,
            "execution_deadline_utc": authority["new_deadline_utc"],
        }
        with record.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=("pusht", "navigation", "recovery"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--tests", type=int, required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    metadata = vars(args).copy()
    for field in ("authority_sha256", "protocol_sha256"):
        value = metadata[field]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("An exact root-reviewed SHA256 is required: " + field)
    metadata["original_entrypoint"] = metadata.pop("entrypoint")
    metadata["expected_protocol_sha256"] = args.protocol_sha256
    metadata["orchestration_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    spec = importlib.util.spec_from_file_location(
        "control", ROOT / "experiments/scripts/lewm_campaign_pods.py"
    )
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    remote = inspect.getsource(remote_launch) + "\nremote_launch(" + repr(metadata) + ")\n"
    ssh, host = control.ssh_args(args.lane)
    subprocess.run(["ssh", *ssh, host, "python3 -c " + shlex.quote(remote)], check=True)


if __name__ == "__main__":
    main()
