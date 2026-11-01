"""Local-only control of the explicitly authorized seven-hour, $10 campaign."""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/runpod-lewm-campaign-20261031"
CLI = str(Path.home() / ".local/bin/runpodctl")
ENV = {**os.environ, "RUNPOD_NO_UPDATE_CHECK": "1"}
START = dt.datetime.fromtimestamp(1793406194, dt.timezone.utc)
DEADLINE = START + dt.timedelta(hours=7)
RATE_CEILING = 0.60
DISK_RATE = 40 * 0.10 / 720


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def cli(*args, timeout=60):
    result = subprocess.run([CLI, *args], env=ENV, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"RunPod {args[0]} {args[1]} failed: " + result.stderr[:220])
    return json.loads(result.stdout) if result.stdout.strip() else {}


def pods():
    value = cli("pod", "list", "--all")
    return value.get("pods", []) if isinstance(value, dict) else value


def ledger():
    return read(OUT / "campaign.json")


def lane_dir(lane):
    if lane not in ("pusht", "navigation", "recovery"):
        raise ValueError("Unknown authorized lane")
    return OUT / lane


def ssh_args(lane, scp=False):
    folder = lane_dir(lane)
    connection = read(folder / "connection.json")
    return [
        "-i",
        str(Path.home() / ".ssh/runpod_ed25519"),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UserKnownHostsFile=" + str(folder / "known_hosts"),
        "-P" if scp else "-p",
        str(connection["port"]),
    ], "root@" + connection["host"]


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "campaign.json").exists():
        return
    write(
        OUT / "campaign.json",
        {
            "goal_started_utc": START.isoformat(),
            "deadline_utc": DEADLINE.isoformat(),
            "fresh_cap_usd": 10.0,
            "minimum_parallel_pods": 2,
            "maximum_pods": 3,
            "base_lanes": ["pusht", "navigation"],
            "provider_rate_ceiling": RATE_CEILING,
            "disk_hourly_upper_bound": DISK_RATE,
            "authorization": "User authorized up to three pods, at least two parallel experiment lanes, seven hours of persistent research and a $10 shared cap.",
            "prior_allocations": "Closed; no budget inherited and no historical jobs resumed.",
        },
    )
    with (OUT / "guard.log").open("ab", buffering=0) as log:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "guard"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(
        OUT / "guard-process.json",
        {
            "pid": worker.pid,
            "start_ticks": Path(f"/proc/{worker.pid}/stat").read_text().split()[21],
        },
    )


def arm_guard():
    # Called from a network-enabled local shell; provider credentials stay here.
    pods()
    old = read(OUT / "guard-process.json")
    proc = Path("/proc") / str(old["pid"])
    if proc.exists():
        stat = (proc / "stat").read_text().split()
        if stat[2] != "Z" and stat[21] == str(old["start_ticks"]):
            import signal

            os.kill(old["pid"], signal.SIGTERM)
    write(OUT / "initial-guard-process.json", old)
    with (OUT / "guard.log").open("ab", buffering=0) as log:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "guard"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    value = {
        "pid": worker.pid,
        "start_ticks": Path(f"/proc/{worker.pid}/stat").read_text().split()[21],
        "provider_network_read_verified": True,
    }
    write(OUT / "guard-process.json", value)
    print(json.dumps(value))


def projected_cost(at):
    total = 0.0
    for folder in OUT.iterdir():
        path = folder / "session.json"
        if not path.is_file():
            continue
        session = read(path)
        start = dt.datetime.fromisoformat(session["requested_utc"])
        close = folder / "closeout.json"
        end = dt.datetime.fromisoformat(read(close)["checked_utc"]) if close.exists() else at
        total += (
            max(0, (end - start).total_seconds()) / 3600 * (float(session["costPerHr"]) + DISK_RATE)
        )
    return total


def create(lane, *, gpu_id="NVIDIA A40", cloud_type="SECURE", rate_ceiling=RATE_CEILING):
    initialize()
    if not 0 < rate_ceiling <= RATE_CEILING:
        raise ValueError("Allocation rate must stay within the authorized provider ceiling")
    if lane != "recovery" and (
        gpu_id != "NVIDIA A40" or cloud_type != "SECURE" or rate_ceiling != RATE_CEILING
    ):
        raise ValueError("Only the unallocated recovery lane accepts a new GPU choice")
    folder = lane_dir(lane)
    folder.mkdir(exist_ok=True)
    with (OUT / "allocation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / "session.json").exists() or (folder / "intent.json").exists():
            raise RuntimeError("Existing allocation or unresolved intent; inspect, never duplicate")
        now = dt.datetime.now(dt.timezone.utc)
        if now + dt.timedelta(minutes=20) >= DEADLINE:
            raise RuntimeError("No useful rental time remains")
        allocations = list(OUT.glob("*/intent.json"))
        if len(allocations) >= 3:
            raise RuntimeError("Three-allocation safety ceiling reached")
        remaining = (DEADLINE - now).total_seconds() / 3600
        if lane == "recovery":
            headroom = 9.60 - projected_cost(DEADLINE)
            duration = min(remaining, headroom / (rate_ceiling + DISK_RATE))
            if duration < 0.25:
                raise RuntimeError("No shared-budget headroom for a recovery pod")
            cutoff = now + dt.timedelta(hours=duration)
        else:
            # Reserve both required base pods through the common deadline before launch.
            if projected_cost(DEADLINE) + remaining * (RATE_CEILING + DISK_RATE) > 9.60:
                raise RuntimeError("Shared budget would be exceeded")
            cutoff = DEADLINE
        intent = {
            "lane": lane,
            "name": "safetydial-lewm-campaign-20261031-" + lane,
            "requested_utc": now.isoformat(),
            "deadline_utc": cutoff.isoformat(),
            "requested_gpu_id": gpu_id,
            "requested_cloud_type": cloud_type,
            "allocation_rate_ceiling": rate_ceiling,
        }
        write(folder / "intent.json", intent)
        response = cli(
            "pod",
            "create",
            "--name",
            intent["name"],
            "--cloud-type",
            cloud_type,
            "--gpu-id",
            gpu_id,
            "--gpu-count",
            "1",
            "--image",
            "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            "--container-disk-in-gb",
            "30",
            "--volume-in-gb",
            "10",
            "--ports",
            "22/tcp",
            timeout=90,
        )
        response = response.get("pod", response)
        record = {
            **intent,
            **{
                k: response[k]
                for k in ("id", "costPerHr", "desiredStatus", "imageName")
                if k in response
            },
        }
        if not record.get("id"):
            raise RuntimeError("Missing allocation ID; inspect exact provider name before retry")
        write(folder / "session.json", record)
        if float(record.get("costPerHr", 999)) > rate_ceiling:
            cli("pod", "stop", record["id"])
            raise RuntimeError("Rate exceeds campaign ceiling; stopped exact pod")
        print(json.dumps(record), flush=True)


def connect(lane):
    folder = lane_dir(lane)
    session = read(folder / "session.json")
    value = cli("pod", "get", session["id"])
    allowed = (
        "id",
        "name",
        "costPerHr",
        "desiredStatus",
        "publicIp",
        "portMappings",
        "runtime",
        "machineId",
        "gpuCount",
    )
    write(folder / "latest-provider.json", {k: value[k] for k in allowed if k in value})
    ssh_info = value.get("ssh", {})
    host = value.get("publicIp") or ssh_info.get("ip")
    port = value.get("portMappings", {}).get("22") or ssh_info.get("port")
    if not host or not port:
        print(json.dumps({"id": session["id"], "ssh_ready": False}))
        return
    connection = {"host": host, "port": int(port), "pod_id": session["id"]}
    known = folder / "known_hosts"
    scan = subprocess.run(
        ["ssh-keyscan", "-T", "10", "-p", str(port), host],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if scan.returncode or not scan.stdout.strip():
        print(json.dumps({"id": session["id"], "ssh_ready": False}))
        return
    if known.exists() and known.read_text() != scan.stdout:
        existing = set(known.read_text().splitlines())
        if not existing.intersection(scan.stdout.splitlines()):
            raise RuntimeError("Host keys changed; investigate exact pod")
    else:
        known.write_text(scan.stdout)
    write(folder / "connection.json", connection)
    args, target = ssh_args(lane)
    result = subprocess.run(["ssh", *args, target, "true"], timeout=15)
    if result.returncode:
        raise RuntimeError("Pod SSH check failed")
    print(json.dumps({**connection, "ssh_ready": True}))


def guard():
    while not (OUT / "guard-cancelled").exists():
        now = dt.datetime.now(dt.timezone.utc)
        amount = projected_cost(now)
        write(
            OUT / "guard-state.json",
            {
                "heartbeat_utc": now.isoformat(),
                "deadline_utc": DEADLINE.isoformat(),
                "estimated_spend_usd": amount,
                "cap_usd": 10,
                "credentials_remain_local": True,
            },
        )
        for path in OUT.glob("*/session.json"):
            session = read(path)
            folder = path.parent
            if (folder / "closeout.json").exists() or (folder / "deadline-stop.json").exists():
                continue
            cutoff = dt.datetime.fromisoformat(session["deadline_utc"])
            if now >= cutoff or amount >= 9.75:
                try:
                    cli("pod", "stop", session["id"], timeout=20)
                    write(
                        folder / "deadline-stop.json",
                        {
                            "pod_id": session["id"],
                            "stopped_utc": now.isoformat(),
                            "reason": "Deadline" if now >= cutoff else "Shared cap reserve",
                        },
                    )
                except Exception as exc:
                    write(folder / "guard-error.json", {"type": type(exc).__name__})
        time.sleep(10)


def delete(lane):
    folder = lane_dir(lane)
    proof = read(folder / "preservation-verified.json")
    if not proof.get("verified") or not proof.get("all_archive_members_verified"):
        raise RuntimeError("Independent local preservation required")
    session = read(folder / "session.json")
    cli("pod", "delete", session["id"])
    absent = not any(p.get("id") == session["id"] for p in pods())
    result = subprocess.run(
        [CLI, "pod", "get", session["id"]], env=ENV, capture_output=True, text=True, timeout=30
    )
    missing = result.returncode != 0 and (
        "404" in result.stdout + result.stderr
        or "not found" in (result.stdout + result.stderr).lower()
    )
    close = {
        "pod_id": session["id"],
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "provider_absence_verified": absent and missing,
        "provider_get_not_found": missing,
        "provider_list_absent": absent,
    }
    if not close["provider_absence_verified"]:
        raise RuntimeError("Deletion provider confirmation incomplete")
    write(folder / "closeout.json", close)
    print(json.dumps(close))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "init",
            "arm",
            "create",
            "connect",
            "guard",
            "get",
            "exec",
            "push",
            "pull",
            "delete",
            "cost",
        ),
    )
    parser.add_argument("lane", nargs="?")
    parser.add_argument("arguments", nargs="*")
    parser.add_argument("--gpu-id", default="NVIDIA A40")
    parser.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"), default="SECURE")
    parser.add_argument("--rate-ceiling", type=float, default=RATE_CEILING)
    args = parser.parse_args()
    if args.operation == "init":
        initialize()
    elif args.operation == "arm":
        arm_guard()
    elif args.operation == "create":
        create(
            args.lane,
            gpu_id=args.gpu_id,
            cloud_type=args.cloud_type,
            rate_ceiling=args.rate_ceiling,
        )
    elif args.operation == "connect":
        connect(args.lane)
    elif args.operation == "guard":
        guard()
    elif args.operation == "cost":
        print(
            json.dumps(
                {
                    "estimated_usd": projected_cost(dt.datetime.now(dt.timezone.utc)),
                    "reserved_through_deadline_usd": projected_cost(DEADLINE),
                    **ledger(),
                }
            )
        )
    elif args.operation == "get":
        folder = lane_dir(args.lane)
        value = cli("pod", "get", read(folder / "session.json")["id"])
        value = {
            k: value[k]
            for k in (
                "id",
                "name",
                "desiredStatus",
                "publicIp",
                "portMappings",
                "costPerHr",
                "machineId",
            )
            if k in value
        }
        write(folder / "latest-provider.json", value)
        print(json.dumps(value))
    elif args.operation in ("exec", "push", "pull"):
        ssh, target = ssh_args(args.lane, scp=args.operation != "exec")
        if args.operation == "exec":
            command = ["ssh", *ssh, target, args.arguments[0]]
        elif args.operation == "push":
            command = ["scp", *ssh, args.arguments[0], target + ":" + args.arguments[1]]
        else:
            command = ["scp", *ssh, target + ":" + args.arguments[0], args.arguments[1]]
        subprocess.run(command, check=True)
    elif args.operation == "delete":
        delete(args.lane)


if __name__ == "__main__":
    main()
