"""Followup process supervision with the user-authorized extension and original clock."""

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
ALLOCATION = ROOT / "configs/maniguard/followup/allocation.json"


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def clock(path=ALLOCATION, now=None):
    allocation = json.loads(Path(path).read_text())
    now = time.time() if now is None else now
    stamps = {
        key: datetime.fromisoformat(allocation[key]).timestamp()
        for key in ("started_utc", "physical_deadline_utc", "hard_deadline_utc")
    }
    original = datetime.fromisoformat(allocation["original_hard_deadline_utc"]).timestamp()
    if original - stamps["started_utc"] != 28800:
        raise ValueError("Original allocation clock changed")
    if stamps["hard_deadline_utc"] - stamps["started_utc"] != 43200:
        raise ValueError("Only the explicitly authorized twelve-hour allocation is valid")
    if stamps["hard_deadline_utc"] - stamps["physical_deadline_utc"] != 3600:
        raise ValueError("Reserve the final hour for preservation and reporting")
    if not allocation.get("extension", {}).get("authority"):
        raise ValueError("An explicit extension receipt is required")
    return {
        "at": now,
        "elapsed_seconds": now - stamps["started_utc"],
        "allocated_gpu_hours": (now - stamps["started_utc"]) / 3600 * allocation["gpus"],
        "physical_remaining": stamps["physical_deadline_utc"] - now,
        "hard_remaining": stamps["hard_deadline_utc"] - now,
        **stamps,
    }


def process_identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def register(directory, process, role):
    if os.getpgid(process.pid) != process.pid:
        raise ValueError("Only owned session leaders may be registered")
    atomic(
        Path(directory) / "groups" / f"{process.pid}.json",
        {
            "pid": process.pid,
            "start_ticks": process_identity(process.pid),
            "role": role,
            "registered_at": time.time(),
        },
    )


def live_group_members(pgid):
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().split(") ", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                members.append(int(path.parent.name))
        except (OSError, IndexError, ValueError):
            continue
    return members


def guard(directory, allocation=ALLOCATION, interval=1):
    """Run in a separate supervised process, independent of the coordinator."""
    directory = Path(directory)
    last_sample = 0
    while True:
        status = clock(allocation)
        for path in sorted((directory / "groups").glob("*.json")):
            group = json.loads(path.read_text())
            pid = group["pid"]
            identity = process_identity(pid)
            if identity is not None and identity != group["start_ticks"]:
                continue
            remaining = status["physical_remaining"]
            # Grace and forced cleanup are inside the declared deadline.
            if remaining <= 15:
                sig = signal.SIGKILL if remaining <= 5 else signal.SIGTERM
                try:
                    os.killpg(pid, sig)
                except ProcessLookupError:
                    pass
                with (directory / "deadline-events.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "at": time.time(),
                                "pid": pid,
                                "signal": sig.name,
                                "role": group["role"],
                            }
                        )
                        + "\n"
                    )
        atomic(directory / "allocation-ledger.json", status)
        if time.time() - last_sample >= 5:
            # Discover this run's policy during startup too, before the coordinator's
            # ready-handshake registration. Configuration identity proves ownership.
            requested_paths = list(directory.glob("service*/requested-policy.conf"))
            installed = Path("/etc/supervisor/conf.d/maniguard_policy.conf")
            if (
                installed.exists()
                and any(p.read_bytes() == installed.read_bytes() for p in requested_paths)
            ):
                try:
                    response = subprocess.run(
                        ["supervisorctl", "pid", "maniguard_policy"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=False,
                    )
                    pid = int(response.stdout.strip())
                    if pid > 0 and not (directory / "groups" / f"{pid}.json").exists():
                        register(directory, SimpleNamespace(pid=pid), "policy")
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            try:
                sample = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=3,
                    check=False,
                )
                with (directory / "gpu-samples.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "at": time.time(),
                                "returncode": sample.returncode,
                                "sample": sample.stdout.strip(),
                            }
                        )
                        + "\n"
                    )
            except (OSError, subprocess.TimeoutExpired):
                pass
            last_sample = time.time()
        if status["hard_remaining"] <= 0:
            return
        time.sleep(interval)


def run_owned(
    command, directory, role, timeout=1200, allocation=ALLOCATION, env=None, registry=None
):
    """Journal before spawn, retain failures, and bound cleanup including descendants."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    status = clock(allocation)
    remaining = status["physical_remaining"]
    timeout = min(timeout, remaining - 30)
    if timeout <= 0:
        raise RuntimeError("No time for process and cleanup")
    receipt = {
        "command": list(map(str, command)),
        "started_at": time.time(),
        "status": "started",
        "role": role,
        "timeout": timeout,
    }
    atomic(directory / "process.json", receipt)
    started = time.monotonic()
    with (directory / "process.log").open("x") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        register(registry or directory.parent, process, role)
        receipt["pid"] = process.pid
        atomic(directory / "process.json", receipt)
        try:
            receipt["returncode"] = process.wait(timeout=timeout)
            receipt["status"] = "completed" if process.returncode == 0 else "failed"
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            receipt["status"] = (
                "deadline-interrupted"
                if isinstance(error, subprocess.TimeoutExpired)
                else "interrupted"
            )
        finally:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                cleanup_end = time.monotonic() + 5
                while live_group_members(process.pid) and time.monotonic() < cleanup_end:
                    time.sleep(0.05)
                if not live_group_members(process.pid):
                    break
            receipt["remaining_live_group_members"] = live_group_members(process.pid)
            if receipt["remaining_live_group_members"]:
                receipt["status"] = "cleanup_failed"
            receipt["finished_at"] = time.time()
            receipt["elapsed_seconds"] = time.monotonic() - started
            atomic(directory / "process.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--allocation", type=Path, default=ALLOCATION)
    args = parser.parse_args()
    guard(args.directory, args.allocation)
