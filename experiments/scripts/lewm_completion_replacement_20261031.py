"""Guard one prospectively declared replacement GPU for the approved completion."""

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

import lewm_campaign_pods as prior
import lewm_completion_pods_20261031 as completion

OUT = completion.OUT / "replacement"
NAME = "safetydial-lewm-final-relocation-20261031"


def stamp():
    return dt.datetime.now(dt.timezone.utc)


def find():
    matches = [p for p in prior.pods() if p.get("name") == NAME]
    if len(matches) > 1:
        raise RuntimeError("Ambiguous replacement identity")
    return matches[0] if matches else None


def record_session(pod):
    record = {
        k: pod[k] for k in ("id", "name", "costPerHr", "desiredStatus", "gpuCount") if k in pod
    }
    path = OUT / "session.json"
    if path.exists() and prior.read(path)["id"] != record["id"]:
        raise RuntimeError("Replacement identity changed")
    prior.write(path, record)
    return record


def amount(at):
    total = completion.estimate(at)
    request = OUT / "create-request.json"
    if request.exists():
        begin = dt.datetime.fromisoformat(prior.read(request)["requested_utc"])
        stop = OUT / "stop-verified.json"
        end = dt.datetime.fromisoformat(prior.read(stop)["checked_utc"]) if stop.exists() else at
        total += max(0, (end - begin).total_seconds()) / 3600 * (0.60 + 80 * 0.10 / 720)
    return total


def arm():
    if OUT.exists():
        raise RuntimeError("Replacement intent exists; inspect instead of replaying")
    authority = prior.read(completion.OUT / "authorization.json")
    OUT.mkdir()
    prior.write(
        OUT / "intent.json",
        {
            "name": NAME,
            "scope": "Availability-driven replacement for all unstarted final profiles; private preservation before every deletion",
            "authority": "User approved full comparison, $3 additional, then destroy pods",
            "deadline_utc": authority["deadline_utc"],
            "total_cap_usd": 13,
            "original_gpu_start_failures": "All three exact original hosts rejected GPU resume; zero-GPU recovery used for verified preservation",
            "maximum_live_campaign_pods": 3,
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
    prior.write(
        OUT / "guard-process.json",
        {
            "pid": worker.pid,
            "start_ticks": Path(f"/proc/{worker.pid}/stat").read_text().split()[21],
        },
    )
    print(
        json.dumps({"replacement_guard_armed": True, "deadline_utc": authority["deadline_utc"]}),
        flush=True,
    )


def deploy():
    if (OUT / "create-request.json").exists():
        raise RuntimeError("Existing creation request; inspect provider without replaying")
    authority = prior.read(completion.OUT / "authorization.json")
    deadline = dt.datetime.fromisoformat(authority["deadline_utc"])
    at = stamp()
    if at + dt.timedelta(minutes=35) >= deadline:
        raise RuntimeError("Insufficient setup, evaluation and preservation reserve")
    heartbeat = prior.read(OUT / "guard-state.json")
    if (at - dt.datetime.fromisoformat(heartbeat["heartbeat_utc"])).total_seconds() > 15:
        raise RuntimeError("Replacement provider guard not live")
    existing = prior.pods()
    ids = {s["id"] for s in authority["sessions"].values()}
    if sum(p.get("id") in ids and p.get("desiredStatus") == "RUNNING" for p in existing) >= 3:
        raise RuntimeError("Preserve and delete an old pod before requesting a fourth live pod")
    if find() is not None:
        raise RuntimeError("Replacement already exists; inspect")
    reserved = amount(at) + (deadline - at).total_seconds() / 3600 * 0.612
    if reserved > 12.70:
        raise RuntimeError("Insufficient revised spending headroom")
    prior.write(
        OUT / "create-request.json",
        {
            "requested_utc": at.isoformat(),
            "name": NAME,
            "maximum_rate_per_hour": 0.60,
            "reserved_total_bound_usd": reserved,
        },
    )
    pod = prior.cli(
        "pod",
        "create",
        "--name",
        NAME,
        "--cloud-type",
        "SECURE",
        "--gpu-id",
        "NVIDIA A40",
        "--gpu-count",
        "1",
        "--image",
        "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "--container-disk-in-gb",
        "30",
        "--volume-in-gb",
        "50",
        "--ports",
        "22/tcp",
        timeout=60,
    )
    record = record_session(pod)
    if not 0 < float(pod["costPerHr"]) <= 0.60:
        prior.cli("pod", "stop", pod["id"])
        raise RuntimeError("Unexpected price, exact replacement stopped")
    print(json.dumps(record), flush=True)


def stop():
    pod = find()
    if pod is None:
        return
    record_session(pod)
    prior.cli("pod", "stop", pod["id"], timeout=20)
    value = prior.cli("pod", "get", pod["id"], timeout=20)
    record = {
        "id": pod["id"],
        "checked_utc": stamp().isoformat(),
        "provider_desired_status": value.get("desiredStatus"),
    }
    if record["provider_desired_status"] == "EXITED":
        prior.write(OUT / "stop-verified.json", record)
    print(json.dumps(record), flush=True)


def guard():
    deadline = dt.datetime.fromisoformat(
        prior.read(completion.OUT / "authorization.json")["deadline_utc"]
    )
    while not (OUT / "guard-cancelled").exists():
        at = stamp()
        total = amount(at)
        prior.write(
            OUT / "guard-state.json",
            {
                "heartbeat_utc": at.isoformat(),
                "deadline_utc": deadline.isoformat(),
                "conservative_total_usd": total,
            },
        )
        if at >= deadline or total >= 12.70:
            try:
                stop()
                if total >= 12.70:
                    for lane in completion.LANES:
                        folder = completion.OUT / lane
                        if (folder / "start-request.json").exists() and not (
                            folder / "deleted-verified.json"
                        ).exists():
                            completion.stop(lane)
            except Exception as exc:
                prior.write(
                    OUT / "guard-error.json", {"utc": at.isoformat(), "type": type(exc).__name__}
                )
        time.sleep(3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("arm", "deploy", "guard", "stop", "cost"))
    a = p.parse_args()
    if a.operation == "cost":
        print(json.dumps({"conservative_total_usd": amount(stamp())}))
    else:
        globals()[a.operation]()


if __name__ == "__main__":
    main()
