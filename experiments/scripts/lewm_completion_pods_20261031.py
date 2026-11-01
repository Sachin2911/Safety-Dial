"""Bounded provider control for the explicitly authorized comparison completion."""

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

import lewm_campaign_pods as prior

OUT = prior.ROOT / "outputs/runpod-lewm-completion-20261031"
LANES = ("navigation", "pusht", "recovery")
STOPPED_STORAGE_HOURLY = 120 * 0.20 / 720
RUNNING_DISK_HOURLY = 40 * 0.10 / 720


def now():
    return dt.datetime.now(dt.timezone.utc)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    prior.write(path, value)


def estimate(at):
    authority = read(OUT / "authorization.json")
    amount = authority["prior_conservative_bound_usd"]
    for lane in LANES:
        folder = OUT / lane
        start = folder / "start-request.json"
        if not start.exists():
            continue
        request = read(start)
        stop = folder / "stop-verified.json"
        end = dt.datetime.fromisoformat(read(stop)["checked_utc"]) if stop.exists() else at
        seconds = max(
            0, (end - dt.datetime.fromisoformat(request["requested_utc"])).total_seconds()
        )
        amount += seconds / 3600 * (request["rate_per_hour"] + RUNNING_DISK_HOURLY)
    elapsed = max(0, (at - dt.datetime.fromisoformat(authority["authorized_utc"])).total_seconds())
    return amount + elapsed / 3600 * STOPPED_STORAGE_HOURLY


def arm():
    if OUT.exists():
        raise RuntimeError("Completion allocation already exists; inspect, never duplicate")
    stamp = now()
    old = read(prior.OUT / "stopped-unpreserved-closeout-v1.json")
    elapsed = max(0, (stamp - dt.datetime.fromisoformat(old["checked_utc"])).total_seconds())
    base = old["conservative_running_rental_estimate_to_verification_usd"]
    base += elapsed / 3600 * STOPPED_STORAGE_HOURLY
    sessions = {}
    for lane in LANES:
        session = read(prior.lane_dir(lane) / "session.json")
        pod = prior.cli("pod", "get", session["id"], timeout=20)
        if pod.get("desiredStatus") != "EXITED":
            raise RuntimeError("Expected the exact previously stopped pod")
        rate = float(pod["costPerHr"])
        if not 0 < rate <= 0.60:
            raise RuntimeError("Unexpected rate; do not start")
        sessions[lane] = {"id": session["id"], "rate_per_hour": rate}
    ceiling = base + 1.5 * sum(v["rate_per_hour"] + RUNNING_DISK_HOURLY for v in sessions.values())
    ceiling += 1.5 * STOPPED_STORAGE_HOURLY
    if ceiling > 12.75 or ceiling - base > 2.75:
        raise RuntimeError("Insufficient reserve within revised $13 / additional $3 caps")
    OUT.mkdir(parents=True)
    authority = {
        "schema": "lewm-comparison-completion-authority-v1",
        "authorized_utc": stamp.isoformat(),
        "deadline_utc": (stamp + dt.timedelta(minutes=90)).isoformat(),
        "user_authority": "Go ahead with full comparison, $3 more; after that destroy the pods.",
        "scope": "Complete frozen final comparison and analysis, privately preserve and independently verify, then delete only verified pods.",
        "total_cap_usd": 13,
        "maximum_additional_usd": 3,
        "prior_conservative_bound_usd": base,
        "conservative_reserved_bound_usd": ceiling,
        "sessions": sessions,
        "storage_rate_source": "https://github.com/runpod/docs/blob/main/accounts-billing/billing.mdx",
        "storage_estimate": "Conservatively charge all 120GB at stopped volume rate in addition to running disk allowances; not a provider invoice.",
        "original_protocols_controllers_cases_seeds_and_scientific_sources_unchanged": True,
        "delete_only_after_independent_every_member_archive_verification": True,
    }
    write(OUT / "authorization.json", authority)
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
    print(json.dumps(authority), flush=True)


def start(lane):
    authority = read(OUT / "authorization.json")
    stamp = now()
    if stamp + dt.timedelta(minutes=20) >= dt.datetime.fromisoformat(authority["deadline_utc"]):
        raise RuntimeError("No useful completion window remains")
    heartbeat = read(OUT / "guard-state.json")
    if (stamp - dt.datetime.fromisoformat(heartbeat["heartbeat_utc"])).total_seconds() > 15:
        raise RuntimeError("Provider stop guard is not fresh")
    folder = OUT / lane
    if (folder / "start-request.json").exists():
        raise RuntimeError("Existing or unresolved start; inspect without replay")
    session = authority["sessions"][lane]
    folder.mkdir(exist_ok=True)
    write(
        folder / "start-request.json",
        {**session, "requested_utc": stamp.isoformat(), "deadline_utc": authority["deadline_utc"]},
    )
    result = prior.cli("pod", "start", session["id"], timeout=40)
    write(
        folder / "start-response.json",
        {k: result[k] for k in ("id", "desiredStatus", "costPerHr") if k in result},
    )
    print(json.dumps({"lane": lane, "id": session["id"], "start_requested": True}), flush=True)


def stop(lane):
    authority = read(OUT / "authorization.json")
    pod_id = authority["sessions"][lane]["id"]
    prior.cli("pod", "stop", pod_id, timeout=20)
    pod = prior.cli("pod", "get", pod_id, timeout=20)
    value = {
        "id": pod_id,
        "checked_utc": now().isoformat(),
        "provider_desired_status": pod.get("desiredStatus"),
    }
    if pod.get("desiredStatus") == "EXITED":
        write(OUT / lane / "stop-verified.json", value)
    print(json.dumps(value), flush=True)


def guard():
    authority = read(OUT / "authorization.json")
    deadline = dt.datetime.fromisoformat(authority["deadline_utc"])
    while not (OUT / "guard-cancelled").exists():
        stamp = now()
        amount = estimate(stamp)
        write(
            OUT / "guard-state.json",
            {
                "heartbeat_utc": stamp.isoformat(),
                "deadline_utc": deadline.isoformat(),
                "conservative_total_usd": amount,
            },
        )
        exhausted = amount >= 12.75 or amount - authority["prior_conservative_bound_usd"] >= 2.75
        if stamp >= deadline or exhausted:
            for lane in LANES:
                folder = OUT / lane
                if (
                    not (folder / "start-request.json").exists()
                    or (folder / "stop-verified.json").exists()
                    or (folder / "deleted-verified.json").exists()
                ):
                    continue
                try:
                    stop(lane)
                except Exception as exc:
                    write(
                        folder / "guard-error.json",
                        {"type": type(exc).__name__, "utc": now().isoformat()},
                    )
        time.sleep(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("arm", "start", "stop", "guard", "cost"))
    p.add_argument("lane", nargs="?", choices=LANES)
    args = p.parse_args()
    if args.operation == "arm":
        arm()
    elif args.operation == "start":
        start(args.lane)
    elif args.operation == "stop":
        stop(args.lane)
    elif args.operation == "guard":
        guard()
    else:
        print(json.dumps({"conservative_total_usd": estimate(now()), "utc": now().isoformat()}))


if __name__ == "__main__":
    main()
