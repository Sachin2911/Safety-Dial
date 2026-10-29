"""Manage the task 19 evolved-adapter campaign rentals (several one-GPU Jar hosts).

Task 19 has its own namespace, a $10 total cap and a hard deadline of 03:20 UTC on
27 October 2026. Every host is labelled jar-evo19-*. The local guard (`guard`) stops
every task 19 host when the conservative spend estimate reaches $9.40 or at the
deadline; each host also runs evo19_remote_guard.py, which stops its own instance at the
same deadline. Hosts are only ever destroyed after their outputs are mirrored locally.
"""

import argparse
import datetime as dt
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import TEMPLATE, atomic_json, instances, ssh_endpoint, utc, vast  # noqa: E402
from helpers.dsrl06_infra import ssh_options as base_ssh_options  # noqa: E402

OUT = ROOT / "outputs/evo19"
RECORD = OUT / "control/rentals.json"
REMOTE_OUT = "/workspace/Safety-Dial/outputs/evo19/"
DEADLINE = dt.datetime(2026, 10, 27, 3, 20, tzinfo=dt.timezone.utc)
TOTAL_USD = 10.0
STOP_USD = 9.4
TRANSFER_GB = 60  # conservative per-host download allowance


def ssh_options(port):
    return [x.replace("/tmp/dsrl06-known-hosts", "/tmp/evo19-known-hosts")
            for x in base_ssh_options(port)]


def load():
    if RECORD.exists():
        return json.loads(RECORD.read_text())
    return {"schema": "evo19-rentals-v1", "total_usd": TOTAL_USD,
            "deadline_utc": DEADLINE.isoformat(), "hosts": []}


def save(record):
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(RECORD, record)


def spend(record, now=None):
    """Conservative spend: elapsed hours at the offer rate plus a fixed transfer allowance."""
    now = now or time.time()
    total = 0.0
    for host in record["hosts"]:
        if host.get("status") == "never_created":
            continue
        start = dt.datetime.fromisoformat(host["created_utc"]).timestamp()
        end = host.get("destroyed_utc")
        end = dt.datetime.fromisoformat(end).timestamp() if end else now
        rate = float(host["offer"]["dph_total"])
        transfer = TRANSFER_GB * float(host["offer"].get("inet_down_cost") or 0)
        total += max(0.0, end - start) / 3600 * rate + transfer
    return total


def owned(record, rows=None):
    rows = instances() if rows is None else rows
    ids = {h["instance_id"] for h in record["hosts"] if h.get("instance_id")}
    return [r for r in rows if r["id"] in ids or str(r.get("label", "")).startswith("jar-evo19-")]


def host_row(name):
    record = load()
    host = next(h for h in record["hosts"] if h["name"] == name)
    row = next((r for r in instances() if r["id"] == host["instance_id"]), None)
    if row is None:
        raise ValueError(f"{name} is absent from provider state")
    return host, row


def create(args):
    record = load()
    if dt.datetime.now(dt.timezone.utc) >= DEADLINE - dt.timedelta(minutes=90):
        raise SystemExit("Too close to the deadline to add a host")
    rows = []
    for gpu in ("RTX_3090", "RTX_4090"):
        found = vast(["search", "offers",
                      f"gpu_name={gpu} num_gpus=1 cpu_ram>=48 cpu_cores_effective>=8 "
                      "disk_space>=160 reliability>0.95 direct_port_count>=2 rented=False "
                      "verified=any", "--storage", "160", "--order", "dph", "--limit", "200"])
        rows += found if isinstance(found, list) else found.get("offers", [])
    rows = [r for r in rows if r["id"] == args.offer]
    if len(rows) != 1:
        if args.rate is None:
            raise SystemExit("Offer not in this search; pass --rate from a fresh listing")
        # Search results vary between calls; the provider still rejects unavailable offers.
        rows = [{"id": args.offer, "gpu_name": args.gpu, "num_gpus": 1, "dph_total": args.rate,
                 "driver_version": "580.unverified", "inet_down_cost": 0.01}]
    offer = rows[0]
    if not str(offer.get("driver_version", "")).startswith("580."):
        raise SystemExit("Isaac Sim 4.5 needs driver 580.x")
    if offer["gpu_name"] not in ("RTX 3090", "RTX 4090") or int(offer["num_gpus"]) != 1:
        raise SystemExit("One RTX 3090 or RTX 4090 only")
    hours = (DEADLINE.timestamp() - time.time()) / 3600
    future = sum(hours * float(h["offer"]["dph_total"]) for h in record["hosts"]
                 if h.get("status") not in ("destroyed", "never_created"))
    projected = spend(record) + future + hours * float(offer["dph_total"]) + TRANSFER_GB * float(
        offer.get("inet_down_cost") or 0)
    if projected > TOTAL_USD and not args.allow_partial:
        raise SystemExit(f"Projected spend to the deadline {projected:.2f} exceeds the cap")
    name = f"h{len(record['hosts']) + 1}"
    label = f"jar-evo19-{name}-{dt.datetime.now(dt.timezone.utc).strftime('%m%d%H%M%S')}"
    keys = ("id", "machine_id", "gpu_name", "num_gpus", "dph_total", "cpu_ram",
            "cpu_cores_effective", "cpu_name", "driver_version", "inet_down_cost",
            "inet_up_cost", "inet_down", "geolocation", "reliability")
    host = {"name": name, "label": label, "created_utc": utc(),
            "offer": {k: offer.get(k) for k in keys}, "instance_id": None,
            "status": "create_requested"}
    record["hosts"].append(host)
    save(record)
    try:
        result = vast(["create", "instance", str(args.offer), "--template_hash", TEMPLATE,
                       "--disk", "160", "--label", label, "--cancel-unavail"], timeout=60)
        if result.get("success") is True and isinstance(result.get("new_contract"), int):
            host["instance_id"] = result["new_contract"]
            host["status"] = "created"
    except RuntimeError:
        host["status"] = "create_uncertain"
    save(record)
    for _ in range(36):
        matches = [r for r in instances() if r.get("label") == label]
        if matches:
            host["instance_id"] = matches[0]["id"]
            host["status"] = "created"
            break
        time.sleep(5)
    save(record)
    print(json.dumps({"name": name, "instance_id": host["instance_id"], "label": label,
                      "status": host["status"], "projected_usd": round(projected, 2)}))


def status():
    record = load()
    rows = instances()
    mine = {r["id"]: r for r in owned(record, rows)}
    for host in record["hosts"]:
        row = mine.get(host.get("instance_id"))
        print(json.dumps({"name": host["name"], "id": host.get("instance_id"),
                          "gpu": host["offer"]["gpu_name"], "rate": host["offer"]["dph_total"],
                          "status": host.get("status"),
                          "provider": row and row.get("actual_status")}))
    print(json.dumps({"spend_estimate_usd": round(spend(record), 3), "utc": utc()}))


def ssh(args):
    _, row = host_row(args.host)
    address, port = ssh_endpoint(row)
    result = subprocess.run(["ssh", *ssh_options(port), f"root@{address}", args.command],
                            timeout=args.timeout, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr[-2000:], file=sys.stderr, end="")
        raise SystemExit(result.returncode)


def sync_host(name, row, timeout=600):
    address, port = ssh_endpoint(row)
    destination = OUT / "remote" / name
    destination.mkdir(parents=True, exist_ok=True)
    transport = shlex.join(["ssh", *ssh_options(port)])
    result = subprocess.run(
        ["rsync", "-az", "--partial", "--timeout=90", "-e", transport,
         "--exclude", "episodes/*/recordings/*.mp4",
         f"root@{address}:{REMOTE_OUT}", str(destination) + "/"],
        capture_output=True, text=True, timeout=timeout)
    return result.returncode


def sync(args):
    record = load()
    rows = {r["id"]: r for r in instances()}
    for host in record["hosts"]:
        if args.host and host["name"] != args.host:
            continue
        row = rows.get(host.get("instance_id"))
        if row is None or row.get("actual_status") != "running":
            continue
        try:
            rc = sync_host(host["name"], row)
        except (subprocess.TimeoutExpired, ValueError) as error:
            rc = type(error).__name__
        print(json.dumps({"host": host["name"], "sync": rc, "utc": utc()}))


def push(args):
    """Copy a file or directory to the same relative path on one or all running hosts."""
    record = load()
    rows = {r["id"]: r for r in instances()}
    for host in record["hosts"]:
        if args.host and host["name"] != args.host:
            continue
        row = rows.get(host.get("instance_id"))
        if row is None or row.get("actual_status") != "running":
            continue
        address, port = ssh_endpoint(row)
        transport = shlex.join(["ssh", *ssh_options(port)])
        source = Path(args.path)
        remote = f"/workspace/Safety-Dial/{source.as_posix().rstrip('/')}"
        if source.is_dir():
            source, remote = Path(str(source) + "/"), remote + "/"
        result = subprocess.run(["rsync", "-az", "--timeout=60", "-e", transport,
                                 str(source), f"root@{address}:{remote}"],
                                capture_output=True, text=True, timeout=300)
        print(json.dumps({"host": host["name"], "push": args.path, "rc": result.returncode}))


def deploy(args):
    _, row = host_row(args.host)
    address, port = ssh_endpoint(row)
    bundle = args.bundle.resolve(strict=True)
    transport = shlex.join(["ssh", *ssh_options(port)])
    subprocess.run(["rsync", "-az", "--timeout=90", "-e", transport, str(bundle),
                    f"root@{address}:/tmp/evo19-source.tar.gz"],
                   capture_output=True, text=True, timeout=300, check=True)
    check = subprocess.run(["ssh", *ssh_options(port), f"root@{address}",
                            "sha256sum /tmp/evo19-source.tar.gz"],
                           capture_output=True, text=True, timeout=60, check=True)
    if check.stdout.split()[0] != hashlib.sha256(bundle.read_bytes()).hexdigest():
        raise ValueError("Transferred source bundle hash differs")
    remote = ("tar -xOzf /tmp/evo19-source.tar.gz experiments/scripts/bootstrap_evo19.sh | "
              f"EVO19_HOST={args.host} bash")
    result = subprocess.run(["ssh", *ssh_options(port), f"root@{address}", remote],
                            capture_output=True, text=True, timeout=180)
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit(result.returncode)


def stop_all(reason):
    record = load()
    for row in owned(record):
        if row.get("actual_status") in ("exited", "stopped"):
            continue
        try:
            vast(["stop", "instance", str(row["id"])])
        except RuntimeError:
            pass
    record.setdefault("stops", []).append({"utc": utc(), "reason": reason})
    save(record)


def destroy(args):
    record = load()
    host = next(h for h in record["hosts"] if h["name"] == args.host)
    marker = OUT / "remote" / args.host / "preserved.json"
    if not args.force_absent_data and not marker.exists():
        raise SystemExit("Mirror the host's outputs (write preserved.json) before destroying")
    try:
        vast(["destroy", "instance", str(host["instance_id"]), "--yes"])
    except RuntimeError:
        pass
    for _ in range(24):
        if not any(r["id"] == host["instance_id"] for r in instances()):
            break
        time.sleep(5)
    else:
        raise SystemExit("Destruction not yet verified by provider absence")
    host["status"] = "destroyed"
    host["destroyed_utc"] = utc()
    save(record)
    print(json.dumps({"destroyed": host["name"], "id": host["instance_id"],
                      "spend_estimate_usd": round(spend(record), 3)}))


def guard(args):
    status_path = OUT / "control/guard-status.json"
    while True:
        record = load()
        now = time.time()
        total = spend(record, now)
        try:
            live = [r for r in owned(record) if r.get("actual_status") not in ("exited",
                                                                              "stopped")]
        except RuntimeError:
            live = None
        atomic_json(status_path, {"utc": utc(), "spend_estimate_usd": round(total, 3),
                                  "live": None if live is None else [r["id"] for r in live]})
        if total >= STOP_USD or now >= DEADLINE.timestamp():
            stop_all("budget" if total >= STOP_USD else "deadline")
            if live is not None and not live:
                return
        time.sleep(args.poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    making = sub.add_parser("create")
    making.add_argument("--offer", type=int, required=True)
    making.add_argument("--allow-partial", action="store_true")
    making.add_argument("--rate", type=float)
    making.add_argument("--gpu", default="RTX 3090")
    sub.add_parser("status")
    remote = sub.add_parser("ssh")
    remote.add_argument("host")
    remote.add_argument("command")
    remote.add_argument("--timeout", type=float, default=120)
    syncing = sub.add_parser("sync")
    syncing.add_argument("--host")
    pushing = sub.add_parser("push")
    pushing.add_argument("path")
    pushing.add_argument("--host")
    deployment = sub.add_parser("deploy")
    deployment.add_argument("host")
    deployment.add_argument("--bundle", type=Path, required=True)
    destruction = sub.add_parser("destroy")
    destruction.add_argument("host")
    destruction.add_argument("--force-absent-data", action="store_true")
    guarding = sub.add_parser("guard")
    guarding.add_argument("--poll", type=float, default=60)
    sub.add_parser("stop-all")
    args = parser.parse_args()
    {"create": create, "ssh": ssh, "sync": sync, "push": push, "deploy": deploy,
     "destroy": destroy, "guard": guard}.get(args.action, lambda a: None)(args)
    if args.action == "status":
        status()
    elif args.action == "stop-all":
        stop_all("manual")


if __name__ == "__main__":
    main()
