"""Fresh PushT/Cube rental, scoped to a new label and a $5 all-in allowance."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, instances, ssh_endpoint, ssh_options, utc, vast

OUT = ROOT / os.environ.get("LEWM18_CONTROL_DIR", "outputs/lewm18")
RENTAL = ROOT / os.environ.get("LEWM18_RENTAL_RECORD", "progress/rental-lewm18-20261026.json")
IMAGE = "vastai/base-image@sha256:c1d2b5326fae806b04d2c2d97a2b3948d0ccb0026dd3085e9a2ad55e304193f8"
DISK = 240


def options(port):
    return [v.replace("dsrl06-known-hosts", "lewm18-known-hosts") for v in ssh_options(port)]


def endpoint(row):
    # This host's advertised direct endpoint timed out; the provider proxy was verified.
    if row.get("ssh_host") and row.get("ssh_port"):
        return row["ssh_host"], int(row["ssh_port"])
    return ssh_endpoint(row)


def admission(row):
    rate = float(row["dph_total"])
    transfer = 85 * float(row["inet_down_cost"]) + 12 * float(row["inet_up_cost"])
    if not 0 < rate <= 0.65 or transfer > 0.75 or 5 * rate + transfer + 0.75 > 5:
        raise ValueError("Five-hour rental plus transfers and closeout exceeds allowance")
    if row["gpu_name"] not in {"RTX 3090", "RTX 4090"} or row["num_gpus"] != 1:
        raise ValueError("Unexpected hardware")
    if float(row["cpu_ram"]) < 45000 or float(row["cpu_cores_effective"]) < 12:
        raise ValueError("Insufficient host resources")
    if float(row["disk_space"]) < DISK:
        raise ValueError("Insufficient storage")
    if int(str(row["driver_version"]).split(".")[0]) < 570:
        raise ValueError("CUDA driver too old")
    return transfer


def offers():
    occupied = {r["machine_id"] for r in instances()}
    fields = (
        "id",
        "machine_id",
        "gpu_name",
        "num_gpus",
        "dph_total",
        "dph_base",
        "dph_storage",
        "cpu_ram",
        "cpu_cores_effective",
        "driver_version",
        "inet_down_cost",
        "inet_up_cost",
        "inet_down",
        "inet_up",
        "disk_space",
        "reliability",
        "geolocation",
        "storage_cost",
    )
    valid = []
    for gpu in ("RTX_4090", "RTX_3090"):
        result = vast(
            [
                "search",
                "offers",
                f"gpu_name={gpu} num_gpus=1 cpu_ram>=45 "
                f"cpu_cores_effective>=12 disk_space>={DISK} reliability>0.98 "
                "direct_port_count>=2 rented=False rentable=True verified=True inet_down>=400",
                "--storage",
                str(DISK),
                "--order",
                "dph",
                "--limit",
                "80",
            ]
        )
        rows = result if isinstance(result, list) else result["offers"]
        for row in rows:
            if row["machine_id"] in occupied:
                continue
            try:
                transfer = admission(row)
            except (ValueError, KeyError, TypeError):
                continue
            valid.append({**{k: row.get(k) for k in fields}, "transfer_reserve": transfer})
    return sorted(
        valid,
        key=lambda r: (r["gpu_name"] != "RTX 4090", 5 * r["dph_total"] + r["transfer_reserve"]),
    )


def current():
    rental = json.loads(RENTAL.read_text())
    rows = [r for r in instances() if r["label"] == rental["label"]]
    if len(rows) > 1:
        raise RuntimeError("Duplicate task-18 label")
    row = rows[0] if rows else None
    if row and rental.get("instance_id") not in (None, row["id"]):
        raise RuntimeError("Rental identity mismatch")
    return rental, row


def finance(rental):
    start = dt.datetime.fromisoformat(rental["created_utc"]).timestamp()
    end = (
        dt.datetime.fromisoformat(rental["destroyed_utc"]).timestamp()
        if rental.get("destroyed_utc")
        else time.time()
    )
    active = max(0, end - start) / 3600 * rental["offer"]["dph_total"]
    return {
        "active_upper_usd": active,
        "reserved_total_usd": active
        + rental["transfer_reserve_usd"]
        + rental.get("prior_spend_upper_usd", 0),
        "deadline_utc": rental["deadline_utc"],
    }


def remote(command, timeout=90):
    _, row = current()
    if row is None:
        raise RuntimeError("Rental absent")
    host, port = endpoint(row)
    return subprocess.run(
        ["ssh", *options(port), f"root@{host}", command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def create_http(row, rental):
    """Minimal documented request; keep the provider credential in an HTTP header."""
    key_path = Path.home() / ".config/vastai/vast_api_key"
    if not key_path.exists():
        key_path = Path.home() / ".vast_api_key"
    key = key_path.read_text().strip()
    payload = {
        "client_id": "me",
        "image": IMAGE,
        "disk": DISK,
        "label": rental["label"],
        "runtype": "ssh_direc ssh_proxy",
        "onstart": "exec /opt/instance-tools/bin/entrypoint.sh",
        "env": {
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "PYTHONUNBUFFERED": "1",
            "UV_HTTP_TIMEOUT": "300",
            "UV_LINK_MODE": "copy",
        },
        "cancel_unavail": True,
    }
    request = urllib.request.Request(
        f"https://console.vast.ai/api/v0/asks/{row['id']}/",
        method="PUT",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        result = json.loads(exc.read())
    safe = {
        k: v for k, v in result.items() if k in ("success", "new_contract", "status_code", "msg")
    }
    if isinstance(safe.get("msg"), str):
        safe["msg"] = safe["msg"].replace(key, "[redacted]")[:500]
    atomic_json(OUT / "create-response.json", safe)
    print(json.dumps(safe), flush=True)
    return result


def local_guard():
    while True:
        try:
            rental, row = current()
            atomic_json(
                OUT / "local-guard.json",
                {"utc": utc(), "pid": os.getpid(), "provider": row, "finance": finance(rental)},
            )
            if row is None and rental.get("instance_id"):
                return
            end = dt.datetime.fromisoformat(rental["deadline_utc"]).timestamp()
            if row and (time.time() >= end or finance(rental)["reserved_total_usd"] >= 4):
                vast(["stop", "instance", str(row["id"])])
                # Preserve locally before destruction where SSH is still available.
                # Stopped storage is bounded by a ten-minute terminal grace period.
                grace = end + 600
                if time.time() >= grace or finance(rental)["reserved_total_usd"] >= 4.65:
                    vast(["destroy", "instance", str(row["id"]), "--yes"])
                    return
        except Exception as exc:
            atomic_json(OUT / "local-guard-error.json", {"utc": utc(), "kind": type(exc).__name__})
        time.sleep(20)


def remote_guard():
    rental = json.loads(RENTAL.read_text())
    instance = rental["instance_id"]
    assert os.environ.get("CONTAINER_ID") == str(instance)
    assert os.environ.get("CONTAINER_API_KEY")
    OUT.mkdir(parents=True, exist_ok=True)
    end = dt.datetime.fromisoformat(rental["deadline_utc"]).timestamp()
    atomic_json(OUT / "remote-guard.json", {"utc": utc(), "pid": os.getpid(), "instance": instance})
    while True:
        rx = tx = 0
        for nic in Path("/sys/class/net").iterdir():
            if nic.name != "lo":
                rx += int((nic / "statistics/rx_bytes").read_text())
                tx += int((nic / "statistics/tx_bytes").read_text())
        offer = rental["offer"]
        transfer = rx / 1e9 * offer["inet_down_cost"] + tx / 1e9 * offer["inet_up_cost"]
        total = (
            finance(rental)["active_upper_usd"]
            + max(transfer, rental["transfer_reserve_usd"])
            + rental.get("prior_spend_upper_usd", 0)
        )
        atomic_json(
            OUT / "network-cost.json",
            {"utc": utc(), "rx": rx, "tx": tx, "transfer_usd": transfer, "total_upper_usd": total},
        )
        if time.time() >= end - 600 or total >= 4:
            (OUT / "STOP").touch()
        if time.time() >= end or total >= 4.35:
            for _ in range(12):
                try:
                    req = urllib.request.Request(
                        f"https://console.vast.ai/api/v0/instances/{instance}/",
                        data=b'{"state":"stopped"}',
                        method="PUT",
                        headers={
                            "Authorization": "Bearer " + os.environ["CONTAINER_API_KEY"],
                            "Content-Type": "application/json",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=20) as response:
                        if json.load(response).get("success"):
                            return
                except Exception:
                    pass
                time.sleep(5)
            raise RuntimeError("Provider did not acknowledge stop")
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "offers",
            "create",
            "status",
            "ssh",
            "sync",
            "stop",
            "destroy",
            "guard-local",
            "guard-remote",
        ],
    )
    parser.add_argument("--command")
    parser.add_argument("--offer", type=int)
    parser.add_argument("--machine", type=int)
    parser.add_argument("--deadline")
    parser.add_argument("--prior-spend", type=float, default=0)
    parser.add_argument("--http", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == "offers":
        rows = offers()
        atomic_json(OUT / "offers.json", rows)
        print(json.dumps(rows[:8]))
    elif args.action == "create":
        if RENTAL.exists():
            raise RuntimeError("Rental reservation already exists; reconcile instead of creating")
        auth = json.loads((ROOT / "configs/lewm18/authorization.json").read_text())
        assert auth["total_usd"] == 5 and auth["rent_and_run"]
        row = next(
            r
            for r in offers()
            if (r["machine_id"] == args.machine if args.machine else r["id"] == args.offer)
        )
        now = dt.datetime.now(dt.timezone.utc)
        deadline = (
            dt.datetime.fromisoformat(args.deadline)
            if args.deadline
            else now + dt.timedelta(hours=5)
        )
        hours = (deadline - now).total_seconds() / 3600
        if not 0 < hours <= 5 or args.prior_spend < 0:
            raise ValueError("Invalid deadline or prior allocation")
        if args.prior_spend + hours * row["dph_total"] + admission(row) + 0.25 > 5:
            raise ValueError("Combined replacement cost exceeds $5")
        rental = {
            "created_utc": now.isoformat(),
            "deadline_utc": deadline.isoformat(),
            "prior_spend_upper_usd": args.prior_spend,
            "label": "lewm18-" + now.strftime("%Y%m%d%H%M%S"),
            "offer": row,
            "transfer_reserve_usd": admission(row),
            "instance_id": None,
            "total_usd": 5,
            "status": "reserved",
            "disk_gb": DISK,
            "image": IMAGE,
        }
        atomic_json(RENTAL, rental)
        log = (OUT / "local-guard.log").open("a")
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "guard-local"],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        result = (
            create_http(row, rental)
            if args.http
            else vast(
                [
                    "create",
                    "instance",
                    str(row["id"]),
                    "--image",
                    IMAGE,
                    "--disk",
                    str(DISK),
                    "--label",
                    rental["label"],
                    "--cancel-unavail",
                    "--ssh",
                    "--direct",
                    "--env",
                    "-e NVIDIA_DRIVER_CAPABILITIES=all "
                    "-e PYTHONUNBUFFERED=1 -e UV_HTTP_TIMEOUT=300 -e UV_LINK_MODE=copy",
                    "--onstart-cmd",
                    "exec /opt/instance-tools/bin/entrypoint.sh",
                ],
                timeout=90,
            )
        )
        rental["instance_id"] = result.get("new_contract")
        rental["status"] = "create_acknowledged" if result.get("success") else "reconcile_required"
        atomic_json(RENTAL, rental)
        print(json.dumps({"instance_id": rental["instance_id"], "status": rental["status"]}))
    elif args.action == "guard-local":
        local_guard()
    elif args.action == "guard-remote":
        remote_guard()
    elif args.action == "status":
        rental, row = current()
        print(json.dumps({"provider": row, "finance": finance(rental)}))
    elif args.action == "ssh":
        p = remote(args.command)
        print(p.stdout)
        if p.returncode:
            print("SSH returned", p.returncode, file=sys.stderr)
            return p.returncode
    elif args.action == "sync":
        _, row = current()
        host, port = endpoint(row)
        (OUT / "remote").mkdir(exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-az",
                "--partial",
                "-e",
                shlex.join(["ssh", *options(port)]),
                f"root@{host}:/workspace/Safety-Dial/outputs/lewm18/",
                str(OUT / "remote") + "/",
            ],
            check=True,
        )
    else:
        rental, row = current()
        if row:
            command = [args.action, "instance", str(row["id"])]
            if args.action == "destroy":
                command.append("--yes")
            try:
                vast(command)
            except RuntimeError:
                # This CLI can return empty stdout after a successful deletion.
                # Reconcile absence instead of issuing another destructive call.
                if args.action != "destroy":
                    raise
                _, reconciled = current()
                if reconciled is not None:
                    raise
        _, after = current()
        if args.action == "destroy" and after is None:
            rental.update(destroyed_utc=utc(), status="destroyed_absence_verified")
            atomic_json(RENTAL, rental)
        print(json.dumps({"provider_after": after, "finance": finance(rental)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
