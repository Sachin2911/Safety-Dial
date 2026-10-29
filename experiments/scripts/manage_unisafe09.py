"""Manage only the authorized task-9 rental, with persistent financial reservations."""

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.unisafe09_infra import (
    IMAGE,
    FinancialBook,
    atomic_json,
    instances,
    provider_stopped,
    ssh_endpoint,
    ssh_options,
    utc,
    vast,
)

OUT = ROOT / "outputs/unisafe09"
BOOK = OUT / "control/finance.json"
RENTAL = ROOT / "progress/rental-unisafe09-20261025.json"


def offers():
    occupied = {r["machine_id"] for r in instances()}
    valid = []
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
        "cpu_name",
        "driver_version",
        "inet_down_cost",
        "inet_up_cost",
        "storage_cost",
        "inet_down",
        "inet_up",
        "geolocation",
        "reliability",
        "disk_space",
        "gpu_ram",
    )
    for gpu in ("RTX_3090", "RTX_4090"):
        query = (
            f"gpu_name={gpu} num_gpus=1 cpu_ram>=64 cpu_cores_effective>=12 "
            "disk_space>=200 reliability>0.98 direct_port_count>=2 rented=False"
        )
        result = vast(
            ["search", "offers", query, "--storage", "200", "--order", "dph", "--limit", "100"]
        )
        rows = result if isinstance(result, list) else result.get("offers")
        if not isinstance(rows, list):
            raise RuntimeError("Unexpected offer schema")
        for row in rows:
            if row.get("machine_id") in occupied:
                continue
            try:
                FinancialBook.validate_offer(row)
            except (ValueError, KeyError, TypeError):
                continue
            valid.append({key: row.get(key) for key in fields})
    return sorted(
        valid,
        key=lambda r: (
            r["gpu_name"] != "RTX 3090",
            r["dph_total"] + (150 * r["inet_down_cost"] + 40 * r["inet_up_cost"]) / 10,
            -r["cpu_cores_effective"],
        ),
    )


def verify_freeze(path):
    plan = json.loads(Path(path).read_text())
    if not plan.get("authorization", {}).get("user_launch"):
        raise ValueError("The approved launch must be recorded in the source freeze")
    for name, expected in plan["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if plan["money"]["total_usd"] != 10:
        raise ValueError("The approved total cap is $10")
    return plan


def start_local_watchdog():
    folder = OUT / "control"
    folder.mkdir(parents=True, exist_ok=True)
    pid_file = folder / "watchdog.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text())
        try:
            os.kill(pid, 0)
            # Do not mistake an unrelated reused PID for our guard.
            if b"unisafe09_watchdog.py" in Path(f"/proc/{pid}/cmdline").read_bytes():
                heartbeat = json.loads((folder / "watchdog-status.json").read_text())
                heartbeat_age = (
                    time.time() - dt.datetime.fromisoformat(heartbeat["utc"]).timestamp()
                )
                allocation = json.loads(RENTAL.read_text())
                observed = allocation["label"] in heartbeat.get("pending_labels", []) or (
                    allocation.get("instance_id") is not None
                    and allocation["instance_id"] in heartbeat.get("active_instances", [])
                )
                if heartbeat.get("pid") == pid and 0 <= heartbeat_age <= 30 and observed:
                    return pid
                raise RuntimeError("Existing financial watchdog has no recent heartbeat")
        except (OSError, ValueError):
            pass
    log = (folder / "watchdog.log").open("a", buffering=1)
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "experiments/scripts/unisafe09_watchdog.py"),
            "--book",
            str(BOOK),
            "--rental",
            str(RENTAL),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_file.write_text(str(process.pid) + "\n")
    deadline = time.monotonic() + 10
    status_file = folder / "watchdog-status.json"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Local financial watchdog exited before rental creation")
        if status_file.exists():
            heartbeat = json.loads(status_file.read_text())
            if heartbeat.get("pid") == process.pid:
                return process.pid
        time.sleep(0.1)
    raise RuntimeError("Local financial watchdog did not confirm a live heartbeat")


def create(args):
    verify_freeze(args.plan)
    candidates = offers()  # Fresh quote immediately before the one create request.
    if args.offer is not None:
        candidates = [r for r in candidates if r["id"] == args.offer]
    if not candidates:
        raise ValueError("No available offer meets the frozen host and price limits")
    chosen = candidates[0]
    book = FinancialBook(BOOK)
    before = book.snapshot()
    created = utc()
    label = f"block-unisafe09-{dt.datetime.now(dt.timezone.utc).strftime('%m%d%H%M%S')}"
    reservation = book.reserve(label, created, chosen)
    partial = {
        "schema": "unisafe09-rental-v1",
        "created_utc": created,
        "label": label,
        "image": IMAGE,
        "disk_gb": 200,
        "offer": chosen,
        "instance_id": None,
        "status": "reserved_before_create",
        "spending_cap_usd": 10,
        "prior_active_estimate_usd": before["active_estimate_usd"],
        "prior_transfer_estimate_usd": before["aggregate_transfer_estimate_usd"],
        "transfer_estimate_usd": reservation["transfer_estimate_usd"],
    }
    atomic_json(RENTAL, partial)
    try:
        watchdog_pid = start_local_watchdog()
    except RuntimeError:
        # The create command has not been issued, so there can be no allocation.
        book.cancel_rejected(label, absence_verified_utc=utc())
        partial["status"] = "guard_not_ready_create_not_requested"
        atomic_json(RENTAL, partial)
        raise
    rejected = False
    try:
        result = vast(
            [
                "create",
                "instance",
                str(chosen["id"]),
                "--image",
                IMAGE,
                "--disk",
                "200",
                "--label",
                label,
                "--cancel-unavail",
                "--ssh",
                "--direct",
                "--env",
                "-e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=yes "
                "-e OMNI_KIT_ALLOW_ROOT=1 -e PYTHONUNBUFFERED=1 -e UV_HTTP_TIMEOUT=300 "
                "-e UV_LINK_MODE=copy",
                "--onstart-cmd",
                "exec /opt/instance-tools/bin/entrypoint.sh",
            ],
            timeout=60,
        )
        if result.get("success") is True and isinstance(result.get("new_contract"), int):
            book.attach(label, result["new_contract"])
            partial["instance_id"] = result["new_contract"]
            partial["status"] = "create_acknowledged"
            atomic_json(RENTAL, partial)
        elif result.get("success") is False:
            rejected = True
            partial["status"] = "provider_rejected_create"
            atomic_json(RENTAL, partial)
    except RuntimeError:
        partial["status"] = "create_acknowledgement_uncertain"
        atomic_json(RENTAL, partial)
        # Reconcile by the unique label. Repeating create could rent two machines.
    current = None
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            rows = instances()
        except RuntimeError:
            time.sleep(5)
            continue
        matches = [r for r in rows if r.get("label") == label]
        if len(matches) > 1:
            book.reconcile_duplicate_label(label, [r["id"] for r in matches])
            for row in matches:
                try:
                    vast(["stop", "instance", str(row["id"])])
                except RuntimeError:
                    pass
            raise RuntimeError(
                "Provider created duplicate labels; stop/reconcile before continuing"
            )
        if matches:
            current = matches[0]
            book.attach(label, current["id"])
            break
        if rejected:
            book.cancel_rejected(label, absence_verified_utc=utc())
            partial["status"] = "create_rejected_absent"
            partial["absence_verified_utc"] = utc()
            atomic_json(RENTAL, partial)
            print(json.dumps({"status": "create_rejected_absent", "label": label}))
            return
        time.sleep(5)
    if current is None:
        print(
            json.dumps(
                {
                    "status": "create_unresolved",
                    "label": label,
                    "watchdog_pid": watchdog_pid,
                    "finance": book.snapshot(),
                }
            )
        )
        return
    snapshot = book.snapshot()
    nominal = dt.datetime.fromisoformat(snapshot["deadline_utc"])
    # Begin independent stop requests five minutes early; the original money
    # margin remains available for delayed provider acknowledgements.
    effective = nominal - dt.timedelta(minutes=5)
    partial.update(
        {
            "instance_id": current["id"],
            "provider": current,
            "status": "created",
            "allocation_deadline_utc": effective.isoformat(),
            "nominal_financial_deadline_utc": nominal.isoformat(),
            "watchdog_pid": watchdog_pid,
            "finance_snapshot": snapshot,
        }
    )
    atomic_json(RENTAL, partial)
    print(
        json.dumps(
            {
                "instance_id": current["id"],
                "label": label,
                "offer": chosen,
                "allocation_deadline_utc": effective.isoformat(),
                "watchdog_pid": watchdog_pid,
            }
        )
    )


def resume(args):
    """Resume the existing stopped rental once, retaining its original deadline."""
    verify_freeze(args.plan)
    rental = json.loads(RENTAL.read_text())
    provider = next((r for r in instances() if r["id"] == rental["instance_id"]), None)
    if provider is None or not provider_stopped(provider):
        raise ValueError("Resume requires the existing provider-verified stopped instance")
    if rental.get("approval_pending"):
        raise ValueError("Record the user's model-file transfer approval before resuming")
    book = FinancialBook(BOOK)
    before = book.snapshot()
    started = utc()
    if dt.datetime.fromisoformat(rental["allocation_deadline_utc"]) <= (
        dt.datetime.fromisoformat(started) + dt.timedelta(minutes=60)
    ):
        raise ValueError("Insufficient time remains in the original rental window")
    label = rental["label"] + ":resume:" + started
    book.reserve_resume(rental["instance_id"], label, started, rental["allocation_deadline_utc"])
    rental.setdefault("original_created_utc", rental["created_utc"])
    rental["created_utc"] = started
    rental["prior_active_estimate_usd"] = before["active_estimate_usd"]
    # The transfer allowance is for this physical rental, including its earlier segment.
    rental["prior_transfer_estimate_usd"] = 0
    rental["status"] = "resume_reserved"
    rental["resume_segment_label"] = label
    atomic_json(RENTAL, rental)
    try:
        rental["watchdog_pid"] = start_local_watchdog()
    except RuntimeError:
        # No start request was issued. The earlier provider state remains stopped.
        book.stopped(rental["instance_id"], utc(), status="exited")
        rental["status"] = "resume_guard_failed_start_not_requested"
        atomic_json(RENTAL, rental)
        raise
    atomic_json(RENTAL, rental)
    try:
        result = vast(["start", "instance", str(rental["instance_id"])])
        rental["resume_acknowledged"] = result.get("success") is True
    except RuntimeError:
        rental["resume_acknowledged"] = False
    # Never repeat start after a missing acknowledgement. Reconcile this same instance.
    rental["status"] = "resume_requested"
    atomic_json(RENTAL, rental)
    print(json.dumps({"instance_id": rental["instance_id"], "status": rental["status"],
                      "watchdog_pid": rental["watchdog_pid"],
                      "allocation_deadline_utc": rental["allocation_deadline_utc"]}))


def status():
    rows = instances()
    rental = json.loads(RENTAL.read_text()) if RENTAL.exists() else None
    owned = [r for r in rows if rental and r["id"] == rental.get("instance_id")]
    if rental and owned:
        rental["provider"] = owned[0]
        rental["provider_checked_utc"] = utc()
        atomic_json(RENTAL, rental)
    print(json.dumps({"owned": owned, "finance": FinancialBook(BOOK).snapshot()}))


def ssh(args):
    rental = json.loads(RENTAL.read_text())
    rows = instances()
    provider = next((r for r in rows if r["id"] == rental["instance_id"]), None)
    if provider is None:
        raise ValueError("Rental is absent from provider state")
    host, port = ssh_endpoint(provider)
    command = ["ssh", *ssh_options(port), f"root@{host}", args.command]
    result = subprocess.run(command, timeout=args.timeout, capture_output=True, text=True)
    # SSH output comes only from our supplied commands; never run env/printenv.
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr[-2000:], file=sys.stderr, end="")
        raise SystemExit(result.returncode)


def sync():
    rental = json.loads(RENTAL.read_text())
    provider = next(r for r in instances() if r["id"] == rental["instance_id"])
    host, port = ssh_endpoint(provider)
    destination = OUT / "remote"
    destination.mkdir(parents=True, exist_ok=True)
    transport = shlex.join(["ssh", *ssh_options(port)])
    command = [
        "rsync",
        "-az",
        "--partial",
        "--timeout=90",
        "-e",
        transport,
        f"root@{host}:/workspace/Safety-Dial/outputs/unisafe09/",
        str(destination) + "/",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    print(json.dumps({"sync_exit": result.returncode, "utc": utc()}))
    if result.returncode:
        raise SystemExit(result.returncode)


def upload_weights():
    """Transfer only hash-listed checkpoint files. Authentication stays local."""
    rental = json.loads(RENTAL.read_text())
    provider = next(r for r in instances() if r["id"] == rental["instance_id"])
    host, port = ssh_endpoint(provider)
    source = ROOT / "data/unisafe09/pretrained"
    manifest_path = source / "manifest.json"
    pin = json.loads((ROOT / "configs/unisafe09/checkpoint.json").read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != pin["manifest_sha256"]:
        raise ValueError("Checkpoint manifest is not the pinned one")
    manifest = json.loads(manifest_path.read_text())
    names = ["manifest.json"]
    for item in manifest["files"] + pin.get("derived_files", []):
        name = item["path"]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe checkpoint path")
        path = source / name
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
        names.append(name)
    listing = OUT / "source/checkpoint-files.txt"
    listing.write_text("\n".join(names) + "\n")
    subprocess.run(
        [
            "ssh",
            *ssh_options(port),
            f"root@{host}",
            "mkdir -p /workspace/Safety-Dial/data/unisafe09/pretrained",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            "--partial",
            "--timeout=90",
            "--files-from",
            str(listing),
            "-e",
            shlex.join(["ssh", *ssh_options(port)]),
            str(source) + "/",
            f"root@{host}:/workspace/Safety-Dial/data/unisafe09/pretrained/",
        ],
        check=True,
        capture_output=True,
        timeout=1800,
    )
    verify_command = (
        "cd /workspace/Safety-Dial && python3 experiments/scripts/download_unisafe09.py "
        "--verify-only && python3 -c "
        + shlex.quote(
            "from pathlib import Path; Path('data/unisafe09/pretrained/transfer-ready.json')"
            ".write_text('{\"verified\": true}\\n')"
        )
    )
    result = subprocess.run(
        ["ssh", *ssh_options(port), f"root@{host}", verify_command],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    print(result.stdout, end="")


def deploy(args):
    verify_freeze(args.plan)
    rental = json.loads(RENTAL.read_text())
    provider = next(r for r in instances() if r["id"] == rental["instance_id"])
    host, port = ssh_endpoint(provider)
    bundle = args.bundle.resolve(strict=True)
    transport = shlex.join(["ssh", *ssh_options(port)])
    subprocess.run(
        [
            "rsync",
            "-az",
            "--timeout=90",
            "-e",
            transport,
            str(bundle),
            f"root@{host}:/tmp/unisafe09-source.tar.gz",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    check = subprocess.run(
        ["ssh", *ssh_options(port), f"root@{host}", "sha256sum /tmp/unisafe09-source.tar.gz"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    if check.stdout.split()[0] != hashlib.sha256(bundle.read_bytes()).hexdigest():
        raise ValueError("Transferred source bundle hash differs")
    result = subprocess.run(
        [
            "ssh",
            *ssh_options(port),
            f"root@{host}",
            "tar -xOzf /tmp/unisafe09-source.tar.gz experiments/scripts/bootstrap_unisafe09.sh | bash",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    print(result.stdout, end="")


def destroy(args):
    rental = json.loads(RENTAL.read_text())
    marker = json.loads(args.marker.read_text())
    if not (
        marker.get("instance_id") == rental["instance_id"]
        and marker.get("local_verified") is True
        and marker.get("private_hf_verified") is True
        and isinstance(marker.get("pinned_revisions"), dict)
        and marker["pinned_revisions"]
        and len(marker.get("manifest_sha256", "")) == 64
    ):
        raise ValueError("Complete local and private pinned preservation is required")
    manifest = Path(marker["manifest_path"])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != marker["manifest_sha256"]:
        raise ValueError("Preservation manifest changed after private verification")
    # Verification includes the complete local inventory, not just a success flag.
    from preserve_unisafe09 import verify_local

    verify_local(manifest.parent)
    rows = instances()
    row = next((r for r in rows if r["id"] == rental["instance_id"]), None)
    if not provider_stopped(row):
        raise ValueError("Stop and verify compute before destroying its disk")
    if row is not None:
        try:
            vast(["destroy", "instance", str(rental["instance_id"]), "--yes"])
        except RuntimeError:
            pass  # Never repeat this mutation on an uncertain acknowledgement.
        for _ in range(12):
            rows = instances()
            if not any(r["id"] == rental["instance_id"] for r in rows):
                break
            time.sleep(5)
        else:
            raise RuntimeError("Destruction is not yet verified by provider absence")
    rental["status"] = "destroyed"
    rental["destroy_verified_utc"] = utc()
    rental["preservation_marker_sha256"] = hashlib.sha256(args.marker.read_bytes()).hexdigest()
    atomic_json(RENTAL, rental)
    print(json.dumps({"destroyed": rental["instance_id"], "absence_verified": True}))


def stop():
    rental = json.loads(RENTAL.read_text())
    try:
        vast(["stop", "instance", str(rental["instance_id"])])
    except RuntimeError:
        pass
    for _ in range(12):
        row = next((r for r in instances() if r["id"] == rental["instance_id"]), None)
        if provider_stopped(row):
            FinancialBook(BOOK).stopped(
                rental["instance_id"], utc(), status="destroyed" if row is None else "exited"
            )
            rental["status"] = "stopped" if row else "destroyed"
            rental["stop_verified_utc"] = utc()
            rental["provider_stop_check"] = row
            atomic_json(RENTAL, rental)
            print(json.dumps({"stop_verified": True, "provider": row}))
            return
        time.sleep(5)
    raise RuntimeError("Provider has not verified stop; watchdog remains responsible")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("offers")
    sub.add_parser("status")
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--offer", type=int)
    create_parser.add_argument(
        "--plan", type=Path, default=ROOT / "progress/receipts/09-unisafe-preparation.json"
    )
    resume_parser = sub.add_parser("resume")
    resume_parser.add_argument("--plan", type=Path, required=True)
    remote = sub.add_parser("ssh")
    remote.add_argument("command")
    remote.add_argument("--timeout", type=float, default=60)
    sub.add_parser("sync")
    sub.add_parser("upload-weights")
    sub.add_parser("stop")
    deployment = sub.add_parser("deploy")
    deployment.add_argument("--bundle", type=Path, required=True)
    deployment.add_argument(
        "--plan", type=Path, default=ROOT / "progress/receipts/09-unisafe-preparation.json"
    )
    destruction = sub.add_parser("destroy")
    destruction.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "offers":
        print(json.dumps({"checked_utc": utc(), "offers": offers()}))
    elif args.action == "create":
        create(args)
    elif args.action == "resume":
        resume(args)
    elif args.action == "status":
        status()
    elif args.action == "ssh":
        ssh(args)
    elif args.action == "sync":
        sync()
    elif args.action == "upload-weights":
        upload_weights()
    elif args.action == "deploy":
        deploy(args)
    elif args.action == "destroy":
        destroy(args)
    else:
        stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
