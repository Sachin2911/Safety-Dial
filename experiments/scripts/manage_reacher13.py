"""Manage only the authorized task-13 rental, with persistent financial reservations."""

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.reacher13_infra import (
    DISK_GB,
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

OUT = ROOT / "outputs/reacher13"
BOOK = OUT / "control/finance.json"
RENTAL = ROOT / "progress/rental-reacher13-20261025.json"
REMOTE_ROOT = "/workspace/Safety-Dial"
DEFAULT_PLAN = ROOT / "progress/receipts/13-reacher-preparation-v1.json"


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
        "cuda_max_good",
    )
    for gpu, gpus in (("RTX_3090", 1), ("RTX_4090", 1), ("RTX_3090", 2), ("RTX_4090", 2)):
        query = (
            f"gpu_name={gpu} num_gpus={gpus} cpu_ram>={30 * (1.6 if gpus == 2 else 1):g} "
            f"cpu_cores_effective>={8 * (1.5 if gpus == 2 else 1):g} "
            f"disk_space>={DISK_GB} reliability>0.98 direct_port_count>=2 rented=False "
            "inet_down>=200"
        )
        result = vast(
            [
                "search",
                "offers",
                query,
                "--storage",
                str(DISK_GB),
                "--order",
                "dph",
                "--limit",
                "100",
            ]
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
            r["dph_total"] + (25 * r["inet_down_cost"] + 10 * r["inet_up_cost"]) / 10,
            -r["cpu_cores_effective"],
        ),
    )


def redact(text):
    """Strip anything that could be a credential before any CLI text is written to disk."""
    import re

    text = re.sub(r"api_key=[^&\s\"']+", "api_key=REDACTED", text)
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer REDACTED", text)
    return re.sub(r"\b[0-9a-f]{32,}\b", "HEX_REDACTED", text)


def vast_create(args, timeout=90):
    """One create request whose failure category is recorded redacted, never retried here."""
    from helpers.dsrl06_infra import VAST

    try:
        result = subprocess.run(
            [VAST, *args, "--raw"], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        (OUT / "control/create-error.txt").write_text(f"{utc()} timeout after {timeout}s\n")
        raise RuntimeError("Vast create timed out") from error
    try:
        parsed = json.loads(result.stdout)
    except ValueError:
        parsed = None
    if result.returncode or not isinstance(parsed, dict):
        (OUT / "control/create-error.txt").write_text(
            f"{utc()} exit={result.returncode}\nstdout:\n{redact(result.stdout)[:1500]}\n"
            f"stderr:\n{redact(result.stderr)[:1500]}\n"
        )
        raise RuntimeError("Vast create failed; redacted output recorded")
    return parsed


def reconcile(args):
    """Resolve an uncertain create by label: attach when present, release when absent."""
    rental = json.loads(RENTAL.read_text())
    book = FinancialBook(BOOK)
    label = rental["label"]
    matches = [r for r in instances() if r.get("label") == label]
    if len(matches) > 1:
        book.reconcile_duplicate_label(label, [r["id"] for r in matches])
        raise RuntimeError("Duplicate labels; stop them before continuing")
    if matches:
        current = matches[0]
        book.attach(label, current["id"])
        snapshot = book.snapshot()
        nominal = dt.datetime.fromisoformat(snapshot["deadline_utc"])
        effective = nominal - dt.timedelta(minutes=5)
        rental.update(
            {
                "instance_id": current["id"],
                "provider": current,
                "status": "created",
                "allocation_deadline_utc": effective.isoformat(),
                "nominal_financial_deadline_utc": nominal.isoformat(),
                "finance_snapshot": snapshot,
            }
        )
        atomic_json(RENTAL, rental)
        print(
            json.dumps(
                {"instance_id": current["id"], "allocation_deadline_utc": effective.isoformat()}
            )
        )
        return
    if not args.release:
        print(json.dumps({"status": "absent", "label": label, "released": False}))
        return
    if rental.get("instance_id") is not None:
        raise ValueError("A rental with an instance id cannot be released by absence")
    # Two authoritative listings at least a minute apart, both without the label.
    time.sleep(60)
    if any(r.get("label") == label for r in instances()):
        raise RuntimeError("The instance appeared late; run reconcile again without --release")
    book.cancel_rejected(label, absence_verified_utc=utc())
    rental["status"] = "create_rejected_absent"
    rental["absence_verified_utc"] = utc()
    atomic_json(RENTAL, rental)
    print(json.dumps({"status": "create_rejected_absent", "label": label}))


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
            if b"reacher13_watchdog.py" in Path(f"/proc/{pid}/cmdline").read_bytes():
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
                # A second rental's label reaches the heartbeat on the watchdog's next cycle.
                for _ in range(60):
                    time.sleep(0.5)
                    heartbeat = json.loads((folder / "watchdog-status.json").read_text())
                    heartbeat_age = (
                        time.time() - dt.datetime.fromisoformat(heartbeat["utc"]).timestamp()
                    )
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
            str(ROOT / "experiments/scripts/reacher13_watchdog.py"),
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


class HelperBook(FinancialBook):
    """Allows exactly one helper rental beside the primary.

    Added 25 October 2026 at the user's request ("rent another GPU ... to get it done
    sooner"). The shared ledger, its limits and the local watchdog still bound money and
    aggregate hours for both rentals; the combined hourly rate must stay under the ceiling.
    """

    def reserve(self, label, created_utc, offer):
        from helpers.dsrl06_infra import timestamp

        self.validate_offer(offer)
        with self.transaction():
            running = [r for r in self.data["allocations"] if r["active_stop_verified_utc"] is None]
            if len(running) >= 2:
                raise ValueError("At most one helper rental may run beside the primary")
            if any(r.get("label") == label for r in self.data["allocations"]):
                raise ValueError("Rental label already reserved")
            combined = sum(r["rate_usd_hour"] for r in running) + float(offer["dph_total"])
            if combined > self.limits()["rate_usd_hour"]:
                raise ValueError("Combined hourly rate would exceed the ceiling")
            if self.snapshot(timestamp(created_utc), refresh=False)["stop"]:
                raise ValueError("No financial allowance remains")
            row = self._row(None, created_utc, offer)
            row["label"] = label
            row["role"] = "helper"
            self.data["allocations"].append(row)
        return row


def create(args):
    verify_freeze(args.plan)
    candidates = [
        r
        for r in offers()
        if not any(code in (r.get("geolocation") or "") for code in args.exclude_geo)
        and float(r.get("reliability") or 0) >= args.min_reliability
    ]
    if not candidates:
        raise ValueError("No available offer meets the frozen host and price limits")
    preferred = [r for r in candidates if r["id"] == args.offer]
    # Offer ids churn between listings; fall back to the best current candidate.
    chosen = preferred[0] if preferred else candidates[0]
    print(json.dumps({"chosen_offer": chosen}), flush=True)
    book = HelperBook(BOOK) if args.helper else FinancialBook(BOOK)
    before = book.snapshot()
    created = utc()
    prefix = "reacher13h" if args.helper else "reacher13"
    label = f"{prefix}-{dt.datetime.now(dt.timezone.utc).strftime('%m%d%H%M%S')}"
    reservation = book.reserve(label, created, chosen)
    partial = {
        "schema": "reacher13-rental-v1",
        "created_utc": created,
        "label": label,
        "image": IMAGE,
        "disk_gb": DISK_GB,
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
        book.cancel_rejected(label, absence_verified_utc=utc())
        partial["status"] = "guard_not_ready_create_not_requested"
        atomic_json(RENTAL, partial)
        raise
    rejected = False
    try:
        result = vast_create(
            [
                "create",
                "instance",
                str(chosen["id"]),
                "--image",
                IMAGE,
                "--disk",
                str(DISK_GB),
                "--label",
                label,
                "--cancel-unavail",
                "--ssh",
                "--direct",
                "--env",
                "-e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 "
                "-e UV_HTTP_TIMEOUT=300 -e UV_LINK_MODE=copy",
                "--onstart-cmd",
                "exec /opt/instance-tools/bin/entrypoint.sh",
            ],
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
    verify_freeze(args.plan)
    rental = json.loads(RENTAL.read_text())
    provider = next((r for r in instances() if r["id"] == rental["instance_id"]), None)
    if provider is None or not provider_stopped(provider):
        raise ValueError("Resume requires the existing provider-verified stopped instance")
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
    rental["prior_transfer_estimate_usd"] = 0
    rental["status"] = "resume_reserved"
    rental["resume_segment_label"] = label
    atomic_json(RENTAL, rental)
    try:
        rental["watchdog_pid"] = start_local_watchdog()
    except RuntimeError:
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
    rental["status"] = "resume_requested"
    atomic_json(RENTAL, rental)
    print(json.dumps({"instance_id": rental["instance_id"], "status": rental["status"]}))


def status():
    rows = instances()
    rental = json.loads(RENTAL.read_text()) if RENTAL.exists() else None
    owned = [r for r in rows if rental and r["id"] == rental.get("instance_id")]
    if rental and owned:
        rental["provider"] = owned[0]
        rental["provider_checked_utc"] = utc()
        atomic_json(RENTAL, rental)
    print(json.dumps({"owned": owned, "finance": FinancialBook(BOOK).snapshot()}))


def endpoint():
    rental = json.loads(RENTAL.read_text())
    provider = next((r for r in instances() if r["id"] == rental["instance_id"]), None)
    if provider is None:
        raise ValueError("Rental is absent from provider state")
    return ssh_endpoint(provider)


def ssh(args):
    host, port = endpoint()
    command = ["ssh", *ssh_options(port), f"root@{host}", args.command]
    result = subprocess.run(command, timeout=args.timeout, capture_output=True, text=True)
    # SSH output comes only from our supplied commands; never run env/printenv.
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr[-2000:], file=sys.stderr, end="")
        raise SystemExit(result.returncode)


def sync(args):
    host, port = endpoint()
    destination = OUT / "remote"
    destination.mkdir(parents=True, exist_ok=True)
    transport = shlex.join(["ssh", *ssh_options(port)])
    command = [
        "rsync",
        "-az",
        "--partial",
        "--timeout=120",
        "--exclude",
        "queue/",
        "--exclude",
        "runner.lock",
        "-e",
        transport,
        f"root@{host}:{REMOTE_ROOT}/outputs/reacher13/",
        str(destination) + "/",
    ]
    if args.light:
        command[3:3] = ["--exclude", "attempts/*/history.json", "--exclude", "logs/"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    print(json.dumps({"sync_exit": result.returncode, "utc": utc()}))
    if result.returncode:
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(result.returncode)


def bundle(plan_path):
    """Build the source bundle from the freeze's hash list plus the records it needs."""
    plan = verify_freeze(plan_path)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "source" / f"bundle-{plan_path.stem}.tar.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        for name in plan["source_sha256"]:
            archive.add(ROOT / name, arcname=name)
        for path in (plan_path, ROOT / "progress/13-reacher-dial-plan.md", RENTAL):
            if path.exists():
                archive.add(path, arcname=str(path.resolve().relative_to(ROOT)))
    return target


def deploy(args):
    plan_path = args.plan.resolve()
    archive = bundle(plan_path)
    host, port = endpoint()
    transport = shlex.join(["ssh", *ssh_options(port)])
    subprocess.run(
        [
            "rsync",
            "-az",
            "--timeout=90",
            "-e",
            transport,
            str(archive),
            f"root@{host}:/tmp/reacher13-source.tar.gz",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    check = subprocess.run(
        ["ssh", *ssh_options(port), f"root@{host}", "sha256sum /tmp/reacher13-source.tar.gz"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    if check.stdout.split()[0] != hashlib.sha256(archive.read_bytes()).hexdigest():
        raise ValueError("Transferred source bundle hash differs")
    if args.extract_only:
        command = (
            f"mkdir -p {REMOTE_ROOT} && tar -xzf /tmp/reacher13-source.tar.gz -C {REMOTE_ROOT}"
        )
    else:
        command = (
            "tar -xOzf /tmp/reacher13-source.tar.gz "
            "experiments/scripts/bootstrap_reacher13.sh | bash"
        )
    result = subprocess.run(
        ["ssh", *ssh_options(port), f"root@{host}", command],
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit(result.returncode)


def prepare(args):
    """Restart the remote setup and preparation session after a source repair."""
    host, port = endpoint()
    session = f"prepare13-{dt.datetime.now(dt.timezone.utc).strftime('%H%M%S')}"
    command = (
        f"cd {REMOTE_ROOT} && rm -f outputs/reacher13/prepare-ended.txt && "
        f"tmux new-session -d -s {session} "
        f"'cd {REMOTE_ROOT} && bash experiments/scripts/setup_reacher13.sh "
        f">> outputs/reacher13/logs/preparation.log 2>&1; "
        f"date -u +%FT%TZ > outputs/reacher13/prepare-ended.txt' && tmux list-sessions"
    )
    result = subprocess.run(
        ["ssh", *ssh_options(port), f"root@{host}", command],
        capture_output=True,
        text=True,
        timeout=60,
    )
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(result.returncode)


def run(args):
    """Start the frozen stage on the rental inside a detached tmux session."""
    plan = verify_freeze(args.plan)
    host, port = endpoint()
    stage = plan["stage"]
    session = f"run13-{stage}-{dt.datetime.now(dt.timezone.utc).strftime('%H%M%S')}"
    remote_plan = str(args.plan.resolve().relative_to(ROOT))
    command = (
        f"cd {REMOTE_ROOT} && mkdir -p outputs/reacher13/logs && "
        f"tmux new-session -d -s {session} "
        f"'cd {REMOTE_ROOT} && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy "
        f"/venv/main/bin/python -u experiments/scripts/run_reacher13.py --freeze {remote_plan} "
        f"--workers {args.workers} --gpus {args.gpus} "
        f">> outputs/reacher13/logs/run-{stage}.log 2>&1; "
        f"date -u +%FT%TZ > outputs/reacher13/run-{stage}-ended.txt' && tmux list-sessions"
    )
    result = subprocess.run(
        ["ssh", *ssh_options(port), f"root@{host}", command],
        capture_output=True,
        text=True,
        timeout=60,
    )
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(result.returncode)


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
    sys.path.insert(0, str(ROOT / "experiments/scripts"))
    from preserve_reacher13 import verify_local

    verify_local(manifest.parent)
    rows = instances()
    row = next((r for r in rows if r["id"] == rental["instance_id"]), None)
    if not provider_stopped(row):
        raise ValueError("Stop and verify compute before destroying its disk")
    if row is not None:
        try:
            vast(["destroy", "instance", str(rental["instance_id"]), "--yes"])
        except RuntimeError:
            pass
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
    global RENTAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rental",
        type=Path,
        default=RENTAL,
        help="rental record to act on (a helper rental has its own record)",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("offers")
    sub.add_parser("status")
    reconcile_parser = sub.add_parser("reconcile")
    reconcile_parser.add_argument("--release", action="store_true")
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--offer", type=int)
    create_parser.add_argument("--exclude-geo", nargs="*", default=["CN"])
    create_parser.add_argument("--min-reliability", type=float, default=0.99)
    create_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    create_parser.add_argument(
        "--helper", action="store_true", help="one extra rental beside the primary"
    )
    resume_parser = sub.add_parser("resume")
    resume_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    remote = sub.add_parser("ssh")
    remote.add_argument("command")
    remote.add_argument("--timeout", type=float, default=60)
    sync_parser = sub.add_parser("sync")
    sync_parser.add_argument("--light", action="store_true")
    sync_parser.add_argument("--timeout", type=float, default=600)
    sub.add_parser("stop")
    sub.add_parser("prepare")
    runner = sub.add_parser("run")
    runner.add_argument("--plan", type=Path, required=True)
    runner.add_argument("--workers", type=int, default=3)
    runner.add_argument("--gpus", type=int, default=1)
    deployment = sub.add_parser("deploy")
    deployment.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    deployment.add_argument("--extract-only", action="store_true")
    destruction = sub.add_parser("destroy")
    destruction.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    RENTAL = args.rental.resolve()
    if args.action == "offers":
        print(json.dumps({"checked_utc": utc(), "offers": offers()}))
    elif args.action == "create":
        create(args)
    elif args.action == "resume":
        resume(args)
    elif args.action == "status":
        status()
    elif args.action == "reconcile":
        reconcile(args)
    elif args.action == "ssh":
        ssh(args)
    elif args.action == "sync":
        sync(args)
    elif args.action == "deploy":
        deploy(args)
    elif args.action == "prepare":
        prepare(args)
    elif args.action == "run":
        run(args)
    elif args.action == "destroy":
        destroy(args)
    else:
        stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps({"error_type": type(error).__name__, "detail": str(error)[:300]}),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
