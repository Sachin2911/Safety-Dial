"""Provider access and conservative, persistent financial accounting for task 6."""

from __future__ import annotations

import datetime as dt
import contextlib
import fcntl
import json
import os
import subprocess
import tempfile
from pathlib import Path

VAST = "/home/sachin/.local/bin/vastai"
TEMPLATE = "62414e8fca6befba085051de7ed36190"
SAFE_PROVIDER_FIELDS = (
    "id",
    "machine_id",
    "actual_status",
    "cur_state",
    "intended_status",
    "public_ipaddr",
    "ports",
    "ssh_host",
    "ssh_port",
    "gpu_name",
    "num_gpus",
    "driver_version",
    "dph_total",
    "start_date",
    "label",
    "disk_space",
)


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def timestamp(value: str) -> float:
    result = dt.datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Financial timestamps must include a timezone")
    return result.timestamp()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def vast(args: list[str], *, timeout: float = 45, executable: str = VAST):
    """Never expose raw CLI errors: the CLI sometimes embeds an API key in URLs."""
    try:
        result = subprocess.run(
            [executable, *args, "--raw"], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Vast command timed out; inspect provider state before retrying"
        ) from error
    if result.returncode:
        raise RuntimeError(
            f"Vast command failed with exit {result.returncode}; raw output withheld"
        )
    try:
        return json.loads(result.stdout)
    except (ValueError, TypeError) as error:
        # Some mutation commands return plain text. Do not assume failure and retry them.
        raise RuntimeError("Vast returned non-JSON output; reconcile provider state") from error


def instances() -> list[dict]:
    result = vast(["show", "instances"])
    if isinstance(result, dict) and (result.get("error") or result.get("success") is False):
        raise RuntimeError("Provider reported an error; absence is unproven")
    if isinstance(result, list):
        rows = result
    elif isinstance(result, dict) and isinstance(result.get("instances"), list):
        rows = result["instances"]
    else:
        raise RuntimeError("Unexpected provider instance schema; absence is unproven")
    if not all(isinstance(row, dict) and isinstance(row.get("id"), int) for row in rows):
        raise RuntimeError("Invalid provider instance rows; absence is unproven")
    return [{key: row.get(key) for key in SAFE_PROVIDER_FIELDS} for row in rows]


def provider_stopped(row: dict | None) -> bool:
    """Call only with a row from a successfully validated complete instance list."""
    return row is None or (
        row.get("actual_status") in {"exited", "stopped"}
        and row.get("intended_status") == "stopped"
        and row.get("cur_state") == "stopped"
    )


def ssh_endpoint(provider: dict) -> tuple[str, int]:
    direct = (provider.get("ports") or {}).get("22/tcp") or []
    if provider.get("public_ipaddr") and direct:
        return provider["public_ipaddr"], int(direct[0]["HostPort"])
    if provider.get("ssh_host") and provider.get("ssh_port"):
        return provider["ssh_host"], int(provider["ssh_port"])
    raise ValueError("The provider has not exposed an SSH endpoint yet")


def ssh_options(port: int) -> list[str]:
    return [
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "UserKnownHostsFile=/tmp/dsrl06-known-hosts",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]


class FinancialBook:
    """All allocations share one ledger; elapsed setup is conservatively charged."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                self.data = json.loads(self.path.read_text())
                if self.data.get("limits") != self.limits():
                    raise ValueError("Financial limits changed after initialization")
            else:
                self.data = {
                    "schema": "dsrl06-finance-v1",
                    "created_utc": utc(),
                    "limits": self.limits(),
                    "allocations": [],
                    "other_actual_usd": 0.0,
                }
                self.save()

    @staticmethod
    def limits() -> dict:
        return {
            "total_usd": 15.0,
            "active_usd": 13.0,
            "active_shutdown_usd": 12.9,
            "other_reserve_usd": 2.0,
            "aggregate_hours": 6.0,
            "rate_usd_hour": 2.2,
        }

    def save(self) -> None:
        atomic_json(self.path, self.data)

    @contextlib.contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.data = json.loads(self.path.read_text())
            if self.data.get("limits") != self.limits():
                raise ValueError("Financial limits changed after initialization")
            yield
            self.save()

    def reserve(self, label: str, created_utc: str, offer: dict) -> dict:
        """Reserve money/time before a create request, including uncertain acknowledgements."""
        self.validate_offer(offer)
        with self.transaction():
            if any(r["active_stop_verified_utc"] is None for r in self.data["allocations"]):
                raise ValueError("Reconcile/stop the existing allocation before another rental")
            if any(r.get("label") == label for r in self.data["allocations"]):
                raise ValueError("Rental label already reserved")
            if self.snapshot(timestamp(created_utc), refresh=False)["stop"]:
                raise ValueError("No financial allowance remains")
            row = self._row(None, created_utc, offer)
            row["label"] = label
            self.data["allocations"].append(row)
        return row

    def attach(self, label: str, instance_id: int) -> None:
        with self.transaction():
            row = next(r for r in self.data["allocations"] if r.get("label") == label)
            if row["instance_id"] not in {None, int(instance_id)}:
                raise ValueError("Reservation is already attached to a different instance")
            row["instance_id"] = int(instance_id)
            row["provider_status"] = "creating"

    def cancel_rejected(self, label: str, *, absence_verified_utc: str) -> None:
        """Only a rejected create plus an authoritative absence check can release it."""
        with self.transaction():
            row = next(r for r in self.data["allocations"] if r.get("label") == label)
            if row["instance_id"] is not None:
                raise ValueError("An attached rental must be stopped and verified")
            row["provider_status"] = "create_rejected_absent"
            row["active_stop_verified_utc"] = row["created_utc"]
            row["absence_verified_utc"] = absence_verified_utc
            row["transfer_actual_usd"] = 0.0

    def reconcile_duplicate_label(self, label: str, instance_ids: list[int]) -> None:
        """Every unexpected allocation is charged and stopped, never ignored."""
        with self.transaction():
            original = next(r for r in self.data["allocations"] if r.get("label") == label)
            known = {r["instance_id"] for r in self.data["allocations"]}
            for instance_id in sorted(set(instance_ids)):
                if instance_id in known:
                    continue
                if original["instance_id"] is None:
                    row = original
                else:
                    row = dict(original)
                    row["label"] = f"{label}:ambiguous:{instance_id}"
                    self.data["allocations"].append(row)
                row["instance_id"] = instance_id
                row["provider_status"] = "duplicate_label_force_stop"
                row["force_stop"] = True
                known.add(instance_id)
            original["force_stop"] = True

    def other_charges(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("Negative other charges")
        with self.transaction():
            self.data["other_actual_usd"] = max(self.data["other_actual_usd"], amount)

    @staticmethod
    def validate_offer(offer: dict) -> None:
        rate = float(offer["dph_total"])
        if not 0 < rate <= 2.2:
            raise ValueError("Offer exceeds the approved hourly rate")
        if not str(offer["driver_version"]).startswith("580."):
            raise ValueError("Offer is not on the pinned driver family")
        if offer["gpu_name"] != "RTX 4090" or int(offer["num_gpus"]) != 4:
            raise ValueError("Wrong GPU allocation")
        if float(offer["cpu_ram"]) < 250000 or float(offer["cpu_cores_effective"]) < 48:
            raise ValueError("Insufficient allocated host resources")
        transfer = 100 * float(offer["inet_down_cost"]) + 8 * float(offer["inet_up_cost"])
        if transfer > 1.8:
            raise ValueError("Offer leaves insufficient transfer/storage reserve")

    @staticmethod
    def _row(instance_id, created_utc, offer):
        return {
            "instance_id": instance_id,
            "created_utc": created_utc,
            "rate_usd_hour": float(offer["dph_total"]),
            "offer": offer,
            "active_stop_verified_utc": None,
            "provider_status": "creating",
            "transfer_estimate_usd": 100 * float(offer["inet_down_cost"])
            + 8 * float(offer["inet_up_cost"]),
        }

    def add(self, instance_id: int, created_utc: str, offer: dict) -> dict:
        self.validate_offer(offer)
        with self.transaction():
            if any(row["instance_id"] == instance_id for row in self.data["allocations"]):
                raise ValueError("Rental already recorded")
            if self.snapshot(timestamp(created_utc), refresh=False)["stop"]:
                raise ValueError("No financial allowance remains")
            row = self._row(int(instance_id), created_utc, offer)
            self.data["allocations"].append(row)
        return row

    def stopped(self, instance_id: int, stopped_utc: str, *, status: str) -> None:
        if status not in {"exited", "stopped", "destroyed"}:
            raise ValueError("A provider terminal state is required")
        with self.transaction():
            row = next(r for r in self.data["allocations"] if r["instance_id"] == instance_id)
            if timestamp(stopped_utc) < timestamp(row["created_utc"]):
                raise ValueError("Invalid stop timestamp")
            # Never clear a stop or move it later on a stale running response.
            if row["active_stop_verified_utc"] is None:
                row["active_stop_verified_utc"] = stopped_utc
                row["provider_status"] = status

    def snapshot(self, now: float | None = None, *, refresh: bool = True) -> dict:
        if refresh:
            with self.path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_SH)
                self.data = json.loads(self.path.read_text())
                if self.data.get("limits") != self.limits():
                    raise ValueError("Financial limits changed after initialization")
        now = dt.datetime.now(dt.timezone.utc).timestamp() if now is None else now
        hours = 0.0
        cost = 0.0
        running = []
        for row in self.data["allocations"]:
            end = (
                timestamp(row["active_stop_verified_utc"])
                if (row["active_stop_verified_utc"] is not None)
                else now
            )
            duration = max(0, end - timestamp(row["created_utc"])) / 3600
            hours += duration
            cost += duration * row["rate_usd_hour"]
            if row["active_stop_verified_utc"] is None:
                running.append(row)
        transfers = sum(
            row.get("transfer_actual_usd", row["transfer_estimate_usd"])
            for row in self.data["allocations"]
        )
        limits = self.limits()
        other = max(limits["other_reserve_usd"], transfers, float(self.data["other_actual_usd"]))
        total_remaining = limits["total_usd"] - other - cost
        money_remaining = min(limits["active_shutdown_usd"] - cost, total_remaining - 0.1)
        hours_remaining = max(0.0, limits["aggregate_hours"] - hours)
        # If allocations accidentally overlap, sum their rates and active hours.
        rate = sum(r["rate_usd_hour"] for r in running)
        count = len(running)
        remaining_seconds = min(
            money_remaining / rate * 3600 if rate else float("inf"),
            hours_remaining / count * 3600 if count else float("inf"),
        )
        stop = (
            money_remaining <= 0
            or hours_remaining <= 0
            or any(r.get("force_stop", False) for r in running)
        )
        deadline = now + max(0, remaining_seconds) if running else None
        return {
            "active_estimate_usd": cost,
            "aggregate_hours": hours,
            "other_reserved_or_actual_usd": other,
            "aggregate_transfer_estimate_usd": transfers,
            "total_estimate_usd": cost + other,
            "active_instances": [r["instance_id"] for r in running if r["instance_id"] is not None],
            "pending_labels": [r["label"] for r in running if r["instance_id"] is None],
            "stop": stop,
            "deadline_utc": dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat()
            if deadline is not None
            else None,
            "remaining_seconds": max(0.0, remaining_seconds) if running else None,
        }
