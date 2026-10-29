"""A separate task-10 rental and financial scope; never operates task-7/8/9 hosts."""

import datetime as dt

from helpers.dsrl06_infra import FinancialBook as BaseFinancialBook
from helpers.dsrl06_infra import atomic_json as atomic_json
from helpers.dsrl06_infra import instances as instances
from helpers.dsrl06_infra import provider_stopped as provider_stopped
from helpers.dsrl06_infra import ssh_endpoint as ssh_endpoint
from helpers.dsrl06_infra import ssh_options as base_ssh_options
from helpers.dsrl06_infra import utc as utc
from helpers.dsrl06_infra import vast as vast

IMAGE = "vastai/base-image@sha256:c1d2b5326fae806b04d2c2d97a2b3948d0ccb0026dd3085e9a2ad55e304193f8"
# 13.2 GB compressed dataset, its decompressed HDF5 copy, packages and evidence.
DISK_GB = 80
# Planned transfer: 13.2 GB dataset plus checkpoint and packages inbound, evidence outbound.
INBOUND_GB = 25
OUTBOUND_GB = 10


def ssh_options(port):
    return [
        x.replace("/tmp/dsrl06-known-hosts", "/tmp/safecem10-known-hosts")
        for x in base_ssh_options(port)
    ]


def transfer_estimate(offer):
    return INBOUND_GB * float(offer["inet_down_cost"]) + OUTBOUND_GB * float(offer["inet_up_cost"])


class FinancialBook(BaseFinancialBook):
    def reserve_resume(self, instance_id, label, created_utc, deadline_utc):
        """Reserve another active segment on the SAME stopped rental, before start."""
        created = dt.datetime.fromisoformat(created_utc)
        deadline = dt.datetime.fromisoformat(deadline_utc)
        if created.tzinfo is None or deadline.tzinfo is None or created >= deadline:
            raise ValueError("Resume cannot extend the original allocation deadline")
        with self.transaction():
            rows = self.data["allocations"]
            if any(row["active_stop_verified_utc"] is None for row in rows):
                raise ValueError("An active or uncertain segment already exists")
            matching = [row for row in rows if row["instance_id"] == int(instance_id)]
            if not matching or matching[-1]["provider_status"] not in {"stopped", "exited"}:
                raise ValueError("Only this study's verified stopped rental may resume")
            previous = matching[-1]
            first_created = min(dt.datetime.fromisoformat(r["created_utc"]) for r in matching)
            if deadline > first_created + dt.timedelta(hours=self.limits()["aggregate_hours"]):
                raise ValueError("Resume would extend the original rental window")
            if created < dt.datetime.fromisoformat(previous["active_stop_verified_utc"]):
                raise ValueError("Resume precedes the verified stop")
            if any(row.get("label") == label for row in rows):
                raise ValueError("Resume reservation already exists")
            if self.snapshot(created.timestamp(), refresh=False)["stop"]:
                raise ValueError("Financial allowance exhausted")
            self.validate_offer(previous["offer"])
            row = self._row(int(instance_id), created_utc, previous["offer"])
            row.update(
                label=label,
                resume_of=previous["label"],
                transfer_estimate_usd=0.0,
                provider_status="resume_reserved",
            )
            rows.append(row)
        return row

    def stopped(self, instance_id, stopped_utc, *, status):
        """Close the active segment, retaining earlier stops on a resumed instance."""
        if status not in {"exited", "stopped", "destroyed"}:
            raise ValueError("A provider terminal state is required")
        ended = dt.datetime.fromisoformat(stopped_utc)
        if ended.tzinfo is None:
            raise ValueError("Stop timestamp must include a timezone")
        with self.transaction():
            matching = [r for r in self.data["allocations"] if r["instance_id"] == int(instance_id)]
            if not matching:
                raise ValueError("Unknown rental")
            for row in matching:
                if row["active_stop_verified_utc"] is not None:
                    continue
                if ended < dt.datetime.fromisoformat(row["created_utc"]):
                    raise ValueError("Stop precedes reservation")
                row["active_stop_verified_utc"] = stopped_utc
                row["provider_status"] = status

    @staticmethod
    def limits():
        return {
            "total_usd": 10.0,
            "active_usd": 8.0,
            "active_shutdown_usd": 7.9,
            "other_reserve_usd": 2.0,
            "aggregate_hours": 14.0,
            "rate_usd_hour": 0.6,
        }

    @staticmethod
    def validate_offer(offer):
        if not 0 < float(offer["dph_total"]) <= 0.6:
            raise ValueError("Offer exceeds the task-10 inclusive hourly ceiling")
        driver = tuple(int(p) for p in str(offer["driver_version"]).split(".")[:2])
        if driver < (570, 0):
            raise ValueError("Driver older than the CUDA 12.8 floor")
        if offer["gpu_name"] not in {"RTX 3090", "RTX 4090"} or int(offer["num_gpus"]) != 1:
            raise ValueError("Task 10 requires one RTX 3090 or 4090")
        if float(offer["gpu_ram"]) < 20000:
            raise ValueError("Insufficient GPU memory")
        if float(offer["cpu_ram"]) < 30000 or float(offer["cpu_cores_effective"]) < 8:
            raise ValueError("Insufficient allocated host resources")
        if float(offer["disk_space"]) < DISK_GB:
            raise ValueError("Insufficient disk")
        if float(offer.get("inet_down") or 0) < 200:
            raise ValueError("Download bandwidth too low for the dataset")
        if transfer_estimate(offer) > 1.0:
            raise ValueError("Transfer allowance leaves insufficient cleanup reserve")

    @staticmethod
    def _row(instance_id, created_utc, offer):
        row = BaseFinancialBook._row(instance_id, created_utc, offer)
        row["transfer_estimate_usd"] = transfer_estimate(offer)
        return row
