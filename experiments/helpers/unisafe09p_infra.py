"""Additional UNISafe host within the explicitly approved combined $15 cap."""

from pathlib import Path

from helpers.dsrl06_infra import FinancialBook as BaseFinancialBook
from helpers.unisafe09_infra import IMAGE as IMAGE
from helpers.unisafe09_infra import FinancialBook as OriginalFinancialBook
from helpers.unisafe09_infra import atomic_json as atomic_json
from helpers.unisafe09_infra import instances as instances
from helpers.unisafe09_infra import provider_stopped as provider_stopped
from helpers.unisafe09_infra import ssh_endpoint as ssh_endpoint
from helpers.unisafe09_infra import ssh_options as ssh_options
from helpers.unisafe09_infra import utc as utc
from helpers.unisafe09_infra import vast as vast

ROOT = Path(__file__).resolve().parents[2]
OLD_BOOK = ROOT / "outputs/unisafe09/control/finance.json"
NEW_BOOK = ROOT / "outputs/unisafe09p/control/finance.json"
RENTAL = ROOT / "progress/rental-unisafe09p-20261025.json"


class FinancialBook(BaseFinancialBook):
    @staticmethod
    def limits():
        # The original allocation is additionally bounded by 15 h at $0.415556/h.
        # $6.50 new active charges + $6.234 original + $2 shared reserve < $15.
        return {
            "total_usd": 8.6,
            "active_usd": 6.6,
            "active_shutdown_usd": 6.5,
            "other_reserve_usd": 2.0,
            "aggregate_hours": 3.5,
            "rate_usd_hour": 2.0,
        }

    @staticmethod
    def validate_offer(offer):
        if not 0 < float(offer["dph_total"]) <= 2.0:
            raise ValueError("Additional host exceeds the $2.00/h ceiling")
        driver = tuple(int(part) for part in str(offer["driver_version"]).split("."))
        if driver[0] != 580 or driver < (580, 95, 5):
            raise ValueError("Additional host needs the supported 580 driver family")
        if offer["gpu_name"] != "RTX 4090" or int(offer["num_gpus"]) != 4:
            raise ValueError("Additional host must have four RTX 4090 GPUs")
        if float(offer["cpu_ram"]) < 250000 or float(offer["cpu_cores_effective"]) < 48:
            raise ValueError("Four workers require at least 250 GB RAM and 48 effective CPUs")
        transfer = 150 * float(offer["inet_down_cost"]) + 40 * float(offer["inet_up_cost"])
        if transfer > 0.8:
            raise ValueError("Transfer reservation exceeds the shared preservation allowance")

    @staticmethod
    def _row(instance_id, created_utc, offer):
        row = BaseFinancialBook._row(instance_id, created_utc, offer)
        row["transfer_estimate_usd"] = (
            150 * float(offer["inet_down_cost"]) + 40 * float(offer["inet_up_cost"])
        )
        return row


def combined_snapshot(old, new):
    active = old["active_estimate_usd"] + new["active_estimate_usd"]
    transfers = old["aggregate_transfer_estimate_usd"] + new["aggregate_transfer_estimate_usd"]
    actual_other = old.get("other_actual_usd", old["other_reserved_or_actual_usd"]) + new.get(
        "other_actual_usd", new["other_reserved_or_actual_usd"]
    )
    reserve = max(2.0, transfers, actual_other)
    return {
        "combined_cap_usd": 15.0,
        "active_compute_disk_estimate_usd": active,
        "transfer_estimate_usd": transfers,
        "other_reserved_or_actual_usd": reserve,
        "total_estimate_usd": active + reserve,
        "combined_stop": active + reserve >= 14.85,
        "active_instances": old["active_instances"] + new["active_instances"],
    }


def current_finances():
    old_book, new_book = OriginalFinancialBook(OLD_BOOK), FinancialBook(NEW_BOOK)
    old, new = old_book.snapshot(), new_book.snapshot()
    old["other_actual_usd"] = old_book.data["other_actual_usd"]
    new["other_actual_usd"] = new_book.data["other_actual_usd"]
    return old, new, combined_snapshot(old, new)
