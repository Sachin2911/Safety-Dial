"""The extra host must retain all historical costs inside the combined ceiling."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe09p_infra import FinancialBook, combined_snapshot


def snapshot(active, transfer=0.5, other=0):
    return {
        "active_estimate_usd": active,
        "aggregate_transfer_estimate_usd": transfer,
        "other_reserved_or_actual_usd": max(2, transfer, other),
        "other_actual_usd": other,
        "active_instances": [],
    }


def test_old_charges_and_shared_reserve_are_counted():
    combined = combined_snapshot(snapshot(4, 1), snapshot(6, 0.6))
    assert combined["total_estimate_usd"] == 12
    assert not combined["combined_stop"]
    assert combined_snapshot(snapshot(6), snapshot(6.9))["combined_stop"]


def test_actual_other_charges_from_both_books_are_retained():
    combined = combined_snapshot(snapshot(3, other=2.2), snapshot(4, other=1.5))
    assert combined["other_reserved_or_actual_usd"] == pytest.approx(3.7)
    assert combined["total_estimate_usd"] == pytest.approx(10.7)


def test_independent_active_ceilings_fit_combined_authorization():
    original_upper_bound = 15 * (0.36 + 200 * 0.2 / 720)
    new_upper_bound = FinancialBook.limits()["active_shutdown_usd"]
    assert original_upper_bound + new_upper_bound + 2 < 15


def test_offer_rejects_unsupported_hardware_and_cost():
    offer = {
        "dph_total": 1.8155555555555556, "driver_version": "580.159.03",
        "gpu_name": "RTX 4090", "num_gpus": 4, "cpu_ram": 257812,
        "cpu_cores_effective": 48, "inet_down_cost": 0.0026041666666666665,
        "inet_up_cost": 0.00390625,
    }
    FinancialBook.validate_offer(offer)
    for changed in ({"num_gpus": 1}, {"dph_total": 2.1}, {"cpu_ram": 128000},
                    {"driver_version": "560.35.03"}, {"inet_down_cost": 0.1}):
        with pytest.raises(ValueError):
            FinancialBook.validate_offer({**offer, **changed})
