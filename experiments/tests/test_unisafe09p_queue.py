"""Protect shared reservations and the earlier financial shutdown boundary."""

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.run_unisafe09p import reserve, work_deadline


def job(index, condition="d00_g100"):
    return {"id": f"job-{index}", "condition": condition, "continuation": True}


def freeze(controls=70720, offline=4360):
    return {"local_caps": {"attempts": 208, "controls": controls,
                           "offline_queries": offline, "replacements": 4}}


def test_concurrent_lanes_reserve_each_full_unfinished_budget(tmp_path):
    rental = {"instance_id": 57690181}
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(
            lambda lane: reserve(tmp_path, job(lane), freeze(), "sha", rental, lane), range(4)
        ))
    assert len({r[0].name for r in result}) == 4
    with pytest.raises(RuntimeError, match="Control-step cap"):
        reserve(tmp_path, job(5), freeze(controls=1360), "sha", rental, 0)
    assert len(list((tmp_path / "attempts").glob("*/intent.json"))) == 4


def test_inflight_job_cannot_be_claimed_again(tmp_path):
    rental = {"instance_id": 57690181}
    reserve(tmp_path, job(0), freeze(), "sha", rental, 0)
    with pytest.raises(RuntimeError, match="recorded diagnosis"):
        reserve(tmp_path, job(0), freeze(), "sha", rental, 0)


def test_diagnostic_pause_prevents_reservation_without_deleting_evidence(tmp_path):
    (tmp_path / "PAUSE").write_text(json.dumps({"invalid_attempt": "prior"}))
    with pytest.raises(RuntimeError, match="diagnostic pause"):
        reserve(tmp_path, job(0), freeze(), "sha", {"instance_id": 57690181}, 0)
    assert not (tmp_path / "attempts").exists()


def test_identity_queries_are_reserved_even_if_worker_has_not_started(tmp_path):
    rental = {"instance_id": 57690181}
    reserve(tmp_path, job(0, "native_direct"), freeze(), "sha", rental, 0)
    with pytest.raises(RuntimeError, match="Diagnostic query cap"):
        reserve(tmp_path, job(1, "native_direct"), freeze(offline=500), "sha", rental, 1)


def test_money_boundary_precedes_absolute_deadline_with_two_margins():
    rental = {"created_utc": "2026-10-25T16:00:00+00:00",
              "allocation_deadline_utc": "2026-10-25T20:00:00+00:00",
              "offer": {"dph_total": 2.0}}
    expected = dt.datetime.fromisoformat("2026-10-25T18:40:00+00:00").timestamp()
    assert work_deadline(rental) == expected


def test_allocation_keeps_every_seed_on_one_host_and_gpu():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    allocation = json.loads(
        (root / "progress/receipts/09-unisafe-parallel-allocation.json").read_text()
    )
    rows = allocation["assignments"]
    assert len(rows) == len({r["job_id"] for r in rows}) == 384
    for seed in {r["seed"] for r in rows}:
        block = [r for r in rows if r["seed"] == seed]
        assert len(block) == 6
        assert len({(r["host"], r["gpu_lane"]) for r in block}) == 1
    for lane in range(4):
        assert sum(r["host"] == "parallel" and r["gpu_lane"] == lane for r in rows) == 48
