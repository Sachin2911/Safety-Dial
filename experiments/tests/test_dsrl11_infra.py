"""Task-11 financial envelope, schedule projection and deploy-time bundle verification."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

from helpers.dsrl11_infra import LIMITS, FinancialBook, full_schedule_projection

ROOT = Path(__file__).resolve().parents[2]
TIMINGS = {
    "bootstrap": 300,
    "preparation": 2700,
    "numerical": 900,
    "cold_episode": 1500,
    "four_lane_episode": 560,
    "three_lane_episode": 470,
    "update": 0.023,
    "restarts": 900,
    "local_copy": 1200,
}


def manager():
    spec = importlib.util.spec_from_file_location(
        "manage_dsrl11", ROOT / "experiments/scripts/manage_dsrl11.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_approved_envelope_and_a_fresh_book(tmp_path):
    assert LIMITS == {
        "total_usd": 18.0,
        "active_usd": 15.0,
        "active_shutdown_usd": 14.9,
        "other_reserve_usd": 3.0,
        "aggregate_hours": 8.0,
        "rate_usd_hour": 2.2,
    }
    book = FinancialBook(tmp_path / "finance.json")
    assert book.data["schema"] == "dsrl11-finance-v1" and book.data["limits"] == LIMITS


def test_projection_charges_waiting_time_and_respects_both_limits():
    base = full_schedule_projection(TIMINGS, 1.922)
    waiting = full_schedule_projection(TIMINGS, 1.922, waiting_seconds=4500)
    assert waiting["total_seconds"] - base["total_seconds"] == pytest.approx(4500)
    assert base["phases_seconds"]["online_collection"] == 12 * 470
    assert waiting["allowed_hours"] == pytest.approx(14.9 / 1.922)
    assert waiting["fits"] and waiting["margin_hours"] > 0.5
    assert not full_schedule_projection(TIMINGS, 2.2, waiting_seconds=9000)["fits"]
    with pytest.raises(ValueError):
        full_schedule_projection(TIMINGS, 2.3)


def build(tmp_path, *, tamper=None, mode="setup", money=None, instance=7):
    files = {
        "experiments/a.py": b"print(1)\n",
        "progress/receipts/ready.md": b"Launch decision: admitted\n",
    }
    plan = {
        "schema": {"setup": "dsrl11-setup-v1", "run": "dsrl11-freeze-v1"}[mode],
        "authorization": {"user_launch": True},
        "source_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
        "input_sha256": {},
        "money": money or dict(LIMITS),
        "launch_readiness_receipt": "progress/receipts/ready.md",
    }
    module = manager()
    files[module.PLAN_PATHS[mode]] = json.dumps(plan).encode()
    files["progress/rental-dsrl11-20261025.json"] = json.dumps({"instance_id": instance}).encode()
    if tamper:
        files[tamper] = b"changed"
    path = tmp_path / "bundle.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    # The local rental record names instance 7; the bundle names ``instance``.
    module.ROOT = tmp_path
    module.RENTAL = tmp_path / "progress/rental-dsrl11-20261025.json"
    module.RENTAL.parent.mkdir(parents=True, exist_ok=True)
    module.RENTAL.write_text(json.dumps({"instance_id": 7}))
    return module, path


def test_deploy_verifies_the_shipped_archive_not_the_working_tree(tmp_path):
    module, path = build(tmp_path)
    assert module.bundle_plan(path, "setup")["schema"] == "dsrl11-setup-v1"
    with pytest.raises(ValueError, match="lacks the regular file"):
        module.bundle_plan(path, "run")


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"tamper": "experiments/a.py"}, "differs from its plan"),
        ({"money": {**LIMITS, "total_usd": 15.0}}, "financial envelope"),
        ({"instance": 8}, "different rental"),
    ],
)
def test_bundle_tampering_money_and_rental_mismatches_are_refused(tmp_path, changes, match):
    module, path = build(tmp_path, **changes)
    with pytest.raises(ValueError, match=match):
        module.bundle_plan(path, "setup")
