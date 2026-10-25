"""A learning freeze must match actual engineering and the remaining allocation."""

import datetime as dt
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freeze_dsrl07_pilot import CAPS, digest, frozen_jobs, stage_b_plan


@pytest.fixture
def completed_engineering(tmp_path):
    def write(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    for name in (
        "experiments/scripts/freeze_dsrl07_pilot.py",
        "experiments/tests/test_dsrl07_pilot_freeze.py",
    ):
        write(name, "new stage B source")
    source = write("source.py", "immutable stage A source")
    plan = write(
        "stage-a.json",
        {
            "authorization": {"user_launch": True},
            "jobs": frozen_jobs(),
            "caps": dict(CAPS),
            "money": {"total_usd": 15},
            "source_sha256": {"source.py": digest(source)},
        },
    )
    numerical = write(
        "numerical-summary.json",
        {
            "admit_live_integration": True,
            "same_process_pass": True,
            "native_draw_passthrough_pass": True,
            "influence_present": True,
            "learner_mode": "deterministic",
        },
    )
    summary = write(
        "stage-a-summary.json",
        {
            "admit": True,
            "source_parity_valid": True,
            "cpu_actor_parity": {"valid": True},
            "instance_id": 7,
            "plan_sha256": digest(plan),
            "numerical_summary_sha256": digest(numerical),
            "learner_mode": "deterministic",
            "projected_stage_b_seconds_including_preservation": 7200,
        },
    )
    rental = write(
        "rental.json", {"instance_id": 7, "allocation_deadline_utc": "2026-10-25T13:00:00+00:00"}
    )
    now = dt.datetime.fromisoformat("2026-10-25T10:00:00+00:00")
    return tmp_path, plan, summary, rental, now


def test_completed_checks_bind_source_mode_and_full_schedule(completed_engineering):
    root, plan, summary, rental, now = completed_engineering
    before = plan.read_bytes()
    result = stage_b_plan(root, plan, summary, rental, now=now)
    assert result["stage"] == "B"
    assert result["learner_mode"] == "deterministic"
    assert result["stage_a_freeze_sha256"] == digest(plan)
    assert result["stage_a_summary_sha256"] == digest(summary)
    assert len(result["jobs"]["training_jobs"]) == 24
    assert len(result["jobs"]["evaluation_jobs"]) == 48
    assert result["remaining_allocation_seconds_at_freeze"] == 10800
    assert plan.read_bytes() == before


@pytest.mark.parametrize("failure", ["different_host", "failed_parity", "changed_numerical"])
def test_positive_admit_flag_cannot_hide_invalid_evidence(completed_engineering, failure):
    root, plan, summary, rental, now = completed_engineering
    values = json.loads(summary.read_text())
    if failure == "different_host":
        values["instance_id"] = 8
    elif failure == "failed_parity":
        values["cpu_actor_parity"]["valid"] = False
    else:
        numerical = root / "numerical-summary.json"
        numerical.write_text(numerical.read_text() + "\n")
    summary.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="does not admit"):
        stage_b_plan(root, plan, summary, rental, now=now)


def test_elapsed_time_can_invalidate_an_earlier_projection(completed_engineering):
    root, plan, summary, rental, now = completed_engineering
    with pytest.raises(ValueError, match="no longer fits"):
        stage_b_plan(root, plan, summary, rental, now=now + dt.timedelta(hours=2))


def test_source_drift_cannot_be_frozen_as_passed_engineering(completed_engineering):
    root, plan, summary, rental, now = completed_engineering
    (root / "source.py").write_text("changed after engineering")
    with pytest.raises(ValueError, match="Frozen source changed"):
        stage_b_plan(root, plan, summary, rental, now=now)
