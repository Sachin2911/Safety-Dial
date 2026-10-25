"""Fresh task-7 episodes are validated against their own source-frozen schedule."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import test_dsrl06_validation as fixtures
from helpers.dsrl06_validation import build_episode
from helpers.dsrl07_protocol import frozen_jobs
from run_dsrl07 import Coordinator


def test_task7_episode_requires_its_fresh_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "frozen_jobs", frozen_jobs)
    directory = tmp_path / "E5"
    fixtures.fixture(directory)
    with pytest.raises(ValueError, match="seeds differ"):
        build_episode(directory)
    result = build_episode(directory, job_schedule=frozen_jobs())
    assert result["validation"]["job_id"] == "E5"
    assert result["validation"]["valid"]


def test_live_queue_uses_its_verified_schedule_for_recording_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "frozen_jobs", frozen_jobs)
    directory = tmp_path / "E5"
    fixtures.fixture(directory)
    queue = Coordinator.__new__(Coordinator)
    queue.plan = {"jobs": frozen_jobs()}
    queue.state = {"episodes": {"E5": {"directory": str(directory)}}}
    result = queue.episode("E5")
    assert result["validation"]["job_id"] == "E5"
    assert result["validation"]["valid"]


def test_task7_schedule_cannot_admit_an_old_task6_episode(tmp_path):
    directory = tmp_path / "E5"
    fixtures.fixture(directory)
    with pytest.raises(ValueError, match="seeds differ"):
        build_episode(directory, job_schedule=frozen_jobs())
    assert build_episode(directory)["validation"]["valid"]
