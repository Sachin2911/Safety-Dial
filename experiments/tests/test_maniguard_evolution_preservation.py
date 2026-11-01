"""File-preservation checks without simulator interactions or network writes."""

import json
import pytest

from preserve_maniguard_evolution import stage


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, completed=False):
    mirror = tmp_path / "mirror"
    run = mirror / "run-v2"
    jobs = {str(i): {"valid": True} for i in range(8)} if completed else {"done": {"valid": True}}
    write(run / "config.json", {
        "search": {"generations": 1, "population": 4, "tasks": [0]},
        "validation": {"tasks": [3], "policy_seeds": [1]},
    })
    # Five search/reference episodes and two validation episodes.
    if completed:
        jobs.pop("7")
    write(run / "status.json", {"status": "completed" if completed else "running",
                                "generation": int(completed), "jobs": jobs,
                                "validation": {}, "active": None})
    write(run / "source-manifest.json", {})
    write(run / "generation-snapshots/g000/controller.json", {"immutable": True})
    write(mirror / "review-v3/source.py", {"next_version": True})
    write(run / "episodes/done/outcome.json", {"valid": True})
    write(run / "episodes/done/trace.json", {"closed": True})
    write(run / "episodes/live/trace.json", {"open": True})
    write(mirror / "experiments/source.py", {"source": "immutable"})
    return mirror


def test_live_snapshot_excludes_open_episode_and_mutable_state(tmp_path):
    mirror = fixture(tmp_path)
    release = tmp_path / "release"
    index = stage(mirror, release)
    assert "run-v2/episodes/done/trace.json" in index["files"]
    assert "run-v2/episodes/live/trace.json" not in index["files"]
    assert "run-v2/status.json" not in index["files"]
    assert "run-v2/source-manifest.json" in index["files"]
    assert "run-v2/generation-snapshots/g000/controller.json" in index["files"]
    assert "review-v3/source.py" in index["files"]


def test_final_snapshot_rejects_incomplete_schedule(tmp_path):
    mirror = fixture(tmp_path)
    with pytest.raises(ValueError, match="every declared"):
        stage(mirror, tmp_path / "release", final=True)


def test_completed_release_preserves_all_files_and_rejects_credentials(tmp_path):
    mirror = fixture(tmp_path, completed=True)
    index = stage(mirror, tmp_path / "release", final=True)
    assert index["final"] and "run-v2/status.json" in index["files"]
    assert "run-v2/episodes/live/trace.json" in index["files"]
    write(mirror / ".env", {"dummy": "refuse this filename"})
    with pytest.raises(ValueError, match="credential-like"):
        stage(mirror, tmp_path / "unsafe-release", final=True)


def test_release_cannot_be_staged_inside_live_mirror(tmp_path):
    mirror = fixture(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        stage(mirror, mirror / "release")


def test_v3_final_schedule_includes_selection_and_uses_its_fresh_prefix(tmp_path):
    mirror = fixture(tmp_path, completed=True)
    run = mirror / "run-v2"
    config = json.loads((run / "config.json").read_text())
    config["selection"] = {"tasks": [6], "policy_seeds": [2]}
    config["preservation"] = {"base_prefix": "22_conservative_adapter_development/test"}
    write(run / "config.json", config)
    with pytest.raises(ValueError, match="every declared"):
        stage(mirror, tmp_path / "too-early", final=True)
    state = json.loads((run / "status.json").read_text())
    state["jobs"].update({f"selection_{i}": {"valid": True} for i in range(3)})
    write(run / "status.json", state)
    actual = stage(mirror, tmp_path / "complete", final=True)
    assert actual["base_prefix"] == config["preservation"]["base_prefix"]
    assert actual["observation"]["planned_episodes"] == 10


def test_terminal_incomplete_archive_preserves_every_file_without_claiming_completion(tmp_path):
    mirror = fixture(tmp_path)
    path = mirror / "run-v2/status.json"
    state = json.loads(path.read_text())
    state["status"] = "budget_exhausted"
    write(path, state)
    index = stage(mirror, tmp_path / "terminal", terminal=True)
    assert index["final"] and not index["complete_schedule"]
    assert index["observation"]["run_status"] == "budget_exhausted"
    assert "run-v2/episodes/live/trace.json" in index["files"]
