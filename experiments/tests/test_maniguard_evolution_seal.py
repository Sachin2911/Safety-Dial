"""Byte-transfer and terminal-state safeguards; no simulator or network calls."""

import hashlib
import json
import shutil

import pytest

from seal_maniguard_evolution import seal, verify_local


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    project = tmp_path / "remote"
    checkout = tmp_path / "checkout"
    upstream = checkout / "maniguard/eval/benchmark.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("# unchanged upstream\n")
    write(project / "run-v2/status.json", {"status": "completed"})
    write(project / "supervision/final-audit.json", {
        "final": True, "status": "completed", "valid_replayed_episodes": 68,
        "declared_episodes": 68})
    write(project / "service-v2.json", {"pid": 99999999})
    write(project / "run-v2/generated/evaluator-source.json", {
        "checkout": str(checkout),
        "upstream_sha256": hashlib.sha256(upstream.read_bytes()).hexdigest()})
    write(project / "run-v2/artifact.json", {"retained": True})
    return project


def transferred(tmp_path):
    remote = fixture(tmp_path)
    output = remote / "supervision/remote-manifest.json"
    record = seal(remote, "run-v2", output)
    local = tmp_path / "local"
    shutil.copytree(remote, local)
    return local, local / "supervision/remote-manifest.json", record


def test_complete_transfer_matches_exact_remote_inventory(tmp_path):
    local, output, record = transferred(tmp_path)
    result = verify_local(local, output, record["remote_seal_sha256"])
    assert result["local_verified"]
    assert result["files_including_seal"] == record["files_excluding_seal"] + 1


def test_changed_local_file_cannot_pass_transfer_verification(tmp_path):
    local, output, record = transferred(tmp_path)
    write(local / "run-v2/artifact.json", {"retained": False})
    with pytest.raises(ValueError, match="differs from remote seal"):
        verify_local(local, output, record["remote_seal_sha256"])


def test_extra_local_file_cannot_be_silently_included_in_release(tmp_path):
    local, output, record = transferred(tmp_path)
    write(local / "unexpected.json", {"unlisted": True})
    with pytest.raises(ValueError, match="inventory differs"):
        verify_local(local, output, record["remote_seal_sha256"])


def test_incomplete_run_cannot_be_sealed_for_teardown(tmp_path):
    remote = fixture(tmp_path)
    write(remote / "run-v2/status.json", {"status": "running"})
    with pytest.raises(ValueError, match="Completed training"):
        seal(remote, "run-v2", remote / "supervision/remote-manifest.json")


def test_honestly_incomplete_terminal_run_can_be_preserved_explicitly(tmp_path):
    remote = fixture(tmp_path)
    write(remote / "run-v2/status.json", {"status": "budget_exhausted"})
    write(remote / "supervision/final-audit.json", {
        "final": True, "status": "budget_exhausted", "valid_replayed_episodes": 4,
        "declared_episodes": 108, "complete_schedule": False})
    output = remote / "supervision/remote-manifest.json"
    seal(remote, "run-v2", output, allow_incomplete=True)
    manifest = json.loads(output.read_text())
    assert manifest["status"] == "budget_exhausted" and not manifest["complete_schedule"]
