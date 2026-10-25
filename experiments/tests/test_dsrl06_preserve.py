"""Preservation tests use local SQLite and an in-memory fake private Hub."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from helpers.dsrl06_server import QueryLedger

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "preserve_dsrl06", ROOT / "experiments/scripts/preserve_dsrl06.py"
)
preserve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preserve)


def write(path, value="evidence"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def fixture(tmp_path, checkpoint=False):
    repository = tmp_path / "repository"
    write(repository / "AGENTS.md", "Task 6 rules")
    write(repository / "progress/06-dsrl-plan.md", "Frozen plan")
    write(repository / "progress/receipts/06-build.md", "Build checks")
    local = tmp_path / "local"
    remote = local / "remote"
    write(remote / "captures/E0/query_00000001.npz", "raw native query")
    write(
        remote / "episodes/E0/job.json",
        json.dumps({"job_id": "E0", "task": 1, "stage": "engineering"}),
    )
    write(remote / "episodes/E0/failed-attempt.log", "Preserve failure too")
    write(remote / "episodes/E0/base_main.mp4", "video bytes")
    write(remote / "authority_abc.state.json", "send intent")
    write(remote / "source/frozen-source.py", "print('frozen')")
    write(remote / "environment/gpus.txt", "four RTX4090")
    write(remote / "network-cost.json", '{"cost":1}')
    write(remote / "logs/remote-watchdog.log", "watchdog started")
    write(local / "control/finance.json", '{"cap":15}')
    write(local / "control/watchdog-status.json", '{"remaining":100}')
    write(local / "control/watchdog.log", "running")
    if checkpoint:
        write(remote / "learner/checkpoints/final/learner.msgpack", "weights")
        write(remote / "learner/checkpoints/final/replay.npz", "replay state")
        write(remote / "learner/checkpoints/final/state.json", "{}")
        write(remote / "learner/actor_exports/final/actor.msgpack", "published actor weights")
        write(remote / "learner/actor_exports/final/state.json", "{}")
        write(remote / "learner/actor_exports/final/manifest.json", "{}")
    ledger = QueryLedger(remote / "control/queries.sqlite")
    # Retain an open connection so the test covers a real uncheckpointed WAL.
    live = ledger.connect()
    ledger.reserve(
        {
            "job_id": "E0",
            "query_index": 0,
            "mode": "native",
            "purpose": "policy",
            "actor_version": -1,
            "packet_id": "E0:0",
        }
    )
    stopped = write(
        remote / "workers-stopped.json",
        json.dumps(
            {
                "instance_id": 123,
                "workers_stopped": True,
                "worker_pids": [],
            }
        ),
    )
    return repository, local, remote, ledger, live, stopped


def prepared_stage(tmp_path, checkpoint=False):
    repository, local, remote, ledger, live, stopped = fixture(tmp_path, checkpoint)
    preserve.prepare_remote(remote, 123, ledger.path, stopped)
    stage = local / "hf-stage"
    preserve.stage_evidence([remote], local, stage, 123, repository)
    return repository, local, remote, stage, live


class Hub:
    def __init__(self, root):
        self.root = root
        self.private = {preserve.DATASET: True}
        self.revisions = {}
        self.uploads = []
        self.downloads = []
        self.created = []

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        assert private and exist_ok and repo_type == "model"
        self.created.append(repo_id)
        self.private.setdefault(repo_id, True)

    def repo_info(self, repo, repo_type):
        return SimpleNamespace(private=self.private[repo], sha="previous")

    def upload_folder(self, folder_path, repo_id, repo_type, commit_message, parent_commit):
        assert parent_commit == "previous"
        revision = f"revision{len(self.uploads)}"
        destination = self.root / revision
        shutil.copytree(folder_path, destination)
        self.revisions[(repo_id, revision)] = destination
        self.uploads.append((repo_id, repo_type, revision))
        return SimpleNamespace(oid=revision)

    def download(self, repo_id, filename, repo_type, revision, token, cache_dir):
        assert token == "process-only-token" and revision.startswith("revision")
        self.downloads.append((repo_id, revision, filename))
        return self.revisions[(repo_id, revision)] / filename


def upload(stage, hub, marker):
    return preserve.upload_verified(
        stage,
        marker,
        123,
        api=hub,
        download=hub.download,
        token="process-only-token",
        workers=2,
    )


def test_live_sqlite_backup_retains_wal_rows_and_excludes_original_db_files(tmp_path):
    repository, local, remote, ledger, live, stopped = fixture(tmp_path)
    assert Path(str(ledger.path) + "-wal").exists()
    result = preserve.prepare_remote(remote, 123, ledger.path, stopped)
    assert result["query_backup"]["queries"] == 1
    stage = local / "hf-stage"
    manifest = preserve.stage_evidence([remote], local, stage, 123, repository)
    names = set(manifest["dataset"])
    assert any(name.endswith("preservation/query-ledger.sqlite") for name in names)
    assert not any(name.endswith(("queries.sqlite", "-wal", "-shm")) for name in names)
    assert any("failed-attempt.log" in name for name in names)
    assert any("authority_abc.state.json" in name for name in names)
    assert preserve.verify_local(stage)["instance_id"] == 123
    live.close()


def test_prepare_refuses_alive_workers_and_another_rental(tmp_path):
    _, _, remote, ledger, live, stopped = fixture(tmp_path)
    stopped.write_text(
        json.dumps({"instance_id": 123, "workers_stopped": True, "worker_pids": [os.getpid()]})
    )
    with pytest.raises(ValueError, match="still present"):
        preserve.prepare_remote(remote, 123, ledger.path, stopped)
    with pytest.raises(ValueError, match="matching"):
        preserve.prepare_remote(remote, 999, ledger.path, stopped)
    assert not (remote / "preservation").exists()
    live.close()


def test_engineering_only_upload_downloads_every_file_and_emits_destroy_marker(tmp_path):
    _, local, _, stage, live = prepared_stage(tmp_path)
    hub = Hub(tmp_path / "hub")
    marker = local / "preservation-verification.json"
    result = upload(stage, hub, marker)
    assert result["instance_id"] == 123
    assert result["local_verified"] and result["private_hf_verified"]
    assert result["manifest_sha256"] == preserve.digest(stage / "manifest.json")
    assert set(result["pinned_revisions"]) == {"dataset"}
    assert not result["model_checkpoint_present"] and not hub.created
    assert len(hub.downloads) == len(preserve.files(stage / "dataset"))
    assert any(name == preserve.PUBLISHED_MANIFEST for _, _, name in hub.downloads)
    assert json.loads(marker.read_text()) == result
    live.close()


def test_full_checkpoint_and_replay_state_go_only_to_new_private_model_repo(tmp_path):
    _, local, _, stage, live = prepared_stage(tmp_path, checkpoint=True)
    manifest = preserve.verify_local(stage)
    assert len(manifest["model"]) == 6
    assert not any("learner/checkpoints" in name for name in manifest["dataset"])
    assert not any("actor_exports" in name for name in manifest["dataset"])
    hub = Hub(tmp_path / "hub")
    result = upload(stage, hub, local / "preservation-verification.json")
    assert set(result["pinned_revisions"]) == {"dataset", "model"}
    assert hub.created == [preserve.MODEL]
    assert result["verified_files"]["model"] == 6
    live.close()


def test_mutating_financial_watchdog_keeps_scientific_freeze_stable(tmp_path):
    _, local, remote, stage, live = prepared_stage(tmp_path)
    (remote / "network-cost.json").write_text('{"cost":2}')
    (remote / "logs/remote-watchdog.log").write_text("new watchdog line")
    (local / "control/watchdog-status.json").write_text('{"remaining":90}')
    (local / "control/watchdog.log").write_text("still running")
    assert preserve.verify_local(stage)
    snapshots = list((stage / "dataset").rglob("network-cost.json"))
    assert len(snapshots) == 1 and snapshots[0].read_text() == '{"cost":1}'
    live.close()


def test_hub_cache_symlinks_and_previous_stage_trees_are_excluded_before_scan(tmp_path):
    _, local, _, stage, live = prepared_stage(tmp_path)
    blob = write(local / "hf-verify-cache/blobs/sha256", "downloaded evidence")
    link = local / "hf-verify-cache/snapshots/revision/linked-file"
    link.parent.mkdir(parents=True)
    link.symlink_to(blob)
    old = local / "hf-stage-old"
    old.mkdir()
    (old / "ignored-cache-link").symlink_to(blob)
    assert preserve.verify_local(stage)
    live.close()


def test_late_source_or_receipt_and_new_remote_evidence_require_restage(tmp_path):
    repository, local, remote, stage, live = prepared_stage(tmp_path)
    late = write(repository / "progress/receipts/06-late.md", "late receipt")
    with pytest.raises(ValueError, match="Late source or receipt"):
        preserve.verify_local(stage)
    late.unlink()
    write(remote / "episodes/E0/late-frame.npz", "new scientific evidence")
    with pytest.raises(ValueError, match="changed after"):
        preserve.verify_local(stage)
    assert not (local / "preservation-verification.json").exists()
    live.close()


@pytest.mark.parametrize("problem", ["public", "download", "secret"])
def test_failed_private_upload_verification_never_leaves_destroy_marker(tmp_path, problem):
    _, local, _, stage, live = prepared_stage(tmp_path)
    marker = write(local / "preservation-verification.json", "stale success")
    hub = Hub(tmp_path / "hub")
    if problem == "public":
        hub.private[preserve.DATASET] = False
    elif problem == "download":
        original = hub.download

        def wrong(**kwargs):
            original(**kwargs)
            return write(tmp_path / "wrong-download", "tampered bytes")

        hub.download = wrong
    with pytest.raises(ValueError):
        preserve.upload_verified(
            stage,
            marker,
            123,
            api=hub,
            download=hub.download,
            token="process-only-token",
            secrets=[b"four RTX4090"] if problem == "secret" else (),
            workers=1,
        )
    assert not marker.exists()
    if problem in {"public", "secret"}:
        assert not hub.uploads
    live.close()


def test_symbolic_links_and_environment_files_are_never_staged(tmp_path):
    _, _, remote, ledger, live, stopped = fixture(tmp_path)
    secret = write(remote / ".env", "HF_TOKEN=never-read-or-print")
    with pytest.raises(ValueError, match="Credential environment"):
        preserve.prepare_remote(remote, 123, ledger.path, stopped)
    secret.unlink()
    shutil.rmtree(remote / "preservation")
    (remote / "linked-source").symlink_to(remote / "source")
    with pytest.raises(ValueError, match="symbolic links"):
        preserve.prepare_remote(remote, 123, ledger.path, stopped)
    live.close()


def test_failed_provisioning_without_query_service_is_explicitly_preserved(tmp_path):
    remote = tmp_path / "local/remote"
    write(remote / "logs/preparation.log", "Install failed before serving")
    write(remote / "source/bootstrap.py", "pinned source")
    stopped = write(
        remote / "quiescence.json",
        json.dumps(
            {
                "instance_id": 123,
                "workers_stopped": True,
                "worker_pids": [],
            }
        ),
    )
    prepared = preserve.prepare_remote(remote, 123, remote / "queries.sqlite", stopped)
    assert prepared["query_backup"] == {
        "source_missing": True,
        "queries": 0,
        "reason": "no_query_service_started",
    }
    assert not (remote / "preservation/query-ledger.sqlite").exists()
    stage = tmp_path / "local/hf-stage"
    preserve.stage_evidence([remote], tmp_path / "local", stage, 123, tmp_path / "repository")
    hub = Hub(tmp_path / "hub")
    result = upload(stage, hub, tmp_path / "local/preservation-verification.json")
    assert result["private_hf_verified"] and result["query_accounting_verified"]
    assert not result["query_ledger_backup_verified"]
    assert result["missing_query_ledgers_no_service"] == [123]


@pytest.mark.parametrize("evidence", ["artifact", "journal", "ledger", "probe"])
def test_missing_query_database_cannot_erase_query_attempts(tmp_path, evidence):
    remote = tmp_path / "remote"
    stopped = write(
        remote / "quiescence.json",
        json.dumps(
            {
                "instance_id": 123,
                "workers_stopped": True,
                "worker_pids": [],
            }
        ),
    )
    if evidence == "artifact":
        write(remote / "queries/E0/query_00000001.npz", "saved call")
    elif evidence == "journal":
        write(remote / "episode.journal.jsonl", json.dumps({"event": "query_attempt"}) + "\n")
    elif evidence == "ledger":
        write(
            remote / "resources.jsonl",
            json.dumps(
                {
                    "event": "charge",
                    "data": {"kind": "model_queries", "count": 1},
                }
            )
            + "\n",
        )
    else:
        write(
            remote / "authority_x.state.json",
            json.dumps(
                {
                    "schema": "dsrl06-probe-state-v1",
                    "requests": {"0": {"sent": True}},
                }
            ),
        )
    with pytest.raises(ValueError, match="Missing query ledger"):
        preserve.prepare_remote(remote, 123, remote / "queries.sqlite", stopped)


def test_terminated_zombie_wrapper_is_not_a_live_science_writer(tmp_path, monkeypatch):
    stopped = write(
        tmp_path / "quiescence.json",
        json.dumps(
            {
                "instance_id": 123,
                "workers_stopped": True,
                "worker_pids": [999999],
            }
        ),
    )
    original = Path.read_text
    monkeypatch.setattr(preserve.os, "kill", lambda pid, signal: None)

    def read(path, *args, **kwargs):
        if str(path) == "/proc/999999/stat":
            return "999999 (terminated wrapper) Z 1 2 3"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert preserve.verify_stopped(stopped, 123)["workers_stopped"]
