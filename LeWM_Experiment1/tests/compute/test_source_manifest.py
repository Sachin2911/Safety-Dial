"""Offline checks for explicitly manifested, ignored source snapshots."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "LeWM_Experiment1/computeConfigs"
SPEC = importlib.util.spec_from_file_location("compute_manifest_test", DIRECTORY / "compute.py")
compute = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compute)


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "PROJECT", tmp_path)
    source = tmp_path / "runs/frozen"
    source.mkdir(parents=True)
    (source / "main.py").write_bytes(b"print('fixture')\n")
    manifest = {"path": "runs/frozen", "instance_id": 42, "session_id": "fixture",
                "files": {"main.py": hashlib.sha256((source / "main.py").read_bytes()).hexdigest()}}
    return source, manifest


def test_manifest_packages_only_explicit_bytes_without_consulting_git(snapshot, monkeypatch):
    source, manifest = snapshot
    (source / "ignored.bin").write_bytes(b"not source")
    (source / ".env").write_bytes(b"excluded")
    monkeypatch.setattr(compute.subprocess, "run", lambda *a, **k: pytest.fail("consulted git"))
    archive, names = compute.bundle_source(source, {}, 1000, manifest=manifest)
    assert names == ["main.py"]
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as handle:
        assert handle.getnames() == names
        assert handle.extractfile("main.py").read() == (source / "main.py").read_bytes()


def test_default_never_falls_back_to_ignored_files(snapshot, monkeypatch):
    source, _ = snapshot
    calls = []

    def git(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(stdout=b"")

    monkeypatch.setattr(compute.subprocess, "run", git)
    with pytest.raises(compute.ComputeError, match="No deployable"):
        compute.bundle_source(source, {}, 1000)
    assert calls[0][:2] == ["git", "ls-files"]


@pytest.mark.parametrize("problem", ["extra", "missing", "changed", "path", "bad-hash", "empty"])
def test_manifest_rejects_any_snapshot_drift(snapshot, problem):
    source, manifest = snapshot
    if problem == "extra":
        (source / "extra.py").write_text("pass\n")
    elif problem == "missing":
        (source / "main.py").unlink()
    elif problem == "changed":
        (source / "main.py").write_text("pass\n")
    elif problem == "path":
        manifest["path"] = "other"
    elif problem == "bad-hash":
        manifest["files"]["main.py"] = "not-a-sha256"
    else:
        manifest["files"] = {}
    with pytest.raises(compute.ComputeError):
        compute.bundle_source(source, {}, 1000, manifest=manifest)


@pytest.mark.parametrize("name", [".secret.py", "cache/file.py", "other.bin", "../other.py", "/other.py"])
def test_manifest_cannot_bypass_source_path_filters(snapshot, name):
    source, manifest = snapshot
    manifest["files"][name] = "a" * 64
    with pytest.raises(compute.ComputeError):
        compute.bundle_source(source, {}, 1000, manifest=manifest)


def test_manifest_rejects_symlink_even_with_matching_hash(snapshot):
    source, manifest = snapshot
    (source / "main.py").rename(source / "original.bin")
    (source / "main.py").symlink_to(source / "original.bin")
    with pytest.raises(compute.ComputeError, match="Symlinks"):
        compute.bundle_source(source, {}, 1000, manifest=manifest)


def test_manifest_cannot_silently_omit_file_removed_during_packaging(snapshot, monkeypatch):
    source, manifest = snapshot
    second = source / "second.py"
    second.write_bytes(b"pass\n")
    manifest["files"]["second.py"] = hashlib.sha256(second.read_bytes()).hexdigest()
    read_bytes = Path.read_bytes

    def read_and_remove(path):
        data = read_bytes(path)
        if path == source / "main.py":
            second.unlink()
        return data

    monkeypatch.setattr(Path, "read_bytes", read_and_remove)
    with pytest.raises(compute.ComputeError, match="changed while packaging"):
        compute.bundle_source(source, {}, 1000, manifest=manifest)


def test_manifest_keeps_secret_scan_and_size_cap(snapshot):
    source, manifest = snapshot
    with pytest.raises(compute.ComputeError):
        compute.bundle_source(source, {"HF_TOKEN": "print('fixture')"}, 1000, manifest=manifest)
    with pytest.raises(compute.ComputeError, match="source_limit_bytes"):
        compute.bundle_source(source, {}, 1, manifest=manifest)


@pytest.fixture
def submission(snapshot, tmp_path, monkeypatch):
    source, manifest = snapshot
    controller = compute.Controller(directory=DIRECTORY, state_dir=tmp_path / "state",
                                    secrets={"HF_TOKEN": "fake-test-token"}, api=object())
    controller.save({"phase": "ready", "session_id": "fixture", "instance_id": 42,
                     "storage": controller.select_storage("ap"), "runs": [],
                     "deadline_epoch": time.time() + 10000, "upload_gb": 1,
                     "image": "fixture-image", "offer": {"machine_id": 7},
                     "remote_tools": "/fake/tools"})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = tmp_path / "config.json"
    config.write_text("{}")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Prospective fixture only.\n")
    args = SimpleNamespace(source=source, source_manifest=manifest_path, config=config,
                           protocol=protocol, max_seconds=1, command=["python", "main.py"],
                           seed=[0], run_id="fixture-run")
    monkeypatch.setattr(controller, "instance", lambda state: {"machine_id": 7})
    monkeypatch.setattr(compute.subprocess, "check_output", lambda *a, **k: "a" * 40)
    return controller, args, manifest


@pytest.mark.parametrize("key", ["instance_id", "session_id"])
def test_run_rejects_other_rental_manifest_before_remote_calls(submission, monkeypatch, key):
    controller, args, manifest = submission
    manifest[key] = "different"
    args.source_manifest.write_text(json.dumps(manifest))
    monkeypatch.setattr(controller, "worker", lambda *a, **k: pytest.fail("remote call"))
    with pytest.raises(compute.ComputeError, match="different instance or session"):
        controller.run(args)
    assert controller.state()["runs"] == []


def test_run_preserves_manifest_and_exact_manifest_bytes_hash(submission, monkeypatch):
    controller, args, manifest = submission
    launches = []

    def worker(state, action, **kwargs):
        if action == "status":
            return {"runs": []}
        assert action == "launch"
        launches.append(kwargs)
        return {"launched": True}

    monkeypatch.setattr(controller, "worker", worker)
    assert controller.run(args) == {"launched": True}
    spec = launches[0]["spec"]
    assert spec["source_manifest"] == manifest
    assert spec["source_manifest_sha256"] == hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()
    assert spec["source_files"] == ["main.py"]
    assert controller.state()["runs"] == ["fixture-run"]


def test_cli_accepts_explicit_manifest_with_named_rental():
    args = compute.parser().parse_args([
        "--session", "ap-cube", "run", "--source", "snapshot", "--source-manifest", "snapshot.json",
        "--config", "config.json", "--protocol", "protocol.md", "--seed", "0",
        "--max-seconds", "1", "--", "python", "main.py",
    ])
    assert args.session == "ap-cube" and args.source_manifest == "snapshot.json"
