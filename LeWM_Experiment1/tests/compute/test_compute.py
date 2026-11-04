"""Control-plane tests with fake providers/processes; no simulator or rental is used."""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "computeConfigs/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "computeConfigs"))

import compute
from compute_common import (
    ApiError,
    ComputeError,
    atomic_json,
    load_secrets,
    locked,
    process_alive,
    process_identity,
    read_json,
    redact,
    reject_secrets,
)
import hf_store
import remote_worker
from run_io import mark_ready, publish_artifact, report_progress


def offer(**changes):
    row = {
        "id": 101,
        "machine_id": 500,
        "cpu_arch": "amd64",
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "verification": "verified",
        "rentable": True,
        "rented": False,
        "cuda_max_good": 12.9,
        "driver_version": "575.51.03",
        "cpu_cores_effective": 16,
        "cpu_ram": 64000,
        "disk_space": 500,
        "direct_port_count": 4,
        "reliability": 0.995,
        "inet_down": 1500,
        "inet_up": 500,
        "dph_total": 0.4,
        "inet_down_cost": 0.01,
        "inet_up_cost": 0.02,
    }
    return row | changes


def options(**changes):
    return argparse.Namespace(
        **(
            {
                "hours": 2,
                "max_hourly": 0.6,
                "download_gb": 10,
                "upload_gb": 2,
                "budget_usd": 2,
                "offer_id": None,
            }
            | changes
        )
    )


class FakeVast:
    def __init__(self):
        self.rows, self.calls = [], []
        self.offers = [offer()]
        self.fail_create = False

    def call(self, path, *, method="GET", payload=None, retry=False):
        self.calls.append((path, method, payload))
        if path == "/bundles/":
            assert "q" not in payload
            assert payload["allocated_storage"] == 200
            return {"offers": self.offers}
        if path == "/instances/":
            return {"instances": list(self.rows)}
        if path.startswith("/asks/"):
            self.rows.append({"id": 1001, "label": payload["label"], "actual_status": "running"})
            if self.fail_create:
                raise ApiError()
            return {"success": True, "new_contract": 1001}
        if method == "DELETE":
            selected = int(path.split("/")[2])
            self.rows = [row for row in self.rows if row["id"] != selected]
            return {"success": True}
        raise AssertionError((path, method))


@pytest.fixture
def controller(tmp_path, monkeypatch):
    api = FakeVast()
    c = compute.Controller(
        state_dir=tmp_path / "state",
        api=api,
        secrets={
            "VAST_API_KEY": "vast-test-secret",
            "GHCR_READ_TOKEN": "registry-test-secret",
            "HF_TOKEN": "hub-test-secret",
        },
    )
    monkeypatch.setattr(c, "credentials_preflight", lambda: None)

    def prepare(state):
        state["phase"] = "ready"
        state["remote_tools"] = "/workspace/LeWM_Experiment1/.control/tools/example"
        c.save(state)
        return {"instance_id": state["instance_id"], "phase": "ready"}

    monkeypatch.setattr(c, "prepare", prepare)
    return c


def test_offer_ranking_accounts_for_transfers_and_disk(controller):
    controller.api.offers = [offer(id=1, dph_total=0.2, inet_down_cost=1), offer(id=2)]
    result = controller.search(options())
    assert [item["offer_id"] for item in result] == [2]
    assert result[0]["estimated_session_usd"] == pytest.approx(0.8 + 21.28 * 0.01 + 2 * 0.02)


@pytest.mark.parametrize(
    "changes",
    [
        {"verification": "unverified"},
        {"gpu_name": "RTX 3090"},
        {"rented": True},
        {"cpu_ram": 32000},
        {"driver_version": "550.90.07"},
        {"inet_up": 20},
        {"reliability": 0.9},
        {"dph_total": float("nan")},
        {"inet_up_cost": None},
        {"end_date": 1},
        {"cpu_arch": "arm64"},
    ],
)
def test_offer_never_relaxes_filters(controller, changes):
    assert (
        compute.offer_summary(offer(**changes), controller.template, controller.policy, options())
        is None
    )


def test_launch_records_identity_and_never_persists_secrets(controller):
    assert controller.up(options())["instance_id"] == 1001
    state = controller.state()
    text = controller.state_path.read_text()
    assert state["phase"] == "ready" and state["label"] == controller.api.rows[0]["label"]
    assert all(secret not in text for secret in controller.secrets.values())
    mutation = next(call for call in controller.api.calls if call[1] == "PUT")
    assert "@sha256:" in mutation[2]["image"]
    assert mutation[2]["runtype"] == "ssh_direct"
    with pytest.raises(ComputeError, match="already recorded"):
        controller.up(options())
    assert sum(call[1] == "PUT" for call in controller.api.calls) == 1


@pytest.mark.parametrize("registered", [True, False])
def test_preflight_reads_vast_public_key_field(controller, tmp_path, monkeypatch, registered):
    key = tmp_path / "id_test"
    key.write_text("fake private key for orchestration test")
    Path(str(key) + ".pub").write_text("ssh-ed25519 test-public-key local-comment\n")
    controller.policy["ssh_key"] = str(key)
    controller.secrets.pop("GHCR_READ_TOKEN")

    def api(path):
        if path.startswith("/template/?"):
            return {"templates": [controller.template]}
        assert path == "/ssh/"
        public = "test-public-key" if registered else "unrelated-key"
        return [{"id": 1, "public_key": f"ssh-ed25519 {public} provider-comment"}]

    def no_cli(*args, **kwargs):
        raise FileNotFoundError("GitHub CLI unavailable in this fixture")

    monkeypatch.setattr(controller.api, "call", api)
    monkeypatch.setattr(compute.subprocess, "run", no_cli)
    expected = "Set GHCR_READ_TOKEN" if registered else "Register the configured SSH public key"
    with pytest.raises(ComputeError, match=expected):
        compute.Controller.credentials_preflight(controller)


def test_registry_prefers_explicit_token_without_reading_cli(controller, monkeypatch):
    monkeypatch.setattr(
        compute.subprocess, "run", lambda *a, **kw: pytest.fail("Unnecessary credential lookup")
    )
    assert controller.registry_token() == "registry-test-secret"


def test_registry_uses_cli_credential_in_memory_only(controller, monkeypatch):
    controller.secrets.pop("GHCR_READ_TOKEN")
    calls = []

    def cli(command, **kwargs):
        calls.append(command)
        assert kwargs["stdout"] == compute.subprocess.PIPE
        assert kwargs["stderr"] == compute.subprocess.PIPE
        return SimpleNamespace(stdout="cli-test-credential\n")

    monkeypatch.setattr(compute.subprocess, "run", cli)
    assert controller.registry_token() == "cli-test-credential"
    assert controller.registry_token() == "cli-test-credential"
    assert calls == [["gh", "auth", "token", "--hostname", "github.com"]]
    assert not controller.state_path.exists()


def test_uncertain_create_is_recovered_without_second_rental(controller):
    controller.api.fail_create = True
    with pytest.raises(ComputeError, match="not confirmed"):
        controller.up(options())
    assert controller.state()["instance_id"] is None
    with pytest.raises(ComputeError):
        controller.up(options())
    assert controller.recover()["instance_id"] == 1001
    assert sum(call[1] == "PUT" for call in controller.api.calls) == 1


def test_instance_label_mismatch_blocks_destruction(controller):
    controller.up(options())
    controller.api.rows[0]["label"] = "another-project"
    with pytest.raises(ComputeError, match="label differs"):
        controller.destroy()
    assert not any(call[1] == "DELETE" for call in controller.api.calls)


@pytest.mark.parametrize("active", ["supervisor_alive", "job_alive"])
def test_destroy_refuses_live_processes(controller, monkeypatch, active):
    controller.up(options())
    run = {"supervisor_alive": False, "job_alive": False, active: True}
    monkeypatch.setattr(controller, "worker", lambda *a, **kw: {"runs": [run]})
    with pytest.raises(ComputeError, match="job is active"):
        controller.destroy()
    assert len(controller.api.rows) == 1


def test_destroy_refuses_failed_backup(controller, monkeypatch):
    controller.up(options())
    monkeypatch.setattr(
        controller,
        "worker",
        lambda *a, **kw: {
            "runs": [{"supervisor_alive": False, "job_alive": False}],
            "ready_to_destroy": False,
        },
    )
    monkeypatch.setattr(controller, "sync", lambda: None)
    with pytest.raises(ComputeError, match="verification is incomplete"):
        controller.destroy()
    assert len(controller.api.rows) == 1


def test_destroy_verifies_manifest_before_deleting_exact_instance(controller, monkeypatch):
    controller.up(options())
    controller.api.rows.append({"id": 2002, "label": "unrelated-work"})
    order = []
    run = {"run_id": "a", "supervisor_alive": False, "job_alive": False, "ready_to_destroy": True}
    monkeypatch.setattr(
        controller,
        "worker",
        lambda *a, **kw: {
            "runs": [run],
            "ready_to_destroy": True,
        },
    )
    monkeypatch.setattr(controller, "sync", lambda: order.append("sync"))
    monkeypatch.setattr(controller, "verify_receipt", lambda item: order.append("verify"))
    assert controller.destroy() == {"instance_id": 1001, "removed": True}
    assert order == ["sync", "verify"]
    assert controller.api.rows == [{"id": 2002, "label": "unrelated-work"}]
    assert controller.state()["phase"] == "destroyed"


def test_boot_failure_without_submitted_jobs_can_be_destroyed(controller, monkeypatch):
    controller.up(options())
    state = controller.state()
    state["phase"] = "boot_failed"
    controller.save(state)
    monkeypatch.setattr(
        controller, "worker", lambda *a, **kw: pytest.fail("SSH should not be needed")
    )
    assert controller.destroy()["removed"]


def test_source_bundle_excludes_credentials_and_data(tmp_path, monkeypatch):
    project = tmp_path / "project"
    source = project / "src"
    source.mkdir(parents=True)
    (source / "main.py").write_text("print('hello')")
    (source / ".env").write_text("HF_TOKEN=secret")
    (source / "data").mkdir()
    (source / "data/sample.csv").write_text("not code")
    monkeypatch.setattr(compute, "PROJECT", project)
    monkeypatch.setattr(
        compute.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout=b"src/main.py\0src/.env\0src/data/sample.csv\0"),
    )
    data, files = compute.bundle_source(source, {}, 1024)
    assert files == ["main.py"]
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        assert archive.getnames() == ["main.py"]
    (source / "main.py").write_text("KNOWN_CREDENTIAL")
    with pytest.raises(ComputeError, match="credential"):
        compute.bundle_source(source, {"token": "KNOWN_CREDENTIAL"}, 1024)


def test_secret_loading_never_evaluates_shell(tmp_path, monkeypatch):
    for key in ("HF_TOKEN", "GHCR_READ_TOKEN", "VAST_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN='$(touch /tmp/should-not-run)'\nGHCR_READ_TOKEN=abc\n")
    values = load_secrets(tmp_path)
    assert values["HF_TOKEN"] == "$(touch /tmp/should-not-run)"
    assert redact("abc and hf_" + "A" * 24, ["abc"]) == "[redacted] and [redacted]"
    with pytest.raises(ComputeError):
        reject_secrets(b"value=abc", ["abc"])


def test_process_identity_and_lock_are_not_stale_pid_checks(tmp_path):
    identity = process_identity(os.getpid())
    assert process_alive(identity)
    assert not process_alive(identity | {"start": "different"})
    with locked(tmp_path / "lock"):
        with pytest.raises(ComputeError, match="Another operation"):
            with locked(tmp_path / "lock"):
                pytest.fail("Acquired a held lock")


def test_stop_never_signals_reused_pid(monkeypatch):
    monkeypatch.setattr(remote_worker, "process_alive", lambda identity: False)
    monkeypatch.setattr(remote_worker.os, "killpg", lambda *a: pytest.fail("Unsafe signal"))
    remote_worker.stop_group({"pid": 123, "start": "old", "boot": "old"})


@pytest.mark.parametrize(
    "name,kind", [("../escape.py", "file"), ("/tmp/escape.py", "file"), ("link.py", "symlink")]
)
def test_archive_extraction_rejects_unsafe_members(tmp_path, name, kind):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        member = tarfile.TarInfo(name)
        if kind == "symlink":
            member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
        archive.addfile(member, io.BytesIO())
    with pytest.raises(ComputeError):
        remote_worker.extract_source(data.getvalue(), tmp_path / "code", 1024)


def test_launch_duplicate_does_not_start_another_process(tmp_path, monkeypatch):
    spec = {"run_id": "same-run"}
    atomic_json(tmp_path / "runs/same-run/run.json", spec)
    monkeypatch.setattr(
        remote_worker.subprocess, "Popen", lambda *a, **kw: pytest.fail("Duplicate job")
    )
    assert remote_worker.launch(tmp_path, {"spec": spec})["run_id"] == "same-run"
    with pytest.raises(ComputeError, match="different launch"):
        remote_worker.launch(tmp_path, {"spec": spec | {"changed": True}})


def test_child_gets_timeout_and_run_paths_but_not_cloud_tokens(tmp_path, monkeypatch):
    atomic_json(
        tmp_path / "run.json",
        {"job_deadline_epoch": time.time() + 100, "seeds": [7], "command": ["python", "main.py"]},
    )
    atomic_json(tmp_path / ".control/status.json", {"supervisor": {"pid": 1}})
    monkeypatch.setattr(remote_worker, "process_alive", lambda value: True)
    monkeypatch.setattr(remote_worker.os, "chdir", lambda directory: None)
    monkeypatch.setenv("HF_TOKEN", "private-hf-token")
    monkeypatch.setenv("VAST_API_KEY", "private-vast-token")
    captured = {}
    monkeypatch.setattr(
        remote_worker.os,
        "execve",
        lambda path, argv, env: captured.update(path=path, argv=argv, env=env),
    )
    remote_worker.child_entry(tmp_path)
    assert captured["path"] == "/usr/bin/timeout"
    assert "--kill-after=10s" in captured["argv"]
    assert captured["env"]["LEWM_RUN_DIR"] == str(tmp_path)
    assert "HF_TOKEN" not in captured["env"] and "VAST_API_KEY" not in captured["env"]


class FakeHub:
    def __init__(self):
        self.remote, self.revisions = {}, []
        self.verify_calls = 0
        self.corrupt = False

    def __call__(self, storage, token):
        return self

    def commit(self, prefix, files, directory):
        for name in files:
            data = (directory / name).read_bytes()
            if name in self.remote and data != self.remote[name]:
                raise ComputeError("Remote collision")
            self.remote[name] = data
        revision = f"{len(self.revisions) + 1:040x}"
        self.revisions.append(revision)
        return revision

    def verify(self, prefix, files, revision):
        self.verify_calls += 1
        for name, expected in files.items():
            data = self.remote[name] + (b"corrupt" if self.corrupt else b"")
            if hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise ComputeError("Read-back mismatch")

    def check_revision(self, revision):
        assert revision in self.revisions


@pytest.fixture
def run_root(tmp_path):
    root = tmp_path / "run"
    atomic_json(
        root / "run.json",
        {"run_id": "run", "max_artifact_bytes": 1000000, "storage": {"path_prefix": "runs"}},
    )
    mark_ready(root, "run.json")
    return root


def test_only_closed_artifacts_upload_until_finalization(run_root):
    hub = FakeHub()
    publish_artifact("metrics/chunk-0001.jsonl", data='{"completed":1}\n', run_dir=run_root)
    (run_root / "metrics/still-writing.jsonl").write_text("unsealed")
    hf_store.sync_run(run_root, "test-token", store_factory=hub)
    assert "metrics/chunk-0001.jsonl" in hub.remote
    assert "metrics/still-writing.jsonl" not in hub.remote
    atomic_json(run_root / "result.json", {"status": "completed", "exit_code": 0})
    result = hf_store.sync_run(run_root, "test-token", final=True, store_factory=hub)
    assert result["verified"] and "metrics/still-writing.jsonl" in hub.remote
    assert hub.verify_calls == 1
    assert hf_store.sync_run(run_root, "test-token", final=True, store_factory=hub)["verified"]
    assert hub.verify_calls == 1  # Successful immutable closeout is not downloaded repeatedly.


def test_changed_published_file_blocks_upload(run_root):
    publish_artifact("artifacts/checkpoint.bin", data=b"original", run_dir=run_root)
    hf_store.sync_run(run_root, "test-token", store_factory=FakeHub())
    (run_root / "artifacts/checkpoint.bin").write_bytes(b"changed")
    with pytest.raises(ComputeError, match="changed"):
        hf_store.sync_run(run_root, "test-token", store_factory=FakeHub())


def test_artifact_publication_cannot_overwrite_prior_checkpoint(run_root):
    publish_artifact("artifacts/checkpoint.bin", data=b"first", run_dir=run_root)
    with pytest.raises(FileExistsError):
        publish_artifact("artifacts/checkpoint.bin", data=b"second", run_dir=run_root)
    assert (run_root / "artifacts/checkpoint.bin").read_bytes() == b"first"


def test_failed_readback_never_marks_run_ready(run_root):
    hub = FakeHub()
    hub.corrupt = True
    atomic_json(run_root / "result.json", {"status": "failed", "exit_code": 1})
    with pytest.raises(ComputeError, match="Read-back"):
        hf_store.sync_run(run_root, "test-token", final=True, store_factory=hub)
    assert read_json(run_root / ".control/upload.json")["verified"] is False
    assert remote_worker.run_status(run_root)["ready_to_destroy"] is False


def test_symlink_and_excess_output_are_not_silently_dropped(run_root):
    (run_root / "artifacts").mkdir()
    (run_root / "artifacts/link").symlink_to("/etc/passwd")
    with pytest.raises(ComputeError, match="symlink"):
        hf_store.inventory(run_root, True)
    (run_root / "artifacts/link").unlink()
    spec = read_json(run_root / "run.json")
    spec["max_artifact_bytes"] = 1
    atomic_json(run_root / "run.json", spec)
    mark_ready(run_root, "run.json")
    with pytest.raises(ComputeError, match="transfer allowance"):
        hf_store.sync_run(run_root, "test-token", store_factory=FakeHub())


def test_output_secret_is_refused_before_upload(run_root):
    hub = FakeHub()
    publish_artifact("logs/mistake.log", data="token=private-test-token", run_dir=run_root)
    with pytest.raises(ComputeError, match="credential"):
        hf_store.sync_run(run_root, "private-test-token", store_factory=hub)
    assert not hub.remote


def test_progress_and_rotated_logs_are_visible(run_root):
    report_progress(2, 10, run_dir=run_root)
    log = remote_worker.RunLog(run_root)
    log.write(b"hello\n")
    log.rotate()
    assert (run_root / "logs/stdout-000000.log").read_bytes() == b"hello\n"
    assert "logs/stdout-000000.log" in hf_store.inventory(run_root, False)
    assert remote_worker.run_status(run_root)["progress"]["completed"] == 2


def test_hub_matches_git_blobs_and_lfs_without_trusting_filenames(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"content")
    expected = {"bytes": 7, "sha256": hashlib.sha256(b"content").hexdigest()}
    git_oid = hashlib.sha1(b"blob 7\0content", usedforsecurity=False).hexdigest()
    assert hf_store.HubStore.matches(
        SimpleNamespace(size=7, lfs=None, blob_id=git_oid), path, expected
    )
    assert hf_store.HubStore.matches(
        SimpleNamespace(size=7, lfs={"sha256": expected["sha256"]}), path, expected
    )
    assert not hf_store.HubStore.matches(SimpleNamespace(size=8, lfs=None), path, expected)


def test_private_destination_is_checked_before_upload(monkeypatch):
    api = SimpleNamespace(repo_info=lambda *a, **kw: SimpleNamespace(private=False, sha="a" * 40))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda **kw: api))
    store = hf_store.HubStore(
        {
            "repo_id": "Sachioster/lewm-experiment1",
            "repo_type": "dataset",
            "private": True,
            "branch": "main",
        },
        "test-token",
    )
    with pytest.raises(ComputeError, match="public"):
        store.head()


def test_manifest_verification_failure_retains_rental(controller, monkeypatch):
    controller.up(options())
    monkeypatch.setattr(
        controller,
        "worker",
        lambda *a, **kw: {
            "runs": [{"supervisor_alive": False, "job_alive": False}],
            "ready_to_destroy": True,
        },
    )
    monkeypatch.setattr(controller, "sync", lambda: None)

    def refuse(run):
        raise ComputeError("Manifest mismatch")

    monkeypatch.setattr(controller, "verify_receipt", refuse)
    with pytest.raises(ComputeError, match="Manifest mismatch"):
        controller.destroy()
    assert not any(call[1] == "DELETE" for call in controller.api.calls)


@pytest.mark.parametrize("mode", ["completed", "stopped", "timed_out", "backup_failed"])
def test_supervisor_closes_logs_and_records_process_outcome(run_root, monkeypatch, mode):
    spec = read_json(run_root / "run.json") | {
        "job_deadline_epoch": 100,
        "sync_interval_seconds": 300,
    }
    atomic_json(run_root / "run.json", spec)
    clock = [0]
    monkeypatch.setattr(
        remote_worker,
        "time",
        SimpleNamespace(
            time=lambda: clock[0],
            monotonic=lambda: clock[0],
            sleep=lambda value: None,
        ),
    )
    monkeypatch.setattr(
        remote_worker,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "start": "test",
            "boot": "test",
        },
    )
    signals, uploads = [], []
    monkeypatch.setattr(remote_worker, "stop_group", lambda identity: signals.append(identity))
    monkeypatch.setattr(remote_worker.os, "killpg", lambda *a: None)

    class FakeProcess:
        pid = 999999

        def __init__(self, *args, **kwargs):
            self.stdout = io.BytesIO(b"job output\n")
            if mode == "stopped":
                (run_root / ".control/stop.request").touch()
            if mode == "timed_out":
                clock[0] = 101

        def poll(self):
            return None if mode in ("stopped", "timed_out") else 0

        def wait(self, timeout=None):
            return -15 if mode in ("stopped", "timed_out") else 0

    monkeypatch.setattr(remote_worker.subprocess, "Popen", FakeProcess)

    def upload(root, token, final=False):
        uploads.append(final)
        if final and mode == "backup_failed":
            raise ComputeError("HF unavailable")

    monkeypatch.setattr(remote_worker, "sync_run", upload)
    remote_worker.supervise(run_root, "test-token")
    result = read_json(run_root / "result.json")
    assert result["status"] == ("completed" if mode == "backup_failed" else mode)
    assert uploads == [False, True]
    assert (run_root / "logs/stdout-000000.log").read_bytes() == b"job output\n"
    assert read_json(run_root / ".control/status.json")["phase"] == (
        "backup_failed" if mode == "backup_failed" else "finished"
    )
    if mode in ("stopped", "timed_out"):
        assert signals


def test_launch_detaches_supervisor_and_preserves_exact_source(tmp_path, monkeypatch):
    source = io.BytesIO()
    with tarfile.open(fileobj=source, mode="w:gz") as archive:
        member = tarfile.TarInfo("entry.py")
        data = b"# transfer fixture only\n"
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    config, protocol = {"setting": 1}, "Infrastructure fixture; no experiment.\n"
    spec = {
        "run_id": "detached",
        "job_deadline_epoch": time.time() + 60,
        "source_limit_bytes": 1024,
        "source_sha256": hashlib.sha256(source.getvalue()).hexdigest(),
        "config_sha256": hashlib.sha256(
            (json.dumps(config, sort_keys=True, indent=2) + "\n").encode()
        ).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol.encode()).hexdigest(),
    }
    captured = {}

    class Input:
        def write(self, data):
            captured["input"] = data

        def close(self):
            pass

    def spawn(*args, **kwargs):
        captured.update(kwargs)
        atomic_json(tmp_path / "runs/detached/.control/status.json", {"supervisor": {"pid": 123}})
        return SimpleNamespace(stdin=Input(), poll=lambda: None)

    monkeypatch.setattr(remote_worker.subprocess, "Popen", spawn)
    monkeypatch.setattr(remote_worker, "process_alive", lambda identity: bool(identity))
    remote_worker.launch(
        tmp_path,
        {
            "spec": spec,
            "experiment_config": config,
            "protocol": protocol,
            "source": base64.b64encode(source.getvalue()).decode(),
            "token": "secret-test-token",
        },
    )
    assert captured["start_new_session"] and captured["close_fds"]
    assert captured["stdout"] == remote_worker.subprocess.DEVNULL
    assert json.loads(captured["input"]) == {"token": "secret-test-token"}
    root = tmp_path / "runs/detached"
    assert (root / ".control/code/entry.py").read_bytes() == b"# transfer fixture only\n"
    assert all(
        b"secret-test-token" not in path.read_bytes() for path in root.rglob("*") if path.is_file()
    )


def test_status_uses_one_ssh_call_and_does_not_rewrite_state(controller, monkeypatch):
    controller.up(options())
    controller.api.rows[0].update(
        public_ipaddr="192.0.2.1", ports={"22/tcp": [{"HostPort": "2222"}]}
    )
    original = controller.state_path.read_bytes()
    requests = []

    def ssh(state, endpoint, command, payload=None, timeout=45):
        requests.append(payload)
        return json.dumps({"runs": [], "ready_to_destroy": True})

    monkeypatch.setattr(controller, "ssh_call", ssh)
    monkeypatch.setattr(
        controller, "save", lambda value: pytest.fail("Read-only status wrote state")
    )
    assert controller.status()["ready_to_destroy"]
    assert len(requests) == 1 and requests[0]["action"] == "status"
    assert controller.state_path.read_bytes() == original


def test_prepare_records_actual_runtime_report(controller, monkeypatch):
    controller.up(options())
    state = controller.state()
    monkeypatch.setattr(controller, "connected", lambda state: {"host": "192.0.2.1", "port": 22})
    health = {"python": "3.11.14", "gpu": {"cuda_and_bfloat16": "passed"}}

    def ssh(state, endpoint, command, payload=None, timeout=45):
        if "--gpu" in command:
            return json.dumps(health)
        return json.dumps({"deployed": True})

    monkeypatch.setattr(controller, "ssh_call", ssh)
    compute.Controller.prepare(controller, state)
    assert controller.state()["readiness"] == health


def test_empty_rental_destruction_does_not_require_hf_credentials(controller, monkeypatch):
    controller.up(options())
    controller.secrets.pop("HF_TOKEN")
    monkeypatch.setattr(
        controller,
        "worker",
        lambda *a, **kw: {
            "runs": [],
            "ready_to_destroy": True,
        },
    )
    monkeypatch.setattr(controller, "sync", lambda: pytest.fail("No artifacts need syncing"))
    assert controller.destroy()["removed"]
