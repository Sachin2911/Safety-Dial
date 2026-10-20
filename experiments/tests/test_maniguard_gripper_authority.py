"""The proposed diagnostic must not silently repeat or expand model requests."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "gripper_authority", Path(__file__).parents[1] / "scripts/maniguard_gripper_authority.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_offsets_isolate_gripper_noise_and_preserve_native_reference():
    offsets = dict(MODULE.offsets())
    assert len(offsets) == 6 and offsets.pop("native") is None
    np.testing.assert_array_equal(offsets.pop("same-noise"), np.zeros((16, 32)))
    for label, offset in offsets.items():
        assert offset.shape == (16, 32)
        assert np.count_nonzero(offset) == 16
        assert not np.any(np.delete(offset, 7, axis=1))
        expected = float(label.split("-s")[0][1:]) * int(label.split("-s")[1])
        np.testing.assert_array_equal(offset[:, 7], np.full(16, expected))


def test_lost_response_cannot_be_retried_and_full_allowance_cannot_expand(tmp_path):
    ledger = tmp_path / "requests.jsonl"
    MODULE.assert_admission(ledger, "o00-native")
    ledger.write_text(json.dumps({"id": "o00-native"}) + "\n")
    with pytest.raises(RuntimeError, match="unresolved"):
        MODULE.assert_admission(ledger, "o00-native")
    ledger.write_text("".join(json.dumps({"id": f"request-{i}"}) + "\n" for i in range(48)))
    with pytest.raises(RuntimeError, match="exhausted"):
        MODULE.assert_admission(ledger, "extra-request")


def test_pending_budget_authorization_prevents_client_creation(tmp_path, monkeypatch):
    def forbidden_client(**kwargs):
        pytest.fail("Pending authorization must be checked before any service connection")

    monkeypatch.setattr(MODULE, "StudyClient", forbidden_client)
    with pytest.raises(RuntimeError, match="authorization"):
        MODULE.run({"status": "prospective_pending_budget_authorization"},
                   "unused", "jar", tmp_path / "output", 18002)
    assert not (tmp_path / "output").exists()


def test_command_metric_uses_native_deadband():
    values = np.array([-.3, -.01, 0., .01, .01001, .3])
    np.testing.assert_array_equal(MODULE.command(values), [-1., -1., -1., -1., 1., 1.])


def test_different_outputs_and_concurrent_clients_share_one_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    protocol = {"admission_ledger": "canonical/requests.jsonl"}
    barrier = Barrier(2)

    def attempt(index):
        barrier.wait()
        try:
            MODULE.admit_request(protocol, "same-request", tmp_path / str(index), "jar", "hash")
        except RuntimeError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt, (0, 1))) == [False, True]
    assert len((tmp_path / protocol["admission_ledger"]).read_text().splitlines()) == 1


def test_interrupted_client_return_recovers_without_another_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    protocol = {"admission_ledger": "canonical/requests.jsonl",
                "server_ledgers": {"jar": "server.jsonl"}}
    output = tmp_path / "results"
    output.mkdir()
    MODULE.admit_request(protocol, "probe", output, "jar", "frozen-hash")
    actions = np.zeros((16, 8), np.float32)
    np.savez(output / "probe.npz", actions=actions)
    server = {"episode": str(output / "probe"), "status": "returned",
              "actions_sha256": MODULE.array_hash(actions)}
    (tmp_path / "server.jsonl").write_text(json.dumps(server) + "\n")
    (tmp_path / "canonical/jar-metadata.json").write_text("{}")
    monkeypatch.setattr(MODULE, "plan", lambda p: [{"id": "probe", "family": "jar"}])
    result = MODULE.reconcile(protocol, "probe", output, tmp_path / "server.jsonl")
    assert result["status"] == "client_artifact_recovered_without_inference"
    assert result["charged"] and not result["replay_allowed"]
    assert json.loads((output / "probe.json").read_text())["protocol_sha256"] == "frozen-hash"
    with pytest.raises(RuntimeError, match="unresolved"):
        MODULE.admit_request(protocol, "probe", tmp_path / "another-output", "jar", "hash")
