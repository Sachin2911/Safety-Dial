"""Model-free authority probes exercise recovery without spending model calls."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.dsrl06_protocol import decoder_vectors
from helpers.dsrl06_server import PolicyBridge, QueryLedger

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "probe_dsrl06", ROOT / "experiments/scripts/probe_dsrl06.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class Backend:
    def seed(self, key):
        return int(key)

    def pack(self, key):
        return np.array([key, 0], np.uint32)

    def unpack(self, value):
        return int(value[0])

    def split(self, key):
        return key + 1, key + 100

    def normal(self, key, shape):
        return np.random.default_rng(key).standard_normal(shape).astype(np.float32)


class Policy:
    def __init__(self):
        self._rng = 0
        self._model = SimpleNamespace(action_horizon=16, action_dim=32)
        self._sample_kwargs = {}
        self.calls = 0

    def infer(self, observation, noise=None):
        self.calls += 1
        self._rng, sample = Backend().split(self._rng)
        if noise is None:
            noise = Backend().normal(sample, (16, 32))
        return {"actions": np.array(noise[:, :8], copy=True)}


def observation():
    return {
        "observation/image_left": np.zeros((256, 256, 3), np.uint8),
        "observation/wrist_image": np.ones((256, 256, 3), np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "Close and carry.",
    }


def setup(tmp_path):
    ledger = QueryLedger(tmp_path / "queries.sqlite")
    policy = Policy()
    server = PolicyBridge(policy, tmp_path / "captures", ledger, backend=Backend())
    return ledger, policy, server


def runner(tmp_path, ledger, transport, freeze=None, output="result.json"):
    return probe.ProbeRunner(
        ledger=ledger,
        record_root=tmp_path / "captures",
        output=tmp_path / output,
        freeze=freeze or probe.startup_freeze("localhost", 8000, 3, "server0"),
        transport=transport,
        poll_seconds=0,
    )


def test_startup_has_three_charged_queries_and_reuses_them_on_resume(tmp_path):
    ledger, policy, server = setup(tmp_path)
    first = runner(tmp_path, ledger, server.infer)
    report = probe.run_startup(first)
    assert report["valid"] and report["query_counts"]["reserved"] == 3
    assert report["query_counts"]["client_confirmed"] == 3
    first.close()
    resumed = runner(tmp_path, ledger, lambda _: pytest.fail("Resent a completed startup"))
    assert probe.run_startup(resumed)["query_counts"]["reserved"] == 3
    assert policy.calls == 3 and resumed.sends_this_invocation == 0
    resumed.close()
    with pytest.raises(ValueError, match="one to three"):
        probe.startup_freeze("localhost", 8000, 4, "bad")


def test_response_lost_after_model_return_recovers_artifact_without_resend(tmp_path):
    ledger, policy, server = setup(tmp_path)

    def lost(request):
        server.infer(request)
        raise TimeoutError("Response was lost")

    run = runner(tmp_path, ledger, lost)
    arrays, reference = run.query(0, observation(), mode="native")
    assert arrays["actions"].shape == (16, 8)
    assert reference["recovered_without_rpc_response"] and not reference["rpc_confirmed"]
    assert run.counts()["reserved"] == policy.calls == 1
    run.close()


@pytest.mark.parametrize("reserve", [False, True])
def test_unknown_or_reserved_send_never_repeats_even_with_new_output_path(tmp_path, reserve):
    ledger, policy, _ = setup(tmp_path)

    def unknown(request):
        if reserve:
            ledger.reserve(request["_dsrl06"])
        raise TimeoutError("Outcome unknown")

    first = runner(tmp_path, ledger, unknown)
    with pytest.raises(probe.PendingQuery, match="without resending"):
        first.query(0, observation(), mode="native")
    first.close()
    resumed = runner(
        tmp_path, ledger, lambda _: pytest.fail("Resent an uncertain request"), output="other.json"
    )
    with pytest.raises(probe.PendingQuery):
        resumed.query(0, observation(), mode="native")
    assert policy.calls == 0 and resumed.sends_this_invocation == 0
    resumed.close()


def test_reserved_request_can_finish_while_runner_polls_without_resending(tmp_path):
    ledger, _, server = setup(tmp_path)
    delayed = []

    def transport(request):
        delayed.append(request)
        raise TimeoutError("Lost response while server is busy")

    run = runner(tmp_path, ledger, transport)
    run.poll_seconds = 2
    ticks = iter([0, 0, 1, 1])
    run.clock = lambda: next(ticks)
    run.sleep = lambda _: server.infer(delayed[0])
    _, reference = run.query(0, observation(), mode="native")
    assert reference["recovered_without_rpc_response"]
    assert run.counts()["reserved"] == 1 and run.sends_this_invocation == 1
    run.close()


def test_failed_request_invalidates_without_using_a_replacement(tmp_path):
    ledger, _, server = setup(tmp_path)

    def fail(request):
        request["_dsrl06"]["noise"] = np.full((16, 32), 2, np.float32)
        return server.infer(request)

    run = runner(tmp_path, ledger, fail)
    with pytest.raises(probe.FailedQuery, match="failed"):
        run.query(0, observation(), mode="explicit", noise=np.zeros((16, 32), np.float32))
    assert run.counts()["reserved"] == run.counts()["failed"] == 1
    run.close()


def frozen_panel(tmp_path, server):
    descriptors = []
    for index, task in enumerate([1, 1, 4, 4, 10, 10, 11, 11]):
        result = server.infer(
            {
                **observation(),
                "_dsrl06": {
                    "job_id": f"E{index // 2}",
                    "query_index": index % 2,
                    "step": (index % 2) * 8,
                    "mode": "native",
                    "purpose": "policy",
                    "packet_id": f"factual:{index}",
                },
            }
        )
        path = result["_dsrl06"]["record_path"]
        descriptors.append(
            {
                "task": task,
                "snapshot_id": f"s{index}",
                "record_path": path,
                "sha256": probe.sha256(path),
            }
        )
    panel = tmp_path / "panel.json"
    probe.write_json(panel, descriptors)
    return panel


def test_authority_exact_128_original_replays_and_frozen_candidates_then_resume(tmp_path):
    ledger, policy, server = setup(tmp_path)
    panel = frozen_panel(tmp_path, server)
    candidate_path = ROOT / "outputs/dsrl06-preflight/decoder-probe-vectors.npy"
    freeze, snapshots, vectors, labels = probe.authority_inputs(
        panel, ROOT / "progress/receipts/06-dsrl-decoder-probe.json", candidate_path
    )
    run = runner(tmp_path, ledger, server.infer, freeze)
    report = probe.run_authority(run, snapshots, vectors, labels)
    assert report["gate"]["valid"] and report["gate"]["admit"]
    assert report["query_counts"]["reserved"] == report["query_counts"]["model_returned"] == 128
    assert len(report["cases"]) == 8 and all(len(x["queries"]) == 16 for x in report["cases"])
    assert policy.calls == 136  # Eight factual records plus exactly 128 probes.
    assert [x["mode"] for x in report["cases"][0]["queries"]] == ["native_replay"] * 2 + [
        "explicit"
    ] * 14
    run.close()
    resumed = runner(tmp_path, ledger, lambda _: pytest.fail("Repeated a frozen probe"), freeze)
    assert probe.run_authority(resumed, snapshots, vectors, labels)["gate"] == report["gate"]
    assert policy.calls == 136 and resumed.sends_this_invocation == 0
    resumed.close()


def test_changed_spec_candidate_or_snapshot_hash_is_rejected_before_calls(tmp_path):
    ledger, policy, server = setup(tmp_path)
    panel = frozen_panel(tmp_path, server)
    spec_path = ROOT / "progress/receipts/06-dsrl-decoder-probe.json"
    candidate_path = ROOT / "outputs/dsrl06-preflight/decoder-probe-vectors.npy"
    spec = json.loads(spec_path.read_text())
    spec["minimum_raw_open"] = 0.01001
    changed = tmp_path / "changed.json"
    probe.write_json(changed, spec)
    with pytest.raises(ValueError, match="specification differs"):
        probe.authority_inputs(panel, changed, candidate_path)
    descriptors = json.loads(panel.read_text())
    descriptors[0]["sha256"] = "0" * 64
    probe.write_json(panel, descriptors)
    with pytest.raises(ValueError, match="hash changed"):
        probe.authority_inputs(panel, spec_path, candidate_path)
    assert policy.calls == 8
    # The tensor hash is independent of the .npy container hash.
    assert (
        hashlib.sha256(decoder_vectors()[1].tobytes()).hexdigest()
        == spec["candidate_float32_bytes_sha256"]
    )
