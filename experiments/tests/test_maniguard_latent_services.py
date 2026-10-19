"""A family or mode switch must never interrupt a live physical worker."""

import json
import os

import pytest

from helpers import maniguard_latent_services as services
from helpers.maniguard_latent_queue import process_identity


def metadata(family="jar", audit=False):
    sources = json.loads((services.ROOT / "configs/maniguard/phase2/sources.json").read_text())
    spec = sources["policies"][family]
    return {"audit_service": audit, "serve_config": spec["config"],
            "checkpoint_revision": spec["revision"], "model_shape": [16, 32],
            "native_noise_steering": True}


def live_queue(tmp_path):
    (tmp_path / "queue.json").write_text(json.dumps({"attempts": [
        {"id": "physical-a0", "process_identity": process_identity(os.getpid())}]}))


def test_matching_service_can_be_observed_during_live_rollout(tmp_path, monkeypatch):
    live_queue(tmp_path)
    monkeypatch.setattr(services, "metadata", metadata)

    def forbidden(*args, **kwargs):
        pytest.fail("Observing the matching service must not invoke supervisor")

    monkeypatch.setattr(services, "supervisor", forbidden)
    services.ensure_policy(tmp_path, "jar", audit=False)
    with pytest.raises(RuntimeError, match="live"):
        services.ensure_policy(tmp_path, "jar", audit=True)
    with pytest.raises(RuntimeError, match="live"):
        services.ensure_policy(tmp_path, "clutter", audit=False)


def test_mode_revision_and_shape_are_part_of_serving_contract():
    good = metadata()
    assert services.policy_matches(good, "jar", False)
    assert not services.policy_matches(good, "jar", True)
    assert not services.policy_matches(good, "clutter", False)
    for key, value in [("checkpoint_revision", "unverified"), ("model_shape", [16, 8]),
                       ("native_noise_steering", False), ("audit_service", 0)]:
        assert not services.policy_matches({**good, key: value}, "jar", False)


def test_two_worker_reuse_verifies_device_before_any_live_replacement(tmp_path, monkeypatch):
    live_queue(tmp_path)

    def observed(port):
        return {**metadata("clutter"), "physical_gpu_uuid": services.GPU_UUIDS[port - 18000]}

    monkeypatch.setattr(services, "metadata", observed)
    services.ensure_worker_policies(tmp_path, "clutter")
    monkeypatch.setattr(services, "metadata", lambda port: observed(18000))
    with pytest.raises(RuntimeError, match="live"):
        services.ensure_worker_policies(tmp_path, "clutter")
