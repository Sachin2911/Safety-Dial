"""Coordinator routing and the serialized per-condition learner sequence, without a host."""

import importlib.util
from pathlib import Path

import pytest

from helpers.dsrl11_protocol import CONDITIONS, all_jobs, condition_order, frozen_jobs

ROOT = Path(__file__).resolve().parents[2]


def coordinator_module():
    spec = importlib.util.spec_from_file_location(
        "run_dsrl11", ROOT / "experiments/scripts/run_dsrl11.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = coordinator_module()


class Queue(MODULE.Coordinator):
    """The real coordinator methods over in-memory state and a scripted learner."""

    def __init__(self, utd=20):
        self.plan = {"jobs": frozen_jobs(), "condition_order": condition_order()}
        self.state = {"learner_ops": {}, "published": {}, "exports": {}, "episodes": {}}
        self.utd, self.requests, self.replies = utd, [], {}
        self.headroom = []

    def save(self):
        pass

    def valid(self, job_id):
        return self.state["episodes"].get(job_id, {}).get("valid") is True

    def descriptor(self, job_id):
        return {"job_id": job_id}

    def selected_utd(self):
        return self.utd

    def require_headroom(self, kind, amount, *, phase=None):
        self.headroom.append((kind, amount, phase))

    def begin_http(self, key, route, body, port=MODULE.LEARNER_PORT):
        self.requests.append((key, route, body))

    def poll_http(self, key):
        return self.replies.pop(key, None)


def finish(queue, key, **result):
    queue.replies[key] = result
    queue.learner_step()


def test_routing_sends_training_actors_to_the_learner_and_evaluation_to_cpu():
    queue = Queue()
    jobs = all_jobs()
    uniform = queue.routing(jobs["T_r0_none_0"], 1)
    assert uniform["residual_source"] == "uniform" and "actor_url" not in uniform
    assert uniform["uniform_seed"] == jobs["T_r0_none_0"]["uniform_seed"]
    zero = queue.routing(jobs["B03"], 2)
    assert zero["attempt_id"] == "2" and set(zero) == {
        "mode",
        "job_id",
        "attempt_id",
        "purpose",
        "residual_source",
        "epsilon_seed",
    }
    training = queue.routing(jobs["T_r1_prior_4"], 1)
    assert training["actor_url"] == f"http://127.0.0.1:{MODULE.LEARNER_PORT}"
    assert training["actor_version"] == 0 and training["actor_condition"] == "r1_prior"
    for condition in CONDITIONS:
        evaluation = queue.routing(jobs[f"V2_{condition}"], 1)
        port = MODULE.EVALUATION_ACTOR_PORTS[condition]
        assert evaluation["actor_url"] == f"http://127.0.0.1:{port}"
        assert evaluation["actor_version"] == 2
    assert queue.routing(jobs["V2_native"], 1)["residual_source"] == "zero"


def test_money_envelope_is_the_shared_task11_limits():
    from helpers.dsrl11_infra import LIMITS

    assert MODULE.MONEY == LIMITS and LIMITS["total_usd"] == 18.0


def complete(queue, condition, phase):
    for job in frozen_jobs()["online_jobs"]:
        if job["condition"] == condition and job["phase"] == phase:
            queue.state["episodes"][job["job_id"]] = {"valid": True}


def test_learner_sequence_is_serialized_and_follows_the_frozen_phases():
    queue = Queue(utd=20)
    first, second = condition_order()[:2]
    queue.learner_step()
    assert queue.requests == []  # Nothing is ready before a whole round is valid.
    complete(queue, second, "warmup")
    complete(queue, first, "warmup")
    queue.learner_step()
    key, route, body = queue.requests[-1]
    # The frozen condition order decides which ready condition the learner serves first.
    assert (key, route) == (f"append-{first}-warmup", "/append")
    assert [d["job_id"] for d in body["episodes"]] == [f"T_{first}_{k}" for k in range(3)]
    queue.learner_step()
    assert len(queue.requests) == 1  # One active learner request at a time.
    finish(queue, f"append-{first}-warmup", new_transitions=750)
    assert queue.requests[-1][0] == f"append-{second}-warmup"
    finish(queue, f"append-{second}-warmup", new_transitions=750)
    complete(queue, first, "batch_a")
    queue.learner_step()
    finish(queue, f"append-{first}-batch_a", new_transitions=612)
    key, route, body = queue.requests[-1]
    assert route == "/train" and body["expected_new_transitions"] == 612
    assert queue.headroom[-1] == ("learner_updates", 15_000, "learning")
    finish(queue, key, updates=12_240)
    assert queue.requests[-1][1] == "/publish"
    finish(queue, f"publish-{first}-batch_a", actor_version=1)
    assert queue.state["published"][first] == 1
    assert queue.requests[-1] == (
        f"save-{first}-batch_a",
        "/save",
        {"name": f"{first}-batch_a"},
    )
    finish(queue, f"save-{first}-batch_a", path="p", manifest_sha256="s")
    # Batch B cannot be appended before its own episodes are valid.
    assert len(queue.requests) == 6
    complete(queue, first, "batch_b")
    queue.learner_step()
    finish(queue, f"append-{first}-batch_b", new_transitions=750)
    finish(queue, f"train-{first}-batch_b", updates=15_000)
    finish(queue, f"publish-{first}-batch_b", actor_version=2)
    finish(queue, f"save-{first}-batch_b", path="p", manifest_sha256="s")
    key, route, body = queue.requests[-1]
    assert route == "/export-actor" and body == {
        "condition": first,
        "name": f"{first}-final",
        "actor_version": 2,
    }
    finish(queue, key, actor_version=2, path="export", manifest_sha256="m")
    assert queue.state["exports"][first]["actor_version"] == 2
    routes = [route for key, route, _ in queue.requests]
    assert routes.count("/train") == 2


@pytest.mark.parametrize("utd", [5, 20])
def test_train_headroom_reserves_a_full_round_at_the_selected_rate(utd):
    queue = Queue(utd=utd)
    condition = condition_order()[0]
    operations = queue.condition_operations(condition)
    assert [op["key"] for op in operations] == [
        f"append-{condition}-warmup",
        f"append-{condition}-batch_a",
        f"train-{condition}-batch_a",
        f"publish-{condition}-batch_a",
        f"save-{condition}-batch_a",
        f"append-{condition}-batch_b",
        f"train-{condition}-batch_b",
        f"publish-{condition}-batch_b",
        f"save-{condition}-batch_b",
        f"export-{condition}-final",
    ]
    assert {op["updates"] for op in operations if op["route"] == "/train"} == {750 * utd}
