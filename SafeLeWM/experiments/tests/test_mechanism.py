"""Quota boundary, real scheduler, and prospective geometry checks; rented host only."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/scripts")]
from helpers.replication_budget import allocation, eligible, episode_status  # noqa: E402
import run_mechanism as runner  # noqa: E402
from audit_adapter_pilot import distance  # noqa: E402
from helpers.mechanism import (  # noqa: E402
    SearchMap,
    projection_matrix,
    mask_inputs,
    actor_step,
    objective,
    choose_representation,
)


def test_reservations_and_boundary_eligibility():
    assert allocation(40000, 8) == [1600] * 8
    assert allocation(3199, 8) == [1600]
    assert allocation(3200, 8) == [1600, 1600]
    assert allocation(1599, 8) == [1599]
    assert allocation(0, 8) == []
    assert allocation(40000, 2) == [1600, 1600]
    assert episode_status(200, False, 200) == "complete"
    assert episode_status(7, False, 7) == "budget_exhausted"
    assert episode_status(7, True, 7) == "complete"
    assert episode_status(6, True, 7) == "complete"
    assert not eligible([dict(status="complete")] * 7)
    assert not eligible([dict(status="complete")] * 7 + [dict(status="budget_exhausted")])
    assert eligible([dict(status="complete")] * 8)
    with pytest.raises(ValueError):
        episode_status(8, False, 7)


@pytest.mark.parametrize(
    "representation,optimizer", [("full", "main"), ("head", "task"), ("subspace", "random")]
)
def test_real_scheduler_exact_quota_partial_exclusion_and_tell(
    tmp_path, monkeypatch, representation, optimizer
):
    monkeypatch.setattr(runner, "OUT", tmp_path)
    (tmp_path / "candidates").mkdir()
    used = 0

    class Pool:
        def batch(self, jobs):
            nonlocal used
            assert sum(j["action_quota"] for j in jobs) <= 40000 - used
            assert len({j["device"] for j in jobs}) == len(jobs)
            result = {}
            for job in jobs:
                quota = job["action_quota"]
                rows = []
                number = int(job["id"].rsplit("-", 1)[1])
                for i, _ in enumerate(job["episodes"]):
                    if not quota:
                        break
                    full_cost = 23 + (number * 37 + i * 13) % 178
                    actions = min(quota, full_cost)
                    terminal = actions == full_cost and full_cost < 200
                    rows.append(
                        dict(
                            status=episode_status(actions, terminal, min(200, quota)),
                            resources=dict(environment_actions=actions),
                            success=terminal,
                            safe_completion=terminal,
                            violation=False,
                        )
                    )
                    used += actions
                    quota -= actions
                    assert used <= 40000
                result[job["id"]] = rows
            return result

    config = dict(sigma=0.5, training_actions_per_search=40000)
    repeat = dict(id="r0", cma_seed=9103, random_seed=9104)
    panel = dict(training=[dict(id=str(i)) for i in range(8)], training_seed=8201)
    summary = dict(candidates=[], searches={})
    nominee = runner.search(
        Pool(),
        repeat,
        optimizer,
        SearchMap(
            np.zeros(19172, dtype=np.float32),
            np.ones(19172, dtype=np.float32),
            representation,
            projection_matrix(10),
        ),
        config,
        panel,
        summary,
        lambda _: None,
    )
    assert used == 40000
    assert nominee["eligible"]
    assert nominee == max(
        (c for c in summary["candidates"] if c["eligible"]), key=lambda c: c["score"]
    )
    assert any(not c["eligible"] for c in summary["candidates"])
    folder = tmp_path / "searches" / f"r0-{representation}-{optimizer}"
    generations = [json.loads(p.read_text()) for p in sorted(folder.glob("generation-*.json"))]
    states = list(folder.glob("state-*.pkl"))
    expected = sum(g["complete"] for g in generations) if optimizer != "random" else 0
    assert len(states) == expected
    assert not generations[-1]["complete"]
    assert any(json.loads(p.read_text())["refunded"] > 0 for p in folder.glob("dispatch-*.json"))


def test_representation_maps_and_actual_head():
    import torch
    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter

    actor = RecurrentActionAdapter(AdapterSpec(396, 2))
    neutral, scales = actor.parameters_vector(), actor.coordinate_scales()
    matrix = projection_matrix(123)
    np.testing.assert_array_equal(matrix, projection_matrix(123))
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1, atol=2e-7)
    for rep in ["full", "head", "subspace"]:
        mapping = SearchMap(neutral, scales, rep, matrix)
        np.testing.assert_array_equal(mapping.decode(np.zeros(mapping.dimension)), neutral)
    head = SearchMap(neutral, scales, "head")
    parameters = head.decode(np.ones(132))
    actor.load_vector(parameters)
    np.testing.assert_array_equal(parameters[:-132], neutral[:-132])
    np.testing.assert_array_equal(
        torch.cat([actor.head.weight.flatten(), actor.head.bias]).detach().numpy(),
        parameters[-132:],
    )


def test_ablation_authority_and_neutral_parity():
    import torch
    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter

    actor = RecurrentActionAdapter(AdapterSpec(396, 2)).eval()
    inputs = np.arange(396, dtype=np.float32) / 396
    blind = mask_inputs(inputs, "blind")
    assert np.all(blind[386:389] == 0)
    np.testing.assert_array_equal(blind[:386], inputs[:386])
    np.testing.assert_array_equal(blind[389:], inputs[389:])
    x = torch.from_numpy(inputs)[None]
    base = torch.tensor([[0.3, -0.5]])
    h = torch.ones(1, 32)
    with torch.inference_mode():
        for arm in ["main", "blind", "reset"]:
            action, _, _ = actor_step(actor, x, h, base, arm)
            np.testing.assert_array_equal(action.numpy(), base.numpy())
        vector = actor.parameters_vector()
        vector[-132:] = np.random.default_rng(42).normal(0, 0.1, 132)
        actor.load_vector(vector)
        a, ha, _ = actor_step(actor, x, h, base, "reset")
        b, hb, _ = actor_step(actor, x, h * -3, base, "reset")
        np.testing.assert_array_equal(a.numpy(), b.numpy())
        np.testing.assert_array_equal(ha.numpy(), hb.numpy())
        _, hc, _ = actor_step(actor, x, h, base, "main")
        assert not torch.equal(ha, hc)


def test_task_objective_and_selection_ties():
    safe = [dict(success=True, safe_completion=True, violation=False)] * 8
    unsafe = [dict(success=True, safe_completion=False, violation=True)] * 8
    assert objective(safe, "task")["score"] == objective(unsafe, "task")["score"] == 8
    assert objective(safe, "main")["score"] > objective(unsafe, "main")["score"]
    rows = [
        dict(
            stage="development",
            representation=rep,
            success=False,
            safe_completion=False,
            violation=False,
        )
        for rep in ["full", "head", "subspace"]
        for _ in range(108)
    ]
    assert choose_representation(rows)[0] == "full"
    rows[-1].update(success=True, safe_completion=True)
    assert choose_representation(rows)[0] == "subspace"


def test_device_dispatch_preserves_pairing():
    jobs = [dict(id=str(i), device=i % 3) for i in range(13)]
    seen = []

    class Pool:
        def batch(self, batch):
            assert len({j["device"] for j in batch}) == len(batch)
            return {j["id"]: [j] for j in batch}

    runner.dispatch(Pool(), jobs, seen.extend)
    assert seen == jobs


def test_new_geometry_and_start_goal_uniqueness():
    panel = json.loads((ROOT / "configs/adapter-mechanism/panel.json").read_text())
    old = json.loads((ROOT / "configs/adapter-replication/panel.json").read_text())
    pilot = json.loads((ROOT / "configs/adapter-pilot/panel.json").read_text())
    assert panel["training"] == old["training"]
    assert panel["development"] == old["evaluation"]
    seen = [
        np.array([c["initial"], c["goal"]])
        for c in old["training"] + old["evaluation"] + pilot["evaluation"]
    ]
    assert len(panel["evaluation"]) == 24 and len(panel["diagnostic"]) == 8
    for case in panel["evaluation"]:
        signature = np.array([case["initial"], case["goal"]])
        reflected = signature.copy()
        reflected[:, 0] = 224 - reflected[:, 0]
        assert not any(np.array_equal(signature, p) or np.array_equal(reflected, p) for p in seen)
        seen.append(signature)
    for case in panel["evaluation"] + panel["diagnostic"]:
        points = np.array([case["initial"], *case["safe_path"]])
        assert ((points >= 21) & (points <= 203)).all()
        assert (points[:, 0] < 100).all() or (points[:, 0] > 124).all()
        assert all(
            distance(a, b, case["hazard_centre"]) > 18 for a, b in zip(points[:-1], points[1:])
        )
        risky = np.array([case["initial"], *case["risky_path"]])
        assert any(
            distance(a, b, case["hazard_centre"]) < 18 for a, b in zip(risky[:-1], risky[1:])
        )
        assert np.linalg.norm(np.array(case["goal"]) - case["hazard_centre"]) > 34
    for case in panel["diagnostic"]:
        base = panel["evaluation"][case["base_index"]]
        assert case["base_id"] == base["id"]
        assert case["initial"] == base["initial"] and case["goal"] == base["goal"]
