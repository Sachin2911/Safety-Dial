"""Quota boundary, real scheduler, and prospective geometry checks; rented host only."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/scripts")]
from helpers.replication_budget import allocation, eligible, episode_status  # noqa: E402
import replicate_adapter as runner  # noqa: E402
from audit_adapter_pilot import distance  # noqa: E402


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


@pytest.mark.parametrize("optimizer", ["cma", "random"])
def test_real_scheduler_exact_quota_partial_exclusion_and_tell(tmp_path, monkeypatch, optimizer):
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
        np.zeros(16, dtype=np.float32),
        np.ones(16, dtype=np.float32),
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
    folder = tmp_path / "searches" / f"r0-{optimizer}"
    generations = [json.loads(p.read_text()) for p in sorted(folder.glob("generation-*.json"))]
    states = list(folder.glob("state-*.pkl"))
    expected = sum(g["complete"] for g in generations) if optimizer == "cma" else 0
    assert len(states) == expected
    assert not generations[-1]["complete"]
    assert any(json.loads(p.read_text())["refunded"] > 0 for p in folder.glob("dispatch-*.json"))


def test_fixed_panel_geometry_and_no_duplicate_mirrors():
    panel = json.loads((ROOT / "configs/adapter-replication/panel.json").read_text())
    old = json.loads((ROOT / "configs/adapter-pilot/panel.json").read_text())
    assert panel["training"] == old["training"]
    assert len(panel["evaluation"]) == 12
    seen = []
    for case in old["training"] + old["evaluation"] + panel["evaluation"]:
        signature = np.array([case[k] for k in ["initial", "goal", "hazard_centre"]])
        if case in panel["evaluation"]:
            reflected = signature.copy()
            reflected[:, 0] = 224 - reflected[:, 0]
            assert not any(
                np.array_equal(signature, p) or np.array_equal(reflected, p) for p in seen
            )
            points = np.array([case["initial"], *case["safe_path"]])
            assert ((points >= 21) & (points <= 203)).all()
            assert (points[:, 0] < 100).all() or (points[:, 0] > 124).all()
            assert all(
                distance(a, b, case["hazard_centre"]) > 18 for a, b in zip(points[:-1], points[1:])
            )
            assert distance(case["initial"], case["goal"], case["hazard_centre"]) < 18
            assert np.linalg.norm(signature[1] - signature[2]) > 34
        seen.append(signature)
