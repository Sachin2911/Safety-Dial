"""Exercise complete optimizer scheduling against a synthetic categorical problem."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.lewm17_selector import probabilities


@pytest.mark.parametrize("method", ["random", "reinforce", "evolution"])
def test_optimizer_charges_all_anchors_training_and_validation(tmp_path, monkeypatch, method):
    spec = importlib.util.spec_from_file_location(
        "optimizer", "experiments/scripts/optimize_lewm17.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class FakeExecutor:
        def __init__(self, gpu, manifest, group, allowance):
            self.used, self.allowance = 0, allowance
            self.actual = {"environment_actions": 0, "predictor_steps": 0}

        def run(self, job):
            self.used += 1
            self.actual["environment_actions"] += 1
            self.actual["predictor_steps"] += 1
            x = np.array([1.0, 0.1, 0.1, 0.8, 0.0, job["margin"] / 6])
            p = probabilities(job["params"]["weights"], x)
            rng = np.random.default_rng(job["params"]["mode_seed"])
            mode = int(rng.choice(6, p=p)) if job["params"]["stochastic"] else int(p.argmax())
            calls.append((job["stage"], job["params"]["stochastic"]))
            row = {
                "requested_safe_completion": mode == 0,
                "J": 1 if mode == 0 else 50,
                "any_violation": False,
                "margin_breach": False,
                "counters": {"predictor_steps": 1},
            }
            return row, [{"mode": mode, "probabilities": p, "x": x}]

    monkeypatch.setattr(module, "Executor", FakeExecutor)
    monkeypatch.setattr(module, "OUT", tmp_path)
    monkeypatch.setattr(module, "ready", lambda: None)
    module.optimize(
        "tworoom",
        0,
        method,
        SimpleNamespace(gpu=0, manifest="synthetic"),
        {"rounds": 6, "validation_panel_sha256": "synthetic"},
    )
    assert len(calls) == 192
    assert calls.count(("A", False)) == 24
    assert calls.count(("T", True)) == 144
    assert calls.count(("V", False)) == 24
    import json

    selected = json.loads(
        (tmp_path / "optimizers" / f"tworoom-{method}-r0" / "selected.json").read_text()
    )
    assert selected["episodes"] == 192 and selected["status"] == "complete"
    assert np.array(selected["weights"]).shape == (6, 6)
    assert selected["resources"]["environment_actions"] == 192
