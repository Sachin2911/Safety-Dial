"""Schedule, safety dominance and crash recovery checks; rented hosts only."""

import copy
import json

import numpy as np
import pytest

from helpers.lewm_proposal_pilot import ProposalPilot, panel_fitness


def row(success=False, unsafe=False, progress=0.0):
    return {
        "safe_completion": success,
        "margin_breach": unsafe,
        "arena_margin_breach": False,
        "unfinished": not success,
        "actions": 30,
        "pusher_path_length_px": 4,
        "safe_progress": progress,
        "error": None,
    }


def design():
    return {
        "search": {"seed": 22, "population": 4, "sigma": 0.1, "order_seed": 77, "generations": 2},
        "training": {"cases": ["island", "bar"], "seed_start": 100},
        "selection": {"cases": [["island", 200], ["bar", 201]]},
        "validation": {"cases": [["gate", 300], ["bar", 301]]},
        "maximum_valid_episodes": 32,
    }


def test_completion_and_safety_dominate_progress():
    assert panel_fitness([row(True, progress=1), row()]) > panel_fitness([row(progress=1)] * 2)
    assert panel_fitness([row(unsafe=True)]) < panel_fitness([row()])
    with pytest.raises(ValueError):
        panel_fitness([row(progress=1.1), row(progress=-0.1)])
    invalid = row()
    invalid["error"] = "engineering failure"
    with pytest.raises(ValueError):
        panel_fitness([invalid])


def test_resume_retains_candidates_and_does_not_repeat_jobs(tmp_path):
    calls, interrupted = [], False

    def evaluator(key, parameters, case, seed):
        nonlocal interrupted
        if len(calls) == 7 and not interrupted:
            interrupted = True
            raise RuntimeError("synthetic process interruption")
        calls.append(key)
        np.random.seed(seed)  # Real simulator seeding must not affect CMA.
        return row(progress=0.2)

    d = design()
    pilot = ProposalPilot(d, np.zeros(8), tmp_path / "resume", evaluator)
    with pytest.raises(RuntimeError, match="synthetic"):
        pilot.run()
    pending = copy.deepcopy(pilot.es.pending)
    recovered = ProposalPilot(d, np.zeros(8), tmp_path / "resume", evaluator, resume=True)
    assert recovered.es.pending == pending
    state = recovered.run()
    assert len(calls) == len(set(calls)) == 32
    assert len(recovered.es.history) == 2
    assert state["selection"]["winner"] == "native"
    selected = json.loads((tmp_path / "resume/selection.json").read_text())
    assert selected["parameters"] is None

    uninterrupted = ProposalPilot(
        d, np.zeros(8), tmp_path / "fresh", lambda *args: row(progress=0.2)
    )
    uninterrupted.run()
    assert uninterrupted.es.history == recovered.es.history
    again = ProposalPilot(d, np.zeros(8), tmp_path / "resume", evaluator, resume=True)
    again.run()
    assert len(calls) == 32


def test_selection_is_frozen_before_validation(tmp_path):
    def evaluator(key, parameters, case, seed):
        if key.startswith("validation"):
            assert (tmp_path / "selection.json").exists()
        return row(progress=0.9 if key.startswith("selection_elite") else 0.1)

    pilot = ProposalPilot(design(), np.zeros(8), tmp_path, evaluator)
    state = pilot.run()
    assert state["selection"]["winner"] == "elite"
    assert len(state["selection"]["parameters"]) == 8
    changed = design()
    changed["search"]["sigma"] = 0.2
    with pytest.raises(ValueError, match="changed"):
        ProposalPilot(changed, np.zeros(8), tmp_path, evaluator, resume=True)
