"""Dummy-evaluator optimizer/accounting checks, executed only on a rented GPU."""

import copy
import datetime as dt
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.lewm_campaign_training import (
    CampaignTraining,
    EpisodeOutcome,
    IncompleteJobError,
    lexicographic_episode_reward,
    panel_fitness,
)
from helpers.lewm_navigation_retrieval import RetrievalController

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("RUNPOD_POD_ID"),
    reason="All experiment software tests belong on a declared rented GPU",
)


def protocol(method="cma", *, budget=100, panels=4):
    return {
        "schema": "lewm-campaign-training-v1",
        "allowed_pod_ids": [os.environ.get("RUNPOD_POD_ID", "not-local")],
        "deadline_utc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat(),
        "method": method,
        "repeat_id": 0,
        "optimizer_seed": 72,
        "initial_parameters": RetrievalController.initial_parameters().tolist(),
        "training_panel": [{"case_id": "a", "seed": 101}, {"case_id": "b", "seed": 102}],
        "max_environment_actions": budget,
        "max_episode_actions": 3,
        "max_episode_starts": 100,
        "max_candidate_panels": panels,
        "population": 4,
        "sigma": 0.1,
        "parameter_bounds": [-6, 6],
        "source_ids": {"purpose": "synthetic evaluator; zero actual simulator actions"},
        "reinforce": {
            "learning_rate": 0.05,
            "baseline_decay": 0.9,
            "temperature": 1.0,
            "max_grad_norm": 10.0,
        },
    }


class DummyEvaluator:
    def __init__(self, *, fail_on=None, decisions=False, seed_noise=0):
        self.calls, self.fail_on, self.decisions, self.seed_noise = (
            [],
            fail_on,
            decisions,
            seed_noise,
        )

    def __call__(self, job, report_progress):
        assert job.folder == job.directory and job.seed == job.environment_seed
        assert job.parameters == job.weights
        self.calls.append(job.job_id)
        np.random.seed(job.seed + self.seed_noise)
        report_progress(0, {"environment_resets": 1, "model_query_attempts": 0})
        report_progress(1, {"environment_resets": 1, "model_query_attempts": 2})
        if self.fail_on == len(self.calls):
            raise RuntimeError("synthetic evaluator failure after a charged action")
        report_progress(2, {"environment_resets": 1, "model_query_attempts": 3})
        decisions = []
        if self.decisions and job.mode == "categorical":
            # A prescribed valid categorical trace tests the exact score-law gradient.
            features = np.zeros((2, 9), dtype=np.float32)
            features[0, 0], features[1, -1] = 1, 1
            decisions = [(features, np.ones(2, dtype=bool), 0)]
        return EpisodeOutcome(
            safe_completion=False,
            violation=False,
            unfinished=True,
            engaged=True,
            progress=0.2,
            environment_actions=2,
            resources={"environment_resets": 1, "model_query_attempts": 3},
            decisions=decisions,
            extra={"synthetic": True},
        )


def row(*, safe=False, violation=False, progress=0.0):
    return {
        "safe_completion": safe,
        "violation": violation,
        "unfinished": not safe,
        "engaged": True,
        "progress": progress,
        "environment_actions": 1,
        "error": None,
    }


def test_lexicographic_rank_and_shared_reward_embedding():
    high = [row(safe=True), row(violation=True)]
    middle = [row(progress=1), row(progress=1)]
    low = [row(violation=True, progress=1), row(progress=1)]
    assert panel_fitness(high) > panel_fitness(middle) > panel_fitness(low)
    encoded = [
        sum(lexicographic_episode_reward(r, 2) for r in panel) for panel in (high, middle, low)
    ]
    assert encoded[0] > encoded[1] > encoded[2]
    with pytest.raises(ValueError, match="Invalid"):
        panel_fitness([row(progress=1.1)])


@pytest.mark.parametrize("method", ["cma", "random"])
def test_actual_allowance_stops_on_complete_panels_with_unspent_balance(tmp_path, method):
    evaluator = DummyEvaluator()
    run = CampaignTraining(protocol(method, budget=18, panels=20), tmp_path / method, evaluator)
    state = run.run()
    assert state["actual_environment_actions"] == state["charged_environment_actions"] == 16
    assert state["remaining_action_allowance"] == 2
    assert state["candidate_panels_completed"] == 3
    assert state["optimizer_updates"] == 0  # CMA did not tell an unfinished four-member generation.
    assert len(state["completed_job_ids"]) == len(evaluator.calls) == 8
    assert state["pending"] is not None and len(state["pending"]["scores"]) == 3
    assert state["selected_is_trained"] is False
    assert state["selected_training_nominee"]["provenance"]["kind"] == "initial_anchor"
    again = CampaignTraining(run.protocol, tmp_path / method, evaluator, resume=True)
    again.run()
    assert len(evaluator.calls) == 8


def test_failure_charges_observed_and_conservative_work_and_never_replays(tmp_path):
    evaluator = DummyEvaluator(fail_on=1)
    d = protocol("random")
    run = CampaignTraining(d, tmp_path / "failure", evaluator)
    with pytest.raises(RuntimeError, match="synthetic evaluator"):
        run.run()
    state = json.loads((tmp_path / "failure/state.json").read_text())
    assert state["episode_starts"] == 1
    assert state["actual_environment_actions"] == 1
    assert state["charged_environment_actions"] == 3
    assert state["action_accounting_exact"] is False
    assert state["resource_totals"]["environment_resets"] == 1
    assert state["resource_totals"]["model_query_attempts"] == 2
    recovered = CampaignTraining(d, tmp_path / "failure", evaluator, resume=True)
    with pytest.raises(IncompleteJobError):
        recovered.run()
    assert evaluator.calls == ["anchor_p000"]


def test_resume_reuses_completed_jobs_and_preserves_private_cma_rng(tmp_path):
    class InterruptAfterFirstJob(CampaignTraining):
        interrupted = False

        def _save(self):
            super()._save()
            if not self.interrupted and len(self.state["completed_job_ids"]) == 1:
                self.interrupted = True
                raise KeyboardInterrupt("synthetic interruption after durable result")

    d = protocol(panels=8)
    evaluator = DummyEvaluator(seed_noise=500)
    interrupted = InterruptAfterFirstJob(d, tmp_path / "resume", evaluator)
    with pytest.raises(KeyboardInterrupt):
        interrupted.run()
    resumed = CampaignTraining(d, tmp_path / "resume", evaluator, resume=True)
    state = resumed.run()
    assert len(evaluator.calls) == len(set(evaluator.calls)) == 18
    fresh = CampaignTraining(d, tmp_path / "fresh", DummyEvaluator(seed_noise=900))
    fresh_state = fresh.run()
    assert state["history"] == fresh_state["history"]
    assert state["optimizer_updates"] == 2
    assert state["selected_is_trained"] is False
    changed = copy.deepcopy(d)
    changed["optimizer_seed"] += 1
    with pytest.raises(ValueError, match="identity changed"):
        CampaignTraining(changed, tmp_path / "resume", evaluator, resume=True)


def test_reinforce_updates_exact_categorical_family_and_saves_compact_trace(tmp_path):
    evaluator = DummyEvaluator(decisions=True)
    d = protocol("reinforce", panels=1)
    calls = []

    def exact_law(features, available, parameters, selected, temperature):
        calls.append((features.shape, parameters.requires_grad))
        return RetrievalController.log_probability(
            features, available, parameters, selected, temperature
        )

    run = CampaignTraining(d, tmp_path / "pg", evaluator, policy_log_probability=exact_law)
    state = run.run()
    assert state["optimizer_updates"] == 1
    assert state["candidate_panels_completed"] == 1
    assert state["training_panels_completed"] == 2
    assert state["actual_environment_actions"] == 12  # Anchor, categorical, deterministic panels.
    assert state["history"][0]["statistics"]["parameter_change_l2"] > 0
    assert any(shape == (2, 9) and gradient for shape, gradient in calls)
    assert run.parameter.detach().cpu().numpy()[7] > d["initial_parameters"][7]
    assert state["selected_is_trained"] is False  # A tie retains the untrained anchor.
    assert any("_sample_" in key for key in evaluator.calls)
    assert any("_evaluate_" in key for key in evaluator.calls)
    state_text = (tmp_path / "pg/state.json").read_text()
    assert '"features"' not in state_text and '"available"' not in state_text
    paths = sorted((tmp_path / "pg/jobs").glob("*/decisions.npz"))
    assert len(paths) == 2
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            assert data["features"].shape == (1, 2, 9)
            assert data["selected"].tolist() == [0]
            assert np.isfinite(data["behavior_log_probability"]).all()
    recovered = CampaignTraining(
        d, tmp_path / "pg", evaluator, policy_log_probability=exact_law, resume=True
    )
    np.testing.assert_array_equal(recovered.parameter.detach().cpu(), run.parameter.detach().cpu())
    assert recovered.run()["optimizer_updates"] == 1
    assert len(evaluator.calls) == 6


def test_hard_cap_excess_is_recorded_before_failure(tmp_path):
    def evaluator(job, report_progress):
        report_progress(job.max_actions + 1, {"environment_resets": 1})
        raise AssertionError("The budget callback should already have rejected excess")

    run = CampaignTraining(protocol("random"), tmp_path / "overrun", evaluator)
    with pytest.raises(RuntimeError, match="hard actual-action cap"):
        run.run()
    state = json.loads((tmp_path / "overrun/state.json").read_text())
    assert state["actual_environment_actions"] == 4
    assert state["charged_environment_actions"] == 4
    assert state["episode_starts"] == 1
