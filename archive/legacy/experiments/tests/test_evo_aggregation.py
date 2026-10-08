"""Fit-only root selection, paired mixture budgets, and exact teacher branching."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
from helpers.evoAggregation import augmentation_indices, select_training_starts, teacher_label  # noqa: E402
from helpers.evoInputs import load_stage_config  # noqa: E402
from helpers.evoTeacherDiagnostic import teacher_branch  # noqa: E402
import test_evo_real as fixtures  # noqa: E402
from test_evo_teacher import Teacher  # noqa: E402

roots = fixtures.roots


def test_training_starts_are_whole_episode_fit_only_and_deterministic():
    lengths = np.arange(20) + 120
    args = dict(count=5, seed=31)
    a = select_training_starts(range(12), range(12, 20), lengths, **args)
    assert a == select_training_starts(range(12), range(12, 20), lengths, **args)
    assert len({e for e, t in a}) == 5
    assert all(e < 12 and 20 <= t <= lengths[e] - 100 for e, t in a)
    with pytest.raises(ValueError):
        select_training_starts(range(12), [11], lengths, **args)


def test_augmentation_has_matched_counts_without_outcome_weighting():
    old, new = augmentation_indices(22993, 640, fraction=.25, seed=42)
    assert len(old) == len(set(old)) == 17245
    assert len(new) == 5748
    counts = np.bincount(new, minlength=640)
    assert counts.max() - counts.min() == 1
    again = augmentation_indices(22993, 640, fraction=.25, seed=42)
    assert all(np.array_equal(a, b) for a, b in zip((old, new), again))


def test_teacher_labels_preserve_solver_prefix_and_match_teacher_rollout(roots):
    env = fixtures.physics_env()
    root = roots[0]
    tape = np.random.default_rng(19).uniform(-1, 1, (100, 6))
    teacher = Teacher()
    try:
        log = teacher_branch(env, root, tape, teacher, takeover_step=0)
        label, qp, qv, charged = teacher_label(env, root, log['actions'], teacher, boundary=40)
        assert charged == 50
        assert np.array_equal(qp, log['qpos'][:51])
        assert np.array_equal(qv, log['qvel'][:51])
        assert np.array_equal(label.reshape(10, 6), log['actions'][40:50])
        replay = teacher_branch(env, root, tape, teacher)
        _, qp, qv, _ = teacher_label(env, root, tape, teacher, boundary=70)
        assert np.array_equal(qp[:71], replay['qpos'][:71])
        assert np.array_equal(qv[:71], replay['qvel'][:71])
    finally:
        env.close()


def test_aggregation_protocol_keeps_architecture_validation_and_budgets():
    cfg = load_stage_config('stage2_aggregation')
    prior = load_stage_config('stage2_coverage')
    assert cfg.policy == prior.policy
    for key in ['seeds', 'epochs', 'batch_size', 'learning_rate', 'weight_decay']:
        assert cfg.training[key] == prior.training[key]
    assert cfg.collection.simulator_steps_per_arm == 128 * (100 + sum(range(10, 101, 10)))
    assert cfg.collection.aggregate_iterations == 1
    assert cfg.interpretation.gate0 == 'not_run'


def test_aggregation_features_use_only_current_and_previous_block():
    import importlib.util
    import torch
    from helpers.evoMlpPolicy import MLPBlockPolicy

    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('aggregation_runner', scripts / 'evo_aggregation_diagnostic.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(22)
    history = rng.normal(size=(2, 3, 192)).astype(np.float32)
    ends = rng.normal(size=(2, 10, 192)).astype(np.float32)
    policy = MLPBlockPolicy((np.zeros(6), np.ones(6)), device='cpu')
    got = module.features_from_rollout(policy, history, ends)
    assert got.shape == (2, 10, 384)
    for b in range(10):
        current = history[:, -1] if b == 0 else ends[:, b-1]
        previous = history[:, -2] if b == 0 else history[:, -1] if b == 1 else ends[:, b-2]
        expected = policy.features(torch.tensor(current), torch.tensor(previous)).numpy()
        assert np.array_equal(got[:, b], expected)
