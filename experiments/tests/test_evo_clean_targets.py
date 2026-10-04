"""Temporal alignment, split isolation and source-teacher dynamics for clean targets."""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
from helpers.evoCleanTargets import cached_steps, check_fit_indices, clean_teacher_block  # noqa: E402
from helpers.evoInputs import load_stage_config  # noqa: E402
from helpers.evoTeacherDiagnostic import teacher_branch  # noqa: E402
import test_evo_real as fixtures  # noqa: E402
from test_evo_teacher import Teacher  # noqa: E402

roots = fixtures.roots


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_cache_times_and_fit_roles_reject_misalignment_or_validation(tmp_path):
    path = tmp_path / 'data.h5'
    a = np.arange(60*6).reshape(60, 6).astype(np.float32)
    with h5py.File(path, 'w') as f:
        f['ep_offset'], f['ep_len'], f['action'] = [0, 30], [30, 30], a
    episodes = np.repeat([0, 1], 3)
    targets = np.stack([a[30*e+t:30*e+t+10].reshape(60) for e in [0, 1] for t in [10, 15, 20]])
    assert np.array_equal(cached_steps(path, episodes, targets), [10, 15, 20, 10, 15, 20])
    bad = targets.copy()
    bad[0, 0] += 1
    with pytest.raises(ValueError, match='temporal alignment'):
        cached_steps(path, episodes, bad)
    check_fit_indices(episodes, [0, 1, 2], [0], [1])
    with pytest.raises(ValueError, match='fit episodes only'):
        check_fit_indices(episodes, [0, 3], [0], [1])


def test_clean_branch_matches_teacher_prefix_and_counts_every_attempt(roots, monkeypatch):
    from helpers import evoCleanTargets as module
    env = fixtures.physics_env()
    root, teacher = roots[0], Teacher()
    counter = dict(real_steps=0, teacher_queries=0)
    try:
        expected = teacher_branch(env, root, root.policy_tape, teacher, takeover_step=0)
        got = clean_teacher_block(env, root.qpos, root.qvel, teacher, counter)
        for key in ['qpos', 'qvel']:
            assert np.array_equal(got[key], expected[key][:11])
        assert np.array_equal(got['actions'], expected['actions'][:10])
        assert counter == dict(real_steps=10, teacher_queries=10)
        original_step = module._step
        def fail_at_second(*args):
            if args[-1] == 1:
                raise RuntimeError('simulator stopped')
            return original_step(*args)
        monkeypatch.setattr(module, '_step', fail_at_second)
        with pytest.raises(RuntimeError, match='simulator stopped'):
            clean_teacher_block(env, root.qpos, root.qvel, teacher, counter)
        assert counter == dict(real_steps=12, teacher_queries=12)
    finally:
        env.close()


def test_clean_target_protocol_matches_reused_control_and_keeps_gate():
    cfg = load_stage_config('stage2_clean_targets')
    control = load_stage_config('stage2_information_resume')
    assert cfg.policy.hidden == control.policy.hidden
    for key in ['seeds', 'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'checkpoint_epochs']:
        assert cfg.training[key] == control.training[key]
    assert cfg.collection.simulator_steps == cfg.collection.examples * 10
    assert cfg.collection.outcome_filtering is False
    assert cfg.collection.validation_teacher_queries is False
    assert cfg.interpretation.privileged_state_input is False
    assert cfg.interpretation.gate0 == 'not_run'
    assert cfg.validation_control.progress_floor == .5
