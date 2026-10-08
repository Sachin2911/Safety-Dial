"""Exact branch replay, intervention timing, and pre-failure error accounting."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
from helpers.evoTeacherDiagnostic import action_error_summary, teacher_branch  # noqa: E402
from helpers.walkerRules import execute_branch  # noqa: E402
import test_evo_real as real_fixtures  # noqa: E402

roots = real_fixtures.roots


@pytest.fixture
def physics_env():
    env = real_fixtures.physics_env()
    yield env
    env.close()


class Teacher:
    def reset(self):
        pass

    def act(self, obs):
        return np.tanh(np.asarray(obs[-6:]) * .02)


def test_teacher_queries_preserve_bitwise_tape_replay(physics_env, roots):
    root = roots[0]
    tape = np.random.default_rng(21).uniform(-1, 1, (100, 6))
    out = teacher_branch(physics_env, root, tape, Teacher())
    ref = execute_branch(physics_env, root.qpos, root.qvel, tape)
    assert np.array_equal(out['qpos'], ref.qpos)
    assert np.array_equal(out['qvel'], ref.qvel)
    assert np.array_equal(out['actions'], np.clip(tape, -1, 1))


def test_takeover_prefix_and_hold_timing(physics_env, roots):
    root = roots[0]
    tape = np.random.default_rng(22).uniform(-1, 1, (100, 6))
    ref = teacher_branch(physics_env, root, tape, Teacher())
    out = teacher_branch(physics_env, root, tape, Teacher(),
                         takeover_step=50, feedback_interval=10)
    assert np.array_equal(out['qpos'][:51], ref['qpos'][:51])
    assert np.array_equal(out['qvel'][:51], ref['qvel'][:51])
    for t in range(50, 100, 10):
        assert np.array_equal(out['actions'][t:t+10],
                              np.broadcast_to(out['teacher_actions'][t], (10, 6)))
    replay = execute_branch(physics_env, root.qpos, root.qvel, out['actions'])
    assert np.array_equal(replay.qpos, out['qpos'])
    assert np.array_equal(replay.qvel, out['qvel'])


def test_error_excludes_post_failure_states_and_reports_counts():
    actions, targets = np.zeros((2, 100, 6)), np.ones((2, 100, 6))
    actions[0, 4:] = 100  # already unsafe, must not contaminate the summary
    out = action_error_summary(actions, targets, [3, 100])
    assert out['all'] == {'n_actions': 104, 'mse': 1.0}
    assert [g['n_actions'] for g in out['block_position']] == [11]*4 + [10]*6
    with pytest.raises(ValueError):
        action_error_summary(actions, targets[:1], [3, 100])
