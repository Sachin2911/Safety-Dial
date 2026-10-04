"""Fresh episode accounting, preserved root rule and honest readiness inference."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'experiments'))
from helpers.evoReadiness import baseline_audit, generate_episode, representative_root, reference_arrays  # noqa: E402
from helpers.evoInputs import load_stage_config  # noqa: E402
import test_evo_real as fixtures  # noqa: E402
from test_evo_teacher import Teacher  # noqa: E402


def test_fresh_episode_reproducibility_and_rejected_prefix_cost():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    env = fixtures.physics_env()
    counter = dict(real_steps=0, teacher_queries=0)
    try:
        a = generate_episode(env, Teacher(), seed=172, max_steps=12, counter=counter)
        b = generate_episode(env, Teacher(), seed=172, max_steps=12, counter=counter)
        assert all(np.array_equal(a[k], b[k]) for k in a)
        assert counter['real_steps'] == 2*len(a['action']) == counter['teacher_queries']
        assert representative_root(a, episode_id=0, rng=np.random.default_rng(1)) is None
    finally:
        env.close()
        torch.set_num_threads(before)


def test_representative_root_keeps_one_episode_and_never_pads_tape():
    n = 190
    episode = dict(qpos=np.arange((n+1)*9).reshape(n+1, 9), qvel=np.ones((n+1, 9)),
                   action=np.arange(n*6).reshape(n, 6), x_velocity=np.ones(n))
    r = representative_root(episode, episode_id=32, rng=np.random.default_rng(9))
    assert 20 <= r.step < n-100 and r.episode == 32
    assert np.array_equal(r.qpos, episode['qpos'][r.step])
    assert np.array_equal(r.policy_tape, episode['action'][r.step:r.step+100])
    assert r.meta['policy_tape_padding_steps'] == 0


def test_readiness_can_pass_without_certifying_safety_and_fails_no_progress():
    real = np.zeros(256, bool)
    real[:40] = True
    args = dict(n_boot=1000, seed=1)
    got = baseline_audit(real, np.zeros(256), np.ones(256), np.ones(256), **args)
    assert got['research_pilot_ready']
    assert not got['safe_controller_claim'] and not got['original_gate0_passed']
    bad = baseline_audit(real, np.zeros(256), np.zeros(256), np.ones(256), **args)
    assert not bad['research_pilot_ready'] and not bad['checks']['progress_retained']
    zero = baseline_audit(np.zeros(256), np.zeros(256), np.ones(256), np.ones(256), **args)
    assert zero['failure']['hi'] > 0
    with pytest.raises(ValueError, match='finite'):
        baseline_audit(real, real, np.full(256, np.nan), np.ones(256), **args)


def test_readiness_protocol_preserves_old_gate_and_freezes_new_roles():
    cfg = load_stage_config('stage2_readiness_baseline')
    assert cfg.readiness.preserves_original_gate0 and not cfg.readiness.original_gate0_passed
    assert not cfg.readiness.fit_or_select_on_fresh_bank
    assert cfg.fresh_bank.n_roots == 256 and cfg.fresh_bank.one_root_per_episode
    assert len(cfg.fresh_bank.actor_names) == 32
    assert cfg.fresh_bank.episode_seed_base > 900000000


roots = fixtures.roots


def test_recorded_reference_matches_physics_only_executor_contract(roots):
    from helpers.evoReal import RealExecutor
    ex = RealExecutor(None, None, None, None, device='cpu')
    try:
        tapes = np.stack([r.policy_tape for r in roots])
        logs = ex.run_tapes(roots, tapes)
        out = reference_arrays(logs)
        assert out['dense_violated'].tolist() == [log.unsafe()['health'] for log in logs]
        assert np.array_equal(out['actions'], tapes)
        assert ex.counters.real_steps == len(roots)*100
    finally:
        ex.close()
