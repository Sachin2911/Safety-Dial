"""Residual control equivalence, declared candidate design and undefined-rank handling."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'experiments'))
from helpers.evoEnsemblePolicy import EnsembleHistoryPolicy  # noqa: E402
from helpers.evoHistoryPolicy import HistoryBlockPolicy  # noqa: E402
from helpers.evoHistoryReal import HistoryRealExecutor  # noqa: E402
from helpers.evoHistoryImagine import HistoryImaginer  # noqa: E402
from helpers.walkerRules import execute_branch  # noqa: E402
from helpers.evoResidualPolicy import ResidualHistoryPolicy, residual_candidates  # noqa: E402
from helpers.evoTransferPilot import rank_audit, selected_effect  # noqa: E402
import test_evo_real as fixtures  # noqa: E402

roots, ctx, tiny = fixtures.roots, fixtures.ctx, fixtures.tiny


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def policy_fixture():
    torch.manual_seed(71)
    member = HistoryBlockPolicy(fixtures.SCALER, hidden=(8, 4))
    base = EnsembleHistoryPolicy(member, 2)
    theta = np.concatenate([member.network_theta(member.network()) for _ in range(2)])
    projection = np.random.default_rng(8).normal(size=(444, 16)).astype(np.float32)/np.sqrt(444)
    return ResidualHistoryPolicy(base, theta, projection)


def test_zero_residual_roundtrip_and_declared_rms_candidates():
    policy = policy_fixture()
    feats = torch.randn(32, 444)
    zero = np.zeros((1, policy.n_params), np.float32)
    assert policy.n_params == 1021
    assert torch.equal(policy.act(zero, feats[None]), policy.base.act(policy.base_theta[None], feats[None]))
    restored = ResidualHistoryPolicy.from_state(policy.state())
    assert torch.equal(restored.act(zero, feats[None]), policy.act(zero, feats[None]))
    th, meta = residual_candidates(policy, feats, directions=12, rms_scales=[.025, .05, .1, .2],
                                   gains=[0, .25, .5, .75], seed=12)
    assert th.shape == (101, 1021) and len(meta) == 101
    for j, m in enumerate(meta):
        if m['kind'] == 'residual':
            rms = float(policy.residual(th[j:j+1], feats[None]).square().mean().sqrt())
            assert rms == pytest.approx(m['target_raw_rms'], rel=1e-6)
    assert np.array_equal(th[1], -th[2])
    assert torch.count_nonzero(policy.act(th[97:98], feats[None])) == 0


def test_zero_residual_real_physics_and_imagination_equal_base(tiny, ctx, roots):
    model, probe, _ = tiny
    policy = policy_fixture()
    zero = np.zeros((1, policy.n_params), np.float32)
    ex = HistoryRealExecutor(model, fixtures.SCALER, probe, policy.base, device='cpu', ctx=ctx, max_envs=2, encode_batch=4)
    try:
        zh = ex.history_latents(roots).numpy()
        base = ex.run(policy.base_theta[None], roots, record_qpos=True, z_hist=zh)
        ex.policy = policy
        got = ex.run(zero, roots, record_qpos=True, z_hist=zh)
        assert np.array_equal(base['qpos'], got['qpos']) and np.array_equal(base['actions'], got['actions'])
        ex.max_envs = 1
        again = ex.run(zero, roots, record_qpos=True, z_hist=zh)
        assert np.array_equal(got['qpos'], again['qpos'])
        changed = np.random.default_rng(4).normal(0, .002, zero.shape).astype(np.float32)
        changed[:, -1] = 0
        out = ex.run(changed, roots, record_qpos=True, z_hist=zh)
        for j, root in enumerate(roots):
            replay = execute_branch(ex._physics(1)[0], root.qpos, root.qvel, out['actions'][j].reshape(100, 6))
            assert np.array_equal(replay.qpos, out['qpos'][j])
            for b in range(10):
                zt = zh[j, -1] if b == 0 else out['z'][j, b-1]
                zp = zh[j, -2] if b == 0 else zh[j, -1] if b == 1 else out['z'][j, b-2]
                past = root.history_actions[-1].reshape(60) if b == 0 else out['actions'][j, b-1]
                feats = policy.features(torch.tensor(zt[None]), torch.tensor(zp[None]), past[None])
                assert np.array_equal(policy.act(changed, feats[:, None])[0, 0], out['actions'][j, b])
    finally:
        ex.close()
    im = HistoryImaginer(model, fixtures.SCALER, probe, device='cpu')
    hist = np.stack([r.history_actions for r in roots])
    old = im.rollout_policies(policy.base, policy.base_theta[None], zh, hist, return_latents=True)
    got = im.rollout_policies(policy, zero, zh, hist, return_latents=True)
    assert np.array_equal(old['z'], got['z'])
    im.sigma = np.ones(192)*.01
    noisy = im.rollout_policies(policy, np.repeat(zero, 2, axis=0), zh, hist, k=.5, n_samples=2)
    assert noisy['actions'].shape == (2, 2, 2, 10, 60)


def test_rank_audit_reports_undefined_and_paired_gap_can_exceed_one():
    a = np.zeros((3, 20))
    b = np.tile(np.arange(3)[:, None], (1, 20))
    r = rank_audit(a, b, np.arange(20), n_boot=20, seed=3)
    assert r['point'] is None and not r['transfer_screen_pass'] and r['n_boot'] == 0
    r = rank_audit(b, b, np.arange(20), n_boot=20, seed=3)
    assert r['transfer_screen_pass'] and r['point'] == 1
    real = np.stack([np.zeros(20), np.ones(20)])
    imagined = 1-real
    result = selected_effect(real, imagined, np.ones_like(real), selected=1,
        assessment_indices=np.arange(10, 20), episodes=np.arange(20), n_boot=30, seed=4)
    assert result['gap_change']['point'] == 2
    assert result['root_only_power_planning']['range_bound_variance'] == 4
    assert result['root_only_power_planning']['plug_in_n'] is None
