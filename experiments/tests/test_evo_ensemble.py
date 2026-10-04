"""Mean policy interface and identical feedback through real and imagined execution."""
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
import test_evo_real as fixtures  # noqa: E402

roots, ctx, tiny = fixtures.roots, fixtures.ctx, fixtures.tiny


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def make_policy():
    torch.manual_seed(31)
    member = HistoryBlockPolicy(fixtures.SCALER, hidden=(8, 4))
    members = [member.network_theta(member.network()) for _ in range(3)]
    return EnsembleHistoryPolicy(member, 3), np.concatenate(members)[None]


def test_ensemble_matches_explicit_member_mean_and_roundtrip():
    policy, theta = make_policy()
    feats = torch.randn(1, 7, 444)
    expected = torch.stack([policy.member.act(m[None], feats) for m in
                            theta.reshape(3, policy.member.n_params)]).mean(0)
    assert torch.equal(policy.act(theta, feats), expected)
    restored = EnsembleHistoryPolicy.from_state(policy.state())
    assert torch.equal(restored.act(theta, feats), expected)
    with pytest.raises(ValueError):
        policy.act(theta[:, :-1], feats)


def test_ensemble_real_grouping_and_executed_feedback(tiny, ctx, roots):
    model, probe, _ = tiny
    policy, theta = make_policy()
    ex = HistoryRealExecutor(model, fixtures.SCALER, probe, policy, device='cpu', ctx=ctx,
                             max_envs=2, encode_batch=4)
    try:
        zh = ex.history_latents(roots).numpy()
        out = ex.run(theta, roots, record_qpos=True, z_hist=zh)
        ex.max_envs = 1
        again = ex.run(theta, roots, record_qpos=True, z_hist=zh)
        assert np.array_equal(out['qpos'], again['qpos'])
        for i, root in enumerate(roots):
            replay = execute_branch(ex._physics(1)[0], root.qpos, root.qvel, out['actions'][i].reshape(100, 6))
            assert np.array_equal(out['qpos'][i], replay.qpos)
            for b in range(10):
                zt = zh[i, -1] if b == 0 else out['z'][i, b-1]
                zp = zh[i, -2] if b == 0 else zh[i, -1] if b == 1 else out['z'][i, b-2]
                past = root.history_actions[-1].reshape(60) if b == 0 else out['actions'][i, b-1]
                f = policy.features(torch.tensor(zt[None]), torch.tensor(zp[None]), past[None])
                assert np.array_equal(policy.act(theta, f[:, None])[0, 0], out['actions'][i, b])
    finally:
        ex.close()


def test_ensemble_imagination_uses_combined_past_with_candidate_noise_layout(tiny):
    model, probe, _ = tiny
    policy, theta = make_policy()
    im = HistoryImaginer(model, fixtures.SCALER, probe, device='cpu')
    zh, hist = torch.randn(2, 3, 192), torch.randn(2, 2, 10, 6)
    out = im.rollout_policies(policy, theta, zh, hist, return_latents=True)
    for b in range(10):
        zt = zh[:, -1] if b == 0 else torch.tensor(out['z'][0, :, 0, b-1])
        zp = zh[:, -2] if b == 0 else zh[:, -1] if b == 1 else torch.tensor(out['z'][0, :, 0, b-2])
        past = hist[:, -1].reshape(2, 60) if b == 0 else torch.tensor(out['actions'][0, :, 0, b-1])
        want = policy.act(theta, policy.features(zt, zp, past)[None]).numpy()[0]
        np.testing.assert_allclose(out['actions'][0, :, 0, b], want, atol=1e-7)
    im.sigma = np.ones(192)*.01
    noisy = im.rollout_policies(policy, np.repeat(theta, 2, axis=0), zh, hist, k=.5, n_samples=2)
    assert noisy['actions'].shape == (2, 2, 2, 10, 60)
