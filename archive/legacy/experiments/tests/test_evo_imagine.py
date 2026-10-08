"""Closed-loop LeWM imagination (evolution harness, Module A): exactness, noise and accounting.

A fixed tape through `ClosedLoopImaginer.rollout_tapes` must be bitwise equal to
`WalkerImaginer.rollout`, which is what catches errors in the action window. Noise is seeded and
shared across candidates (common random numbers). On CPU, population batching and chunking leave
results unchanged within 1e-6; on CUDA they move them slightly (module doc), so there the tests ask
that a fixed configuration repeats bitwise. At the few CPU threads used here the kernels ignore the
batch shape, so the shapes the model is called with (the full action buffer at every step, the
requested chunks) are recorded directly, and the bitwise tape comparison is repeated on CUDA, where
the shape does change the numbers. Tests use a tiny random LeWM built with hydra from a config
modelled on the real `config.json` (the CUDA ones skip without CUDA); one CUDA test uses the real
model on two S4 development roots and skips when the assets, CUDA or EGL rendering are missing.
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.evoImagine import ClosedLoopImaginer, Counters, one_step_residuals, open_loop_errors, sigma_from_residuals  # noqa: E402
from helpers.evoPolicy import FEATURE_DIM, N_PARAMS, LinearLatentPolicy  # noqa: E402
from helpers.poseProbes import Probe, ProbeSpec  # noqa: E402
from helpers.walkerLewm import WalkerImaginer  # noqa: E402

D = 192
MODEL_DIR = ROOT / "data/hf/safetydial-walker2d/lewm-a/walker2d-lewm-a-recovery-20260927-1"
PROBE_PATH = ROOT / "data/hf/safetydial-walker2d/probes/walker2d-probes-recovery-20260927-1/walker_mlp.pt"
DEV_BANK = ROOT / "data/hf/safetydial-walker2d-data/banks/walker2d-s4-recovery-20260927-1/dev"
_BN = {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}
TINY = {  # the real config.json with every size shrunk (latent and action widths kept)
    "_target_": "stable_worldmodel.wm.lewm.LeWM",
    "encoder": {"_target_": "stable_pretraining.backbone.utils.vit_hf", "size": "tiny", "patch_size": 8, "image_size": 32,
                "pretrained": False, "use_mask_token": False, "num_hidden_layers": 2},
    "predictor": {"_target_": "stable_worldmodel.wm.lewm.module.Predictor", "num_frames": 3, "input_dim": D, "hidden_dim": D,
                  "output_dim": D, "depth": 2, "heads": 4, "mlp_dim": 256, "dim_head": 16, "dropout": 0.1, "emb_dropout": 0.0},
    "action_encoder": {"_target_": "stable_worldmodel.wm.lewm.module.Embedder", "input_dim": 60, "emb_dim": D},
    "projector": {"_target_": "stable_worldmodel.wm.lewm.module.MLP", "input_dim": D, "output_dim": D, "hidden_dim": 256, "norm_fn": _BN},
    "pred_proj": {"_target_": "stable_worldmodel.wm.lewm.module.MLP", "input_dim": D, "output_dim": D, "hidden_dim": 256, "norm_fn": _BN},
}


@pytest.fixture(scope="module", autouse=True)
def few_threads():
    """Tiny CPU workloads: torch's default pool (every visible core) oversubscribes the cgroup quota."""
    n = torch.get_num_threads()
    torch.set_num_threads(min(n, 4))
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def model():
    from hydra.utils import instantiate

    torch.manual_seed(0)
    m = instantiate(TINY)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():  # adaLN-zero makes every block the identity at init: let actions reach the output
        for blk in m.predictor.transformer.layers:
            for p in blk.adaLN_modulation[-1].parameters():
                p.copy_(0.05 * torch.randn(p.shape, generator=gen))
    return m.eval()


@pytest.fixture(scope="module")
def scaler():
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 0.1, 6), rng.uniform(0.3, 0.7, 6)


@pytest.fixture(scope="module")
def probe():
    torch.manual_seed(2)
    return Probe(ProbeSpec(target="walker", kind="mlp", hidden=(32,)), np.array([1.2, 0.0, 1.0]), np.array([0.2, 0.3, 0.8]),
                 np.zeros(D), np.ones(D)).eval()


@pytest.fixture(scope="module")
def policy(scaler):
    return LinearLatentPolicy(scaler, device="cpu")


def roots(rng, R):
    """Random history latents (R, 3, D) and raw history action blocks (R, 2, 10, 6)."""
    return torch.as_tensor(rng.normal(size=(R, 3, D)), dtype=torch.float32), rng.uniform(-1, 1, (R, 2, 10, 6))


def thetas(rng, P):
    return rng.normal(0.0, 0.03, (P, N_PARAMS))


def sigma_vec(seed=3):
    return np.random.default_rng(seed).uniform(0.05, 0.2, D)


def spec_rollout(model, z_hist, hist_norm, action_fn, K, k=0.0, sigma=None, eps=None):
    """The rollout_core algorithm exactly as the harness spec writes it."""
    B = z_hist.shape[0]
    emb = [z_hist[:, 0], z_hist[:, 1], z_hist[:, 2]]
    buf = torch.zeros(B, 2 + K, 60)
    buf[:, :2] = hist_norm
    for t in range(K):
        buf[:, 2 + t] = action_fn(t, emb[-1], emb[-2])
        ae = model.action_encoder(buf)
        z = model.predict(torch.stack(emb[-3:], 1), ae[:, t : t + 3])[:, -1]
        if k > 0:
            z = z + k * sigma * eps[:, t]
        emb.append(z)
    return torch.stack(emb[3:], 1), buf[:, 2:]


@pytest.mark.parametrize("S,K,chunk", [(13, 10, 512), (13, 10, 5), (7, 4, 512), (520, 10, 512)])
def test_rollout_tapes_is_bitwise_walker_imaginer(model, scaler, S, K, chunk):
    rng = np.random.default_rng(S + K + chunk)
    z_hist, hist_blocks = roots(rng, 1)
    tapes = rng.uniform(-1, 1, (S, K, 10, 6))
    ref = WalkerImaginer(model, scaler, "cpu").rollout(z_hist[0], hist_blocks[0], tapes, chunk=chunk)
    im = ClosedLoopImaginer(model, scaler, None, device="cpu")
    got = im.rollout_tapes(z_hist[0], hist_blocks[0], tapes, chunk=chunk)
    assert got.shape == (S, K, D) and got.dtype == torch.float32
    assert torch.equal(got, ref)
    assert im.counters.imagined_rows == S * K


@pytest.mark.parametrize("chunk", [512, 64])
def test_rollout_tapes_is_bitwise_walker_imaginer_on_cuda(model, scaler, chunk):
    """CUDA picks kernels by batch shape, so here an action encoder call of another shape or a chunk
    size other than the reference's changes the numbers (by about 4e-7 with this model)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    m = copy.deepcopy(model)  # WalkerImaginer moves its model to the device in place
    rng = np.random.default_rng(chunk)
    z_hist, hist_blocks = roots(rng, 1)
    S, K = 700, 10  # a partial last chunk either way
    tapes = rng.uniform(-1, 1, (S, K, 10, 6))
    ref = WalkerImaginer(m, scaler, "cuda").rollout(z_hist[0].cuda(), hist_blocks[0], tapes, chunk=chunk)
    im = ClosedLoopImaginer(m, scaler, None, device="cuda")
    got = im.rollout_tapes(z_hist[0], hist_blocks[0], tapes, chunk=chunk)
    assert got.shape == (S, K, D) and torch.equal(got, ref)
    assert im.counters.imagined_rows == S * K


def test_model_sees_the_full_action_buffer_and_the_requested_chunks(model, scaler, probe, policy):
    """At a few CPU threads the kernels give the same numbers whatever the batch shape, so the bitwise
    tests cannot see a prefix-only action encoding or an ignored chunk size: record the calls. Every
    step re-encodes the full (B, 2 + K, 60) buffer (history, decided blocks, undecided blocks still
    zero), and the batches follow `chunk` (rollout_tapes) and `max_batch` (rollout_policies)."""
    rng = np.random.default_rng(13)
    z_hist, hist_blocks = roots(rng, 3)
    S, K, chunk = 13, 10, 5
    tapes = rng.uniform(-1, 1, (S, K, 10, 6))
    im = ClosedLoopImaginer(model, scaler, probe, device="cpu", sigma=sigma_vec())
    enc, pred = [], []
    hooks = [model.action_encoder.register_forward_pre_hook(lambda mod, args: enc.append(args[0].clone())),
             model.predictor.register_forward_pre_hook(lambda mod, args: pred.append([tuple(a.shape) for a in args]))]
    try:
        im.rollout_tapes(z_hist[0], hist_blocks[0], tapes, chunk=chunk)
        sizes = (5, 5, 3)
        assert [tuple(x.shape) for x in enc] == [(s, 2 + K, 60) for s in sizes for _ in range(K)]
        assert pred == [[(s, 3, D), (s, 3, D)] for s in sizes for _ in range(K)]
        hist, fut = im.flat(hist_blocks[0]), im.flat(tapes)
        for j, x in enumerate(enc):
            i0, t = chunk * (j // K), j % K
            assert torch.equal(x[:, :2], hist.expand(len(x), -1, -1))
            assert torch.equal(x[:, 2 : 3 + t], fut[i0 : i0 + len(x), : t + 1])
            assert not x[:, 3 + t :].any()
        P, n, K = 5, 2, 4
        M = len(z_hist) * n
        for max_batch, rows in ((16384, (P * M,)), (2 * M + 1, (2 * M, 2 * M, M))):  # whole candidates per chunk
            enc.clear()
            pred.clear()
            im.rollout_policies(policy, thetas(rng, P), z_hist, hist_blocks, n_samples=n, k=1.0, n_blocks=K, max_batch=max_batch)
            assert [tuple(x.shape) for x in enc] == [(b, 2 + K, 60) for b in rows for _ in range(K)]
            assert [p[0] for p in pred] == [(b, 3, D) for b in rows for _ in range(K)]
    finally:
        for h in hooks:
            h.remove()


def test_tiny_model_is_action_sensitive_so_equivalence_is_not_vacuous(model, scaler):
    rng = np.random.default_rng(1)
    z_hist, hist_blocks = roots(rng, 1)
    tapes = rng.uniform(-1, 1, (4, 10, 10, 6))
    im = ClosedLoopImaginer(model, scaler, None, device="cpu")
    base = im.rollout_tapes(z_hist[0], hist_blocks[0], tapes)
    for j in (0, 3, 9):
        moved = tapes.copy()
        moved[:, j] += 0.5
        diff = (im.rollout_tapes(z_hist[0], hist_blocks[0], moved) - base).abs().amax((0, 2))
        assert (diff[:j] == 0).all() and diff[j] > 1e-4  # block j moves its own prediction, never earlier ones
    for h in (0, 1):
        moved = hist_blocks.copy()
        moved[0, h] += 0.5
        assert (im.rollout_tapes(z_hist[0], moved[0], tapes) - base).abs()[:, 0].max() > 1e-4


def test_rollout_core_matches_spec_algorithm_and_policy_sees_noisy_latents(model, scaler):
    rng = np.random.default_rng(2)
    B, K, k = 5, 6, 1.5
    z_hist, hist_blocks = roots(rng, B)
    sigma = sigma_vec()
    eps = torch.randn((B, K, D), generator=torch.Generator().manual_seed(9))
    im = ClosedLoopImaginer(model, scaler, None, device="cpu", sigma=sigma)
    hist_norm = im.flat(hist_blocks)
    seen = []

    def react(t, z_t, z_prev):
        seen.append((z_t.clone(), z_prev.clone()))
        return torch.tanh(z_t[:, :60] - 0.5 * z_prev[:, 60:120])

    z, blocks = im.rollout_core(z_hist, hist_norm, react, n_blocks=K, k=k, eps=eps)
    assert len(seen) == K and z.shape == (B, K, D) and blocks.shape == (B, K, 60)
    assert torch.equal(seen[0][0], z_hist[:, 2]) and torch.equal(seen[0][1], z_hist[:, 1])
    for t in range(1, K):  # the noisy latent is what the policy sees next, and the one before it
        assert torch.equal(seen[t][0], z[:, t - 1])
        assert torch.equal(seen[t][1], z[:, t - 2] if t > 1 else z_hist[:, 2])
    with torch.inference_mode():
        z_ref, blocks_ref = spec_rollout(model, z_hist, hist_norm, react, K, k, torch.as_tensor(sigma, dtype=torch.float32), eps)
    assert torch.equal(z, z_ref) and torch.equal(blocks, blocks_ref)
    clean, _ = im.rollout_core(z_hist, hist_norm, react, n_blocks=K)
    assert not torch.allclose(clean, z)
    with pytest.raises(ValueError):
        im.rollout_core(z_hist, hist_norm, react, n_blocks=K, k=k)  # eps missing
    with pytest.raises(ValueError):
        ClosedLoopImaginer(model, scaler, None, device="cpu").rollout_core(z_hist, hist_norm, react, n_blocks=K, k=k, eps=eps)


def test_k0_policy_rollouts_are_deterministic_and_well_formed(model, scaler, probe, policy):
    rng = np.random.default_rng(3)
    P, R, K = 4, 3, 10
    z_hist, hist_blocks = roots(rng, R)
    theta = thetas(rng, P)
    im = ClosedLoopImaginer(model, scaler, probe, device="cpu")
    a = im.rollout_policies(policy, theta, z_hist, hist_blocks, return_latents=True)
    b = im.rollout_policies(policy, torch.as_tensor(theta), z_hist.numpy(), hist_blocks, return_latents=True)
    for key in ("readout", "actions", "root_readout", "z"):
        assert np.array_equal(a[key], b[key]), key
    assert a["readout"].shape == (P, R, 1, K, 3) and a["readout"].dtype == np.float64
    assert a["actions"].shape == (P, R, 1, K, 6) and a["actions"].dtype == np.float32
    assert a["z"].shape == (P, R, 1, K, D) and a["z"].dtype == np.float32
    assert a["root_readout"].shape == (R, 3) and a["root_readout"].dtype == np.float64
    assert np.abs(a["actions"]).max() <= 1.0 and (np.abs(a["actions"]) == 1.0).any() and (np.abs(a["actions"]) < 1.0).any()
    with torch.inference_mode():
        assert np.array_equal(a["root_readout"], probe(z_hist[:, -1]).double().numpy())
        assert np.array_equal(a["readout"], probe(torch.from_numpy(a["z"])).double().numpy())
        first = policy.act(torch.as_tensor(theta, dtype=torch.float32), policy.features(z_hist[:, 2], z_hist[:, 1]).expand(P, R, FEATURE_DIM))
    assert np.allclose(a["actions"][:, :, 0, 0], first.numpy(), rtol=0, atol=1e-6)  # step 0 reacts to z_t, z_(t-1)
    assert "z" not in im.rollout_policies(policy, theta, z_hist, hist_blocks)
    with pytest.raises(ValueError):
        im.rollout_policies(policy, theta, z_hist, hist_blocks, n_samples=2)  # k == 0 needs n_samples == 1
    with pytest.raises(ValueError):
        im.rollout_policies(policy, theta, z_hist, hist_blocks, n_samples=2, k=1.0)  # no sigma
    with pytest.raises(ValueError):
        im.rollout_policies(LinearLatentPolicy((scaler[0] + 0.1, scaler[1])), theta, z_hist, hist_blocks)


def test_noise_is_seeded_and_shared_across_candidates(model, scaler, probe, policy):
    rng = np.random.default_rng(4)
    R, n, K, k, seed = 2, 3, 10, 1.0, 5
    z1, h1 = roots(rng, 1)
    z_hist, hist_blocks = z1.repeat(R, 1, 1), np.repeat(h1, R, 0)  # two identical roots
    theta = thetas(rng, 3)
    theta[1] = theta[0]  # two identical candidates
    im = ClosedLoopImaginer(model, scaler, probe, device="cpu", sigma=sigma_vec())
    a = im.rollout_policies(policy, theta, z_hist, hist_blocks, n_samples=n, k=k, seed=seed, return_latents=True)
    b = im.rollout_policies(policy, theta, z_hist, hist_blocks, n_samples=n, k=k, seed=seed, return_latents=True)
    c = im.rollout_policies(policy, theta, z_hist, hist_blocks, n_samples=n, k=k, seed=seed + 1, return_latents=True)
    for key in ("readout", "actions", "z"):
        assert np.array_equal(a[key], b[key]), key
        assert np.array_equal(a[key][0], a[key][1]), key  # identical candidates, identical rollouts
    assert not np.allclose(a["z"], c["z"])  # another seed, other noise
    assert not np.allclose(a["z"][0, 0], a["z"][0, 1])  # same candidate, identical roots, different eps
    assert not np.allclose(a["z"][0, 0, 0], a["z"][0, 0, 1])  # samples differ
    eps = torch.randn((R, n, K, D), generator=torch.Generator(device="cpu").manual_seed(seed))
    th = torch.as_tensor(theta[2:], dtype=torch.float32)

    def act(t, z_t, z_prev):
        return policy.model_block(policy.act(th, policy.features(z_t, z_prev).view(1, R * n, FEATURE_DIM))).reshape(R * n, 60)

    rows = ClosedLoopImaginer(model, scaler, probe, device="cpu", sigma=sigma_vec())
    z_rows = z_hist.unsqueeze(1).expand(R, n, 3, D).reshape(R * n, 3, D)
    h_rows = rows.flat(hist_blocks).unsqueeze(1).expand(R, n, 2, 60).reshape(R * n, 2, 60)
    z_ref, _ = rows.rollout_core(z_rows, h_rows, act, n_blocks=K, k=k, eps=eps.reshape(R * n, K, D))
    assert np.allclose(a["z"][2].reshape(R * n, K, D), z_ref.numpy(), rtol=1e-6, atol=1e-6)  # row (p, r, s) uses eps[r, s]


def test_population_batching_equals_candidate_loop(model, scaler, probe, policy):
    rng = np.random.default_rng(5)
    R, n = 3, 2
    z_hist, hist_blocks = roots(rng, R)
    theta = thetas(rng, 5)
    im = ClosedLoopImaginer(model, scaler, probe, device="cpu", sigma=sigma_vec())
    kw = dict(n_samples=n, k=0.5, seed=11, return_latents=True)
    pop = im.rollout_policies(policy, theta, z_hist, hist_blocks, **kw)
    for p in range(len(theta)):
        one = im.rollout_policies(policy, theta[[p]], z_hist, hist_blocks, **kw)
        for key in ("readout", "actions", "z"):
            assert np.allclose(pop[key][p], one[key][0], rtol=1e-6, atol=1e-6), (p, key)
        assert np.array_equal(pop["root_readout"], one["root_readout"])


@pytest.mark.parametrize("k,n", [(0.0, 1), (2.0, 2)])
def test_chunking_does_not_change_results(model, scaler, probe, policy, k, n):
    rng = np.random.default_rng(6)
    R, P = 3, 5
    z_hist, hist_blocks = roots(rng, R)
    theta = thetas(rng, P)
    im = ClosedLoopImaginer(model, scaler, probe, device="cpu", sigma=sigma_vec())
    kw = dict(n_samples=n, k=k, seed=2, return_latents=True)
    full = im.rollout_policies(policy, theta, z_hist, hist_blocks, **kw)
    for max_batch in (R * n, 2 * R * n + 1):  # one and two candidates per chunk
        part = im.rollout_policies(policy, theta, z_hist, hist_blocks, max_batch=max_batch, **kw)
        for key in ("readout", "actions", "z", "root_readout"):
            assert np.allclose(full[key], part[key], rtol=1e-6, atol=1e-6), (max_batch, key)
    assert im.counters.imagined_rows == 3 * P * R * n * 10
    with pytest.raises(ValueError):
        im.rollout_policies(policy, theta, z_hist, hist_blocks, max_batch=R * n - 1, **kw)


def test_cuda_closed_loop_repeats_bitwise_for_a_fixed_configuration(model, scaler, probe):
    """On CUDA the chunk shapes, so P and max_batch, move the numbers slightly (readouts by up to about
    4e-3 with the real model, module doc); a fixed configuration repeats bitwise, which resumed runs
    rely on, and other chunk shapes stay close because the row layout is the same."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    im = ClosedLoopImaginer(copy.deepcopy(model), scaler, copy.deepcopy(probe), device="cuda", sigma=sigma_vec())
    pol = LinearLatentPolicy(scaler, device="cuda")
    rng = np.random.default_rng(14)
    R, P = 3, 5
    z_hist, hist_blocks = roots(rng, R)
    theta = thetas(rng, P)
    for kw in (dict(), dict(n_samples=2, k=1.0, seed=4)):
        runs = []
        for max_batch in (16384, 2 * R * kw.get("n_samples", 1) + 1):  # one chunk, then chunks of 2, 2 and 1 candidates
            a = im.rollout_policies(pol, theta, z_hist, hist_blocks, max_batch=max_batch, return_latents=True, **kw)
            b = im.rollout_policies(pol, theta, z_hist, hist_blocks, max_batch=max_batch, return_latents=True, **kw)
            assert all(np.array_equal(a[key], b[key]) for key in a), (kw, max_batch)
            runs.append(a)
        for key in runs[0]:
            assert np.allclose(runs[0][key], runs[1][key], rtol=0, atol=1e-4), (kw, key)


def test_counters_count_rows_encodes_and_time(model, scaler, probe, policy):
    rng = np.random.default_rng(7)
    P, R, n = 4, 3, 2
    z_hist, hist_blocks = roots(rng, R)
    im = ClosedLoopImaginer(model, scaler, probe, device="cpu", sigma=sigma_vec())
    im.rollout_policies(policy, thetas(rng, P), z_hist, hist_blocks, n_samples=n, k=1.0, max_batch=R * n)
    assert im.counters.imagined_rows == P * R * n * 10
    im.rollout_policies(policy, thetas(rng, P), z_hist, hist_blocks, n_blocks=5)
    assert im.counters.imagined_rows == P * R * n * 10 + P * R * 5
    im.rollout_tapes(z_hist[0], hist_blocks[0], rng.uniform(-1, 1, (6, 4, 10, 6)))
    assert im.counters.imagined_rows == P * R * n * 10 + P * R * 5 + 6 * 4
    frames = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
    assert torch.equal(im.encode(frames), WalkerImaginer(model, scaler, "cpu").encode(frames))
    c = im.counters
    assert c.encodes == 5 and c.real_steps == 0 and c.renders == 0 and c.wall_s > 0
    total = Counters()
    total.add(c)
    total.add({"real_steps": 100, "renders": 11, "wall_s": 0.5})
    assert total.as_dict() == {"imagined_rows": c.imagined_rows, "real_steps": 100, "renders": 11, "encodes": 5, "wall_s": c.wall_s + 0.5}
    with pytest.raises(ValueError):
        total.add({"imagined_steps": 1})


def test_one_step_residuals_equal_manual_teacher_forcing(model):
    rng = np.random.default_rng(8)
    z_win = torch.as_tensor(rng.normal(size=(9, 4, D)), dtype=torch.float32)
    a_win = torch.as_tensor(rng.normal(size=(9, 3, 60)), dtype=torch.float32)
    res = one_step_residuals(model, z_win, a_win)
    with torch.inference_mode():
        manual = model.predict(z_win[:, :3], model.action_encoder(a_win))[:, -1] - z_win[:, 3]
    assert res.shape == (9, D) and torch.equal(res, manual)
    assert torch.allclose(one_step_residuals(model, z_win.numpy(), a_win.numpy(), batch=4), manual, rtol=1e-6, atol=1e-6)
    sig = sigma_from_residuals(res)
    assert sig.shape == (D,) and sig.dtype == np.float64
    assert np.allclose(sig, res.double().numpy().std(0, ddof=1), rtol=1e-12, atol=0)
    with pytest.raises(ValueError):
        sigma_from_residuals(res[:1])


@pytest.mark.parametrize("sub", ["", "predictor", "predictor.transformer.layers.1.attn", "action_encoder", "pred_proj.net.1",
                                 "projector.net.1"])
def test_every_module_must_be_in_eval_mode(model, scaler, probe, policy, sub):
    """Dropout (0.1 in the predictor; attention dropout follows each attention module's own flag) or
    BatchNorm batch statistics in any submodule would make rollouts random or rows depend on each
    other, so one module left in train mode under an eval top level is refused like model.train()."""
    m = copy.deepcopy(model)  # a BatchNorm in train mode would update the shared fixture's statistics
    im = ClosedLoopImaginer(m, scaler, probe, device="cpu", sigma=sigma_vec())
    rng = np.random.default_rng(12)
    z_hist, hist_blocks = roots(rng, 2)
    z_seq = torch.as_tensor(rng.normal(size=(3, 13, D)), dtype=torch.float32)
    a_seq = torch.as_tensor(rng.normal(size=(3, 12, 60)), dtype=torch.float32)
    calls = {
        "rollout_core": lambda: im.rollout_core(z_hist, im.flat(hist_blocks), lambda t, z_t, z_prev: torch.zeros(2, 60)),
        "rollout_tapes": lambda: im.rollout_tapes(z_hist[0], hist_blocks[0], rng.uniform(-1, 1, (3, 10, 10, 6))),
        "rollout_policies": lambda: im.rollout_policies(policy, thetas(rng, 2), z_hist, hist_blocks, n_samples=2, k=1.0),
        "encode": lambda: im.encode(rng.integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)),
        "one_step_residuals": lambda: one_step_residuals(m, z_seq[:, :4], a_seq[:, :3]),
        "open_loop_errors": lambda: open_loop_errors(m, z_seq, a_seq, 10),
    }
    m.get_submodule(sub).train()
    assert m.training == (sub == "")  # a submodule alone leaves the top-level flag False
    for name, call in calls.items():
        try:
            call()
        except ValueError as e:
            assert "eval mode" in str(e), name
        else:
            pytest.fail(f"{name} accepted a model with {sub or 'every module'} in train mode")
    assert im.counters.imagined_rows == 0 and im.counters.encodes == 0
    m.eval()
    for call in calls.values():
        call()


def test_open_loop_errors_follow_the_tape_rollout(model, scaler):
    rng = np.random.default_rng(9)
    N, K = 6, 5
    z_seq = torch.as_tensor(rng.normal(size=(N, 3 + K, D)), dtype=torch.float32)
    a_seq = torch.as_tensor(rng.normal(size=(N, 2 + K, 60)), dtype=torch.float32)
    err = open_loop_errors(model, z_seq, a_seq, K)
    im = ClosedLoopImaginer(model, scaler, None, device="cpu")

    def tape(t, z_t, z_prev):
        return a_seq[:, 2 + t]

    pred, _ = im.rollout_core(z_seq[:, :3], a_seq[:, :2], tape, n_blocks=K)
    manual = np.sqrt(((pred.double() - z_seq[:, 3:].double()) ** 2).numpy().mean((0, 2)))
    assert err.shape == (K,) and err.dtype == np.float64
    assert np.allclose(err, manual, rtol=1e-12, atol=0)
    res = one_step_residuals(model, z_seq[:, :4], a_seq[:, :3])  # horizon one is the teacher-forced step
    assert np.isclose(err[0], np.sqrt((res.double() ** 2).mean().item()), rtol=1e-5)
    assert np.allclose(open_loop_errors(model, z_seq, a_seq, K, batch=4), err, rtol=1e-5, atol=0)
    assert np.allclose(open_loop_errors(model, z_seq, a_seq, 3), err[:3], rtol=1e-5, atol=0)


def test_real_model_on_two_dev_roots_is_bitwise_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not (MODEL_DIR / "weights.pt").is_file() or not (DEV_BANK / "branches.h5").is_file():
        pytest.skip("Walker LeWM or S4 dev bank not downloaded")
    from helpers.walkerBank import WalkerBank
    from helpers.walkerLewm import load_walker_model

    bank, ctx = WalkerBank(DEV_BANK), None
    try:
        try:
            from helpers.locoData import RenderContext

            ctx = RenderContext()
            frames = [ctx.render_many(r.history_qpos, r.history_qvel) for r in bank.roots[:2]]
        except Exception as e:  # noqa: BLE001  no EGL or MuJoCo on this machine
            pytest.skip(f"EGL rendering unavailable: {type(e).__name__}")
        model, scaler = load_walker_model(MODEL_DIR, "cuda")
        ref = WalkerImaginer(model, scaler, "cuda")
        im = ClosedLoopImaginer(model, scaler, None, device="cuda")
        all_tapes = bank.h5["tape"][:].astype(np.float64).reshape(-1, 10, 10, 6)
        z_hist = []
        for ri, root in enumerate(bank.roots[:2]):
            z = im.encode(frames[ri])
            assert torch.equal(z, ref.encode(frames[ri]))
            z_hist.append(z)
            for tapes in (all_tapes[bank.indices_for_root(ri)], np.concatenate([all_tapes] * 4)[:700]):  # 700 > chunk 512
                assert torch.equal(im.rollout_tapes(z, root.history_actions, tapes), ref.rollout(z, root.history_actions, tapes))
        assert im.counters.encodes == 6
        if PROBE_PATH.is_file():  # the closed loop is deterministic on CUDA with the real model and probe
            from helpers.poseProbes import load_probe

            probe, _ = load_probe(PROBE_PATH, "cuda")
            im = ClosedLoopImaginer(model, scaler, probe, device="cuda", sigma=np.full(D, 0.05))
            pol = LinearLatentPolicy(scaler, device="cuda")
            theta = thetas(np.random.default_rng(10), 3)
            hist = np.stack([r.history_actions for r in bank.roots[:2]])
            for kw in (dict(), dict(n_samples=2, k=1.0, seed=3)):
                a = im.rollout_policies(pol, theta, torch.stack(z_hist), hist, **kw)
                b = im.rollout_policies(pol, theta, torch.stack(z_hist), hist, **kw)
                assert all(np.array_equal(a[key], b[key]) for key in a)
                assert np.isfinite(a["readout"]).all()
    finally:
        bank.h5.close()
        if ctx is not None:
            ctx.close()
