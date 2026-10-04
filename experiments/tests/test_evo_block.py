"""Sequential action order, fixed-tape equivalence, feedback, and worker invariance."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.evoBlockPolicy import BLOCK_DIM, FEATURE_DIM, N_PARAMS, LinearBlockPolicy  # noqa: E402
from helpers.evoBlockReal import BlockRealExecutor, evaluate_blocks_parallel  # noqa: E402
from helpers.evoImagine import ClosedLoopImaginer  # noqa: E402
from helpers.evoInputs import load_stage_config  # noqa: E402
from helpers.evoRoots import load_roots_json  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.walkerLewm import load_walker_model  # noqa: E402
from helpers.walkerRules import execute_branch  # noqa: E402
import test_evo_real as real_fixtures  # noqa: E402
from test_evo_real import MODEL_DIR, PROBE, S4_DEV, SCALER, physics_env  # noqa: E402

ctx, roots, tiny = real_fixtures.ctx, real_fixtures.roots, real_fixtures.tiny


@pytest.fixture(scope="module", autouse=True)
def few_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def block_theta(seed=0, scale=0.001):
    return np.random.default_rng(seed).normal(0, scale, (2, N_PARAMS))


def test_parameter_roundtrip_and_population_actions():
    policy = LinearBlockPolicy(SCALER)
    theta = block_theta(scale=0.03)
    w, b = policy.unpack(theta)
    assert np.array_equal(policy.pack(w, b), theta)
    tt = torch.tensor(theta)
    assert torch.equal(policy.pack(*policy.unpack(tt)), tt)
    feats = torch.randn(2, 3, 4, FEATURE_DIM)
    out = policy.act(theta, feats)
    expected = torch.stack([(feats[i] @ torch.tensor(w[i], dtype=torch.float32).T
                             + torch.tensor(b[i], dtype=torch.float32)).clamp(-1, 1)
                            for i in range(2)])
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)
    assert out.shape == (2, 3, 4, BLOCK_DIM)
    restored = LinearBlockPolicy.from_state(policy.state())
    assert torch.equal(restored.act(theta, feats), out)
    with pytest.raises(ValueError):
        LinearBlockPolicy.from_state({k: v for k, v in policy.state().items() if k != "policy_kind"})
    with pytest.raises(TypeError, match="do not hold"):
        policy.hold(out)


def test_time_order_and_scaler_match_reference(tiny):
    model, probe, _ = tiny
    policy = LinearBlockPolicy(SCALER)
    im = ClosedLoopImaginer(model, SCALER, probe, device="cpu")
    actions = torch.linspace(-1.5, 1.5, 60).reshape(1, 60)
    block = policy.raw_block(actions)
    assert torch.equal(block.reshape(1, 60), actions.clamp(-1, 1))
    assert not torch.equal(block[:, 0], block[:, 1])
    assert torch.equal(policy.model_block(actions), im.flat(block.numpy()))


def test_full_block_imagination_equals_fixed_tape(tiny):
    model, probe, _ = tiny
    policy = LinearBlockPolicy(SCALER)
    im = ClosedLoopImaginer(model, SCALER, probe, device="cpu")
    theta = np.zeros((1, N_PARAMS))
    theta[0, -60:] = np.linspace(-0.9, 0.9, 60)
    zh = torch.randn(1, 3, 192)
    hist = np.zeros((1, 2, 10, 6), np.float32)
    out = im.rollout_policies(policy, theta, zh, hist, return_latents=True)
    tapes = out["actions"][0, 0].reshape(1, 10, 10, 6)
    ref = im.rollout_tapes(zh[0], hist[0], tapes)
    assert np.array_equal(out["z"][0, 0], ref.numpy())
    assert out["actions"].shape == (1, 1, 1, 10, 60)
    # A nonconstant policy must feed each new imagined latent into the next decision.
    theta = block_theta(scale=0.005)[:1]
    out = im.rollout_policies(policy, theta, zh, hist, return_latents=True)
    for t in range(10):
        zt = zh[:, -1] if t == 0 else torch.tensor(out["z"][0, :, 0, t-1])
        zp = zh[:, -2] if t == 0 else (zh[:, -1] if t == 1 else torch.tensor(out["z"][0, :, 0, t-2]))
        expected = policy.act(theta, policy.features(zt, zp)[:, None]).numpy()
        np.testing.assert_allclose(out["actions"][0, :, 0, t], expected[0], atol=1e-6)


def test_real_sequential_actions_equal_exact_physics_and_grouping(tiny, ctx, roots):
    model, probe, _ = tiny
    policy = LinearBlockPolicy(SCALER)
    theta = block_theta(scale=0.004)
    with torch.inference_mode():
        ex = BlockRealExecutor(model, SCALER, probe, policy, device="cpu", ctx=ctx,
                               max_envs=4, encode_batch=4)
        out = ex.run(theta, roots, record_qpos=True)
        other = BlockRealExecutor(model, SCALER, probe, policy, device="cpu", ctx=ctx,
                                  max_envs=1, encode_batch=4)
        again = other.run(theta, roots, record_qpos=True)
    assert ex.counters.real_steps == 400
    env = physics_env()
    for i, (_, r) in enumerate(out["pairs"]):
        tape = out["actions"][i].reshape(100, 6)
        assert not np.array_equal(tape[0], tape[1])
        log = execute_branch(env, roots[r].qpos, roots[r].qvel, tape)
        for key in ("qpos", "qvel", "x_velocity"):
            assert np.array_equal(out[key][i], getattr(log, key)), key
    for key in out:
        assert np.array_equal(out[key], again[key]), key
    env.close()
    ex.close()
    other.close()


def test_bc_targets_are_future_actions_in_time_order(tmp_path):
    path = ROOT / "experiments/scripts/evo_block_bc_init.py"
    spec = importlib.util.spec_from_file_location("block_bc_test", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    h5 = tmp_path / "states.h5"
    acts = np.arange(35*6).reshape(35, 6) / 300
    with h5py.File(h5, "w") as f:
        f["ep_offset"], f["ep_len"] = [0], [35]
        f["qpos"] = np.repeat(np.arange(35)[:, None], 9, axis=1)
        f["qvel"] = np.zeros((35, 9))
        f["action"] = acts

    class Context:
        def render_many(self, qp, qv):
            return qp[:, :1]

    X, Y, E, n = script.bc_frames(h5, [0], Context(),
                                  lambda x: torch.tensor(x).repeat(1, 192), 5)
    assert n == 7 and Y.shape == (4, 60)
    for row, t in enumerate((10, 15, 20, 25)):
        assert np.array_equal(Y[row], acts[t:t+10].reshape(-1))
        assert np.all(X[row, :192] == t) and np.all(X[row, 192:] == 10)
    assert not E.any()


def test_gate_configuration_preserves_thresholds_data_and_roots():
    old, new = load_stage_config("stage2"), load_stage_config("stage2_block")
    for key in ("inputs", "roots", "gate0", "evaluation"):
        assert old[key] == new[key]
    for key in old.bc:
        assert old.bc[key] == new.bc[key]
    assert new.policy.n_params == N_PARAMS
    assert new.interpretation.evaluation_role == "development_gate_rerun"


def test_downloaded_cuda_model_worker_and_chunk_invariance():
    if not torch.cuda.is_available() or not (MODEL_DIR / "weights.pt").is_file():
        pytest.skip("needs CUDA and pinned assets")
    model, scaler = load_walker_model(MODEL_DIR, "cuda")
    probe, _ = load_probe(PROBE, "cuda")
    policy = LinearBlockPolicy(scaler, device="cuda")
    roots = load_roots_json(S4_DEV / "roots.json")[:2]
    theta = block_theta(scale=0.003)
    ctx = RenderContext()
    ex = BlockRealExecutor(model, scaler, probe, policy, device="cuda", ctx=ctx,
                           max_envs=4, encode_batch=64)
    ref = ex.run(theta, roots, record_qpos=True)
    ex.close()
    ctx.close()
    out = evaluate_blocks_parallel(theta, roots, None, model_dir=MODEL_DIR, probe_path=PROBE,
                                    policy_state=policy.state(), n_workers=2, device="cuda",
                                    max_envs=1, chunk_tasks=1, encode_batch=64, record_qpos=True)
    for key in ref:
        assert np.array_equal(out[key], ref[key]), key
    assert out["counters"].real_steps == 400
