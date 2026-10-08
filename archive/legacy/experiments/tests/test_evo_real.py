"""Real executor (evoReal): physics bitwise equal to `execute_branch`, a closed loop that reacts to
real encoded frames, task outcomes that do not depend on the lockstep group, accounting, and
spawned one-thread workers equal to the in-process path.

The physics tests need no assets. The closed-loop tests use a seeded tiny random LeWM and probe on
CPU, with a 4-frame encoder batch to stay fast, and skip cleanly when EGL rendering is unavailable.
One test repeats the group check with the downloaded model on CUDA at the default encoder batch
and skips without them."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import helpers.evoReal as evoReal  # noqa: E402
from helpers.evoPolicy import N_PARAMS, LinearLatentPolicy  # noqa: E402
from helpers.evoReal import _THREAD_KEYS, RealExecutor, RealPool, _worker_threads, evaluate_parallel  # noqa: E402
from helpers.evoRoots import load_roots_json  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.locoEnv import SafetyWalker2dVelocityEnv, make_loco_env  # noqa: E402
from helpers.poseProbes import Probe, ProbeSpec, load_probe, save_probe  # noqa: E402
from helpers.walkerBank import WalkerBank, endpoint_targets  # noqa: E402
from helpers.walkerLewm import build_model, load_walker_model  # noqa: E402
from helpers.walkerRules import execute_branch, health_clearance, roots_from_episode  # noqa: E402

D = 192
ENC = 4  # block-end frames per encoder call in the CPU tests (the default, 64, runs in the CUDA test)
HF = ROOT / "data" / "hf"
S4_DEV = HF / "safetydial-walker2d-data" / "banks" / "walker2d-s4-recovery-20260927-1" / "dev"
MODEL_DIR = HF / "safetydial-walker2d" / "lewm-a" / "walker2d-lewm-a-recovery-20260927-1"
PROBE = HF / "safetydial-walker2d" / "probes" / "walker2d-probes-recovery-20260927-1" / "walker_mlp.pt"
SCALER = (np.linspace(-0.2, 0.2, 6), np.linspace(0.5, 1.0, 6))
# Constant biases (W = 0), float32-exact; 1.5, -1.25 and 1.25 exercise the clip to the action box.
# From both test roots the first falls within 0.8 s and the second stays healthy throughout.
BIASES = np.array([[0.5, -0.25, 1.5, -1.25, 0.125, 0.0], [0.75, -0.75, 0.0, 1.25, 1.0, -0.5]])
_MLP = {"_target_": "stable_worldmodel.wm.lewm.module.MLP", "input_dim": D, "output_dim": D, "hidden_dim": 64,
        "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}}
TINY_LEWM = {  # the real config.json shrunk: 1-layer ViT with 32 px patches, 1-layer predictor
    "_target_": "stable_worldmodel.wm.lewm.LeWM",
    "encoder": {"_target_": "stable_pretraining.backbone.utils.vit_hf", "size": "tiny", "patch_size": 32, "image_size": 224,
                "pretrained": False, "use_mask_token": False, "num_hidden_layers": 1, "intermediate_size": 256},
    "predictor": {"_target_": "stable_worldmodel.wm.lewm.module.Predictor", "num_frames": 3, "input_dim": D, "hidden_dim": D,
                  "output_dim": D, "depth": 1, "heads": 2, "mlp_dim": 64, "dim_head": 16, "dropout": 0.0, "emb_dropout": 0.0},
    "action_encoder": {"_target_": "stable_worldmodel.wm.lewm.module.Embedder", "input_dim": 60, "emb_dim": D},
    "projector": _MLP, "pred_proj": _MLP,
}


def physics_env():
    env = make_loco_env("Walker2d", "v1", render=False, terminate_when_unhealthy=False, six_tuple=True)
    env.reset(seed=0)
    return env


def constant_theta(biases) -> np.ndarray:
    theta = np.zeros((len(biases), N_PARAMS))
    theta[:, -6:] = biases
    return theta


def assert_same_physics(out, i, log) -> None:
    for key in ("qpos", "qvel", "x_velocity"):
        assert np.array_equal(out[key][i], getattr(log, key)), key


def assert_same_tasks(out, ref, idx, label) -> None:
    """Every output of `out` equals the rows `idx` of `ref`, bitwise."""
    assert out.keys() == ref.keys(), label
    for key, v in out.items():
        assert np.array_equal(v, ref[key][idx]), (label, key)


@pytest.fixture(scope="module")
def roots():
    """Two roots of a synthetic episode: 60 seeded random actions from the reset state."""
    env = physics_env()
    u, actions = env.unwrapped, np.random.default_rng(0).uniform(-1.0, 1.0, (60, 6))
    qpos, qvel, xv = [u.data.qpos.copy()], [u.data.qvel.copy()], []
    for a in actions:
        x0 = u.data.qpos[0]
        u.do_simulation(a, u.frame_skip)
        qpos.append(u.data.qpos.copy())
        qvel.append(u.data.qvel.copy())
        xv.append((u.data.qpos[0] - x0) / u.dt)
    ep = {"qpos": np.array(qpos[:-1]), "qvel": np.array(qvel[:-1]), "action": actions, "x_velocity": np.array(xv), "healthy": np.ones(60, bool)}
    out = roots_from_episode(ep, [25, 40], episode=0, prefix="test")
    assert len(out) == 2
    return out


@pytest.fixture(scope="module")
def ctx():
    ctx = RenderContext()
    try:
        ctx.render_state(np.r_[0.0, 1.25, np.zeros(7)], np.zeros(9))
    except Exception as err:
        pytest.skip(f"EGL rendering unavailable: {type(err).__name__}: {err}")
    yield ctx
    ctx.close()


@pytest.fixture(scope="module")
def tiny():
    """A seeded tiny random LeWM, a walker probe and the linear policy, all on CPU in eval mode."""
    torch.manual_seed(0)
    model = build_model(TINY_LEWM).eval()
    probe = Probe(ProbeSpec(target="walker", hidden=(16,)), np.array([1.2, 0.0, 1.0]), np.array([0.1, 0.3, 1.0]), np.zeros(D), np.ones(D)).eval()
    return model, probe, LinearLatentPolicy(SCALER, device="cpu")


def executor(tiny, ctx, **kw) -> RealExecutor:
    model, probe, policy = tiny
    return RealExecutor(model, SCALER, probe, policy, device="cpu", ctx=ctx, **{"encode_batch": ENC, **kw})


# ---- physics only ---------------------------------------------------------------------------
def test_run_tapes_is_execute_branch_bitwise_and_repeatable(roots):
    tapes = np.random.default_rng(1).uniform(-1.3, 1.3, (2, 100, 6))
    ex = RealExecutor(None, None, None, None, device="cpu")
    logs = ex.run_tapes(roots, tapes)
    shared = physics_env()  # one env reused across branches, as the S4 bank builder ran them
    for root, tape, log in zip(roots, tapes, logs):
        for env in (physics_env(), shared):
            ref = execute_branch(env, root.qpos, root.qvel, tape)
            for key in ("qpos", "qvel", "x_velocity", "actions"):
                assert np.array_equal(getattr(log, key), getattr(ref, key)), key
            assert log.frames is None
    one_env = RealExecutor(None, None, None, None, device="cpu", max_envs=1)  # one env restored per root
    for again in (ex.run_tapes(roots, tapes), one_env.run_tapes(roots, tapes), one_env.run_tapes(roots[::-1], tapes[::-1])[::-1]):
        for log, ref in zip(again, logs, strict=True):
            for key in ("qpos", "qvel", "x_velocity"):
                assert np.array_equal(getattr(log, key), getattr(ref, key)), key
    float32_tapes = [root.policy_tape.astype(np.float32) for root in roots]  # bank tapes are float32
    for root, tape, log in zip(roots, float32_tapes, ex.run_tapes(roots, float32_tapes)):
        ref = execute_branch(physics_env(), root.qpos, root.qvel, tape)
        assert np.array_equal(log.qpos, ref.qpos) and np.array_equal(log.x_velocity, ref.x_velocity)
    assert ex.counters.real_steps == 3 * 2 * 100 and ex.counters.renders == ex.counters.encodes == 0
    assert ex.counters.wall_s > 0


def test_run_tapes_reproduces_saved_s4_branch_truth():
    """The downloaded S4 development bank (skipped when absent): stored qpos, qvel and x velocity
    of a policy tape and a perturbed tape per root, bitwise."""
    if not (S4_DEV / "branches.h5").is_file():
        pytest.skip("S4 development bank not downloaded")
    bank = WalkerBank(S4_DEV)
    try:
        js = [int(bank.indices_for_root(i)[k]) for i in range(3) for k in (0, 1)]
        roots = [bank.roots[int(bank.h5["root_index"][j])] for j in js]
        logs = RealExecutor(None, None, None, None, device="cpu").run_tapes(roots, np.stack([bank.h5["tape"][j] for j in js]))
        for j, log in zip(js, logs, strict=True):
            for key in ("qpos", "qvel", "x_velocity"):
                assert np.array_equal(getattr(log, key), bank.h5[key][j]), (j, key)
    finally:
        bank.h5.close()


def test_nonfinite_physics_raises_like_execute_branch(roots, monkeypatch):
    original = SafetyWalker2dVelocityEnv.do_simulation

    def corrupt(self, ctrl, n_frames):
        original(self, ctrl, n_frames)
        self.data.qvel[0] = np.nan

    monkeypatch.setattr(SafetyWalker2dVelocityEnv, "do_simulation", corrupt)
    with pytest.raises(FloatingPointError):
        RealExecutor(None, None, None, None, device="cpu").run_tapes(roots[:1], np.zeros((1, 100, 6)))
    with pytest.raises(FloatingPointError):
        execute_branch(physics_env(), roots[0].qpos, roots[0].qvel, np.zeros((100, 6)))


def test_bad_inputs_are_refused(roots, tmp_path):
    ex = RealExecutor(None, None, None, None, device="cpu")
    with pytest.raises(ValueError):
        ex.run_tapes(roots, np.zeros((1, 100, 6)))
    with pytest.raises(ValueError):
        ex.run(np.zeros((1, N_PARAMS - 1)), roots)
    with pytest.raises(ValueError):
        ex.run(np.zeros((1, N_PARAMS)), roots, pairs=[(0, 2)])
    with pytest.raises(ValueError):
        ex.run(np.zeros((1, N_PARAMS)), roots, pairs=[(-1, 0)])
    for bad in ({"max_envs": 0}, {"encode_batch": 0}, {"encode_batch": 257}):
        with pytest.raises(ValueError):
            RealExecutor(None, None, None, None, device="cpu", **bad)
    # Refused in the parent, before any worker is spawned (a worker failing to load dies as an
    # opaque BrokenProcessPool at the first evaluate). The files are empty: nothing is loaded.
    model_dir = tmp_path / "lewm"
    model_dir.mkdir()
    for name in ("config.json", "weights.pt", "probe.pt"):
        (model_dir / name).touch()
    kw = {"model_dir": model_dir, "probe_path": model_dir / "probe.pt", "policy_state": LinearLatentPolicy(SCALER).state(), "n_workers": 2}
    with pytest.raises(FileNotFoundError, match="scalers.npz"):  # load_walker_model reads it too
        RealPool(**kw)
    with pytest.raises(FileNotFoundError):
        evaluate_parallel(np.zeros((1, N_PARAMS)), roots, None, **{**kw, "model_dir": ROOT / "no-such-model"})
    for bad in ({"n_workers": -1}, {"chunk_tasks": 0}, {"max_envs": 0}, {"encode_batch": 0}, {"encode_batch": 257}):
        with pytest.raises(ValueError):
            RealPool(**{**kw, **bad})


# ---- closed loop (rendering + tiny model) ---------------------------------------------------
def test_constant_policy_run_equals_execute_branch(roots, ctx, tiny):
    ex = executor(tiny, ctx, max_envs=3)  # 4 tasks: lockstep groups of 3 and 1
    out = ex.run(constant_theta(BIASES), roots, record_qpos=True)
    assert out["pairs"].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i, (p, r) in enumerate(out["pairs"]):
        assert np.array_equal(out["actions"][i], np.tile(np.clip(BIASES[p], -1.0, 1.0), (10, 1)))
        log = execute_branch(physics_env(), roots[r].qpos, roots[r].qvel, np.tile(BIASES[p], (100, 1)))
        assert_same_physics(out, i, log)
        clearance = health_clearance(log.qpos[1:, 1], log.qpos[1:, 2])
        unsafe = clearance <= 0
        assert np.array_equal(out["dense_clearance"][i], clearance)
        assert out["dense_violated"][i] == unsafe.any()
        assert out["dense_first_step"][i] == (int(np.argmax(unsafe)) if unsafe.any() else 100)
        assert np.array_equal(out["endpoint_targets"][i], endpoint_targets(log.qpos, log.x_velocity))
        assert np.array_equal(out["endpoint_clearance"][i], clearance[9::10])
        assert np.array_equal(out["block_progress"][i], log.qpos[10::10, 0] - log.qpos[:-1:10, 0])
        assert out["progress"][i] == log.qpos[100, 0] - log.qpos[0, 0]
    assert out["dense_violated"].tolist() == [True, True, False, False]  # both outcomes are covered
    assert (out["dense_first_step"][:2] < 100).all() and out["dense_first_step"][2:].tolist() == [100, 100]
    assert out["z"].shape == (4, 10, D) and out["z"].dtype == np.float32
    assert out["readout"].shape == (4, 10, 3) and out["readout"].dtype == np.float64
    assert out["dense_first_step"].dtype == np.int64 and out["dense_violated"].dtype == bool
    c = ex.counters  # padding frames are not counted
    assert c.real_steps == 4 * 100 and c.renders == c.encodes == 4 * 10 + 2 * 3 and c.imagined_rows == 0
    again = ex.run(constant_theta(BIASES), roots)
    assert "qpos" not in again and "qvel" not in again
    for key, v in again.items():
        assert np.array_equal(v, out[key]), key
    assert ex.counters.renders == 46 + 40  # history latents are cached by root_id


def test_policy_reacts_to_real_encoded_block_ends(roots, ctx, tiny):
    """The spec's per-task formulas, recomputed task by task: each block-end frame encoded as if
    alone, `policy.act(theta[[p]], feats)`, and the probe on the segment's latents."""
    _, probe, policy = tiny
    theta = np.random.default_rng(2).normal(0.0, 0.05, (2, N_PARAMS))
    ex = executor(tiny, ctx)
    out = ex.run(theta, roots, record_qpos=True)  # one lockstep group of 4: one full encoder call per block
    hist = torch.stack([ex.encode(ctx.render_many(root.history_qpos, root.history_qvel)) for root in roots])
    black = np.zeros((ENC - 1, 224, 224, 3), np.uint8)
    p, r = out["pairs"][:, 0], out["pairs"][:, 1]
    z = torch.as_tensor(out["z"])
    for b in range(10):  # block-end latents: the true block-end states rendered, each encoded with black padding only
        frames = ctx.render_many(out["qpos"][:, 10 * (b + 1)], out["qvel"][:, 10 * (b + 1)])
        assert torch.equal(z[:, b], torch.stack([ex.imaginer.encode(np.concatenate([f[None], black]))[0] for f in frames]))
    seq = torch.cat([hist[torch.as_tensor(r)][:, 1:], z], 1)  # z_(t-10), z_t of the root, then the 10 real block ends
    for b in range(10):  # block b acts on (z_t, z_prev): the history latents first, then real ones
        for i in range(4):
            a = policy.act(torch.as_tensor(theta[[p[i]]], dtype=torch.float32), policy.features(seq[i : i + 1, b + 1], seq[i : i + 1, b])[:, None])
            assert np.array_equal(out["actions"][i, b], a[0, 0].double().numpy()), (i, b)
    assert np.abs(out["actions"]).max() > 0 and np.ptp(out["actions"], axis=1).max() > 0  # actions vary by block
    assert np.array_equal(out["readout"], np.stack([probe.predict(seg) for seg in z]).astype(np.float64))


def test_task_outcome_does_not_depend_on_its_group(roots, ctx, tiny):
    """Alone, in full or partial groups, at another slot or next to a copy of itself: a task gives
    bitwise the same segment. Batched, CPU kernels already differ by slot and batch size in the
    last bits, which the closed loop amplifies."""
    theta = np.random.default_rng(5).normal(0.0, 0.05, (2, N_PARAMS))
    alone = executor(tiny, ctx, max_envs=1).run(theta, roots, record_qpos=True)  # 4 tasks, one per group
    pairs = [(1, 1), (0, 0), (1, 1), (1, 0), (0, 1)]  # a duplicate at slots 0 and 2; 5 frames = 4 + 1 padded
    for max_envs, sel, idx in ((4, None, np.arange(4)), (3, None, np.arange(4)), (5, pairs, [3, 0, 3, 2, 1])):
        assert_same_tasks(executor(tiny, ctx, max_envs=max_envs).run(theta, roots, sel, record_qpos=True), alone, idx, max_envs)
    assert len(np.unique(alone["actions"][:, 0], axis=0)) == 4  # four distinct tasks


def test_pairs_select_tasks_and_history_is_cached_or_given(roots, ctx, tiny):
    theta = np.random.default_rng(3).normal(0.0, 0.05, (2, N_PARAMS))
    ex = executor(tiny, ctx)  # one group per run, so the subset below shares its group with other tasks
    full = ex.run(theta, roots)
    assert ex.counters.renders == ex.counters.encodes == 4 * 10 + 2 * 3
    sub = ex.run(theta, roots, pairs=[(1, 1), (0, 0), (1, 0)])
    assert ex.counters.renders == 46 + 3 * 10 and ex.counters.real_steps == 7 * 100
    assert_same_tasks(sub, full, [3, 0, 2], "pairs")
    z_hist = ex.history_latents(roots)
    assert z_hist.shape == (2, 3, D) and ex.counters.renders == 76
    given = executor(tiny, ctx)
    assert_same_tasks(given.run(theta, roots, z_hist=z_hist.numpy()), full, np.arange(4), "z_hist")
    assert given.counters.renders == given.counters.encodes == 4 * 10  # no history frames
    assert all(len(v) == 0 for v in ex.run(theta, roots, pairs=[]).values())
    with pytest.raises(ValueError, match="z_hist"):
        given.run(theta, roots, z_hist=z_hist[:1])
    with pytest.raises(ValueError, match="different history"):
        ex.history_latents([replace(roots[0], history_qpos=roots[0].history_qpos + 0.01)])


def test_real_model_outcomes_do_not_depend_on_the_group(ctx):
    """The downloaded LeWM and probe on CUDA at the default encoder batch, where batched kernels
    differ most (a lone frame's latent by 6e-4): a task alone, in partial groups or next to a copy
    of itself gives the same segment. Skipped without CUDA or the assets."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not all(p.is_file() for p in (MODEL_DIR / "weights.pt", MODEL_DIR / "scalers.npz", PROBE, S4_DEV / "roots.json")):
        pytest.skip("downloaded model, probe or S4 development roots absent")
    model, scaler = load_walker_model(MODEL_DIR, "cuda")
    probe, _ = load_probe(PROBE, "cuda")
    roots = load_roots_json(S4_DEV / "roots.json")[:2]
    policy = LinearLatentPolicy(scaler, device="cuda")
    theta = np.random.default_rng(6).normal(0.0, 0.05, (3, N_PARAMS))

    def run(max_envs, pairs=None):
        return RealExecutor(model, scaler, probe, policy, device="cuda", ctx=ctx, max_envs=max_envs).run(theta, roots, pairs, record_qpos=True)

    alone = run(1)
    for max_envs, sel, idx in ((6, None, np.arange(6)), (4, None, np.arange(6)), (64, [(2, 1), (0, 0), (2, 1)], [5, 0, 5])):
        assert_same_tasks(run(max_envs, sel), alone, idx, max_envs)
    assert np.ptp(alone["actions"], axis=0).max() > 0  # the tasks differ


def test_parallel_workers_equal_the_in_process_path(roots, ctx, tiny, tmp_path, monkeypatch):
    model, probe, policy = tiny
    model_dir = tmp_path / "lewm"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(TINY_LEWM))
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, model_dir / "weights.pt")
    np.savez(model_dir / "scalers.npz", action_mean=SCALER[0], action_std=SCALER[1])
    save_probe(probe, {}, tmp_path / "probe.pt")
    theta = np.vstack([constant_theta(BIASES[:1]), np.random.default_rng(4).normal(0.0, 0.05, (1, N_PARAMS))])
    kw = {"model_dir": model_dir, "probe_path": tmp_path / "probe.pt", "policy_state": policy.state(), "device": "cpu",
          "max_envs": 2, "chunk_tasks": 2, "encode_batch": ENC}
    serial = evaluate_parallel(theta, roots, None, n_workers=0, record_qpos=True, **kw)
    keys = serial.keys() - {"counters", "elapsed_s"}
    for key in _THREAD_KEYS:  # what spawned workers inherit unless they pin themselves to one thread
        monkeypatch.setenv(key, "3")
    with RealPool(n_workers=2, **kw) as pool:
        assert pool._pool._mp_context.get_start_method() == "spawn"
        threads = [f.result() for f in [pool._pool.submit(_worker_threads) for _ in range(4)]]
        parallel = pool.evaluate(theta, roots, record_qpos=True)
        recut = pool.evaluate(theta, roots, chunk_tasks=3)  # the call form of evo_eval_snapshots.py
    assert threads == [{"torch": 1, **dict.fromkeys(_THREAD_KEYS, "1")}] * 4
    for key in keys:
        assert np.array_equal(serial[key], parallel[key]), key
    assert recut.keys() == serial.keys() - {"qpos", "qvel"}
    for key in keys - {"qpos", "qvel"}:
        assert np.array_equal(recut[key], serial[key]), key
    assert serial["pairs"].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i, root in enumerate(roots):  # the constant candidate's physics is exact in the workers too
        assert_same_physics(parallel, i, execute_branch(physics_env(), root.qpos, root.qvel, np.tile(BIASES[0], (100, 1))))
    s, w = serial["counters"], parallel["counters"]
    assert s.real_steps == w.real_steps == 400 and s.renders == s.encodes == 46  # one executor: histories once
    assert w.renders == w.encodes and w.renders in (46, 52)  # each worker renders the histories it needs
    assert s.wall_s > 0 and w.wall_s > 0 and serial["elapsed_s"] > s.wall_s  # elapsed includes loading the executor

    chunks = []  # (tasks, executor seconds) of every in-process chunk
    run_chunk = evoReal._run_chunk

    def spy(*args, **kwargs):
        res = run_chunk(*args, **kwargs)
        chunks.append((len(args[2]), res[1].wall_s))
        return res

    monkeypatch.setattr(evoReal, "_run_chunk", spy)
    with RealPool(n_workers=0, **kw) as pool:  # one executor kept across calls
        first = pool.evaluate(theta, roots, record_qpos=True)
        picked = pool.evaluate(theta, roots, [(1, 1), (0, 1)], record_qpos=True)
        assert [n for n, _ in chunks] == [2, 2, 2]
        chunks.clear()
        cut = pool.evaluate(theta, roots, chunk_tasks=3)
        assert [n for n, _ in chunks] == [3, 1]  # this call's chunk size, not the pool's
        assert cut["counters"].wall_s == sum(t for _, t in chunks) and cut["elapsed_s"] >= cut["counters"].wall_s  # summed
        with pytest.raises(ValueError, match="chunk_tasks"):
            pool.evaluate(theta, roots, chunk_tasks=0)
    for key in keys:
        assert np.array_equal(first[key], serial[key]), key
    for key in keys - {"qpos", "qvel"}:
        assert np.array_equal(cut[key], serial[key]), key
    assert picked["pairs"].tolist() == [[1, 1], [0, 1]] and picked["qpos"].shape == (2, 101, 9)
    assert picked["counters"].renders == 2 * 10 and picked["counters"].real_steps == 200  # histories cached from the first call
    assert_same_physics(picked, 1, execute_branch(physics_env(), roots[1].qpos, roots[1].qvel, np.tile(BIASES[0], (100, 1))))
    with pytest.raises(ValueError, match="closed"):
        pool.evaluate(theta, roots)
