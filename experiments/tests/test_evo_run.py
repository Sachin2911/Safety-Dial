"""CMA-ES loop: rank-based progress, exact resume, snapshots and no use of numpy's global RNG.

The core tests run `run_cmaes` on a toy score function with a small reference ranking that
follows the evoRanking contract, so they need neither the LeWM nor `helpers.evoRanking`; the
tests marked as integration run the real ranking module when it is importable. Others pin
down what pycma does behind the loop's back: the TPA step-size rule's global draws, the
geno-pheno transform that `conditioncov_alleviate` installs, and `out_dir` holding only the
run's own snapshots.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from cma.sigma_adaptation import CMAAdaptSigmaTPA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.evoRun import (RULES, CMAConfig, generation_roots, imagined_score_fn,  # noqa: E402
                            noise_seed, run_cmaes)

N_ROOTS, N_SAMPLES, K = 12, 2, 10
TARGET = np.array([1.0, 0.8, -0.6, 0.4, 0.2, -0.2])
THRESH = 0.5 + 0.1 * np.linspace(-1.0, 1.0, N_ROOTS)  # x[0] above a root's threshold violates
OFFSET = np.linspace(-0.5, 0.5, N_ROOTS)  # per-root return offset, equal for every candidate
THETA0 = np.zeros(6)


# ---- reference ranking (the evoRanking contract) -----------------------------------------------
class RefRosarl:
    def __init__(self):
        self.v_min, self.v_max, self.n_seen = np.inf, -np.inf, 0

    def update(self, task_returns):
        r = np.asarray(task_returns, dtype=np.float64).ravel()
        self.v_min, self.v_max = min(self.v_min, float(r.min())), max(self.v_max, float(r.max()))
        self.n_seen += r.size

    @property
    def penalty(self):
        return self.v_min - self.v_max if self.n_seen else 0.0

    def state(self):
        return {"v_min": self.v_min, "v_max": self.v_max, "n_seen": self.n_seen}

    @classmethod
    def from_state(cls, d):
        out = cls()
        out.v_min, out.v_max, out.n_seen = d["v_min"], d["v_max"], d["n_seen"]
        return out


def ref_rank(rule, *, violated, ret, ret_pre, lam=None, rosarl=None):
    p, mean_ret, idx = violated.mean(1), ret.mean(1), np.arange(len(ret))
    scores = None
    if rule == "deb":
        order = np.lexsort((idx, -mean_ret, p))
    else:
        if rule == "rosarl":
            rosarl.update(np.where(violated, ret_pre, ret))
            scores = np.where(violated, ret_pre + rosarl.penalty, ret).mean(1)
        else:
            scores = (ret - lam * violated).mean(1)
        order = np.lexsort((idx, -scores))
    ranks = np.empty(len(ret), dtype=np.int64)
    ranks[order] = idx
    return ranks, {"p": p, "mean_ret": mean_ret, "scores": scores}


REF = types.SimpleNamespace(rank_candidates=ref_rank, RosarlPenalty=RefRosarl,
                            ranks_to_fitness=lambda ranks: np.asarray(ranks, dtype=np.float64))


# ---- toy problem -------------------------------------------------------------------------------
def toy_score(X, roots_idx, g):
    """Segment (root r, sample s) is violated when x[0] exceeds r's threshold; return is
    5 - ||x - c||^2 plus r's offset, and a violated segment keeps 40% of it before the violation."""
    seg = np.repeat(roots_idx, N_SAMPLES)
    violated = X[:, :1] > THRESH[seg]
    ret = 5.0 - np.sum((X - TARGET) ** 2, 1, keepdims=True) + OFFSET[seg]
    return {"violated": violated, "ret": ret, "ret_pre": np.where(violated, 0.4 * ret, ret),
            "rows": len(X) * len(seg) * K}


def config(rule="deb", **kw):
    base = dict(popsize=8, generations=10, sigma0=0.3, seed=3, k_roots=4, n_samples=N_SAMPLES,
                rule=rule, lam=1.0 if rule == "fixed" else None, eval_generations=(0, 4, 7, 10),
                checkpoint_every=3)
    return CMAConfig(**{**base, **kw})


def run(cfg, out, score=toy_score, theta0=THETA0, **kw):
    return run_cmaes(cfg, theta0, score, out, n_fitness_roots=N_ROOTS, log=None, ranking=REF, **kw)


def crashing(at, score=toy_score):
    def crash(X, roots_idx, g):
        if g == at:
            raise RuntimeError("simulated crash")
        return score(X, roots_idx, g)

    return crash


def saved_strategy(out):
    with open(Path(out) / "checkpoint.pkl", "rb") as f:
        return pickle.load(f)["es"]


def snapshot_files(out):
    return sorted(p.name for p in (Path(out) / "snapshots").iterdir())


def names(res):
    return [Path(p).name for p in res["snapshots"]]


def log_rows(out):
    lines = (Path(out) / "log.jsonl").read_text().splitlines()
    return [{k: v for k, v in json.loads(line).items() if k != "wall_s"} for line in lines]


def assert_same_run(a, b, out_a, out_b):
    """Equal log rows (except wall_s), result, counters and byte-identical snapshot files."""
    assert log_rows(out_a) == log_rows(out_b) == [
        {k: v for k, v in row.items() if k != "wall_s"} for row in a["history"]]
    assert [{k: v for k, v in r.items() if k != "wall_s"} for r in b["history"]] == log_rows(out_b)
    assert np.array_equal(a["best_theta"], b["best_theta"])
    assert np.array_equal(a["mean_theta"], b["mean_theta"]) and a["sigma"] == b["sigma"]
    assert a["best_stats"] == b["best_stats"]
    assert {k: v for k, v in a["counters"].items() if k != "wall_s"} == {
        k: v for k, v in b["counters"].items() if k != "wall_s"}
    assert [Path(p).name for p in a["snapshots"]] == [Path(p).name for p in b["snapshots"]]
    for pa, pb in zip(a["snapshots"], b["snapshots"]):
        assert sha256(pa) == sha256(pb)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_global_state(s, t):
    return s[0] == t[0] and np.array_equal(s[1], t[1]) and tuple(s[2:]) == tuple(t[2:])


# ---- tests -------------------------------------------------------------------------------------
@pytest.mark.parametrize("rule", RULES)
def test_ranked_selection_improves_the_toy_problem_under_every_rule(tmp_path, rule):
    res = run(config(rule, generations=40, sigma0=0.5, eval_generations=()), tmp_path)
    start = np.sum((THETA0 - TARGET) ** 2)
    assert np.sum((res["mean_theta"] - TARGET) ** 2) < 0.5 * start
    assert res["mean_theta"][0] < THRESH.max()  # drawn toward the target, held by the constraint
    hist = res["history"]
    assert [row["g"] for row in hist] == list(range(40))
    assert hist[-1]["mean_ret_median"] > hist[0]["mean_ret_median"] + 1.0
    best = res["best_stats"]
    if rule == "deb":
        assert best["p"] == 0.0 and best["key"] == [0.0, -best["mean_ret"]]
    else:
        assert best["key"] == [-best["score"]]
    if rule == "rosarl":
        assert hist[-1]["penalty"] < hist[0]["penalty"] < 0.0  # V_MIN - V_MAX widens over the run
    else:
        assert all(row["penalty"] == 0.0 for row in hist)
    # the stored best theta reproduces its imagined stats on its own roots
    redo = toy_score(res["best_theta"][None], np.array(best["roots_idx"]), best["generation"])
    assert redo["violated"].mean() == best["p"]
    assert redo["ret"].mean() == pytest.approx(best["mean_ret"], abs=1e-12)


@pytest.mark.parametrize("rule", RULES)
def test_resume_with_more_generations_equals_one_uninterrupted_run(tmp_path, rule):
    cfg = config(rule)
    full = run(cfg, tmp_path / "full")
    part = run(replace(cfg, generations=5), tmp_path / "split")
    assert len(part["history"]) == 5 and len(log_rows(tmp_path / "split")) == 5
    np.random.seed(12345)
    np.random.standard_normal(11)  # perturb numpy's global RNG between the calls
    rest = run(cfg, tmp_path / "split")
    assert_same_run(full, rest, tmp_path / "full", tmp_path / "split")


def test_resume_after_a_crash_drops_log_rows_past_the_checkpoint(tmp_path):
    cfg = config("rosarl")
    full = run(cfg, tmp_path / "full")
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(cfg, tmp_path / "crash", score=crashing(8))
    assert len(log_rows(tmp_path / "crash")) == 8  # rows 0..7 written, last checkpoint at g = 6
    res = run(cfg, tmp_path / "crash")
    assert_same_run(full, res, tmp_path / "full", tmp_path / "crash")


def test_resume_restores_numpy_global_rng_for_score_functions_that_draw_from_it(tmp_path):
    def noisy(X, roots_idx, g):
        out = toy_score(X, roots_idx, g)
        out["ret"] = out["ret"] + 0.05 * np.random.standard_normal(out["ret"].shape)
        return out

    cfg = config("deb")
    np.random.seed(7)
    full = run(cfg, tmp_path / "full", score=noisy)
    np.random.seed(7)
    run(replace(cfg, generations=4), tmp_path / "split", score=noisy)
    np.random.seed(99)
    rest = run(cfg, tmp_path / "split", score=noisy)
    assert_same_run(full, rest, tmp_path / "full", tmp_path / "split")


@pytest.mark.parametrize("cma_options,n_params", [
    ({}, 6), ({"CMA_diagonal": True}, 6), ({"CMA_diagonal": 3}, 6), ({"CMA_diagonal": True}, 300),
    ({"CMA_mirrors": 4, "CMA_mirrormethod": 0}, 6)],
    ids=["full", "diagonal", "diagonal-then-full", "diagonal-tpa", "mirrored"])
def test_numpy_global_rng_is_never_used(tmp_path, cma_options, n_params):
    """pycma samples from the run's private stream only, also after a diagonal-to-full switch.
    Its global draws that change no result are undone: above 299 diagonal parameters (the real
    policy has 2310) the TPA step-size rule draws consistency-check indices in every tell, and
    unconditional mirroring rounds its mirror count with a draw in every ask."""
    cfg = config(cma_options=cma_options, eval_generations=())
    theta0 = np.zeros(n_params)

    def score(X, roots_idx, g):  # the toy problem on the first coordinates
        return toy_score(X[:, :TARGET.size], roots_idx, g)

    np.random.seed(0)
    before = np.random.get_state()
    a = run(cfg, tmp_path / "a", score=score, theta0=theta0)
    assert same_global_state(before, np.random.get_state())
    assert isinstance(saved_strategy(tmp_path / "a").adapt_sigma, CMAAdaptSigmaTPA) == (
        n_params > 299)
    np.random.seed(2024)
    np.random.uniform(size=5)
    b = run(cfg, tmp_path / "b", score=score, theta0=theta0)
    assert_same_run(a, b, tmp_path / "a", tmp_path / "b")
    c = run(replace(cfg, seed=4), tmp_path / "c", score=score, theta0=theta0)
    assert not np.array_equal(a["mean_theta"], c["mean_theta"])


def test_snapshots_hold_the_state_after_g_tells_including_the_start(tmp_path):
    cfg = config(eval_generations=(10, 0, 4, 7, 99))
    res = run(cfg, tmp_path)
    names = ["g0000.npz", "g0004.npz", "g0007.npz", "g0010.npz"]
    assert [Path(p).name for p in res["snapshots"]] == names
    assert not (tmp_path / "snapshots" / "g0099.npz").exists()
    with np.load(res["snapshots"][0], allow_pickle=False) as z:  # g = 0: theta0, nothing evaluated
        assert np.array_equal(z["best_theta"], THETA0) and np.array_equal(z["mean_theta"], THETA0)
        assert z["sigma"] == cfg.sigma0 and z["generation"] == 0 and z["best_generation"] == -1
        assert np.isnan(z["best_p"]) and z["best_key"].size == 0
        assert json.loads(z["best_stats"].item()) == {}
    hist = res["history"]
    for path in res["snapshots"][1:]:
        with np.load(path, allow_pickle=False) as z:
            g, stats = int(z["generation"]), json.loads(z["best_stats"].item())
            assert z["best_key"].tolist() == hist[g - 1]["best_key"] == stats["key"]
            assert int(z["best_generation"]) == hist[g - 1]["best_g"] == stats["generation"] < g
            assert float(z["best_p"]) == stats["p"]
            assert float(z["best_mean_ret"]) == stats["mean_ret"]
            if g < cfg.generations:
                assert float(z["sigma"]) == hist[g]["sigma"]  # the step size sampling generation g
    with np.load(res["snapshots"][-1], allow_pickle=False) as z:
        assert np.array_equal(z["best_theta"], res["best_theta"])
        assert np.array_equal(z["mean_theta"], res["mean_theta"])
        assert float(z["sigma"]) == res["sigma"]
        assert json.loads(z["best_stats"].item()) == res["best_stats"]


def test_mean_theta_stays_the_parameter_space_mean_after_pycma_alleviates_conditioning(tmp_path):
    """pycma's `conditioncov_alleviate` (here at condition 2, by default 1e12) moves the
    covariance into its geno-pheno transform, after which `es.mean` is a genotype. With every
    sample mirrored, a population is centred exactly on the distribution mean, so each
    snapshot's mean_theta must be the mean of the population sampled right after it."""
    rot = np.linalg.qr(np.random.default_rng(0).normal(size=(4, 4)))[0]
    scales, target, pops = 10.0 ** np.linspace(0, 2, 4), np.array([1.0, -2.0, 0.5, 3.0]), {}

    def ellipsoid(X, roots_idx, g):
        pops[g] = X.copy()
        ret = np.repeat(-np.sum((scales * ((X - target) @ rot)) ** 2, 1, keepdims=True),
                        len(roots_idx), 1)
        return {"violated": np.zeros(ret.shape, bool), "ret": ret, "ret_pre": ret, "rows": len(X)}

    cfg = config(generations=30, sigma0=1.0, eval_generations=tuple(range(0, 31, 5)),
                 cma_options={"conditioncov_alleviate": [1e8, 2], "CMA_mirrors": 4,
                              "CMA_mirrormethod": 0})
    res = run(cfg, tmp_path / "full", score=ellipsoid, theta0=np.zeros(4))
    es = saved_strategy(tmp_path / "full")  # a copy, so `result` may call es.stop()
    favorite = np.asarray(es.result.xfavorite)  # pycma's parameter-space mean
    assert not es.gp.isidentity and np.abs(es.mean - favorite).max() > 0.1
    assert np.array_equal(res["mean_theta"], favorite)
    assert len(res["snapshots"]) == 7
    for path in res["snapshots"][:-1]:
        with np.load(path, allow_pickle=False) as z:
            pop = pops[int(z["generation"])]
            assert np.allclose(pop.mean(0), z["mean_theta"], rtol=0, atol=1e-12)
    # resuming across the alleviation is exact too: the transform pickles with the strategy
    run(replace(cfg, generations=12), tmp_path / "split", score=ellipsoid, theta0=np.zeros(4))
    assert not saved_strategy(tmp_path / "split").gp.isidentity
    rest = run(cfg, tmp_path / "split", score=ellipsoid, theta0=np.zeros(4))
    assert_same_run(res, rest, tmp_path / "full", tmp_path / "split")


def test_resume_removes_snapshot_files_written_past_the_checkpoint(tmp_path):
    cfg = config(eval_generations=(0, 4, 8))  # 10 generations, checkpoints every 3
    full = run(cfg, tmp_path / "full")
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(cfg, out, score=crashing(8))
    assert snapshot_files(out) == names(full)  # g0008 written after the checkpoint at g = 6
    short = run(replace(cfg, generations=7), out)  # resumes at 6, stops before g0008
    assert snapshot_files(out) == names(short) == ["g0000.npz", "g0004.npz"]
    assert_same_run(full, run(cfg, out), tmp_path / "full", out)
    assert snapshot_files(out) == names(full)


def test_a_fresh_start_discards_the_previous_run_in_out_dir(tmp_path):
    out, cfg = tmp_path / "run", config(eval_generations=(0, 4, 8))
    run(cfg, out)
    new = replace(cfg, seed=7, generations=5)
    fresh = run(new, out, resume=False)
    assert snapshot_files(out) == names(fresh) == ["g0000.npz", "g0004.npz"]
    assert len(log_rows(out)) == 5

    def changed(X, roots_idx, g):  # e.g. a fixed score function, same config
        return toy_score(X + 0.1, roots_idx, g)

    with pytest.raises(RuntimeError, match="simulated crash"):  # before the first checkpoint
        run(new, out, score=crashing(2, changed), resume=False)
    assert not (out / "checkpoint.pkl").exists() and snapshot_files(out) == ["g0000.npz"]
    again = run(new, out, score=changed)  # no checkpoint to resume: starts afresh
    ref = run(new, tmp_path / "ref", score=changed)
    assert_same_run(ref, again, tmp_path / "ref", out)
    assert snapshot_files(out) == names(ref)


def test_resume_accepts_new_eval_generations_only_from_the_checkpoint_on(tmp_path):
    split = tmp_path / "split"
    cfg = config(eval_generations=(0, 4, 6, 8))
    run(replace(cfg, generations=6, eval_generations=(0, 4)), split)  # checkpoint at g = 6
    for evals, match in (((0, 2, 4, 6, 8), r"missing \[2\]"), ((0, 6, 8), r"undeclared \[4\]")):
        with pytest.raises(ValueError, match=match):  # g0002 cannot be recomputed, g0004 exists
            run(replace(cfg, eval_generations=evals), split)
    assert snapshot_files(split) == ["g0000.npz", "g0004.npz"] and len(log_rows(split)) == 6
    rest = run(cfg, split)  # 6 and 8 are new: both lie at or after the checkpoint
    full = run(cfg, tmp_path / "full")
    assert_same_run(full, rest, tmp_path / "full", split)
    assert names(rest) == ["g0000.npz", "g0004.npz", "g0006.npz", "g0008.npz"]


def test_common_roots_per_generation_and_counters(tmp_path):
    seen = []

    def spy(X, roots_idx, g):
        seen.append((g, roots_idx.copy(), X.shape))
        return toy_score(X, roots_idx, g)

    cfg = config()
    res = run(cfg, tmp_path, score=spy)
    assert [s[0] for s in seen] == list(range(cfg.generations))
    for g, idx, shape in seen:
        rng = np.random.default_rng([cfg.seed, g, 1])
        assert np.array_equal(idx, rng.choice(N_ROOTS, cfg.k_roots, replace=False))
        assert np.array_equal(idx, generation_roots(cfg.seed, g, N_ROOTS, cfg.k_roots))
        assert len(set(idx.tolist())) == cfg.k_roots and shape == (cfg.popsize, THETA0.size)
    assert len({tuple(s[1]) for s in seen}) > 1  # resampled across generations
    c = res["counters"]
    assert c["generations"] == 10 and c["candidates"] == 10 * cfg.popsize
    rows = sum(row["rows"] for row in res["history"])
    assert c["imagined_rows"] == rows == 10 * cfg.popsize * cfg.k_roots * N_SAMPLES * K
    assert c["wall_s"] == pytest.approx(sum(row["wall_s"] for row in res["history"]))


def test_bad_configs_scores_and_foreign_checkpoints_raise(tmp_path):
    with pytest.raises(ValueError, match="rule"):
        run(config(rule="pareto"), tmp_path / "x")
    with pytest.raises(ValueError, match="lam"):
        run(config("fixed", lam=None), tmp_path / "x")
    with pytest.raises(ValueError, match="randn"):
        run(config(cma_options={"randn": np.random.randn}), tmp_path / "x")
    with pytest.raises(ValueError, match="score_fn"):
        run(config(), tmp_path / "x",
            score=lambda X, r, g: {**toy_score(X, r, g), "ret": np.zeros((len(X), 1))})
    run(config(generations=4), tmp_path / "r")
    with pytest.raises(ValueError, match="different run"):
        run(config(generations=6, seed=4), tmp_path / "r")
    with pytest.raises(ValueError, match="beyond"):
        run(config(generations=2), tmp_path / "r")
    res = run(config(generations=6, seed=4), tmp_path / "r", resume=False)  # a fresh start
    assert len(res["history"]) == 6 and len(log_rows(tmp_path / "r")) == 6


def test_violation_flags_must_be_boolean_or_zero_one(tmp_path):
    def flags_as(cast):
        def score(X, roots_idx, g):
            out = toy_score(X, roots_idx, g)
            return {**out, "violated": cast(out["violated"])}

        return score

    ref = run(config(), tmp_path / "bool")
    for i, cast in enumerate((lambda v: v.astype(np.int64), lambda v: v.astype(np.float64))):
        res = run(config(), tmp_path / f"ok{i}", score=flags_as(cast))
        assert_same_run(ref, res, tmp_path / "bool", tmp_path / f"ok{i}")
    for i, cast in enumerate((lambda v: v + 2,  # unsafe-block counts
                              lambda v: 0.5 * v + 0.25,  # fractions
                              lambda v: np.full(v.shape, np.nan))):
        with pytest.raises(ValueError, match="boolean"):
            run(config(), tmp_path / f"bad{i}", score=flags_as(cast))


def test_noise_seeds_are_deterministic_per_generation():
    assert noise_seed(5, 3) == noise_seed(5, 3) != noise_seed(5, 4)
    assert noise_seed(5, 3) != noise_seed(6, 3) and 0 <= noise_seed(5, 3) < 2**63


# ---- integration with helpers.evoRanking ---------------------------------------------------------
@pytest.mark.parametrize("rule", RULES)
def test_integration_real_ranking_module_runs_and_resumes(tmp_path, rule):
    pytest.importorskip("helpers.evoRanking")
    cfg = config(rule, generations=6)

    def go(c, out):
        return run_cmaes(c, THETA0, toy_score, out, n_fitness_roots=N_ROOTS, log=None)

    full = go(cfg, tmp_path / "full")
    go(replace(cfg, generations=4), tmp_path / "split")
    rest = go(cfg, tmp_path / "split")
    assert_same_run(full, rest, tmp_path / "full", tmp_path / "split")
    assert (full["history"][-1]["penalty"] < 0.0) == (rule == "rosarl")


class FakeImaginer:
    """Stands in for ClosedLoopImaginer.rollout_policies: records its inputs and returns healthy
    readouts (height 1.4 m, pitch 0, speed 1 m/s) except that candidate p falls (height 0.5 m)
    at block p on the first root and noise sample."""

    device = "cpu"

    def __init__(self):
        self.calls = []

    def rollout_policies(self, policy, theta, z_hist, hist_blocks, *, n_samples=1, k=0.0, seed=0):
        self.calls.append({"z": z_hist.numpy().copy(), "hist": np.array(hist_blocks),
                           "n": n_samples, "k": k, "seed": seed})
        readout = np.zeros((len(theta), len(z_hist), n_samples, K, 3))
        readout[..., 0], readout[..., 2] = 1.4, 1.0
        for p in range(min(len(theta), K)):
            readout[p, 0, 0, p, 0] = 0.5
        return {"readout": readout}


def test_imagined_score_fn_rejects_unencoded_roots_and_noise_free_samples():
    roots = types.SimpleNamespace(z_hist=None, hist_blocks=np.zeros((2, 2, 10, 6)))
    with pytest.raises(ValueError, match="encoded"):
        imagined_score_fn(FakeImaginer(), None, roots, n_samples=1, noise_k=0.0, seed=0)
    roots.z_hist = np.zeros((2, 3, 4), np.float32)
    with pytest.raises(ValueError, match="n_samples"):
        imagined_score_fn(FakeImaginer(), None, roots, n_samples=4, noise_k=0.0, seed=0)


def test_integration_imagined_score_fn_flattens_roots_and_samples_into_segments():
    pytest.importorskip("torch")
    pytest.importorskip("helpers.evoRanking")
    roots = types.SimpleNamespace(z_hist=np.arange(5 * 3 * 4, dtype=np.float32).reshape(5, 3, 4),
                                  hist_blocks=np.random.default_rng(0).normal(size=(5, 2, 10, 6)))
    fake = FakeImaginer()
    score_fn = imagined_score_fn(fake, None, roots, n_samples=2, noise_k=1.0, seed=11)
    X, idx = np.zeros((4, 3)), np.array([3, 0, 4])
    out = score_fn(X, idx, 5)
    assert out["violated"].shape == out["ret"].shape == out["ret_pre"].shape == (4, 3 * 2)
    assert out["rows"] == 4 * 3 * 2 * K
    call = fake.calls[0]
    assert np.array_equal(call["z"], roots.z_hist[idx])
    assert np.array_equal(call["hist"], roots.hist_blocks[idx])
    assert call["n"] == 2 and call["k"] == 1.0 and call["seed"] == noise_seed(11, 5)
    # segment 0 is (root 3, sample 0): every candidate falls there, nowhere else
    assert out["violated"][:, 0].all() and not out["violated"][:, 1:].any()
    assert np.allclose(out["ret"], 0.8)  # 10 blocks x 1 m/s x 0.08 s
    assert np.allclose(out["ret_pre"][:, 0], 0.08 * np.arange(4))
    assert np.allclose(out["ret_pre"][:, 1:], 0.8)
    score_fn(X, idx, 6)
    assert fake.calls[1]["seed"] == noise_seed(11, 6) != call["seed"]
