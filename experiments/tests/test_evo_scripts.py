"""Pure logic of the evolution-study entry points: grids, selection rules, ridge, policy sets."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from evo_bc_init import clipped_mse, ridge  # noqa: E402
from evo_best_of_n import best_of_first  # noqa: E402
from evo_cheating_curve import choose_generations  # noqa: E402
from evo_cmaes import run_grid  # noqa: E402
from evo_transfer import transfer_policies  # noqa: E402


def stage6_cfg():
    return OmegaConf.create({
        "run_sizes": {"penalty_scale": 2.0, "n_samples_noisy": 4},
        "grid": {"popsize": 64, "generations": 50, "noise_levels": [0, 0.5, 1, 2, 4], "seeds": list(range(10)),
                 "rules": [{"name": "deb", "kind": "deb"}, {"name": "rosarl", "kind": "rosarl"},
                           {"name": "fixed-0.1", "kind": "fixed", "lam_multiple": 0.1},
                           {"name": "fixed-1", "kind": "fixed", "lam_multiple": 1},
                           {"name": "fixed-10", "kind": "fixed", "lam_multiple": 10}]}})


def test_stage6_grid_is_the_planned_250_runs_with_paired_seeds():
    runs = run_grid(stage6_cfg(), "6")
    assert len(runs) == 250 and len({r["name"] for r in runs}) == 250
    fixed = {r["lam_penalty"] for r in runs if r["rule"] == "fixed"}
    assert fixed == {0.2, 2.0, 20.0}
    assert all(r["n_samples"] == (1 if r["noise_k"] == 0 else 4) for r in runs)
    assert all(r["lam_penalty"] is None for r in runs if r["rule"] != "fixed")
    assert {r["seed"] for r in runs} == set(range(10))


def test_stage5b_grid_and_shards_partition_the_runs():
    cfg = OmegaConf.create({"run_sizes": {}, "cmaes": {"popsizes": [16, 64, 256], "seeds": list(range(10)), "generations": 200}})
    runs = run_grid(cfg, "5b")
    assert len(runs) == 30 and all(r["rule"] == "deb" and r["noise_k"] == 0.0 for r in runs)
    shards = [runs[i::3] for i in range(3)]
    assert sorted(r["name"] for s in shards for r in s) == sorted(r["name"] for r in runs)


def test_stage6_generations_rule_needs_persistent_cheating():
    gens = [0, 5, 10, 25, 50, 100, 200]
    lo = {"5": -0.1, "10": 0.02, "25": -0.01, "50": 0.03, "100": 0.04, "200": 0.05}
    diffs = {g: {"lo": v} for g, v in lo.items()} | {"0": {"lo": float("nan")}}
    assert choose_generations(diffs, gens, 200) == 50
    none = {str(g): {"lo": -1.0} for g in gens}
    assert choose_generations(none, gens, 200) == 200


def test_best_of_first_uses_deb_and_breaks_ties_by_index():
    p = np.array([0.5, 0.1, 0.1, 0.0])
    ret = np.array([9.0, 1.0, 2.0, -5.0])
    assert best_of_first(p, ret, 1) == 0
    assert best_of_first(p, ret, 3) == 2      # equal violation rate: higher return wins
    assert best_of_first(p, ret, 4) == 3      # lower violation rate beats any return
    assert best_of_first(np.zeros(3), np.zeros(3), 3) == 0


def test_ridge_recovers_a_linear_map_and_clipping_is_scored():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 8))
    W = rng.normal(size=(6, 8)) * 0.1
    Y = X @ W.T + 0.05
    Wh, bh = ridge(X, Y, 1e-6)
    assert np.allclose(Wh, W, atol=1e-6) and np.allclose(bh, 0.05, atol=1e-6)
    assert clipped_mse(X, np.clip(Y, -1, 1), Wh, bh) == pytest.approx(0.0, abs=1e-10)


def test_transfer_policy_set_spans_good_to_bad():
    cfg = OmegaConf.create({"policies": {"seed": 1, "perturbation_multiples": [0.5, 1, 2, 4], "per_scale": 20,
                                         "n_interpolations": 19, "interpolation_range": [0.0, 0.9]}})
    theta = np.ones(2310)
    thetas, meta = transfer_policies(theta, 0.01, cfg)
    assert thetas.shape == (100, 2310) and len(meta) == 100
    assert np.array_equal(thetas[0], theta) and meta[0]["kind"] == "bc"
    alphas = [m["alpha"] for m in meta if m["kind"] == "interpolation"]
    assert alphas[0] == 0.0 and alphas[-1] == pytest.approx(0.9)
    big = [np.std(t - theta) for t, m in zip(thetas, meta) if m.get("multiple") == 4]
    small = [np.std(t - theta) for t, m in zip(thetas, meta) if m.get("multiple") == 0.5]
    assert np.mean(big) == pytest.approx(8 * np.mean(small), rel=0.05)
