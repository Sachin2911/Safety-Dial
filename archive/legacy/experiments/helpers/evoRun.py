"""CMA-ES over latent policies scored in imagination (docs/evoPlan/README.md, Module E).

Each generation g asks pycma 4.4.4 for a population, scores every candidate on the same
`k_roots` fitness roots (`generation_roots`: common random numbers within a generation,
resampled across generations), ranks the population with the run's rule (`helpers.evoRanking`:
Deb, fixed penalty, or ROSARL with one running penalty per run) and tells pycma the ranks.

A run is a pure function of its config, start point and score function, so a resumed run equals
an uninterrupted one. pycma samples only from a private `RandomState` that pickles with the
strategy (`_pin_sampler` also covers the sampler pycma rebuilds with the global `randn` when an
iteration-count `CMA_diagonal` switches to full covariance). Where pycma does draw from numpy's
global RNG (two indices per `tell` for a consistency check of the TPA step-size rule, pycma's
choice for `CMA_diagonal` True above 299 parameters; the mirror-count rounding of
`CMA_mirrormethod` 0 in `ask`), no result depends on the draws, and every pycma call restores
the global state, so the caller's stream is untouched. Termination is disabled and `es.stop()`
is never called, so exactly `generations` generations run.

`out_dir` always mirrors the run it holds. A fresh start removes the previous run's checkpoint
and snapshot files. A resume rebuilds everything from `checkpoint.pkl`: log rows and snapshot
files past the checkpoint are dropped and recomputed, snapshots are rewritten (np.savez writes
fixed zip timestamps, so equal arrays give byte-identical files) and log rows match except
`wall_s`. `eval_generations` may change on resume only from the checkpoint's generation on,
since earlier snapshots cannot be recomputed.

    cfg = CMAConfig(popsize=64, generations=200, sigma0=0.02, seed=0, k_roots=16, rule="rosarl",
                    eval_generations=(0, 5, 10, 25, 50, 100, 200))
    score_fn = imagined_score_fn(imaginer, policy, fitness_roots, n_samples=1, noise_k=0.0, seed=0)
    result = run_cmaes(cfg, theta_bc, score_fn, run_dir / "cmaes", n_fitness_roots=96)

`out_dir` holds `log.jsonl` (one row per generation), `snapshots/g{g:04d}.npz` (the state after
g tells: best-so-far theta and its imagined stats, distribution mean in parameter space and
step size; g = 0 is the start point) and `checkpoint.pkl` (strategy, ROSARL state, best,
history, snapshots, counters, and numpy's global RNG state, restored on resume in case a score
function draws from it). The distribution mean is pycma's `es.mean` mapped through its
geno-pheno transform (`_mean_theta`): `es.mean` itself stops being a parameter vector once
pycma's default `conditioncov_alleviate` moves the covariance into that transform. Accounting
covers generations, candidates, imagined predictor rows (summed from the score function's
`rows`) and wall-clock seconds of the generations in the result.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import re
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import cma
import numpy as np

RULES = ("deb", "fixed", "rosarl")
NO_STOP = {"tolfun": 0, "tolfunhist": 0, "tolx": 0, "tolstagnation": 10**9,
           "tolflatfitness": 10**9, "maxiter": 10**9, "verbose": -9}
CHECKPOINT_FORMAT = 1
SNAPSHOT_NAME = re.compile(r"g\d+\.npz")


@dataclass
class CMAConfig:
    popsize: int
    generations: int
    sigma0: float
    seed: int
    k_roots: int
    n_samples: int = 1
    noise_k: float = 0.0
    rule: str = "deb"
    lam: float | None = None
    eval_generations: tuple = ()
    checkpoint_every: int = 10
    cma_options: dict = field(default_factory=dict)  # e.g. {"CMA_diagonal": True}


def generation_roots(seed: int, g: int, n_fitness_roots: int, k_roots: int) -> np.ndarray:
    """Indices of the fitness roots that every candidate of generation g is scored on."""
    return np.random.default_rng([seed, g, 1]).choice(n_fitness_roots, k_roots, replace=False)


def noise_seed(seed: int, g: int) -> int:
    """Seed of generation g's imagination noise, shared by all of its candidates."""
    return int(np.random.default_rng([seed, g, 2]).integers(2**63))


def cma_options(cfg: CMAConfig) -> dict:
    """pycma options: the private normal stream, no termination, then `cfg.cma_options`. pycma
    ignores `seed` once `randn` is not numpy's own, so the seed is only a record."""
    owned = sorted({"popsize", "seed", "randn"} & set(cfg.cma_options))
    if owned:
        raise ValueError(f"cma_options must not set {owned}: the config fixes them")
    return {"popsize": cfg.popsize, "seed": cfg.seed + 1,
            "randn": np.random.RandomState(cfg.seed).randn, **NO_STOP, **dict(cfg.cma_options)}


def run_cmaes(cfg: CMAConfig, theta0, score_fn, out_dir, *, n_fitness_roots: int,
              resume: bool = True, log=print, ranking=None) -> dict:
    """Run exactly `cfg.generations` CMA-ES generations from `theta0`, or continue a checkpoint.

    score_fn(X (lam, n_params) float64, roots_idx (k,), g) -> dict(violated (lam, M) bool,
    ret (lam, M), ret_pre (lam, M), rows int). `ranking` supplies rank_candidates,
    ranks_to_fitness and RosarlPenalty (default `helpers.evoRanking`). With `resume`, a
    checkpoint in `out_dir` is continued if it comes from the same run (same theta0 and config
    apart from generations, checkpoint_every and eval_generations from the checkpoint's
    generation on) and raises otherwise; `resume=False` starts afresh and discards the run in
    `out_dir`. Returns best_theta, best_stats, mean_theta (distribution mean in parameter
    space), sigma, counters, history (the log rows) and snapshots (paths).
    """
    if ranking is None:
        from helpers import evoRanking as ranking
    log = log or (lambda msg: None)
    theta0 = np.array(theta0, dtype=np.float64)
    _check_config(cfg, theta0, n_fitness_roots)
    out = Path(out_dir)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    ckpt_path, log_path = out / "checkpoint.pkl", out / "log.jsonl"
    identity = _identity(cfg, theta0, n_fitness_roots)
    evals = {int(g) for g in cfg.eval_generations}
    if resume and ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            st = pickle.load(f)
        _check_resume(st, identity, cfg, evals, ckpt_path)
        es, rosarl = st["es"], ranking.RosarlPenalty.from_state(st["rosarl"])
        g, best, history, snaps, counters = (
            st[k] for k in ("g", "best", "history", "snapshots", "counters"))
        np.random.set_state(st["np_random_state"])
        log(f"cmaes: resuming {out} at generation {g}")
    else:
        with _global_rng_kept():
            es = cma.CMAEvolutionStrategy(theta0, cfg.sigma0, cma_options(cfg))
        ckpt_path.unlink(missing_ok=True)  # never resume the previous run after a fresh start
        rosarl, g, history, snaps = ranking.RosarlPenalty(), 0, [], {}
        best = {"theta": theta0.copy(), "key": None, "stats": {}}
        counters = {"generations": 0, "candidates": 0, "imagined_rows": 0, "wall_s": 0.0}
    text = "".join(json.dumps(row) + "\n" for row in history)
    _atomic(log_path, lambda f: f.write(text.encode()))
    held = {_snapshot_path(out, s).name for s in snaps}
    for path in (out / "snapshots").iterdir():  # another run's, or written past the checkpoint
        if SNAPSHOT_NAME.fullmatch(path.name) and path.name not in held:
            path.unlink()
    for s, arrays in snaps.items():
        _atomic(_snapshot_path(out, s), lambda f, a=arrays: np.savez(f, **a))

    def snapshot():
        if g in evals and g not in snaps:
            snaps[g] = _snapshot(g, es, best)
            _atomic(_snapshot_path(out, g), lambda f: np.savez(f, **snaps[g]))

    def checkpoint():
        state = {"format": CHECKPOINT_FORMAT, "identity": identity, "g": g, "es": es,
                 "rosarl": rosarl.state(), "best": best, "history": history, "snapshots": snaps,
                 "counters": counters, "np_random_state": np.random.get_state(),
                 "config": {f.name: getattr(cfg, f.name) for f in fields(cfg)}}
        _atomic(ckpt_path, lambda f: pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL))

    snapshot()
    while g < cfg.generations:
        t0 = time.perf_counter()
        roots_idx = generation_roots(cfg.seed, g, n_fitness_roots, cfg.k_roots)
        sigma = float(es.sigma)
        with _global_rng_kept():
            _pin_sampler(es)
            asked = es.ask()
        X = np.array(asked, dtype=np.float64)
        sc = _check_scores(score_fn(X.copy(), roots_idx.copy(), g), len(X))
        ranks, info = ranking.rank_candidates(cfg.rule, violated=sc["violated"], ret=sc["ret"],
                                              ret_pre=sc["ret_pre"], lam=cfg.lam, rosarl=rosarl)
        ranks = np.asarray(ranks)
        if not np.array_equal(np.sort(ranks), np.arange(len(X))):
            raise ValueError("rank_candidates must return a permutation of 0..lam-1")
        fitness = [float(v) for v in ranking.ranks_to_fitness(ranks)]
        with _global_rng_kept():
            es.tell(asked, fitness)
        i = int(np.argmin(ranks))
        key, stats = _candidate_stats(cfg.rule, g, i, info, sc, roots_idx, rosarl.penalty)
        if best["key"] is None or key < best["key"]:
            best = {"theta": X[i].copy(), "key": key, "stats": stats}
        wall = time.perf_counter() - t0
        for name, v in (("generations", 1), ("candidates", len(X)), ("imagined_rows", sc["rows"]),
                        ("wall_s", wall)):
            counters[name] += v
        row = _log_row(g, i, key, info, best, sigma, rosarl.penalty, sc["rows"], wall)
        history.append(row)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"cmaes g {g}: p min {row['p_min']:.3f} median {row['p_median']:.3f}, mean_ret max "
            f"{row['mean_ret_max']:.4g} median {row['mean_ret_median']:.4g}, best from g "
            f"{row['best_g']}, sigma {sigma:.4g}, {sc['rows']} rows, {wall:.2f} s")
        g += 1
        snapshot()
        if g % cfg.checkpoint_every == 0 and g < cfg.generations:
            checkpoint()
    checkpoint()
    return {"best_theta": best["theta"].copy(), "best_stats": dict(best["stats"]),
            "mean_theta": _mean_theta(es), "sigma": float(es.sigma),
            "counters": dict(counters), "history": [dict(row) for row in history],
            "snapshots": [str(_snapshot_path(out, s)) for s in sorted(snaps)]}


def imagined_score_fn(imaginer, policy, fitness_rootset, *, n_samples: int, noise_k: float,
                      seed: int):
    """`run_cmaes` score function: closed-loop imagination (`ClosedLoopImaginer.rollout_policies`)
    of every candidate from the generation's fitness roots, `n_samples` noise draws each with seed
    `noise_seed(seed, g)` (shared by the whole population), read out per segment by
    `evoRanking.segment_metrics`. A candidate's (R, n) segments flatten row-major into M = R * n."""
    if fitness_rootset.z_hist is None:
        raise ValueError("the fitness roots need encoded history latents "
                         "(evoRoots.encode_histories)")
    if noise_k == 0 and n_samples != 1:
        raise ValueError("noise-free imagination is deterministic: use n_samples=1")
    import torch

    from helpers.evoRanking import segment_metrics

    z_all = torch.as_tensor(np.asarray(fitness_rootset.z_hist, dtype=np.float32),
                            device=imaginer.device)
    hist_all = np.asarray(fitness_rootset.hist_blocks, dtype=np.float64)

    def score_fn(X, roots_idx, g):
        idx = np.asarray(roots_idx, dtype=np.int64)
        out = imaginer.rollout_policies(policy, X, z_all[torch.as_tensor(idx, device=z_all.device)],
                                        hist_all[idx], n_samples=n_samples, k=float(noise_k),
                                        seed=noise_seed(seed, g))
        readout, P = out["readout"], len(X)
        if readout.shape[:3] != (P, len(idx), n_samples):
            raise ValueError(f"readout {readout.shape} is not (P, R, n, K, 3) for P={P}, "
                             f"R={len(idx)}, n={n_samples}")
        m = segment_metrics(readout)
        return {"violated": m["violated"].reshape(P, -1), "ret": m["ret"].reshape(P, -1),
                "ret_pre": m["ret_pre"].reshape(P, -1), "rows": int(np.prod(readout.shape[:-1]))}

    return score_fn


# ---- internals -----------------------------------------------------------------------------
@contextlib.contextmanager
def _global_rng_kept():
    """Restore numpy's global RNG state after a pycma call. pycma 4.4.4 draws from it where no
    result depends on the draws: the TPA step-size rule's consistency-check indices in `tell`,
    the mirror-count rounding of `CMA_mirrormethod` 0 in `ask`."""
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


def _mean_theta(es) -> np.ndarray:
    """The distribution mean in parameter space, mapped like pycma maps its samples and its
    `result.xfavorite` (`result` itself calls `es.stop()`). `es.mean` is the genotype: it is the
    parameter-space mean only while `es.gp` is the identity, which pycma's default
    `conditioncov_alleviate` ends once the covariance condition number passes 1e12."""
    return np.array(es.gp.pheno(es.mean, into_bounds=es.boundary_handler.repair),
                    dtype=np.float64)


def _pin_sampler(es) -> None:
    """pycma 4.4.4 `tell` rebuilds its sampler with numpy's global `randn` when an iteration-count
    `CMA_diagonal` ends; keep whatever sampler it holds on the private stream."""
    if hasattr(es.sm, "randn") and es.sm.randn is not es.opts["randn"]:
        es.sm.randn = es.opts["randn"]


def _check_config(cfg: CMAConfig, theta0: np.ndarray, n_fitness_roots: int) -> None:
    if cfg.rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {cfg.rule!r}")
    if cfg.rule == "fixed" and (cfg.lam is None or not np.isfinite(cfg.lam)):
        raise ValueError("the fixed-penalty rule needs a finite lam")
    if not 0 <= cfg.seed < 2**32 - 1:
        raise ValueError(f"seed must be in [0, 2**32 - 1), got {cfg.seed}")
    if not 1 <= cfg.k_roots <= n_fitness_roots:
        raise ValueError(f"k_roots must be in [1, {n_fitness_roots}], got {cfg.k_roots}")
    if cfg.generations < 0 or cfg.checkpoint_every < 1 or any(
            int(g) < 0 for g in cfg.eval_generations):
        raise ValueError("generations and eval_generations must be >= 0 and checkpoint_every >= 1")
    if theta0.ndim != 1 or not np.isfinite(theta0).all():
        raise ValueError("theta0 must be a finite 1-d parameter vector")


def _identity(cfg: CMAConfig, theta0: np.ndarray, n_fitness_roots: int) -> dict:
    """Everything that shapes the search; generations and checkpoint_every may change between a
    run and its resumption, eval_generations from the checkpoint's generation on
    (`_check_resume`)."""
    def name(o):
        return o.tolist() if isinstance(o, np.ndarray) else getattr(o, "__qualname__", str(o))

    opts = json.dumps(dict(cfg.cma_options), sort_keys=True, default=name)
    return {"popsize": int(cfg.popsize), "sigma0": float(cfg.sigma0), "seed": int(cfg.seed),
            "k_roots": int(cfg.k_roots), "n_samples": int(cfg.n_samples),
            "noise_k": float(cfg.noise_k), "rule": cfg.rule,
            "lam": None if cfg.lam is None else float(cfg.lam), "cma_options": opts,
            "n_fitness_roots": int(n_fitness_roots), "cma_version": cma.__version__,
            "theta0_sha256": hashlib.sha256(theta0.tobytes()).hexdigest()}


def _check_resume(st: dict, identity: dict, cfg: CMAConfig, evals: set, path: Path) -> None:
    """Same run, not past `generations`, and snapshots that an uninterrupted run with these
    eval_generations would also hold: those before the checkpoint cannot be recomputed."""
    if st.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unknown checkpoint format {st.get('format')!r}")
    old = st["identity"]
    diff = sorted(k for k in identity.keys() | old.keys() if identity.get(k) != old.get(k))
    if diff:
        raise ValueError(f"{path} belongs to a different run (differs in {diff}): "
                         "use another out_dir or resume=False")
    g, held = st["g"], set(st["snapshots"])
    if g > cfg.generations:
        raise ValueError(f"{path} is at generation {g}, beyond generations={cfg.generations}")
    missing, extra = sorted(s for s in evals - held if s < g), sorted(held - evals)
    if missing or extra:
        raise ValueError(f"{path} is at generation {g} with snapshots {sorted(held)}, so "
                         f"eval_generations may only change from {g} on (missing {missing}, "
                         f"undeclared {extra})")


def _check_scores(sc: dict, lam: int) -> dict:
    violated = np.asarray(sc["violated"])
    ret, ret_pre = (np.asarray(sc[k], dtype=np.float64) for k in ("ret", "ret_pre"))
    if violated.ndim != 2 or violated.shape[0] != lam or violated.shape[1] < 1 or not (
            ret.shape == ret_pre.shape == violated.shape):
        raise ValueError(f"score_fn must return (lam={lam}, M) arrays, got violated "
                         f"{violated.shape}, ret {ret.shape}, ret_pre {ret_pre.shape}")
    if violated.dtype != bool:  # as evoRanking: flags, not counts or fractions
        if not np.isin(violated, (0, 1)).all():
            raise ValueError("score_fn violation flags must be boolean (or 0/1)")
        violated = violated != 0
    if not (np.isfinite(ret).all() and np.isfinite(ret_pre).all()):
        raise FloatingPointError("score_fn returned non-finite returns")
    return {"violated": violated, "ret": ret, "ret_pre": ret_pre, "rows": int(sc["rows"])}


def _candidate_stats(rule, g, i, info, sc, roots_idx, penalty) -> tuple[tuple, dict]:
    """The rule's key of candidate i (smaller is better; Deb (p, -mean_ret), else (-score,)) and
    its imagined stats."""
    p, mean_ret, scores = float(info["p"][i]), float(info["mean_ret"][i]), info.get("scores")
    if rule != "deb" and scores is None:
        raise ValueError(f"rank_candidates returned no scores for rule {rule!r}")
    score = None if scores is None else float(np.asarray(scores)[i])
    key = (p, -mean_ret) if rule == "deb" else (-score,)
    return key, {"generation": g, "index": i, "key": list(key), "p": p, "mean_ret": mean_ret,
                 "score": score, "n_violated": int(sc["violated"][i].sum()),
                 "mean_ret_pre": float(sc["ret_pre"][i].mean()),
                 "n_segments": int(sc["violated"].shape[1]), "penalty": float(penalty),
                 "roots_idx": [int(r) for r in roots_idx]}


def _log_row(g, i, key, info, best, sigma, penalty, rows, wall) -> dict:
    """One log.jsonl row: population summaries of generation g, its best (rank 0) candidate, the
    best-so-far key, the sampling step size and the ROSARL penalty after this generation."""
    p, r = np.asarray(info["p"], dtype=np.float64), np.asarray(info["mean_ret"], dtype=np.float64)
    s = None if info.get("scores") is None else np.asarray(info["scores"], dtype=np.float64)
    return {"g": g, "p_min": float(p.min()), "p_median": float(np.median(p)),
            "mean_ret_max": float(r.max()), "mean_ret_median": float(np.median(r)),
            "score_max": None if s is None else float(s.max()),
            "score_median": None if s is None else float(np.median(s)),
            "gen_best": i, "gen_key": list(key), "best_key": list(best["key"]),
            "best_g": best["stats"]["generation"], "sigma": sigma, "penalty": float(penalty),
            "rows": rows, "wall_s": wall}


def _snapshot(g: int, es, best: dict) -> dict:
    s = best["stats"]
    return {"generation": np.int64(g), "best_theta": best["theta"].copy(),
            "mean_theta": _mean_theta(es), "sigma": np.float64(es.sigma),
            "best_generation": np.int64(s.get("generation", -1)),
            "best_key": np.asarray(best["key"] or (), dtype=np.float64),
            "best_p": np.float64(s.get("p", np.nan)),
            "best_mean_ret": np.float64(s.get("mean_ret", np.nan)),
            "best_score": np.float64(np.nan if s.get("score") is None else s["score"]),
            "best_stats": np.array(json.dumps(s, sort_keys=True))}


def _snapshot_path(out: Path, g: int) -> Path:
    return out / "snapshots" / f"g{g:04d}.npz"


def _atomic(path: Path, write) -> None:
    """Write through `write(binary file)` to a temporary sibling, then rename it over `path`."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
