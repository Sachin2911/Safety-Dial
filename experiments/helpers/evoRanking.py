"""Ranking rules for evolution inside the Walker LeWM (docs/evoPlan/README.md, "Ranking rules").

All three rules score the same segments: P candidates, each on M segments (fitness roots x noise
samples; trailing axes are flattened row-major), given every segment's health violation flag, full
return and pre-violation return from `segment_metrics`.

- Deb (feasibility first; credited prior work: Wen and Topcu 2018, SafeDreamer): lexicographic,
  lower violation rate p first, then higher mean return.
- Fixed penalty: mean over segments of `ret - lam * violated`.
- ROSARL (RLC 2026), adapted to evolution: a segment terminates at its first violation, keeps its
  pre-violation return and adds `R_unsafe = V_MIN - V_MAX`, the running minimum minus the running
  maximum of every unpenalised task return seen so far (observed segment returns stand in for the
  paper's value estimates; to be confirmed with the supervisor before stage 6). Averaged over
  segments the score is (1 - p) E[ret | safe] + p (E[ret_pre | violated] + R_unsafe), probability
  weighted rather than lexicographic in p. With one segment per candidate, in exact arithmetic a
  violator scores at most V_MIN (its ret_pre is at most V_MAX) and a safe candidate at least V_MIN,
  so no violator scores above a safe one. That is short of Deb's first rule, the research report's
  bridge to ROSARL, which needs a penalty strictly below every safe return: the scores can tie, a
  tie goes to the candidate index like every tie here (so a violator can rank first), and rounding
  `ret_pre + R_unsafe` can lift a violator above V_MIN. The bridge belongs to the same supervisor
  check.

Exactness. A segment's returns are read off one sequential running sum over its blocks, so the
pre-violation return is an exact prefix of the full return (bitwise equal when nothing is
violated). Means over segments use an exactly rounded sum (`math.fsum`), so scores do not depend
on segment order, batch layout or numpy's summation strategy, and candidates with the same segment
values tie exactly. Every tie then falls to the candidate index, the least significant
`np.lexsort` key, so ranks are a permutation and `ranks_to_fitness` hands pycma distinct values
whose own argsort reproduces this order.

    m = segment_metrics(out["readout"])                      # (P, R, n, K, 3) -> (P, R, n) arrays
    ranks, info = rank_candidates("rosarl", violated=m["violated"], ret=m["ret"],
                                  ret_pre=m["ret_pre"], rosarl=penalty)
    es.tell(X, ranks_to_fitness(ranks))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from helpers.walkerRules import DT, FRAMESKIP, HORIZON_BLOCKS, health_clearance, rule_unsafe

RULES = ("deb", "fixed", "rosarl")
BLOCK_S = FRAMESKIP * DT  # one block, 0.08 s (exactly the float 0.08)


# ---- segments ----------------------------------------------------------------------------
def segment_metrics(readout, *, k_blocks: int | None = HORIZON_BLOCKS) -> dict:
    """Probe readouts at the K block ends, (..., K, 3) = (torso height m, pitch rad, speed m/s) ->
    clearance, unsafe (..., K): health clearance per block end, unsafe iff clearance <= 0;
    violated, first_violation (...): any unsafe block, index of the first one (K if none);
    progress (..., K) = speed x 0.08 s; ret (...) its sum; ret_pre (...) the sum over the blocks
    strictly before first_violation; min_clearance (...).

    `k_blocks` is the expected K (None accepts any). A nonfinite readout raises, so an unseen
    future is never labelled safe.
    """
    r = np.asarray(readout, dtype=np.float64)
    if r.ndim < 2 or r.shape[-1] != 3 or r.shape[-2] < 1 or k_blocks not in (None, r.shape[-2]):
        raise ValueError(f"readout must be (..., K, 3) with K = {k_blocks}, got {r.shape}")
    if not np.isfinite(r).all():
        raise FloatingPointError("nonfinite readout; do not label the unseen future safe")
    clearance = health_clearance(r[..., 0], r[..., 1])
    metrics = _block_metrics(rule_unsafe("health", clearance), r[..., 2] * BLOCK_S)
    return {"clearance": clearance, **metrics, "min_clearance": clearance.min(-1)}


def real_segment_metrics(block_progress, endpoint_clearance=None, dense_violated=None, dense_first_step=None, *,
                         k_blocks: int | None = HORIZON_BLOCKS) -> dict:
    """Real segments (the arrays of `evoReal.RealExecutor.run`) in the layout of `segment_metrics`.

    block_progress (..., K) is the true torso x displacement per block; `progress` and `ret` (its
    sequential sum) are always returned. Each safety reading supplied adds a sub-dict with the
    `segment_metrics` keys it determines: "endpoint" from endpoint_clearance (..., K), the true
    state at block ends (the reading imagination makes), and "dense" from dense_first_step (...),
    the first unsafe environment step (0-based over qpos rows 1..K*FRAMESKIP, K*FRAMESKIP if none).
    Step i is qpos row i + 1 and lies in block i // FRAMESKIP, the dense first violation, so that
    block and the later ones do not count towards ret_pre. Block b ends at step
    (b + 1) * FRAMESKIP - 1, the state the endpoint reading checks, so on one trace the dense first
    violation is never later than the endpoint one and equals it when the first unsafe step is a
    block end. dense_violated alone yields only "violated"; given with dense_first_step it must
    agree with it.
    """
    prog = np.asarray(block_progress, dtype=np.float64)
    if prog.ndim < 1 or prog.shape[-1] < 1 or k_blocks not in (None, prog.shape[-1]):
        raise ValueError(f"block_progress must be (..., K) with K = {k_blocks}, got {prog.shape}")
    if not np.isfinite(prog).all():
        raise FloatingPointError("nonfinite block progress")
    k, lead = prog.shape[-1], prog.shape[:-1]
    out = {"progress": prog, "ret": np.cumsum(prog, -1)[..., -1]}
    if endpoint_clearance is not None:
        c = np.asarray(endpoint_clearance, dtype=np.float64)
        if c.shape != prog.shape:
            raise ValueError(f"endpoint_clearance must be {prog.shape}, got {c.shape}")
        if not np.isfinite(c).all():
            raise FloatingPointError("nonfinite endpoint clearance")
        out["endpoint"] = {"clearance": c, **_block_metrics(rule_unsafe("health", c), prog), "min_clearance": c.min(-1)}
    dense = None if dense_violated is None else _flags(dense_violated, lead)
    if dense_first_step is not None:
        step = np.asarray(dense_first_step)
        if step.shape != lead or not np.issubdtype(step.dtype, np.integer) or ((step < 0) | (step > k * FRAMESKIP)).any():
            raise ValueError(f"dense_first_step must be integers in [0, {k * FRAMESKIP}] of shape {lead}")
        violated = step < k * FRAMESKIP
        if dense is not None and not np.array_equal(dense, violated):
            raise ValueError("dense_violated disagrees with dense_first_step")
        first = (step // FRAMESKIP).astype(np.int64)
        ret, ret_pre = _returns(prog, first)
        out["dense"] = {"violated": violated, "first_violation": first, "first_step": step.astype(np.int64),
                        "progress": prog, "ret": ret, "ret_pre": ret_pre}
    elif dense is not None:
        out["dense"] = {"violated": dense, "progress": prog, "ret": out["ret"]}
    return out


def _block_metrics(unsafe: np.ndarray, progress: np.ndarray) -> dict:
    violated = unsafe.any(-1)
    first = np.where(violated, unsafe.argmax(-1), unsafe.shape[-1]).astype(np.int64)
    ret, ret_pre = _returns(progress, first)
    return {"unsafe": unsafe, "violated": violated, "first_violation": first, "progress": progress, "ret": ret, "ret_pre": ret_pre}


def _returns(progress: np.ndarray, first: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full return and the return of the blocks before `first`, both read off one sequential
    running sum, so the second is an exact prefix of the first (and equal to it when first == K)."""
    csum = np.cumsum(progress, -1)
    prefix = np.concatenate([np.zeros_like(csum[..., :1]), csum], -1)  # prefix[..., f] = blocks < f
    return csum[..., -1], np.take_along_axis(prefix, first[..., None], -1)[..., 0]


# ---- candidate scores ----------------------------------------------------------------------
def candidate_stats(violated, ret) -> dict:
    """violated, ret (P, M) -> p (P,) violation rate (exactly count / M), mean_ret (P,) mean return
    over all M segments, n_violated (P,) and n_segments M."""
    v, r = _segments(violated, ret)
    n = v.sum(1)
    return {"p": n / v.shape[1], "mean_ret": _mean_rows(r), "n_violated": n, "n_segments": v.shape[1]}


def deb_ranks(p, mean_ret) -> np.ndarray:
    """Deb's feasibility-first ranks, 0 best: lower p, then higher mean_ret, then lower index."""
    return _ranks(order_key("deb", {"p": p, "mean_ret": mean_ret}))[1]


def fixed_penalty_scores(violated, ret, lam) -> np.ndarray:
    """Mean over segments of `ret - lam * violated` (P,); lam must be finite and >= 0."""
    lam = float(lam)
    if not (math.isfinite(lam) and lam >= 0):
        raise ValueError(f"lam must be finite and nonnegative, got {lam}")
    v, r = _segments(violated, ret)
    with np.errstate(over="ignore"):  # an overflow reaches _mean_rows, which raises ValueError
        x = np.where(v, r - lam, r)
    return _mean_rows(x)


def rosarl_scores(violated, ret, ret_pre, penalty) -> np.ndarray:
    """Mean over segments of `ret` if safe, else `ret_pre + penalty` (P,): a violated segment
    terminates at its first violation, so later blocks never count. penalty must be <= 0."""
    penalty = float(penalty)
    if not (math.isfinite(penalty) and penalty <= 0):
        raise ValueError(f"the ROSARL penalty must be finite and <= 0, got {penalty}")
    v, r, rp = _segments(violated, ret, ret_pre)
    with np.errstate(over="ignore"):  # an overflow reaches _mean_rows, which raises ValueError
        x = np.where(v, rp + penalty, r)
    return _mean_rows(x)


@dataclass
class RosarlPenalty:
    """ROSARL's minmax penalty `R_unsafe = V_MIN - V_MAX` (<= 0; 0.0 before any update), with
    V_MIN and V_MAX the running minimum and maximum of every unpenalised task return seen so far (a
    violated segment's task return is its pre-violation return). Penalties never feed back into
    the range. One instance lives for a whole run; checkpoint it with `state()`. An update is
    atomic: one that raises (a nonfinite return, or a range too wide for a finite penalty) leaves
    the state unchanged."""

    v_min: float = math.inf
    v_max: float = -math.inf
    n_seen: int = 0

    def update(self, task_returns) -> None:
        x = np.asarray(task_returns, dtype=np.float64).ravel()
        if not np.isfinite(x).all():
            raise ValueError("task returns must be finite")
        if x.size:
            lo, hi = min(self.v_min, float(x.min())), max(self.v_max, float(x.max()))
            if not math.isfinite(lo - hi):
                raise ValueError(f"task returns span [{lo}, {hi}], too wide for a finite penalty")
            self.v_min, self.v_max, self.n_seen = lo, hi, self.n_seen + int(x.size)

    @property
    def penalty(self) -> float:
        return self.v_min - self.v_max if self.n_seen else 0.0

    def state(self) -> dict:
        return {"v_min": self.v_min, "v_max": self.v_max, "n_seen": self.n_seen}

    @classmethod
    def from_state(cls, d: dict) -> "RosarlPenalty":
        out = cls(float(d["v_min"]), float(d["v_max"]), int(d["n_seen"]))
        fresh = out.n_seen == 0 and out.v_min == math.inf and out.v_max == -math.inf
        seen = out.n_seen > 0 and math.isfinite(out.v_min - out.v_max) and out.v_min <= out.v_max
        if out.n_seen != d["n_seen"] or not (fresh or seen):
            raise ValueError(f"inconsistent ROSARL state {d}")
        return out


# ---- ranking -------------------------------------------------------------------------------
def rank_candidates(rule: str, *, violated, ret, ret_pre, lam=None, rosarl: RosarlPenalty | None = None) -> tuple[np.ndarray, dict]:
    """Ranks (P,) int64 of P candidates on their M segments under `rule`, 0 best, ties to the
    lower candidate index, plus info: p, mean_ret, n_violated, n_segments (`candidate_stats`),
    scores (P,) (fixed and ROSARL; None for Deb, which is lexicographic), penalty (the ROSARL
    penalty used, else None), lam (fixed, else None), rule and order (candidate indices, best
    first). ROSARL first absorbs this batch's unpenalised task returns `where(violated, ret_pre,
    ret)` into `rosarl` and then scores with the updated penalty. Arguments a rule does not use
    are ignored. `rosarl` changes only under ROSARL and only when the call succeeds: a call that
    raises leaves it unchanged. Errors are ValueError, including values whose sums leave the
    float range."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    v, r, rp = _segments(violated, ret, ret_pre)
    scores = penalty = trial = None
    if rule == "fixed":
        if lam is None:
            raise ValueError("the fixed rule needs lam")
        scores = fixed_penalty_scores(v, r, lam)
    elif rule == "rosarl":
        if rosarl is None:
            raise ValueError("the ROSARL rule needs a RosarlPenalty")
        trial = replace(rosarl)  # copied back into `rosarl` once nothing below can raise
        trial.update(np.where(v, rp, r))
        penalty = trial.penalty
        scores = rosarl_scores(v, r, rp, penalty)
    info = {**candidate_stats(v, r), "scores": scores, "penalty": penalty, "lam": float(lam) if rule == "fixed" else None, "rule": rule}
    info["order"], ranks = _ranks(order_key(rule, info))
    if trial is not None:
        rosarl.v_min, rosarl.v_max, rosarl.n_seen = trial.v_min, trial.v_max, trial.n_seen
    return ranks, info


def order_key(rule: str, info) -> tuple[np.ndarray, ...]:
    """Lexicographic keys of every candidate under `rule` (from a mapping with p and mean_ret, or
    scores), most significant first and smaller better, ending with the candidate index so that
    nothing ties: Deb (p, -mean_ret, index); fixed and ROSARL (-score, index). The order, best
    first, is `np.lexsort(keys[::-1])`. Without the index the keys compare candidates across
    generations."""
    if rule == "deb":
        keys = (_vector(info["p"]), -_vector(info["mean_ret"]))
    elif rule in ("fixed", "rosarl"):
        keys = (-_vector(info["scores"]),)
    else:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if len({len(k) for k in keys}) != 1:
        raise ValueError("ranking keys must have one entry per candidate")
    return (*keys, np.arange(len(keys[0])))


def ranks_to_fitness(ranks) -> np.ndarray:
    """Ranks (a permutation of 0..P-1, 0 best) -> the f-values pycma minimises: rank r -> float(r).
    The values are distinct, so pycma's own sort reproduces the ranking exactly."""
    r = np.asarray(ranks)
    if r.ndim != 1 or not np.array_equal(np.sort(r), np.arange(r.size)):
        raise ValueError("ranks must be a permutation of 0..P-1")
    return r.astype(np.float64)


# ---- internals ------------------------------------------------------------------------------
def _ranks(keys) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort(keys[::-1])
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size)
    return order, ranks


def _mean_rows(x: np.ndarray) -> np.ndarray:
    """Mean of each row of a (P, M) array from an exactly rounded sum: independent of order.
    Values or sums beyond the float range raise ValueError (not fsum's OverflowError, nor -inf)."""
    try:
        out = np.array([math.fsum(row) / x.shape[1] for row in x.tolist()], dtype=np.float64)
    except OverflowError as e:
        raise ValueError(f"segment values overflow the float range ({e})") from None
    if not np.isfinite(out).all():
        raise ValueError("segment values overflow the float range")
    return out


def _vector(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError(f"expected a finite (P,) vector, got shape {a.shape}")
    return a


def _flags(x, shape) -> np.ndarray:
    f = np.asarray(x)
    if f.shape != tuple(shape):
        raise ValueError(f"violation flags must be {tuple(shape)}, got {f.shape}")
    if f.dtype != bool:
        if not np.isin(f, (0, 1)).all():
            raise ValueError("violation flags must be boolean (or 0/1)")
        f = f != 0
    return f


def _segments(violated, *values) -> list[np.ndarray]:
    """(P, ...) violation flags and same-shape returns -> (P, M) bool and finite float64 arrays."""
    shape = np.shape(violated)
    if len(shape) < 2 or 0 in shape:
        raise ValueError(f"segments must be (P, M) (or (P, ...)) with P, M >= 1, got {shape}")
    out = [_flags(violated, shape).reshape(shape[0], -1)]
    for x in values:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != shape:
            raise ValueError(f"segment arrays must share the shape {shape}, got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("segment returns must be finite")
        out.append(x.reshape(shape[0], -1))
    return out
