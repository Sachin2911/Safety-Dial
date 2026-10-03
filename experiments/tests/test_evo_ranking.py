"""Ranking rules for evolution in imagination: segment metrics, Deb, fixed penalty, ROSARL, exact ties."""
from __future__ import annotations

import math
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.evoRanking import (  # noqa: E402
    BLOCK_S, RosarlPenalty, candidate_stats, deb_ranks, fixed_penalty_scores, order_key, rank_candidates,
    ranks_to_fitness, real_segment_metrics, rosarl_scores, segment_metrics)
from helpers.walkerBank import endpoint_targets  # noqa: E402
from helpers.walkerRules import FRAMESKIP, health_clearance  # noqa: E402

K = 10
SAFE_HEIGHT = 1.25  # with pitch 0 the health clearance is 2.0


def readout(speed, height=SAFE_HEIGHT, pitch=0.0):
    """(K, 3) block-end readout; scalars broadcast over the K blocks."""
    return np.stack([np.broadcast_to(np.asarray(x, dtype=np.float64), (K,)) for x in (height, pitch, speed)], -1)


def seq_sum(x) -> float:
    acc = 0.0
    for v in np.asarray(x, dtype=np.float64).tolist():
        acc += v
    return acc


def population(rng, *, ties=False, p_range=(2, 40), m_range=(1, 25)):
    """Random (P, M) segments; `ties` uses small integers so equal means are common and exact."""
    P, M = int(rng.integers(*p_range)), int(rng.integers(*m_range))
    rates = rng.uniform(0, 1, P) * (rng.random(P) < 0.8)  # some candidates never violate
    violated = rng.random((P, M)) < rates[:, None]
    if ties:
        ret = rng.integers(-3, 4, (P, M)).astype(np.float64)
        ret_pre = np.where(violated, rng.integers(-3, 4, (P, M)), ret).astype(np.float64)
    else:
        ret = rng.normal(0, 2, (P, M))
        ret_pre = np.where(violated, ret - rng.uniform(0, 3, (P, M)), ret)
    return violated, ret, ret_pre


def segments(violated, ret, ret_pre):
    return {"violated": violated, "ret": ret, "ret_pre": ret_pre}


# ---- segment metrics -------------------------------------------------------------------------
def test_segment_metrics_hand_built_first_violation_returns_and_band_edges():
    speed = np.linspace(0.5, 2.3, K)
    h_b = np.full(K, SAFE_HEIGHT)
    h_b[3], h_b[6] = 0.8, 0.5  # clearance exactly 0 at block 3 (unsafe), negative at block 6
    p_c = np.zeros(K)
    p_c[0] = 1.0  # pitch on the band edge at the first block
    h_d = np.full(K, SAFE_HEIGHT)
    h_d[9] = 2.0  # upper height edge at the last block
    h_e = np.full(K, SAFE_HEIGHT)
    h_e[2] = 0.8 + 1e-9  # just inside the band: safe
    heights = np.stack([np.full(K, SAFE_HEIGHT), h_b, np.full(K, SAFE_HEIGHT), h_d, h_e])
    pitches = np.stack([np.zeros(K), np.zeros(K), p_c, np.zeros(K), np.zeros(K)])
    m = segment_metrics(np.stack([readout(speed, h, a) for h, a in zip(heights, pitches)]))

    assert np.array_equal(m["clearance"], health_clearance(heights, pitches))
    assert m["clearance"][1, 3] == m["clearance"][2, 0] == m["clearance"][3, 9] == 0.0
    assert m["clearance"][4, 2] > 0
    assert [np.flatnonzero(u).tolist() for u in m["unsafe"]] == [[], [3, 6], [0], [9], []]
    assert m["violated"].tolist() == [False, True, True, True, False]
    assert m["first_violation"].tolist() == [K, 3, 0, 9, K] and m["first_violation"].dtype == np.int64
    assert np.array_equal(m["progress"], np.broadcast_to(speed * 0.08, (5, K)))
    full = seq_sum(speed * 0.08)
    assert m["ret"].tolist() == [full] * 5
    assert m["ret_pre"].tolist() == [full, seq_sum(speed[:3] * 0.08), 0.0, seq_sum(speed[:9] * 0.08), full]
    assert np.array_equal(m["min_clearance"], m["clearance"].min(-1))
    assert BLOCK_S == 0.08


def test_segment_metrics_batch_shapes_and_validation():
    rng = np.random.default_rng(0)
    shape = (2, 3, 4, K)
    r = np.stack([rng.uniform(0.75, 2.05, shape), rng.uniform(-1.05, 1.05, shape), rng.normal(1, 1, shape)], -1)
    m = segment_metrics(r)
    assert m["clearance"].shape == shape and m["ret"].shape == shape[:-1] and m["violated"].any() and not m["violated"].all()
    flat = segment_metrics(r.reshape(-1, K, 3))
    one = segment_metrics(r[1, 2, 3])
    for key, value in m.items():
        assert np.array_equal(value.reshape(flat[key].shape), flat[key]), key
        assert np.array_equal(value[1, 2, 3], one[key]), key
    assert segment_metrics(r[..., :4, :], k_blocks=4)["first_violation"].max() <= 4
    assert segment_metrics(r[..., :4, :], k_blocks=None)["ret"].shape == shape[:-1]
    with pytest.raises(ValueError):
        segment_metrics(r[..., :4, :])
    with pytest.raises(ValueError):
        segment_metrics(r[..., :2])
    bad = r.copy()
    bad[0, 0, 0, 5, 0] = np.nan
    with pytest.raises(FloatingPointError):
        segment_metrics(bad)


def test_real_segment_metrics_dense_and_endpoint_readings():
    rng = np.random.default_rng(1)
    prog = rng.normal(0.1, 0.05, (3, K))
    step = np.array([37, 100, 0])  # first unsafe env step (0-based), 100 = never
    clear = np.full((3, K), 1.5)
    clear[0, 5], clear[2, 0], clear[2, 7] = 0.0, -0.3, -1.0
    m = real_segment_metrics(prog, clear, np.array([True, False, True]), step)
    ret = [seq_sum(row) for row in prog]
    assert m["ret"].tolist() == ret and m["dense"]["ret"].tolist() == ret
    assert m["dense"]["violated"].tolist() == [True, False, True]
    assert m["dense"]["first_violation"].tolist() == [3, K, 0]
    assert m["dense"]["ret_pre"].tolist() == [seq_sum(prog[0, :3]), ret[1], 0.0]
    assert m["endpoint"]["first_violation"].tolist() == [5, K, 0]
    assert m["endpoint"]["ret_pre"].tolist() == [seq_sum(prog[0, :5]), ret[1], 0.0]
    assert real_segment_metrics(prog, dense_first_step=step)["dense"]["violated"].tolist() == [True, False, True]
    only = real_segment_metrics(prog, dense_violated=[1, 0, 1])
    assert "endpoint" not in only and "ret_pre" not in only["dense"] and only["dense"]["violated"].tolist() == [True, False, True]
    for kwargs in ({"dense_violated": np.array([True, True, True]), "dense_first_step": step},
                   {"dense_first_step": np.array([37, 101, 0])}, {"dense_first_step": step.astype(float)},
                   {"endpoint_clearance": clear[:, :5]}, {"dense_violated": np.array([2, 0, 1])}):
        with pytest.raises(ValueError):
            real_segment_metrics(prog, **kwargs)


def test_real_dense_first_step_maps_to_its_block_at_block_boundaries():
    rng = np.random.default_rng(11)
    T = K * FRAMESKIP
    step = np.array([0, 8, 9, 10, 11, 19, 20, 89, 90, 98, 99, T])  # step i is qpos row i + 1: blocks end at 9, 19, ..
    block = [0, 0, 0, 1, 1, 1, 2, 8, 9, 9, 9, K]
    prog = rng.normal(0.1, 0.05, (len(step), K))
    d = real_segment_metrics(prog, dense_first_step=step)["dense"]
    assert d["first_violation"].tolist() == block and d["first_step"].tolist() == step.tolist()
    assert d["violated"].tolist() == [True] * (len(step) - 1) + [False]
    assert d["ret_pre"].tolist() == [seq_sum(row[:b]) for row, b in zip(prog, block)]
    # dense traces built as evoReal builds them: the torso falls at qpos row s (every s in 1..T)
    # for one row, three rows or for good; the endpoint states are rows 10, 20, .., T of the trace
    start, length = np.tile(np.arange(1, T + 1), 3), np.repeat([1, 3, T], T)
    rows = np.arange(T + 1)
    qp = np.zeros((start.size, T + 1, 9))
    qp[..., 0] = np.cumsum(rng.normal(0.01, 0.002, qp.shape[:2]), 1)
    qp[..., 1] = np.where((rows >= start[:, None]) & (rows < (start + length)[:, None]), 0.5, SAFE_HEIGHT)
    dense = health_clearance(qp[:, 1:, 1], qp[:, 1:, 2])
    unsafe = dense <= 0
    first_step = np.where(unsafe.any(1), unsafe.argmax(1), T)
    ends = np.stack([endpoint_targets(q, np.zeros(T)) for q in qp])
    endpoint = health_clearance(ends[..., 0], ends[..., 1])
    assert np.array_equal(endpoint, dense[:, FRAMESKIP - 1::FRAMESKIP])  # block ends are steps 9, 19, ..
    m = real_segment_metrics(qp[:, FRAMESKIP::FRAMESKIP, 0] - qp[:, :-1:FRAMESKIP, 0], endpoint, unsafe.any(1), first_step)
    d, e = m["dense"], m["endpoint"]
    assert d["violated"].all() and d["first_step"].tolist() == (start - 1).tolist()
    assert (d["first_violation"] <= e["first_violation"]).all()
    on_end = first_step % FRAMESKIP == FRAMESKIP - 1
    assert on_end.sum() == 3 * K
    for key in ("first_violation", "ret_pre"):  # the same block, so the same blocks count towards ret_pre
        assert np.array_equal(d[key][on_end], e[key][on_end]), key
        assert np.array_equal(d[key][length == T], e[key][length == T]), key  # still down at the block end
    assert not e["violated"][(length == 1) & ~on_end].any()  # a dip between block ends only the dense reading sees


def test_real_endpoint_reading_matches_imagined_metrics_on_the_same_states():
    rng = np.random.default_rng(2)
    r = np.stack([rng.uniform(0.6, 2.1, (64, K)), rng.uniform(-1.2, 1.2, (64, K)), rng.normal(1, 1, (64, K))], -1)
    imagined = segment_metrics(r)
    real = real_segment_metrics(r[..., 2] * BLOCK_S, health_clearance(r[..., 0], r[..., 1]))["endpoint"]
    for key in ("clearance", "unsafe", "violated", "first_violation", "progress", "ret", "ret_pre", "min_clearance"):
        assert np.array_equal(imagined[key], real[key]), key


# ---- Deb -----------------------------------------------------------------------------------
def test_deb_puts_every_lower_violation_candidate_first():
    rng = np.random.default_rng(3)
    for trial in range(300):
        violated, ret, ret_pre = population(rng, ties=trial % 2 == 0)
        ranks, info = rank_candidates("deb", **segments(violated, ret, ret_pre))
        p, mean = info["p"], info["mean_ret"]
        assert sorted(ranks.tolist()) == list(range(len(ranks))) and ranks.dtype == np.int64
        lower = p[:, None] < p[None, :]
        assert (ranks[:, None] < ranks[None, :])[lower].all()
        assert info["order"].tolist() == sorted(range(len(ranks)), key=lambda i: (p[i], -mean[i], i))
        assert np.array_equal(ranks, deb_ranks(p, mean))
        assert np.array_equal(info["n_violated"], violated.sum(1)) and np.array_equal(p, violated.sum(1) / violated.shape[1])
        np.testing.assert_allclose(mean, ret.mean(1), rtol=1e-12, atol=1e-12)
        assert info["scores"] is None and info["penalty"] is None and info["lam"] is None


def test_deb_is_feasibility_first_whatever_the_return():
    violated = np.array([[False, True], [False, False], [True, True], [False, False]])
    ret = np.array([[50.0, 50.0], [0.1, 0.1], [99.0, 99.0], [0.3, 0.1]])
    ranks, info = rank_candidates("deb", **segments(violated, ret, ret))
    assert info["p"].tolist() == [0.5, 0.0, 1.0, 0.0]
    assert ranks.tolist() == [2, 1, 3, 0] and info["order"].tolist() == [3, 1, 0, 2]


# ---- fixed penalty -------------------------------------------------------------------------
def test_fixed_penalty_is_monotone_in_lam_and_charges_violations():
    rng = np.random.default_rng(4)
    lams = [0.0, 0.1, 1.0, 10.0, 100.0]
    for _ in range(100):
        violated, ret, _ = population(rng)
        scores = np.stack([fixed_penalty_scores(violated, ret, lam) for lam in lams])
        assert (np.diff(scores, axis=0) <= 0).all()
        clean = ~violated.any(1)
        assert (scores[:, clean] == scores[0, clean]).all()
        assert np.array_equal(scores[0], candidate_stats(violated, ret)["mean_ret"])
        rate = violated.sum(1) / violated.shape[1]
        np.testing.assert_allclose(scores[0] - scores, np.outer(lams, rate), rtol=1e-9, atol=1e-9)
    ret = np.tile(np.arange(8.0), (4, 1))  # identical returns, 0, 1, 3 and 8 violations
    violated = np.arange(8)[None, :] < np.array([0, 1, 3, 8])[:, None]
    base = fixed_penalty_scores(violated, ret, 0.0)
    for lam in lams[1:]:
        s = fixed_penalty_scores(violated, ret, lam)
        assert (np.diff(s) < 0).all() and (np.diff(base - s) > 0).all()
        ranks, info = rank_candidates("fixed", **segments(violated, ret, ret), lam=lam)
        assert ranks.tolist() == [0, 1, 2, 3] and np.array_equal(info["scores"], s) and info["lam"] == lam
    for lam in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            fixed_penalty_scores(violated, ret, lam)


def test_large_fixed_penalty_reduces_to_deb():
    rng = np.random.default_rng(5)
    for _ in range(100):
        seg = segments(*population(rng, ties=True))
        assert np.array_equal(rank_candidates("deb", **seg)[0], rank_candidates("fixed", **seg, lam=2.0 ** 20)[0])


# ---- ROSARL --------------------------------------------------------------------------------
def test_rosarl_scores_use_pre_violation_return_plus_penalty():
    violated = np.array([[True, False], [False, False]])
    ret = np.array([[7.0, 3.0], [1.0, 2.0]])
    ret_pre = np.array([[2.0, 3.0], [1.0, 2.0]])
    assert rosarl_scores(violated, ret, ret_pre, -4.0).tolist() == [0.5, 1.5]
    other = ret_pre.copy()
    other[0, 1], other[1] = 99.0, -50.0  # ret_pre of safe segments never counts
    assert rosarl_scores(violated, ret, other, -4.0).tolist() == [0.5, 1.5]
    for bad in (1.0, math.nan, -math.inf):
        with pytest.raises(ValueError):
            rosarl_scores(violated, ret, ret_pre, bad)


def test_rosarl_terminates_at_the_first_violation():
    speed = np.ones(K)
    h0 = np.full(K, SAFE_HEIGHT)
    h0[4] = 0.7  # falls at block 4 and recovers
    h1 = h0.copy()
    h1[5:] = 0.5  # stays down
    fast = speed.copy()
    fast[4:] = 5.0  # same up to the violation, much faster from the violating block on
    safe = readout(0.5 * speed)
    m = segment_metrics(np.stack([readout(speed, h0), readout(fast, h1), safe])[:, None])
    assert m["first_violation"][:, 0].tolist() == [4, 4, K]
    assert m["ret_pre"][0, 0] == m["ret_pre"][1, 0] == seq_sum(speed[:4] * 0.08)
    assert m["ret"][1, 0] > m["ret"][0, 0]
    seg = segments(m["violated"], m["ret"], m["ret_pre"])
    scores = rosarl_scores(**seg, penalty=-0.25)
    assert scores[0] == scores[1] == m["ret_pre"][0, 0] - 0.25
    ranks, info = rank_candidates("rosarl", **seg, rosarl=RosarlPenalty())
    assert info["scores"][0] == info["scores"][1] and ranks[0] < ranks[1]
    fixed = fixed_penalty_scores(m["violated"], m["ret"], 1.0)
    assert fixed[1] > fixed[0]  # the fixed penalty keeps counting the blocks after the violation


def test_rosarl_penalty_tracks_unpenalised_task_returns_across_calls():
    rng = np.random.default_rng(6)
    pen = RosarlPenalty()
    assert pen.penalty == 0.0 and pen.n_seen == 0 and pen.v_min == math.inf and pen.v_max == -math.inf
    seen = []
    for _ in range(6):
        violated, ret, ret_pre = population(rng)
        ranks, info = rank_candidates("rosarl", **segments(violated, ret, ret_pre), rosarl=pen)
        seen.append(np.where(violated, ret_pre, ret).ravel())
        task = np.concatenate(seen)
        assert (pen.v_min, pen.v_max, pen.n_seen) == (task.min(), task.max(), task.size)
        assert info["penalty"] == pen.penalty == task.min() - task.max() <= 0
        assert np.array_equal(info["scores"], rosarl_scores(violated, ret, ret_pre, pen.penalty))
        assert info["order"].tolist() == sorted(range(len(ranks)), key=lambda i: (-info["scores"][i], i))


def test_rosarl_range_ignores_penalised_values_and_rejected_calls():
    pen = RosarlPenalty()
    seg = segments(np.array([[True, False], [False, False]]), np.array([[2.0, 1.0], [2.0, 1.5]]), np.array([[1.0, 1.0], [2.0, 1.5]]))
    _, info = rank_candidates("rosarl", **seg, rosarl=pen)
    assert (pen.v_min, pen.v_max, info["penalty"]) == (1.0, 2.0, -1.0)  # the first call already uses its own batch
    assert info["scores"].tolist() == [0.5, 1.75]  # violated segment: 1.0 + (-1.0) = 0.0
    _, info = rank_candidates("rosarl", **seg, rosarl=pen)
    assert (pen.v_min, pen.v_max, pen.n_seen, info["penalty"]) == (1.0, 2.0, 8, -1.0)  # 0.0 never entered
    before = pen.state()
    bad = dict(seg, ret=np.array([[2.0, np.nan], [2.0, 1.5]]))
    with pytest.raises(ValueError):
        rank_candidates("rosarl", **bad, rosarl=pen)
    assert pen.state() == before
    for rule, kwargs in (("deb", {}), ("fixed", {"lam": 1.0})):
        rank_candidates(rule, **seg, rosarl=pen, **kwargs)
        assert pen.state() == before  # only the ROSARL rule moves the range


def test_rejected_calls_leave_the_rosarl_range_unchanged_and_overflow_is_a_value_error():
    one = np.array([[True], [False]])  # candidate 0 violates, candidate 1 is safe; M = 1
    huge = np.array([[1e308], [-1e308]])
    cases = ((one, huge, huge),  # V_MIN - V_MAX overflows
             (one, np.array([[0.0], [7e307]]), np.array([[-1e308], [7e307]])),  # finite penalty, ret_pre + penalty overflows
             (np.zeros((2, 2), bool), np.array([[1e308, 1e308], [0.0, 0.0]]), np.array([[1e308, 1e308], [0.0, 0.0]])))  # a sum overflows
    for pen in (RosarlPenalty(), RosarlPenalty.from_state({"v_min": -1.0, "v_max": 2.0, "n_seen": 3})):
        before = pen.state()
        for violated, ret, ret_pre in cases:
            with pytest.raises(ValueError):
                rank_candidates("rosarl", **segments(violated, ret, ret_pre), rosarl=pen)
            assert pen.state() == before
        with pytest.raises(ValueError):
            pen.update(huge)
        assert pen.state() == before
        _, info = rank_candidates("rosarl", **segments(one, np.array([[3.0], [1.0]]), np.array([[0.5], [1.0]])), rosarl=pen)
        assert pen.state() == {"v_min": min(before["v_min"], 0.5), "v_max": max(before["v_max"], 1.0), "n_seen": before["n_seen"] + 2}
        assert info["penalty"] == pen.penalty
    with pytest.raises(ValueError):
        RosarlPenalty.from_state({"v_min": -1e308, "v_max": 1e308, "n_seen": 2})
    for rule, kwargs in (("deb", {}), ("fixed", {"lam": 1.0})):
        with pytest.raises(ValueError):  # not fsum's OverflowError
            rank_candidates(rule, **segments(np.zeros((1, 2), bool), np.full((1, 2), 1e308), np.full((1, 2), 1e308)), **kwargs)
    for score, args in ((fixed_penalty_scores, (one[:1], np.array([[-1e308]]), 1e308)),
                        (rosarl_scores, (one[:1], np.array([[0.0]]), np.array([[-1e308]]), -1e308))):
        with pytest.raises(ValueError):  # a score below the float range, not -inf
            score(*args)


def test_rosarl_state_round_trip_and_validation():
    pen = RosarlPenalty()
    assert RosarlPenalty.from_state(pen.state()) == pen
    pen.update([3.0, -1.0, 2.0])
    clone = RosarlPenalty.from_state(pen.state())
    assert clone == pen and clone.penalty == -4.0
    assert pickle.loads(pickle.dumps(pen)) == pen
    pen.update(np.array([[5.0]]))
    clone.update([5.0])
    assert clone == pen and pen.penalty == -6.0 and pen.n_seen == 4
    pen.update([])
    assert clone == pen
    with pytest.raises(ValueError):
        pen.update([1.0, np.nan])
    assert clone == pen
    for bad in ({"v_min": 1.0, "v_max": 0.0, "n_seen": 2}, {"v_min": math.inf, "v_max": -math.inf, "n_seen": 1},
                {"v_min": 0.0, "v_max": 1.0, "n_seen": 0}, {"v_min": 0.0, "v_max": 1.0, "n_seen": 2.5}):
        with pytest.raises(ValueError):
            RosarlPenalty.from_state(bad)


def test_rosarl_is_probability_weighted_where_deb_is_lexicographic():
    seg = segments(np.array([[False, False, False, True], [False] * 4]), np.array([[10.0] * 4, [1.0] * 4]),
                   np.array([[10.0, 10.0, 10.0, 5.0], [1.0] * 4]))
    assert rank_candidates("deb", **seg)[0].tolist() == [1, 0]
    ranks, info = rank_candidates("rosarl", **seg, rosarl=RosarlPenalty())
    assert ranks.tolist() == [0, 1] and info["penalty"] == -9.0 and info["scores"].tolist() == [6.5, 1.0]


def test_single_segment_rosarl_bounds_violators_by_v_min_but_is_not_debs_first_rule():
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(200):
        violated, ret, ret_pre = population(rng, ties=True, m_range=(1, 2))  # small integers: exact arithmetic
        pen = RosarlPenalty()
        _, info = rank_candidates("rosarl", **segments(violated, ret, ret_pre), rosarl=pen)
        v = violated[:, 0]
        if v.any() and not v.all():
            assert info["scores"][v].max() <= pen.v_min <= info["scores"][~v].min()
            checked += 1
    assert checked > 50
    tie = segments(np.array([[True], [False]]), np.array([[5.0], [1.0]]), np.array([[2.0], [1.0]]))
    ranks, info = rank_candidates("rosarl", **tie, rosarl=RosarlPenalty())  # ret_pre = V_MAX, the safe ret = V_MIN
    assert info["scores"].tolist() == [1.0, 1.0] and ranks.tolist() == [0, 1]  # the index puts the violator first
    assert rank_candidates("deb", **tie)[0].tolist() == [1, 0]
    rounded = segments(np.array([[False], [True]]), np.array([[-1e-17], [1.0]]), np.array([[-1e-17], [1.0]]))
    ranks, info = rank_candidates("rosarl", **rounded, rosarl=RosarlPenalty())  # -1e-17 - 1.0 rounds to -1.0
    assert info["penalty"] == -1.0 and info["scores"].tolist() == [-1e-17, 0.0] and ranks.tolist() == [1, 0]


# ---- ties, layout and pycma ----------------------------------------------------------------
@pytest.mark.parametrize("rule", ["deb", "fixed", "rosarl"])
def test_ties_are_exact_and_broken_by_candidate_index(rule):
    rng = np.random.default_rng(8)
    M = 37
    v0 = rng.random(M) < 0.3
    r0 = rng.normal(0, 1, M)
    rp0 = np.where(v0, r0 - rng.uniform(0, 1, M), r0)
    perms = [np.arange(M), rng.permutation(M), rng.permutation(M), np.arange(M)[::-1]]
    rows = [(v0[q], r0[q], rp0[q]) for q in perms]  # the same segments in four orders
    rows.insert(1, (v0, r0 - 1.0, rp0 - 1.0))
    rows.insert(4, (v0, r0 - 1.0, rp0 - 1.0))  # candidates 1 and 4: a strictly worse twin pair
    seg = segments(*(np.stack(c) for c in zip(*rows)))
    kwargs = {"lam": 1.0} if rule == "fixed" else {"rosarl": RosarlPenalty()} if rule == "rosarl" else {}
    ranks, info = rank_candidates(rule, **seg, **kwargs)
    assert ranks.tolist() == [0, 4, 1, 2, 5, 3] and info["order"].tolist() == [0, 2, 3, 5, 1, 4]
    for key in ("p", "mean_ret") + (("scores",) if rule != "deb" else ()):
        assert len(set(info[key][[0, 2, 3, 5]].tolist())) == 1 and info[key][1] == info[key][4], key
    again, info2 = rank_candidates(rule, **seg, **kwargs)
    assert np.array_equal(again, ranks) and np.array_equal(info2["order"], info["order"])
    same = segments(*(np.stack([c] * 5) for c in (v0, r0, rp0)))
    assert rank_candidates(rule, **same, **kwargs)[0].tolist() == [0, 1, 2, 3, 4]


def test_population_layout_is_flattened_row_major():
    rng = np.random.default_rng(9)
    P, R, n = 5, 4, 3
    shape = (P, R, n, K)
    m = segment_metrics(np.stack([rng.uniform(0.6, 2.1, shape), rng.uniform(-1.2, 1.2, shape), rng.normal(1, 1, shape)], -1))
    for rule, kwargs in (("deb", {}), ("fixed", {"lam": 0.5}), ("rosarl", {})):
        a, ia = rank_candidates(rule, **segments(m["violated"], m["ret"], m["ret_pre"]), rosarl=RosarlPenalty(), **kwargs)
        b, ib = rank_candidates(rule, **segments(*(m[k].reshape(P, R * n) for k in ("violated", "ret", "ret_pre"))),
                                rosarl=RosarlPenalty(), **kwargs)
        assert np.array_equal(a, b) and ia["n_segments"] == R * n
        for key in ("p", "mean_ret", "scores"):
            assert (ia[key] is None and ib[key] is None) or np.array_equal(ia[key], ib[key])


def test_order_key_ends_with_the_candidate_index():
    keys = order_key("deb", {"p": [0.5, 0.0, 0.5], "mean_ret": [1.0, -2.0, 3.0]})
    assert len(keys) == 3 and keys[-1].tolist() == [0, 1, 2]
    assert np.lexsort(keys[::-1]).tolist() == [1, 2, 0]
    assert deb_ranks([0.5, 0.0, 0.5], [1.0, -2.0, 3.0]).tolist() == [2, 0, 1]
    assert np.lexsort(order_key("rosarl", {"scores": [1.0, 3.0, 1.0]})[::-1]).tolist() == [1, 0, 2]
    for rule, info in (("cvar", {"scores": [1.0]}), ("deb", {"p": [0.0, 1.0], "mean_ret": [1.0]}),
                       ("fixed", {"scores": [np.nan, 1.0]})):
        with pytest.raises(ValueError):
            order_key(rule, info)


def test_rank_candidates_rejects_bad_inputs():
    v, r = np.zeros((3, 4), bool), np.ones((3, 4))
    nan = r.copy()
    nan[1, 2] = np.nan
    for rule, kwargs in (("cvar", {}), ("fixed", {}), ("fixed", {"lam": -1.0}), ("rosarl", {}),
                         ("deb", {"ret": r[:, :3]}), ("deb", {"violated": np.full((3, 4), 2)}), ("deb", {"ret": nan}),
                         ("deb", {"violated": v[0], "ret": r[0], "ret_pre": r[0]}), ("deb", {"violated": v[:0], "ret": r[:0], "ret_pre": r[:0]})):
        with pytest.raises(ValueError):
            rank_candidates(rule, **{**segments(v, r, r), **kwargs})
    ranks, info = rank_candidates("deb", **segments(v.astype(int), r, r))  # 0/1 flags are accepted
    assert ranks.tolist() == [0, 1, 2] and info["p"].tolist() == [0.0, 0.0, 0.0]


def test_ranks_to_fitness_gives_pycma_the_same_order(tmp_path, monkeypatch):
    fitness = ranks_to_fitness(np.array([2, 0, 1]))
    assert fitness.dtype == np.float64 and fitness.tolist() == [2.0, 0.0, 1.0]
    for bad in ([0, 0, 1], [1, 2, 3], [[0, 1]], [0.5, 1.5]):
        with pytest.raises(ValueError):
            ranks_to_fitness(bad)
    cma = pytest.importorskip("cma")
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(10)
    violated, ret, ret_pre = population(rng, ties=True, p_range=(12, 13), m_range=(6, 7))
    global_state = np.random.get_state()  # pycma seeds numpy's global RNG; leave it as found
    try:
        es = cma.CMAEvolutionStrategy(np.zeros(5), 0.3, {"popsize": 12, "seed": 3, "verbose": -9, "verb_log": 0, "verb_disp": 0})
        X = es.ask()
        ranks, info = rank_candidates("deb", **segments(violated, ret, ret_pre))
        fitness = ranks_to_fitness(ranks)
        assert fitness[info["order"][0]] == 0.0
        es.tell(X, fitness)
        assert np.array_equal(es.fit.idx, info["order"])
    finally:
        np.random.set_state(global_state)
