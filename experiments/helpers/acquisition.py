"""Acquisition arms for E3 (protocol.md): random, predicted boundary, learned risk, oracle.

All arms choose from the SAME candidate pool per root (one common proposal generator).
Choosing costs model compute; executing costs charged simulator steps. Selection is
balanced across roots. Selection rules and band widths are frozen after development.

    pool = CandidatePool.build(...)            # proposals per root, never executed here
    scores = score_pool(pool, imaginer, probe, layouts)   # predicted clearance + features
    chosen = select("boundary", scores, n=64, rng=rng, margin=m, band=10)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from helpers.branchBank import Proposal, propose
from helpers.decomposition import RootLatentCache, interp_from_endpoints
from helpers.pushtAssets import ACTION_BLOCK
from helpers.pushtGeometry import clearance_trace

ARMS = ("random", "boundary", "learned", "oracle")


@dataclass
class Candidate:
    root_index: int
    proposal: Proposal
    key: tuple  # (root_index, position in pool)
    features: dict = field(default_factory=dict)
    executed: bool = False


@dataclass
class CandidatePool:
    candidates: list[Candidate]

    @staticmethod
    def build(rng, bank, per_root: int, *, n_stress: int, n_toward: int, layouts_by_root: dict) -> "CandidatePool":
        from helpers.pushtReplay import make_env, reset_root

        env = make_env()
        cands = []
        for ri, root in enumerate(bank.roots):
            ctx = reset_root(env, root)
            hz = layouts_by_root[root.root_id]["familiar"]
            props, _ = propose(rng, root, ctx, n_random=per_root - n_stress - n_toward - 1, sigmas=(0.05, 0.1, 0.2), n_stress=n_stress,
                               hazard_centre=hz.centre, n_toward_hazard=n_toward)
            for p_i, p in enumerate(props):
                cands.append(Candidate(ri, p, (ri, p_i)))
        return CandidatePool(cands)

    def by_root(self) -> dict[int, list[Candidate]]:
        out: dict[int, list[Candidate]] = {}
        for c in self.candidates:
            out.setdefault(c.root_index, []).append(c)
        return out

    def unexecuted(self) -> list[Candidate]:
        return [c for c in self.candidates if not c.executed]


def score_pool(pool: CandidatePool, imaginer, probe, layouts_by_root: dict, bank, cache: RootLatentCache | None = None) -> None:
    """Fill `features` for every candidate with the CURRENT model: predicted min clearance,
    when it occurs, predicted displacement/rotation, action statistics. No simulator."""
    cache = cache or RootLatentCache(imaginer, bank)
    for ri, cands in pool.by_root().items():
        root = bank.roots[ri]
        hz = layouts_by_root[root.root_id]["familiar"]
        zh, hb, frames = cache.get(ri)
        tapes = np.stack([c.proposal.tape for c in cands])
        z_imag = imaginer.rollout(zh, hb, tapes)
        n, K, D = z_imag.shape
        pose_imag = probe.predict_pose(z_imag.reshape(-1, D)).reshape(n, K, 3)
        pose0 = probe.predict_pose(zh[-1:])
        state0 = np.asarray(root.meta["state_at_root"])
        for a, c in enumerate(cands):
            p6 = np.concatenate([pose0, pose_imag[a]], 0)
            ends = np.zeros((K + 1, 7))
            ends[:, 2:5] = p6
            tr = interp_from_endpoints(ends)[:, 2:5]
            cl = clearance_trace(tr, hz)
            path = state0[:2] + np.cumsum(c.proposal.tape.reshape(-1, 2) * 100.0, axis=0)
            c.features = {"c_hat": float(cl.min()), "argmin_step": int(np.argmin(cl)) / (K * ACTION_BLOCK),
                          "disp_hat": float(np.linalg.norm(p6[-1, :2] - p6[0, :2])), "rot_hat": float(np.degrees(abs(np.arctan2(np.sin(p6[-1, 2] - p6[0, 2]), np.cos(p6[-1, 2] - p6[0, 2]))))),
                          "pusher_min_dist_to_block": float(np.min(np.linalg.norm(path - state0[2:4], axis=1))),
                          "action_norm_mean": float(np.linalg.norm(c.proposal.tape.reshape(-1, 2), axis=1).mean()), "action_norm_max": float(np.linalg.norm(c.proposal.tape.reshape(-1, 2), axis=1).max()),
                          "kind_stress": float(c.proposal.kind in ("stress", "toward_hazard"))}


FEATURE_KEYS = ("c_hat", "argmin_step", "disp_hat", "rot_hat", "pusher_min_dist_to_block", "action_norm_mean", "action_norm_max", "kind_stress")


def feature_matrix(cands: list[Candidate]) -> np.ndarray:
    return np.array([[c.features[k] for k in FEATURE_KEYS] for c in cands], dtype=np.float64)


def _balanced_pick(groups: dict[int, list], n: int, rng, key=None) -> list:
    """Round-robin over roots; within a root, by `key` (ascending) or random."""
    order = list(groups)
    rng.shuffle(order)
    lists = {}
    for r in order:
        items = list(groups[r])
        if key is None:
            rng.shuffle(items)
        else:
            items.sort(key=key)
        lists[r] = items
    out = []
    while len(out) < n and any(lists.values()):
        for r in order:
            if lists[r]:
                out.append(lists[r].pop(0))
                if len(out) >= n:
                    break
    return out


def select(arm: str, pool: CandidatePool, n: int, rng, *, margin: float = 0.0, band: float = 10.0, risk_model=None, true_errors: dict | None = None) -> list[Candidate]:
    groups = {r: [c for c in cs if not c.executed] for r, cs in pool.by_root().items()}
    groups = {r: cs for r, cs in groups.items() if cs}
    if arm == "random":
        return _balanced_pick(groups, n, rng)
    if arm == "boundary":
        # distance of predicted clearance to the operating margin; in-band first, then nearest
        return _balanced_pick(groups, n, rng, key=lambda c: (abs(c.features["c_hat"] - margin) > band, abs(c.features["c_hat"] - margin) + rng.uniform(0, 1e-3)))
    if arm == "learned":
        assert risk_model is not None
        for r, cs in groups.items():
            X = feature_matrix(cs)
            risk = risk_model.predict(X)
            for c, s in zip(cs, risk):
                c.features["risk_hat"] = float(s)
        return _balanced_pick(groups, n, rng, key=lambda c: -c.features["risk_hat"])
    if arm == "oracle":
        assert true_errors is not None, "oracle needs true optimistic errors for the whole pool"
        return _balanced_pick(groups, n, rng, key=lambda c: -true_errors.get(c.key, -np.inf))
    raise ValueError(arm)


class RiskModel:
    """Ridge regression of the optimistic error (c_hat - c_true) on pre-execution features.

    Retrained each round on branches already executed. Never sees an unqueried future.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.w = None
        self.mu = None
        self.sd = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiskModel":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xn = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        A = Xn.T @ Xn + self.alpha * np.eye(Xn.shape[1])
        self.w = np.linalg.solve(A, Xn.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            return np.zeros(len(X))
        Xn = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return Xn @ self.w


def true_clearance(bank_h5_states: np.ndarray, hz) -> float:
    return float(clearance_trace(bank_h5_states[:, 2:5], hz).min())
