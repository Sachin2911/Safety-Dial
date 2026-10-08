"""The four-source decomposition and per-bank evaluation shared by E1 to E4.

For every branch of a bank and every hazard layout of its root, compute the minimum
signed clearance of the rule under each source (protocol.md):
  dense, endpoint, real_readout, imagined, stationary, coord_mlp (optional).
`evaluate_model_on_bank` is the cheap variant used during adaptation and acquisition:
only the imagined source (with a given model) plus dense truth.
"""

from __future__ import annotations

import time

import numpy as np

from helpers.dialMetrics import cluster_bootstrap, fsa
from helpers.pushtAssets import ACTION_BLOCK
from helpers.pushtGeometry import clearance_trace
from helpers.pushtReplay import endpoint_interpolation, observation_domain_mask

SOURCES = ["dense", "endpoint", "real_readout", "imagined", "stationary", "coord_mlp"]


def contact_metadata(branch: dict) -> dict:
    """Only body-specific observed counters establish pusher-T contact or free motion."""
    aggregate = np.asarray(branch["n_contacts"])
    observed = np.asarray(branch.get("observed", np.ones(len(aggregate), bool)))
    specific = branch.get("pusher_block_contacts")
    typed = specific is not None and branch.get("contact_kind") == "pusher_block"
    wall = branch.get("block_wall_contacts")
    return {"contact": bool(np.any(np.asarray(specific)[observed] > 0)) if typed else None,
            "contact_kind": "pusher_block" if typed else "any_collision",
            "contact_mechanism_identifiable": typed,
            "any_collision": bool(np.any(aggregate[observed] > 0)),
            "block_wall_contact": bool(np.any(np.asarray(wall)[observed] > 0)) if wall is not None else None,
            "contact_counter": branch.get("contact_counter", "upstream_any_collision")}


def outcome_metadata(states, clearances, *, observed=None, observation_valid=None,
                     terminated=None, truncated=None) -> dict:
    """Separate observed hazard failure, image-domain exits and unknown future steps."""
    states, clearances = np.asarray(states), np.asarray(clearances, dtype=float)
    observed = np.ones(len(states), bool) if observed is None else np.asarray(observed, bool)
    if observed.shape != (len(states),) or not observed[0]:
        raise ValueError("Outcome observation mask must include the root state")
    domain = observation_domain_mask(states)
    exit_seen = bool((~domain & observed).any())
    valid = domain & observed if observation_valid is None else np.asarray(observation_valid, bool)
    hazard = bool((clearances[observed] <= 0).any())
    censored = bool((~observed).any())
    return {
        "hazard_unsafe_observed": hazard, "hazard_boundary_convention": "clearance <= 0 includes contact",
        "hazard_only_unsafe": True if hazard else (None if censored else False),
        "hazard_censored": bool(censored and not hazard),
        "observation_domain_exit": exit_seen, "arena_exit": exit_seen,
        "censored": censored, "unsafe_composite": bool(hazard or exit_seen),
        "observed_steps": int(observed.sum()) - 1,
        "horizon_steps": len(states) - 1,
        "observation_valid": bool(valid.all()),
        "terminated": bool(np.any(terminated)) if terminated is not None else None,
        "truncated": bool(np.any(truncated)) if truncated is not None else None,
        "termination_flags_recorded": terminated is not None and truncated is not None,
        "cmin_dense": float(clearances[observed].min()),
    }


def _outcome_from_bank(bank, j, states, clearances):
    return outcome_metadata(states, clearances, **{
        key: bank.h5[key][int(j)] for key in
        ("observed", "observation_valid", "terminated", "truncated") if key in bank.h5
    })


def row_outcomes(rows):
    """Known composite violations and censoring mask for fsa(..., censored=...)."""
    unsafe = np.array([r.get("unsafe_composite", r["cmin_dense"] <= 0) for r in rows], bool)
    censored = np.array([r.get("censored", False) for r in rows], bool)
    return unsafe, censored


def interp_from_endpoints(end_states: np.ndarray) -> np.ndarray:
    """(K+1, 7) endpoint states -> (K*ACTION_BLOCK+1, 7) interpolated states."""
    K = len(end_states) - 1
    full = np.zeros((K * ACTION_BLOCK + 1, 7))
    full[::ACTION_BLOCK] = end_states
    return endpoint_interpolation(full)


def interp_poses_batch(poses: np.ndarray, block: int = ACTION_BLOCK) -> np.ndarray:
    """(N, K+1, 3) endpoint poses -> (N, K*block+1, 3) linearly interpolated (angle on the circle)."""
    poses = np.asarray(poses, dtype=float)
    N, K1, _ = poses.shape
    K = K1 - 1
    w = (np.arange(block) / block)[None, None, :, None]  # (1, 1, block, 1)
    p0 = poses[:, :-1, None, :]
    p1 = poses[:, 1:, None, :]
    out = p0 * (1 - w) + p1 * w  # (N, K, block, 3)
    d = np.arctan2(np.sin(p1[..., 2] - p0[..., 2]), np.cos(p1[..., 2] - p0[..., 2]))
    out[..., 2] = (p0[..., 2] + w[..., 0] * d) % (2 * np.pi)
    out = out.reshape(N, K * block, 3)
    return np.concatenate([out, poses[:, -1:, :]], axis=1)


def ang_err_deg(a, b):
    return np.degrees(np.abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def layouts_by_root(layouts) -> dict:
    out = {}
    for lay in layouts:
        out.setdefault(lay.root_id, {})[lay.family] = lay.shape
    return out


class RootLatentCache:
    """Encoded history frames and history actions per root (re-encoded on demand)."""

    def __init__(self, imaginer, bank):
        from helpers.pushtReplay import make_env

        self.imaginer, self.bank, self.env, self.cache = imaginer, bank, make_env(), {}

    def get(self, ri: int):
        if ri not in self.cache:
            from helpers.pushtReplay import reset_root

            ctx = reset_root(self.env, self.bank.roots[ri])
            self.cache[ri] = (self.imaginer.encode(ctx.frames), ctx.history_actions.copy(), ctx.frames)
        return self.cache[ri]


def analyse_bank(name, bank, layouts, imaginer, probe, coord=None, *, verbose=True, evaluation_ledger=None) -> dict:
    """All sources; optionally charge each attempted root-history replay step.

    Repeated branches/layouts share one cached history per root. The owned replay
    environment is closed on success and failure; the supplied ledger survives both.
    """
    lay = layouts_by_root(layouts)
    N = len(bank)
    rows, horizon_rows, diagnostic_pose_rows = [], [], []
    srcs = ["endpoint", "real_readout", "imagined"] + (["coord_mlp"] if coord is not None else [])
    pose_err = {s: {"centre": [], "angle": []} for s in srcs}
    cache = RootLatentCache(imaginer, bank)
    try:
        if evaluation_ledger is not None:
            from helpers.acquisitionSafety import MeteredEnv

            cache.env = MeteredEnv(cache.env, evaluation_ledger, f"{name}_history")
        t0 = time.time()
        for j in range(N):
            b = bank.branch(j, frames=True)
            root = bank.root_of(j)
            ri = int(b["root_index"])
            st = b["states"]
            tape = b["tape"].astype(np.float64)
            ends = st[::ACTION_BLOCK]
            z_real = imaginer.encode(b["frames"])
            pose_real = probe.predict_pose(z_real)
            z_hist, hist_blocks, _ = cache.get(ri)
            z_imag = imaginer.rollout(z_hist, hist_blocks, tape[None])[0]
            pose_imag = np.concatenate([pose_real[:1], probe.predict_pose(z_imag)], 0)
            traces = {
                "dense": st[:, 2:5],
                "endpoint": interp_from_endpoints(ends)[:, 2:5],
                "real_readout": interp_from_endpoints(np.column_stack([ends[:, :2], pose_real, ends[:, 5:]]))[:, 2:5],
                "imagined": interp_from_endpoints(np.column_stack([ends[:, :2], pose_imag, ends[:, 5:]]))[:, 2:5],
                "stationary": interp_from_endpoints(np.repeat(st[:1], len(ends), 0))[:, 2:5],
            }
            poses6 = {"endpoint": ends[:, 2:5], "real_readout": pose_real, "imagined": pose_imag}
            if coord is not None:
                cs = coord.rollout(st[0], tape[None])[0]
                traces["coord_mlp"] = interp_from_endpoints(cs)[:, 2:5]
                poses6["coord_mlp"] = cs[:, 2:5]
            observed = np.asarray(b.get("observed", np.ones(len(st), bool)))
            valid = np.asarray(b.get("observation_valid", observation_domain_mask(st)))
            endpoint_valid = observed[::ACTION_BLOCK] & valid[::ACTION_BLOCK]
            for s, p6 in poses6.items():
                centre_error = np.linalg.norm(p6[:, :2] - ends[:, 2:4], axis=1)
                angle_error = ang_err_deg(p6[:, 2], ends[:, 4])
                centre_error[~endpoint_valid], angle_error[~endpoint_valid] = np.nan, np.nan
                pose_err[s]["centre"].append(centre_error)
                pose_err[s]["angle"].append(angle_error)
            contact = contact_metadata(b)
            rot = float(ang_err_deg(st[-1, 4], st[0, 4]))
            disp = float(np.linalg.norm(st[-1, 2:4] - st[0, 2:4]))
            disp_imag = float(np.linalg.norm(pose_imag[-1, :2] - pose_imag[0, :2]))
            rot_imag = float(ang_err_deg(pose_imag[-1, 2], pose_imag[0, 2]))
            clearance_by_layout = {}
            for fam, hz in lay.get(root.root_id, {}).items():
                clearance_by_layout[fam] = {s: clearance_trace(tr, hz) for s, tr in traces.items()}
                cd = clearance_by_layout[fam]["dense"]
                cmins = {s: float(values.min()) for s, values in clearance_by_layout[fam].items()}
                rows.append({"bank": name, "branch": j, "root": ri, "layout": fam, "kind": b["kind"], **contact, "rotation_deg": rot,
                             "displacement_px": disp, "imag_displacement_px": disp_imag, "imag_rotation_deg": rot_imag,
                             "dense_argmin_step": int(np.argmin(cd)), **{f"cmin_{s}": v for s, v in cmins.items()},
                             **_outcome_from_bank(bank, j, st, cd)})
            prefix_rows, prefix_pose_rows = cumulative_branch_diagnostics(name, j, ri, root, b, traces, poses6, clearance_by_layout)
            horizon_rows.extend(prefix_rows)
            diagnostic_pose_rows.extend(prefix_pose_rows)
            if verbose and (j + 1) % 200 == 0:
                print(f"[decomp:{name}] {j + 1}/{N} branches {time.time() - t0:.0f}s")
    finally:
        cache.env.close()
    errs = {s: {"centre_by_block": np.nanmean(np.stack(v["centre"]), axis=0).tolist(), "centre_p95_by_block": np.nanpercentile(np.stack(v["centre"]), 95, axis=0).tolist(),
                "angle_by_block": np.nanmean(np.stack(v["angle"]), axis=0).tolist(), "angle_p95_by_block": np.nanpercentile(np.stack(v["angle"]), 95, axis=0).tolist()}
            for s, v in pose_err.items() if v["centre"]}
    return {"rows": rows, "pose_err": errs, "horizon_rows": horizon_rows, "diagnostic_pose_rows": diagnostic_pose_rows}


def evaluate_model_on_bank(name, bank, layouts, imaginer, probe, *, correction=None, cache: RootLatentCache | None = None, chunk_roots: bool = True) -> list[dict]:
    """Imagined source only (fast). Rows carry cmin_dense, cmin_imagined and regime flags.

    `correction` (optional) is a ReadoutCorrection applied to the imagined latents.
    """
    import torch

    lay = layouts_by_root(layouts)
    cache = cache or RootLatentCache(imaginer, bank)
    rows = []
    root_ids = np.unique(bank.h5["root_index"][:])
    for ri in root_ids:
        idx = bank.indices_for_root(int(ri))
        root = bank.roots[int(ri)]
        if root.root_id not in lay:
            continue
        z_hist, hist_blocks, _ = cache.get(int(ri))
        tapes = np.stack([bank.h5["tape"][int(j)].astype(np.float64) for j in idx])
        z_imag = imaginer.rollout(z_hist, hist_blocks, tapes)  # (n, K, D)
        n, K, D = z_imag.shape
        if correction is not None:
            with torch.no_grad():
                steps = torch.arange(1, K + 1, device=z_imag.device).expand(n, K)
                y = probe(z_imag.reshape(-1, D)).reshape(n, K, 4) + correction(z_imag, steps)
                th = torch.atan2(y[..., 2], y[..., 3]) % (2 * np.pi)
                pose_imag = torch.stack([y[..., 0], y[..., 1], th], -1).cpu().numpy()
        else:
            pose_imag = probe.predict_pose(z_imag.reshape(-1, D)).reshape(n, K, 3)
        pose0 = probe.predict_pose(z_hist[-1:])  # current frame readout
        for a, j in enumerate(idx):
            st = bank.h5["states"][int(j)]
            ends = st[::ACTION_BLOCK]
            p6 = np.concatenate([pose0, pose_imag[a]], 0)
            tr_imag = interp_from_endpoints(np.column_stack([ends[:, :2], p6, ends[:, 5:]]))[:, 2:5]
            contact = contact_metadata(bank.branch(int(j)))
            kind = bank.h5["kind"][int(j)]
            kind = kind.decode() if isinstance(kind, bytes) else str(kind)
            for fam, hz in lay[root.root_id].items():
                cd = clearance_trace(st[:, 2:5], hz)
                rows.append({"bank": name, "branch": int(j), "root": int(ri), "layout": fam, "kind": kind, **contact,
                             "cmin_dense": float(cd.min()), "cmin_imagined": float(clearance_trace(tr_imag, hz).min()),
                             "displacement_px": float(np.linalg.norm(st[-1, 2:4] - st[0, 2:4])), "imag_displacement_px": float(np.linalg.norm(p6[-1, :2] - p6[0, :2])),
                             "rotation_deg": float(ang_err_deg(st[-1, 4], st[0, 4])), "imag_rotation_deg": float(ang_err_deg(p6[-1, 2], p6[0, 2])),
                             "centre_err_h5_px": float(np.linalg.norm(p6[-1, :2] - ends[-1, 2:4])), "angle_err_h5_deg": float(ang_err_deg(p6[-1, 2], ends[-1, 4])),
                             **_outcome_from_bank(bank, j, st, cd)})
    return rows


def decision_table(rows, m: float, sources=None, boot: int = 500) -> dict:
    sources = sources or [s for s in SOURCES if rows and f"cmin_{s}" in rows[0]]
    u, censored = row_outcomes(rows)
    root = np.array([r["root"] for r in rows])
    out = {"m": float(m), "n": int(len(rows)), "n_unsafe": int(u.sum()),
           "n_censored": int(censored.sum()),
           "n_observation_domain_exits": sum(r.get("observation_domain_exit", False) for r in rows),
           "outcome": "observed hazard violation OR observation-domain exit; unresolved futures censored"}
    for s in sources:
        c = np.array([r[f"cmin_{s}"] for r in rows])
        out[s] = fsa(c, u, m, censored=censored)
    acc = {s: np.array([r[f"cmin_{s}"] >= m for r in rows]) for s in sources}
    if "imagined" in acc:
        fs4 = acc["imagined"] & u
        domain_exit = np.array([r.get("observation_domain_exit", False) for r in rows])
        attr = {"n_false_safe_imagined": int(fs4.sum()),
                "domain_exit": int((fs4 & domain_exit).sum()),
                "censored": int((fs4 & ~domain_exit & censored).sum())}
        fs4 = fs4 & ~domain_exit & ~censored
        if "endpoint" in acc and "real_readout" in acc:
            attr.update({"temporal": int((fs4 & acc["endpoint"]).sum()), "readout": int((fs4 & ~acc["endpoint"] & acc["real_readout"]).sum()),
                         "imagination": int((fs4 & ~acc["endpoint"] & ~acc["real_readout"]).sum())})
        out["attribution"] = attr
        if len(rows) and boot:
            out["fsa_imagined_ci"] = cluster_bootstrap(lambda c, u, censored: fsa(c, u, m, censored=censored)["fsa"], root, n_boot=boot, c=np.array([r["cmin_imagined"] for r in rows]), u=u, censored=censored)
    return out


def by_regime(rows, m: float, sources=None) -> dict:
    out = {}
    groups = {"contact": lambda r: r["contact"] is True, "free": lambda r: r["contact"] is False,
              "contact_unknown": lambda r: r["contact"] is None,
              "rotation>=10deg": lambda r: r["rotation_deg"] >= 10, "rotation<10deg": lambda r: r["rotation_deg"] < 10,
              "near_boundary(|c|<20)": lambda r: abs(r["cmin_dense"]) < 20, "far(|c|>=20)": lambda r: abs(r["cmin_dense"]) >= 20}
    for g, f in groups.items():
        sub = [r for r in rows if f(r)]
        if sub:
            t = decision_table(sub, m, sources, boot=0)
            out[g] = {"n": len(sub), "n_unsafe": t["n_unsafe"], **{s: {"fsa": t[s]["fsa"], "ar": t[s]["acceptance_rate"], "n_acc": t[s]["n_accepted"]} for s in t if isinstance(t[s], dict) and "fsa" in t[s]}, "attribution": t.get("attribution")}
    return out


def ordinary_motion(rows) -> dict:
    """Predicted against true displacement/rotation on ordinary (random/nominal) tapes."""
    sub = [r for r in rows if r["kind"] in ("nominal", "random") and r["layout"] == "familiar" and not r.get("censored", False) and r.get("observation_valid", True)]
    if not sub:
        return {}
    dt = np.array([r["displacement_px"] for r in sub])
    di = np.array([r["imag_displacement_px"] for r in sub])
    rt = np.array([r["rotation_deg"] for r in sub])
    ri = np.array([r["imag_rotation_deg"] for r in sub])
    return {"n": len(sub), "disp_true_mean_px": float(dt.mean()), "disp_imag_mean_px": float(di.mean()), "disp_ratio": float(di.sum() / max(dt.sum(), 1e-9)),
            "disp_abs_err_mean_px": float(np.abs(di - dt).mean()), "rot_true_mean_deg": float(rt.mean()), "rot_imag_mean_deg": float(ri.mean()), "rot_ratio": float(ri.sum() / max(rt.sum(), 1e-9))}


HORIZON_REPORT_PROTOCOL = {
    "version": "cumulative-prefix-diagnostics-v1",
    "horizon": "root through K action blocks, inclusive; K starts at 1",
    "truth": "observed clearance <= 0 OR observed pusher-domain exit; unknown suffix is censored",
    "acceptance": "predicted cumulative clearance >= margin, including equality",
    "source_availability": {
        "dense": "every prefix state observed",
        "endpoint": "both true endpoints of each interpolated prefix block observed",
        "real_readout": "both endpoints of every interpolated prefix block observed and image-valid; no earlier domain exit",
        "imagined": "valid observed root image; actual future need not remain observed or in-domain",
        "stationary": "observed root pose",
        "coord_mlp": "observed root state",
    },
    "pose_errors": "Endpoint error at K, scored only against an observed, image-valid target with no earlier observed domain exit; one contribution per physical branch, never per layout",
    "clearance_errors": "Predicted prefix minimum minus dense prefix minimum; only complete, image-valid true prefixes and available sources; positive is optimistic",
    "regimes": "Contact and rotation use this prefix only. No observed contact in an incomplete prefix is unknown, not free motion. Boundary-distance strata require complete dense truth.",
    "layout_strata": "A physical branch may occur in both near/far strata under different layouts, once per stratum for pose errors",
    "compatibility": "Supplementary diagnostics only; legacy full-horizon rows, metrics and development gate are unchanged",
}


def cumulative_branch_diagnostics(name, branch_index, root_index, root, branch, traces, poses,
                                  clearances_by_layout):
    """Derive cumulative prefixes from already computed traces; no model/simulator calls."""
    states = np.asarray(branch["states"])
    n_steps = len(states) - 1
    if n_steps % ACTION_BLOCK:
        raise ValueError("Diagnostic branch horizon must contain whole action blocks")
    observed = np.asarray(branch.get("observed", np.ones(len(states), bool)), bool)
    domain = observation_domain_mask(states)
    stored_valid = np.asarray(branch.get("observation_valid", domain), bool)
    if observed.shape != domain.shape or stored_valid.shape != domain.shape:
        raise ValueError("Diagnostic masks must match dense state length")
    # Re-entry into the image domain does not restore a valid observation history.
    domain_history_valid = np.logical_and.accumulate(domain | ~observed)
    image_valid = observed & domain & stored_valid & domain_history_valid
    source_poses = {**poses, "dense": states[::ACTION_BLOCK, 2:5],
                    "stationary": np.repeat(states[:1, 2:5], len(states[::ACTION_BLOCK]), axis=0)}
    horizon_rows, pose_rows = [], []
    for horizon in range(1, n_steps // ACTION_BLOCK + 1):
        stop = horizon * ACTION_BLOCK + 1
        endpoint = stop - 1
        prefix_observed, prefix_valid = observed[:stop], image_valid[:stop]
        contact_branch = {**branch, "observed": prefix_observed}
        for key in ("n_contacts", "pusher_block_contacts", "block_wall_contacts"):
            if branch.get(key) is not None:
                contact_branch[key] = np.asarray(branch[key])[:stop]
        contact = contact_metadata(contact_branch)
        if not prefix_observed.all() and contact["contact"] is False:
            contact["contact"] = None
        available = {"dense": bool(prefix_observed.all()),
            "endpoint": bool(observed[:stop:ACTION_BLOCK].all()),
            "real_readout": bool(image_valid[:stop:ACTION_BLOCK].all()),
            "imagined": bool(image_valid[0]), "stationary": bool(observed[0]), "coord_mlp": bool(observed[0])}
        identity = {"bank": name, "branch": int(branch_index), "root": int(root_index),
            "source_episode": root.meta.get("episode"), "kind": branch["kind"], "horizon_blocks": horizon,
            "horizon_steps": endpoint}
        rotation = float(ang_err_deg(states[endpoint, 4], states[0, 4])) if prefix_observed.all() else None
        pose_row = {**identity, **contact, "rotation_deg": rotation,
                    "target_observed": bool(observed[endpoint]), "target_image_valid": bool(image_valid[endpoint]), "errors": {}}
        for source, predicted in source_poses.items():
            prediction = np.asarray(predicted)[horizon]
            usable = bool(image_valid[endpoint] and available[source] and np.isfinite(prediction).all())
            pose_row["errors"][source] = {
                "available": usable,
                "centre_px": float(np.linalg.norm(prediction[:2] - states[endpoint, 2:4])) if usable else None,
                "angle_deg": float(ang_err_deg(prediction[2], states[endpoint, 4])) if usable else None,
            }
        pose_rows.append(pose_row)
        for family, clearance in clearances_by_layout.items():
            dense = np.asarray(clearance["dense"])[:stop]
            flags = {key: np.asarray(branch[key])[:stop] for key in ("terminated", "truncated") if key in branch}
            truth = outcome_metadata(states[:stop], dense, observed=prefix_observed,
                observation_valid=prefix_valid, **flags)
            source_available = {source: bool(available[source] and np.isfinite(np.asarray(values)[:stop]).all())
                                for source, values in clearance.items()}
            row = {**identity, "layout": family, **contact, "rotation_deg": rotation, **truth,
                "source_available": source_available,
                "clearance_error_support": bool(prefix_observed.all() and prefix_valid.all()),
                "boundary_regime_known": bool(prefix_observed.all()),
                "dense_argmin_step": int(np.flatnonzero(prefix_observed)[np.argmin(dense[prefix_observed])])}
            for source, values in clearance.items():
                # Dense observed minimum stays available as partial truth, not a complete-source decision.
                if source != "dense":
                    row[f"cmin_{source}"] = float(np.asarray(values)[:stop].min()) if source_available[source] else None
            horizon_rows.append(row)
    return horizon_rows, pose_rows


def cumulative_decision_table(rows, margin, sources):
    """Source-specific denominators and four-link attribution on each available prefix."""
    truth, censored = row_outcomes(rows)
    out = {"m": float(margin), "n_rows": len(rows), "n_unsafe_observed": int(truth.sum()),
        "n_censored": int(censored.sum()), "n_domain_exits": sum(r["observation_domain_exit"] for r in rows),
        "n_exact_zero_dense_clearance": sum(r["cmin_dense"] == 0 for r in rows), "sources": {}}
    accepted = {}
    for source in sources:
        mask = np.array([r["source_available"].get(source, False) for r in rows], bool)
        values = np.array([r[f"cmin_{source}"] for r, ok in zip(rows, mask) if ok], float)
        out["sources"][source] = {**fsa(values, truth[mask], margin, censored=censored[mask]),
            "n_available": int(mask.sum()), "n_unavailable": int((~mask).sum())}
        accepted[source] = np.array([bool(ok and r[f"cmin_{source}"] >= margin) for r, ok in zip(rows, mask)], bool)
    if "imagined" in accepted:
        false_safe = accepted["imagined"] & truth
        domain = np.array([r["observation_domain_exit"] for r in rows], bool)
        chain = np.array([all(r["source_available"].get(source, False)
                             for source in ("dense", "endpoint", "real_readout", "imagined")) for r in rows], bool)
        eligible = false_safe & ~domain & ~censored & chain
        attribution = {"n_false_safe_imagined": int(false_safe.sum()),
            "domain_exit": int((false_safe & domain).sum()),
            "censored": int((false_safe & ~domain & censored).sum()),
            "unavailable_chain": int((false_safe & ~domain & ~censored & ~chain).sum()),
            "n_attributed": int(eligible.sum()),
            "n_unresolved_accepted": int((accepted["imagined"] & censored & ~truth).sum()),
            "exact_zero_dense_clearance_false_safe": sum(bool(fs and row["cmin_dense"] == 0) for fs, row in zip(false_safe, rows))}
        if "endpoint" in accepted and "real_readout" in accepted:
            attribution.update({"temporal": int((eligible & accepted["endpoint"]).sum()),
                "readout": int((eligible & ~accepted["endpoint"] & accepted["real_readout"]).sum()),
                "imagination": int((eligible & ~accepted["endpoint"] & ~accepted["real_readout"]).sum())})
        out["attribution"] = attribution
    return out


def _numeric_summary(values, *, signed=False):
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0, "mean": None, "p50": None, "p95": None, "mae": None, "rmse": None,
                **({"frac_optimistic": None} if signed else {})}
    return {"n": int(len(values)), "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)), "mae": float(np.abs(values).mean()),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        **({"frac_optimistic": float((values > 0).mean())} if signed else {})}


def _diagnostic_identity(row):
    return row["bank"], row["branch"], row["horizon_blocks"]


def _diagnostic_group(rows, pose_rows, margins, sources):
    # Defensive deduplication keeps physical pose counts independent of virtual relabeling.
    unique_poses = {_diagnostic_identity(row): row for row in pose_rows}
    physical = list(unique_poses.values())
    pose_error, clearance_error = {}, {}
    for source in sources:
        usable_pose = [r["errors"][source] for r in physical if r["errors"].get(source, {}).get("available", False)]
        pose_error[source] = {"n_branches": len(usable_pose), "n_unavailable": len(physical) - len(usable_pose),
            "centre_px": _numeric_summary([p["centre_px"] for p in usable_pose]),
            "angle_deg": _numeric_summary([p["angle_deg"] for p in usable_pose])}
        usable_rows = [r for r in rows if r["clearance_error_support"] and r["source_available"].get(source, False)]
        errors = [r[f"cmin_{source}"] - r["cmin_dense"] for r in usable_rows]
        clearance_error[source] = {**_numeric_summary(errors, signed=True), "n_rows_excluded": len(rows) - len(usable_rows),
            "n_branches": len({_diagnostic_identity(r) for r in usable_rows})}
    return {"n_rows": len(rows), "n_branches": len(physical), "n_roots": len({r["root"] for r in physical}),
        "n_source_episodes": len({r["source_episode"] for r in physical if r["source_episode"] is not None}),
        "n_target_observed": sum(r["target_observed"] for r in physical),
        "n_target_image_valid": sum(r["target_image_valid"] for r in physical),
        "pose_error": pose_error, "clearance_error": clearance_error,
        "decisions": {label: cumulative_decision_table(rows, margin, sources) for label, margin in margins.items()}}


def cumulative_horizon_report(rows, pose_rows, *, matched_margin=0.0):
    """Counts, four-source attribution and numeric errors by horizon and physical regime."""
    sources = [source for source in SOURCES if any(source in row["source_available"] for row in rows)]
    margins = {"at_m0": 0.0, "at_matched": float(matched_margin)}
    groups = {"contact": lambda r: r["contact"] is True, "free": lambda r: r["contact"] is False,
        "contact_unknown": lambda r: r["contact"] is None,
        "rotation>=10deg": lambda r: r["rotation_deg"] is not None and r["rotation_deg"] >= 10,
        "rotation<10deg": lambda r: r["rotation_deg"] is not None and r["rotation_deg"] < 10,
        "rotation_unknown": lambda r: r["rotation_deg"] is None,
        "near_boundary(|c|<20)": lambda r: r["boundary_regime_known"] and abs(r["cmin_dense"]) < 20,
        "far(|c|>=20)": lambda r: r["boundary_regime_known"] and abs(r["cmin_dense"]) >= 20,
        "boundary_unknown": lambda r: not r["boundary_regime_known"]}
    result = {"protocol": HORIZON_REPORT_PROTOCOL, "sources": sources, "by_horizon": {}}
    for horizon in sorted({r["horizon_blocks"] for r in rows}):
        selected = [r for r in rows if r["horizon_blocks"] == horizon]
        poses = [r for r in pose_rows if r["horizon_blocks"] == horizon]
        entry = _diagnostic_group(selected, poses, margins, sources)
        entry.update({"horizon_blocks": horizon, "horizon_steps": horizon * ACTION_BLOCK, "by_regime": {}})
        for label, predicate in groups.items():
            subset = [r for r in selected if predicate(r)]
            if subset:
                identities = {_diagnostic_identity(r) for r in subset}
                entry["by_regime"][label] = _diagnostic_group(subset, [r for r in poses if _diagnostic_identity(r) in identities], margins, sources)
        result["by_horizon"][str(horizon)] = entry
    return result
