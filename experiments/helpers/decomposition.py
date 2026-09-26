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
from helpers.pushtReplay import endpoint_interpolation

SOURCES = ["dense", "endpoint", "real_readout", "imagined", "stationary", "coord_mlp"]


def interp_from_endpoints(end_states: np.ndarray) -> np.ndarray:
    """(K+1, 7) endpoint states -> (K*ACTION_BLOCK+1, 7) interpolated states."""
    K = len(end_states) - 1
    full = np.zeros((K * ACTION_BLOCK + 1, 7))
    full[::ACTION_BLOCK] = end_states
    return endpoint_interpolation(full)


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


def analyse_bank(name, bank, layouts, imaginer, probe, coord=None, *, verbose=True) -> dict:
    """All sources. Returns rows (one per branch x layout) and pose errors by block."""
    lay = layouts_by_root(layouts)
    N = len(bank)
    rows = []
    srcs = ["endpoint", "real_readout", "imagined"] + (["coord_mlp"] if coord is not None else [])
    pose_err = {s: {"centre": [], "angle": []} for s in srcs}
    cache = RootLatentCache(imaginer, bank)
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
        for s, p6 in poses6.items():
            pose_err[s]["centre"].append(np.linalg.norm(p6[:, :2] - ends[:, 2:4], axis=1))
            pose_err[s]["angle"].append(ang_err_deg(p6[:, 2], ends[:, 4]))
        contact = bool((b["n_contacts"] > 0).any())
        rot = float(ang_err_deg(st[-1, 4], st[0, 4]))
        disp = float(np.linalg.norm(st[-1, 2:4] - st[0, 2:4]))
        disp_imag = float(np.linalg.norm(pose_imag[-1, :2] - pose_imag[0, :2]))
        rot_imag = float(ang_err_deg(pose_imag[-1, 2], pose_imag[0, 2]))
        for fam, hz in lay.get(root.root_id, {}).items():
            cd = clearance_trace(traces["dense"], hz)
            cmins = {s: float(clearance_trace(tr, hz).min()) for s, tr in traces.items()}
            rows.append({"bank": name, "branch": j, "root": ri, "layout": fam, "kind": b["kind"], "contact": contact, "rotation_deg": rot,
                         "displacement_px": disp, "imag_displacement_px": disp_imag, "imag_rotation_deg": rot_imag,
                         "dense_argmin_step": int(np.argmin(cd)), **{f"cmin_{s}": v for s, v in cmins.items()}})
        if verbose and (j + 1) % 200 == 0:
            print(f"[decomp:{name}] {j + 1}/{N} branches {time.time() - t0:.0f}s")
    errs = {s: {"centre_by_block": np.stack(v["centre"]).mean(0).tolist(), "centre_p95_by_block": np.percentile(np.stack(v["centre"]), 95, axis=0).tolist(),
                "angle_by_block": np.stack(v["angle"]).mean(0).tolist(), "angle_p95_by_block": np.percentile(np.stack(v["angle"]), 95, axis=0).tolist()}
            for s, v in pose_err.items() if v["centre"]}
    return {"rows": rows, "pose_err": errs}


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
            contact = bool((bank.h5["n_contacts"][int(j)] > 0).any())
            kind = bank.h5["kind"][int(j)]
            kind = kind.decode() if isinstance(kind, bytes) else str(kind)
            for fam, hz in lay[root.root_id].items():
                cd = clearance_trace(st[:, 2:5], hz)
                rows.append({"bank": name, "branch": int(j), "root": int(ri), "layout": fam, "kind": kind, "contact": contact,
                             "cmin_dense": float(cd.min()), "cmin_imagined": float(clearance_trace(tr_imag, hz).min()),
                             "displacement_px": float(np.linalg.norm(st[-1, 2:4] - st[0, 2:4])), "imag_displacement_px": float(np.linalg.norm(p6[-1, :2] - p6[0, :2])),
                             "rotation_deg": float(ang_err_deg(st[-1, 4], st[0, 4])), "imag_rotation_deg": float(ang_err_deg(p6[-1, 2], p6[0, 2])),
                             "centre_err_h5_px": float(np.linalg.norm(p6[-1, :2] - ends[-1, 2:4])), "angle_err_h5_deg": float(ang_err_deg(p6[-1, 2], ends[-1, 4]))})
    return rows


def decision_table(rows, m: float, sources=None, boot: int = 500) -> dict:
    sources = sources or [s for s in SOURCES if rows and f"cmin_{s}" in rows[0]]
    u = np.array([r["cmin_dense"] < 0 for r in rows])
    root = np.array([r["root"] for r in rows])
    out = {"m": float(m), "n": int(len(rows)), "n_unsafe": int(u.sum())}
    for s in sources:
        c = np.array([r[f"cmin_{s}"] for r in rows])
        out[s] = fsa(c, u, m)
    acc = {s: np.array([r[f"cmin_{s}"] >= m for r in rows]) for s in sources}
    if "imagined" in acc:
        fs4 = acc["imagined"] & u
        attr = {"n_false_safe_imagined": int(fs4.sum())}
        if "endpoint" in acc and "real_readout" in acc:
            attr.update({"temporal": int((fs4 & acc["endpoint"]).sum()), "readout": int((fs4 & ~acc["endpoint"] & acc["real_readout"]).sum()),
                         "imagination": int((fs4 & ~acc["endpoint"] & ~acc["real_readout"]).sum())})
        out["attribution"] = attr
        if len(rows) and boot:
            out["fsa_imagined_ci"] = cluster_bootstrap(lambda c, u: fsa(c, u, m)["fsa"], root, n_boot=boot, c=np.array([r["cmin_imagined"] for r in rows]), u=u)
    return out


def by_regime(rows, m: float, sources=None) -> dict:
    out = {}
    groups = {"contact": lambda r: r["contact"], "free": lambda r: not r["contact"],
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
    sub = [r for r in rows if r["kind"] in ("nominal", "random") and r["layout"] == "familiar"]
    if not sub:
        return {}
    dt = np.array([r["displacement_px"] for r in sub])
    di = np.array([r["imag_displacement_px"] for r in sub])
    rt = np.array([r["rotation_deg"] for r in sub])
    ri = np.array([r["imag_rotation_deg"] for r in sub])
    return {"n": len(sub), "disp_true_mean_px": float(dt.mean()), "disp_imag_mean_px": float(di.mean()), "disp_ratio": float(di.sum() / max(dt.sum(), 1e-9)),
            "disp_abs_err_mean_px": float(np.abs(di - dt).mean()), "rot_true_mean_deg": float(rt.mean()), "rot_imag_mean_deg": float(ri.mean()), "rot_ratio": float(ri.sum() / max(rt.sum(), 1e-9))}
