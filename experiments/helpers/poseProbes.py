"""Trajectory-split probes from frozen latents to physical quantities.

Push-T: block pose (x, y, sin t, cos t) and pusher (x, y) from single-frame 192-d
latents. Walker2d: torso height, pitch and forward speed. Training frames come from
whole source trajectories in the probe split; validation is by trajectory, never by
random frame (AGENTS.md: the historical pusher probe's random-frame split is not a
trajectory-independent validation). Capacity (linear vs small MLP) is chosen on
development data, then frozen. A probe is supervision of a generic physical quantity,
never of the safety concept.

    spec = ProbeSpec(target="block_pose", kind="mlp")
    probe, stats = fit_probe(Z_train, Y_train, Z_val, Y_val, spec)
    pose = probe.predict_pose(z)          # (N, 3): x, y, theta   (Push-T)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

TARGETS = {
    # name: (dim, description)
    "block_pose": (4, "block x, y (px) and sin, cos of its angle"),
    "pusher": (2, "pusher x, y (px)"),
    "walker": (3, "torso height (m), pitch (rad), forward speed (m/s)"),
}


@dataclass
class ProbeSpec:
    target: str = "block_pose"
    kind: str = "mlp"  # "linear" | "mlp"
    hidden: tuple = (256, 128)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    batch_size: int = 1024
    seed: int = 0
    in_dim: int = 192

    @property
    def out_dim(self) -> int:
        return TARGETS[self.target][0]


def pose_to_target(states: np.ndarray) -> np.ndarray:
    """7-d Push-T states -> (N, 4) block target."""
    s = np.asarray(states, float)
    return np.stack([s[:, 2], s[:, 3], np.sin(s[:, 4]), np.cos(s[:, 4])], axis=1).astype(np.float32)


def pusher_to_target(states: np.ndarray) -> np.ndarray:
    return np.asarray(states, float)[:, :2].astype(np.float32)


class Probe(torch.nn.Module):
    def __init__(self, spec: ProbeSpec, y_mean: np.ndarray, y_std: np.ndarray, z_mean: np.ndarray, z_std: np.ndarray):
        super().__init__()
        self.spec = spec
        layers, d = [], spec.in_dim
        if spec.kind == "mlp":
            for h in spec.hidden:
                layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
                d = h
        layers.append(torch.nn.Linear(d, spec.out_dim))
        self.net = torch.nn.Sequential(*layers)
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))
        self.register_buffer("z_mean", torch.as_tensor(z_mean, dtype=torch.float32))
        self.register_buffer("z_std", torch.as_tensor(z_std, dtype=torch.float32))

    def forward(self, z):
        return self.net((z - self.z_mean) / self.z_std) * self.y_std + self.y_mean

    @torch.no_grad()
    def predict(self, z, batch: int = 8192) -> np.ndarray:
        z = torch.as_tensor(z, dtype=torch.float32, device=self.y_mean.device)
        out = [self(z[i : i + batch]).cpu() for i in range(0, len(z), batch)]
        return torch.cat(out).numpy()

    @torch.no_grad()
    def predict_pose(self, z) -> np.ndarray:
        """Push-T block_pose target -> (N, 3) x, y, theta in [0, 2pi)."""
        y = self.predict(z)
        th = np.arctan2(y[:, 2], y[:, 3]) % (2 * np.pi)
        return np.stack([y[:, 0], y[:, 1], th], axis=1)

    def predict_pose_t(self, z: torch.Tensor) -> torch.Tensor:
        """Differentiable/batched torch version: (..., 4) -> (..., 3)."""
        y = self(z)
        th = torch.atan2(y[..., 2], y[..., 3]) % (2 * np.pi)
        return torch.stack([y[..., 0], y[..., 1], th], dim=-1)


def _metrics(target: str, y: np.ndarray, yhat: np.ndarray) -> dict:
    resid = y - yhat
    ss_res = float((resid**2).sum(0).sum())
    ss_tot = float(((y - y.mean(0)) ** 2).sum(0).sum())
    m = {"r2": 1.0 - ss_res / max(ss_tot, 1e-12), "rmse": float(np.sqrt((resid**2).mean())), "n": int(len(y))}
    per = 1.0 - (resid**2).sum(0) / np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-12)
    m["r2_per_dim"] = per.tolist()
    if target in ("block_pose", "pusher"):
        rad = np.linalg.norm(resid[:, :2], axis=1)
        m.update({"centre_median_px": float(np.median(rad)), "centre_p95_px": float(np.percentile(rad, 95)), "centre_mean_px": float(rad.mean())})
    if target == "block_pose":
        th = np.arctan2(y[:, 2], y[:, 3])
        th_hat = np.arctan2(yhat[:, 2], yhat[:, 3])
        d = np.degrees(np.abs(np.arctan2(np.sin(th - th_hat), np.cos(th - th_hat))))
        m.update({"angle_median_deg": float(np.median(d)), "angle_p95_deg": float(np.percentile(d, 95)), "angle_mean_deg": float(d.mean())})
    if target == "walker":
        for i, name in enumerate(("height", "pitch", "speed")):
            m[f"{name}_rmse"] = float(np.sqrt((resid[:, i] ** 2).mean()))
            m[f"{name}_r2"] = float(per[i])
    return m


def fit_probe(Z_tr, Y_tr, Z_va, Y_va, spec: ProbeSpec, *, device: str = "cuda", verbose: bool = True) -> tuple[Probe, dict]:
    """Train on train trajectories, pick the best epoch on validation trajectories."""
    Z_tr, Y_tr = np.asarray(Z_tr, np.float32), np.asarray(Y_tr, np.float32)
    Z_va, Y_va = np.asarray(Z_va, np.float32), np.asarray(Y_va, np.float32)
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    probe = Probe(spec, Y_tr.mean(0), Y_tr.std(0) + 1e-6, Z_tr.mean(0), Z_tr.std(0) + 1e-6).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, spec.epochs)
    zt, yt = torch.from_numpy(Z_tr).to(device), torch.from_numpy(Y_tr).to(device)
    best, best_state, hist = np.inf, None, []
    n = len(zt)
    for ep in range(spec.epochs):
        probe.train()
        idx = torch.randperm(n, device=device)
        for i in range(0, n, spec.batch_size):
            b = idx[i : i + spec.batch_size]
            loss = torch.nn.functional.mse_loss((probe(zt[b]) - probe.y_mean) / probe.y_std, (yt[b] - probe.y_mean) / probe.y_std)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        probe.eval()
        va = _metrics(spec.target, Y_va, probe.predict(Z_va))
        hist.append({"epoch": ep + 1, "val_rmse": va["rmse"], "val_r2": va["r2"]})
        if va["rmse"] < best:
            best, best_state = va["rmse"], {k: v.detach().clone() for k, v in probe.state_dict().items()}
        if verbose and (ep + 1) % 20 == 0:
            print(f"[probe:{spec.target}/{spec.kind}] epoch {ep + 1}/{spec.epochs} val rmse {va['rmse']:.3f} r2 {va['r2']:.4f}")
    probe.load_state_dict(best_state)
    probe.eval()
    stats = {"spec": asdict(spec), "train": _metrics(spec.target, Y_tr, probe.predict(Z_tr)), "val": _metrics(spec.target, Y_va, probe.predict(Z_va)), "history": hist}
    return probe, stats


def save_probe(probe: Probe, stats: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": asdict(probe.spec), "state_dict": {k: v.cpu() for k, v in probe.state_dict().items()}, "stats": stats}, path)
    path.with_suffix(".json").write_text(json.dumps(stats, indent=1, default=float) + "\n")


def load_probe(path: Path, device: str = "cuda") -> tuple[Probe, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    spec = ProbeSpec(**blob["spec"])
    spec.hidden = tuple(spec.hidden)
    sd = blob["state_dict"]
    probe = Probe(spec, sd["y_mean"].numpy(), sd["y_std"].numpy(), sd["z_mean"].numpy(), sd["z_std"].numpy())
    probe.load_state_dict(sd)
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe.to(device).eval(), blob["stats"]


# --------------------------------------------------------------------------------------
# expert-frame collection (Push-T), by whole episodes, contiguous reads
# --------------------------------------------------------------------------------------
def collect_expert_latents(model, h5_path: Path, episodes: list[int], n_frames: int, rng, *, device="cuda", verbose=True):
    """Encode whole expert episodes (in the given list, shuffled) until n_frames.

    Returns Z (N, 192), states (N, 7), episode ids (N,). Reads are contiguous per episode.
    """
    import h5py
    import hdf5plugin  # noqa: F401

    from helpers.imagination import encode

    eps = list(rng.permutation(np.asarray(episodes)))
    Z, S, E = [], [], []
    got = 0
    with h5py.File(h5_path, "r") as f:
        ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
        for ep in eps:
            ep = int(ep)
            a, b = int(ep_off[ep]), int(ep_off[ep] + ep_len[ep])
            frames = f["pixels"][a:b]
            Z.append(encode(model, frames, device).cpu().numpy())
            S.append(f["state"][a:b].astype(np.float64))
            E.append(np.full(b - a, ep, dtype=np.int64))
            got += b - a
            if verbose and len(Z) % 50 == 0:
                print(f"[probe] encoded {got:,} frames from {len(Z)} episodes")
            if got >= n_frames:
                break
    return np.concatenate(Z), np.concatenate(S), np.concatenate(E)


def split_by_trajectory(traj_ids: np.ndarray, rng, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(traj_ids)
    val = set(rng.choice(ids, size=max(1, int(len(ids) * val_frac)), replace=False).tolist())
    is_val = np.array([t in val for t in traj_ids])
    return np.where(~is_val)[0], np.where(is_val)[0]
