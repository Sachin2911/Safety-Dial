"""Walker2d LeWM: data config, model construction and the upstream loss, for our own training.

The model classes are the installed `stable_worldmodel` LeWM modules (the same the released
Push-T checkpoint uses, see the assets `config.json`); only the action encoder input width
changes (frameskip * 6 = 60). The loss mirrors `third_party/le-wm/train.py::lejepa_forward`
at the pinned commit: latent prediction MSE over a 3-frame context plus SIGReg (weight 0.1,
1,024 projections). `SIGReg` is copied from `third_party/le-wm/module.py` (provenance in the
docstring) so training does not depend on that clone being importable.

Data: `helpers.locoData.MujocoStateDataset` (imported, never edited): state-only HDF5 with
lazy EGL rendering at 224 px, frameskip 10, num_steps 4 (history 3 + 1 prediction).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_CONFIG = REPO_ROOT / "runs" / "pusht-assets-20260926-1" / "config.json"

FRAMESKIP = 10
HISTORY = 3
NUM_STEPS = HISTORY + 1
ACTION_DIM = 6
IMG = 224


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single GPU). Copied from le-wm/module.py
    at commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac (26 September 2026)."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def model_config(frameskip: int = FRAMESKIP, history: int = HISTORY) -> dict:
    cfg = json.loads(ASSETS_CONFIG.read_text())
    cfg["action_encoder"]["input_dim"] = frameskip * ACTION_DIM
    cfg["predictor"]["num_frames"] = history
    return cfg


def build_model(cfg: dict | None = None):
    from hydra.utils import instantiate

    return instantiate(cfg or model_config())


def make_dataset(h5_path: Path, *, frameskip=FRAMESKIP, num_steps=NUM_STEPS, clip_stride=1):
    from helpers.locoData import MujocoStateDataset

    return MujocoStateDataset(h5_path, frameskip=frameskip, num_steps=num_steps, keys_to_load=["pixels", "action"], clip_stride=clip_stride)


def action_scaler(dataset) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(dataset.get_col_data("action"), np.float64)
    a = a[~np.isnan(a).any(1)]
    return a.mean(0), a.std(0) + 1e-8


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def preprocess_batch(batch: dict, mean: np.ndarray, std: np.ndarray, device: str) -> dict:
    """uint8 pixels (B, T, H, W, C) -> normalised float (B, T, C, H, W); actions z-scored."""
    px = batch["pixels"].to(device, non_blocking=True)
    if px.dtype == torch.uint8:
        px = px.permute(0, 1, 4, 2, 3).float().div_(255.0)
    px = (px - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    a = batch["action"].to(device).float()  # (B, T, frameskip*6)
    B, T, D = a.shape
    a = ((a.reshape(B, T, -1, ACTION_DIM) - torch.as_tensor(mean, device=device, dtype=torch.float32)) / torch.as_tensor(std, device=device, dtype=torch.float32)).reshape(B, T, D)
    a = torch.nan_to_num(a, 0.0)
    return {"pixels": px, "action": a}


def lewm_loss(model, sigreg, batch: dict, *, history=HISTORY, sigreg_weight=0.1) -> dict:
    """Mirror of `lejepa_forward`: teacher-forced next-latent MSE + SIGReg on all latents."""
    out = model.encode(batch)
    emb, act_emb = out["emb"], out["act_emb"]
    pred = model.predict(emb[:, :history], act_emb[:, :history])
    tgt = emb[:, 1 : history + 1]
    pred_loss = (pred - tgt).pow(2).mean()
    sig = sigreg(emb.transpose(0, 1).float())
    return {"loss": pred_loss + sigreg_weight * sig, "pred_loss": pred_loss.detach(), "sigreg_loss": sig.detach()}


def save_checkpoint(model, run_dir: Path, *, cfg: dict, scaler: tuple, step: int, extra: dict | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run_dir / "weights.pt")
    torch.save(model, run_dir / "lewm_object.ckpt")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    np.savez(run_dir / "scalers.npz", action_mean=scaler[0], action_std=scaler[1])
    (run_dir / "train_state.json").write_text(json.dumps({"step": step, **(extra or {})}) + "\n")


def load_walker_model(run_dir: Path, device="cuda"):
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "weights.pt", map_location="cpu"))
    z = np.load(run_dir / "scalers.npz")
    return model.to(device).eval(), (z["action_mean"], z["action_std"])
