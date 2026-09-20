"""Latent probes for Push-T, with an on-disk cache.

The probe maps a LeWM latent (192-d) to a physical quantity. This is supervision of a
generic physical quantity (pusher position), never of the safety concept; say so in any
reporting that uses it.

Training the probe was previously inline in a notebook, so every fresh kernel paid the
encode cost again. `load_or_train_pusher_probe` caches weights plus the normalisation
stats under `data/probes/`, which is gitignored.
"""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (required before reading the compressed HDF5)
import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class MLPProbe(torch.nn.Module):
    """z (192) -> pusher (x, y)."""

    def __init__(self, in_dim=192, hidden=(256, 128), out_dim=2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            d = h
        layers.append(torch.nn.Linear(d, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def preprocess_pixels(frames_nhwc: np.ndarray, device: str) -> torch.Tensor:
    """uint8 (N, H, W, C) -> normalised float (N, C, H, W).

    Matches le-wm's `get_img_preprocessor` (ImageNet stats, resize to 224) to within
    7e-07 for frames already at 224px, which is what the expert HDF5 stores. Doing it
    directly avoids a per-frame python transform, which is what makes a 100k-frame
    encode take seconds instead of minutes.
    """
    x = torch.from_numpy(frames_nhwc).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    return (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


@torch.inference_mode()
def encode_frames(model, frames_nhwc: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    """Encode uint8 frames to (N, 192) latents."""
    out = np.empty((len(frames_nhwc), 192), dtype=np.float32)
    for i in range(0, len(frames_nhwc), batch_size):
        chunk = frames_nhwc[i : i + batch_size]
        x = preprocess_pixels(chunk, device)
        z = model.encode({"pixels": x.unsqueeze(0)})["emb"][0]
        out[i : i + batch_size] = z.float().cpu().numpy()
    return out


def encode_spread_chunks(
    model,
    h5_path: Path,
    n_samples: int,
    n_chunks: int,
    *,
    device: str = "cuda",
    cols=(0, 2),
    batch_size: int = 256,
    verbose: bool = True,
):
    """Encode `n_samples` frames drawn as `n_chunks` contiguous spans, streaming.

    Two constraints shape this function.

    Reads must be CONTIGUOUS. On this box a contiguous read of the pixels array runs at
    about 19k frames/s while a sorted scattered fancy-index runs at about 46 frames/s, a
    415x penalty. Never fancy-index the pixels array.

    Pixels must NOT be accumulated. Holding 100k frames costs 15 GB, and building them
    with `np.concatenate` over a list of chunks needs that twice over, about 30 GB peak.
    On a shared box that thrashes: an earlier version of this burned 164 s of system time
    against 16 s of user time and never reached the encoder. Each chunk is therefore
    encoded and discarded, so peak memory is one chunk (about 150 MB) plus the 192-d
    latent bank (77 MB for 100k frames).
    """
    chunk = n_samples // n_chunks
    Z = np.empty((chunk * n_chunks, 192), dtype=np.float32)
    Y = np.empty((chunk * n_chunks, cols[1] - cols[0]), dtype=np.float32)

    t0 = time.time()
    with h5py.File(str(h5_path), "r") as f:
        total = len(f["pixels"])
        starts = np.linspace(0, total - chunk, n_chunks, dtype=int)
        for i, s in enumerate(starts):
            frames = f["pixels"][s : s + chunk]
            Z[i * chunk : (i + 1) * chunk] = encode_frames(
                model, frames, device, batch_size=batch_size
            )
            Y[i * chunk : (i + 1) * chunk] = f["state"][s : s + chunk, cols[0] : cols[1]]
            del frames
            if verbose and (i + 1) % 25 == 0:
                done = (i + 1) * chunk
                print(
                    f"[probe]   encoded {done:,}/{len(Z):,} frames "
                    f"({done / (time.time() - t0):.0f}/s)"
                )
    if verbose:
        el = time.time() - t0
        print(f"[probe] read+encoded {len(Z):,} frames in {el:.1f}s ({len(Z) / el:.0f}/s)")
    return Z, Y


def load_or_train_pusher_probe(
    model,
    h5_path: Path,
    cache_path: Path,
    *,
    device: str = "cuda",
    n_samples: int = 100_000,
    n_chunks: int = 100,
    epochs: int = 80,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    force: bool = False,
    verbose: bool = True,
):
    """Return (probe, stats). Trains and caches on first call.

    Targets are pusher (x, y) = `state[:, 0:2]`, confirmed from the simulator's
    `_get_obs`: agent.position, block.position, block.angle % 2pi, agent.velocity.
    """
    cache_path = Path(cache_path)
    if cache_path.is_file() and not force:
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        probe = MLPProbe(in_dim=blob["in_dim"])
        probe.load_state_dict(blob["state_dict"])
        probe = probe.to(device).eval()
        if verbose:
            s = blob["stats"]
            print(
                f"[probe] loaded cache {cache_path.name}: "
                f"R2={s['r2']:.4f} RMSE={s['rmse']:.2f}px (n={s['n_train']:,})"
            )
        return probe, blob["stats"]

    if verbose:
        print(f"[probe] no cache at {cache_path}, training")

    Z, coords = encode_spread_chunks(
        model, h5_path, n_samples, n_chunks, device=device, verbose=verbose
    )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Z))
    n_test = len(Z) // 5
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    Z_tr = torch.from_numpy(Z[train_idx]).float().to(device)
    Y_tr = torch.from_numpy(coords[train_idx]).float().to(device)
    Z_te = torch.from_numpy(Z[test_idx]).float().to(device)
    Y_te = coords[test_idx].astype(np.float32)

    torch.manual_seed(seed)
    probe = MLPProbe(in_dim=Z.shape[1]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    probe.train()
    n = len(Z_tr)
    for epoch in range(epochs):
        idx = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            loss = torch.nn.functional.mse_loss(probe(Z_tr[b]), Y_tr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        if verbose and (epoch + 1) % 20 == 0:
            print(f"[probe]   epoch {epoch + 1}/{epochs} train mse={total / n:.2f}")

    probe.eval()
    with torch.inference_mode():
        Y_pred = probe(Z_te).cpu().numpy()

    resid = Y_te - Y_pred
    ss_res = float((resid**2).sum())
    ss_tot = float(((Y_te - Y_te.mean(0)) ** 2).sum())
    stats = {
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt((resid**2).mean())),
        "radial_median": float(np.median(np.linalg.norm(resid, axis=1))),
        "radial_p95": float(np.percentile(np.linalg.norm(resid, axis=1), 95)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    if verbose:
        print(
            f"[probe] R2={stats['r2']:.4f}  RMSE={stats['rmse']:.2f}px  "
            f"radial p95={stats['radial_p95']:.1f}px"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": probe.state_dict(), "in_dim": Z.shape[1], "stats": stats},
        cache_path,
    )
    if verbose:
        print(f"[probe] cached -> {cache_path}")

    return probe, stats
