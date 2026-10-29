"""Remote preparation: pinned downloads, checkpoint conversion, scalers, probe, imagination.

Public Hub repositories only, so no credential is needed or sent to the rental. Every
artifact is verified by size and sha256 against configs/safecem10/design.json before use.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
from helpers.dsrl06_infra import atomic_json, utc
from helpers.safecem10_protocol import design, digest

DATA = ROOT / "data/safecem10"
OUT = ROOT / "outputs/safecem10"
os.environ.setdefault("STABLEWM_HOME", str(DATA / "stablewm"))


def remap_vit_keys(sd):
    """Map pre-transformers-v5 ViT keys to the current vit_hf layout."""
    out = {}
    for key, value in sd.items():
        nk = key.replace("encoder.encoder.layer.", "encoder.layers.")
        nk = nk.replace("attention.attention.query", "attention.q_proj")
        nk = nk.replace("attention.attention.key", "attention.k_proj")
        nk = nk.replace("attention.attention.value", "attention.v_proj")
        nk = nk.replace("attention.output.dense", "attention.o_proj")
        nk = nk.replace("intermediate.dense", "mlp.fc1")
        nk = nk.replace(".output.dense", ".mlp.fc2")
        out[nk] = value
    return out


def verify(path, expected_sha, expected_bytes):
    path = Path(path)
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Size mismatch for {path.name}")
    if digest(path) != expected_sha:
        raise ValueError(f"Hash mismatch for {path.name}")


def fetch_model(spec):
    from huggingface_hub import hf_hub_download

    pins = spec["pins"]
    folder = DATA / "hf_pusht"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "weights.pt"):
        if not (folder / name).exists():
            hf_hub_download(
                pins["model_repo"],
                name,
                repo_type="model",
                revision=pins["model_revision"],
                local_dir=str(folder),
            )
    verify(folder / "weights.pt", pins["model_weights_sha256"], pins["model_weights_bytes"])
    return folder


def convert(folder, spec):
    import torch
    from hydra.utils import instantiate

    target = Path(os.environ["STABLEWM_HOME"]) / "checkpoints/pusht/lewm_object.ckpt"
    if target.exists():
        return target
    cfg = json.loads((folder / "config.json").read_text())
    model = instantiate(cfg)
    sd = torch.load(folder / "weights.pt", map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(remap_vit_keys(sd), strict=True)
        remapped = True
    except RuntimeError:
        model.load_state_dict(sd, strict=True)
        remapped = False
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, target)
    atomic_json(
        target.with_suffix(".json"),
        {"utc": utc(), "remapped_keys": remapped, "source_sha256": digest(folder / "weights.pt")},
    )
    return target


def fetch_dataset(spec):
    from huggingface_hub import hf_hub_download

    pins = spec["pins"]
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / pins["dataset_file"]
    if not archive.exists():
        hf_hub_download(
            pins["dataset_repo"],
            pins["dataset_file"],
            repo_type="dataset",
            revision=pins["dataset_revision"],
            local_dir=str(raw),
        )
    marker = raw / "verified.json"
    if not marker.exists():
        verify(archive, pins["dataset_sha256"], pins["dataset_bytes"])
        atomic_json(marker, {"utc": utc(), "sha256": pins["dataset_sha256"]})
    processed = DATA / "processed/pusht_expert_train.h5"
    if not processed.exists():
        import zstandard

        processed.parent.mkdir(parents=True, exist_ok=True)
        staging = processed.with_name(processed.name + ".part")
        if staging.exists():
            staging.unlink()
        try:
            with archive.open("rb") as fin, staging.open("wb") as fout:
                zstandard.ZstdDecompressor().copy_stream(fin, fout)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
        staging.rename(processed)
    return processed


def build_scalers(h5_path):
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np

    path = DATA / "scalers.json"
    if path.exists():
        return path
    columns = {}
    with h5py.File(h5_path, "r") as f:
        for col in ("action", "proprio", "state"):
            data = np.asarray(f[col][:], dtype=np.float64)
            data = data[~np.isnan(data).any(axis=1)]
            scale = data.std(axis=0)
            scale[scale == 0] = 1.0
            columns[col] = {
                "mean": data.mean(axis=0).tolist(),
                "scale": scale.tolist(),
                "rows": int(len(data)),
            }
        meta = {
            "steps": int(len(f["state"])),
            "episodes": int(len(f["ep_len"])),
            "pixels_shape": list(f["pixels"].shape[1:]),
        }
    atomic_json(path, {"utc": utc(), "columns": columns, "dataset": meta})
    return path


def encode_frames(model, frames, device, batch_size=256):
    import numpy as np
    import torch

    from helpers.safecem10_planner import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = np.empty((len(frames), 192), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(frames), batch_size):
            chunk = torch.from_numpy(frames[i : i + batch_size]).to(device)
            x = chunk.permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - mean) / std
            z = model.encode({"pixels": x.unsqueeze(0)})["emb"][0]
            out[i : i + batch_size] = z.float().cpu().numpy()
    return out


def train_probe(model, h5_path, spec, device):
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import torch

    from helpers.safecem10_planner import MLPProbe

    path = DATA / "probe.pt"
    if path.exists():
        return path
    cfg = spec["probe"]
    n_chunks = cfg["n_chunks"]
    chunk = cfg["n_samples"] // n_chunks
    Z = np.empty((chunk * n_chunks, 192), dtype=np.float32)
    Y = np.empty((chunk * n_chunks, 2), dtype=np.float32)
    started = time.time()
    with h5py.File(h5_path, "r") as f:
        total = len(f["pixels"])
        starts = np.linspace(0, total - chunk, n_chunks, dtype=int)
        for i, s in enumerate(starts):
            frames = f["pixels"][s : s + chunk]
            Z[i * chunk : (i + 1) * chunk] = encode_frames(model, frames, device)
            Y[i * chunk : (i + 1) * chunk] = f["state"][s : s + chunk, 0:2]
    encode_seconds = time.time() - started
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(len(Z))
    n_test = len(Z) // 5
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    Z_tr = torch.from_numpy(Z[train_idx]).to(device)
    Y_tr = torch.from_numpy(Y[train_idx]).to(device)
    Z_te = torch.from_numpy(Z[test_idx]).to(device)
    torch.manual_seed(cfg["seed"])
    probe = MLPProbe(in_dim=192, hidden=tuple(cfg["hidden"])).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Z_tr)
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(n, device=device)
        for i in range(0, n, 1024):
            b = idx[i : i + 1024]
            loss = torch.nn.functional.mse_loss(probe(Z_tr[b]), Y_tr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    probe.eval()
    with torch.inference_mode():
        pred = probe(Z_te).cpu().numpy()
    resid = Y[test_idx] - pred
    radial = np.linalg.norm(resid, axis=1)
    stats = {
        "r2": float(1 - (resid**2).sum() / ((Y[test_idx] - Y[test_idx].mean(0)) ** 2).sum()),
        "rmse": float(np.sqrt((resid**2).mean())),
        "radial_median": float(np.median(radial)),
        "radial_p90": float(np.percentile(radial, 90)),
        "radial_p95": float(np.percentile(radial, 95)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "encode_seconds": encode_seconds,
        "frames_encoded": int(len(Z)),
    }
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in probe.state_dict().items()},
            "in_dim": 192,
            "hidden": list(cfg["hidden"]),
            "stats": stats,
            "utc": utc(),
        },
        path,
    )
    return path


def imagination(model, h5_path, spec, device):
    """Radial probe error at each imagined depth against the true future pusher position."""
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import torch

    from helpers.safecem10_runtime import load_probe, load_scalers

    path = OUT / "imagination.json"
    if path.exists():
        return json.loads(path.read_text())
    cfg = spec["imagination"]
    block = spec["planner"]["action_block"]
    depth = cfg["depth_blocks"]
    span = block * depth
    probe, _ = load_probe(device)
    process, _ = load_scalers()
    action = process["action"]
    rng = np.random.default_rng(cfg["seed"])
    from helpers.safecem10_planner import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    errors = np.full((cfg["n_snippets"], depth), np.nan)
    probe_only = np.full((cfg["n_snippets"], depth), np.nan)
    with h5py.File(h5_path, "r") as f:
        lengths = f["ep_len"][:]
        offsets = f["ep_offset"][:]
        eligible = np.flatnonzero(lengths > span + 1)
        picks = rng.choice(eligible, size=cfg["n_snippets"], replace=True)
        for i, ep in enumerate(picks):
            start = int(offsets[ep]) + int(rng.integers(0, lengths[ep] - span - 1))
            frames = f["pixels"][start : start + span + 1]
            states = f["state"][start : start + span + 1]
            acts = np.asarray(f["action"][start : start + span], dtype=np.float64)
            acts = action.transform(acts).reshape(1, 1, depth, block * 2)
            with torch.inference_mode():
                x = torch.from_numpy(frames[0:1]).to(device).permute(0, 3, 1, 2).float() / 255.0
                x = ((x - mean) / std).unsqueeze(0).unsqueeze(0)
                info = {"pixels": x}
                info = model.rollout(info, torch.as_tensor(acts, device=device, dtype=x.dtype))
                emb = info["predicted_emb"][0, 0]
                xy = probe(emb.float()).cpu().numpy()
                real = encode_frames(model, frames[block::block], device)
                real_xy = probe(torch.from_numpy(real).to(device)).cpu().numpy()
            truth = states[block::block, 0:2]
            errors[i] = np.linalg.norm(xy[1 : depth + 1] - truth, axis=1)
            probe_only[i] = np.linalg.norm(real_xy - truth, axis=1)
    quantiles = {
        str(q): float(np.percentile(errors[:, depth - 1], q)) for q in cfg["quantiles"]
    }
    result = {
        "utc": utc(),
        "n_snippets": int(cfg["n_snippets"]),
        "depth_blocks": depth,
        "radial_quantiles_px": quantiles,
        "per_depth": [
            {
                "depth": d + 1,
                "env_steps_ahead": (d + 1) * block,
                "imagined_p50": float(np.percentile(errors[:, d], 50)),
                "imagined_p90": float(np.percentile(errors[:, d], 90)),
                "imagined_p95": float(np.percentile(errors[:, d], 95)),
                "imagined_rms": float(np.sqrt((errors[:, d] ** 2).mean())),
                "probe_only_p50": float(np.percentile(probe_only[:, d], 50)),
                "probe_only_p95": float(np.percentile(probe_only[:, d], 95)),
            }
            for d in range(depth)
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(path, result)
    np.savez_compressed(OUT / "imagination-errors.npz", imagined=errors, probe_only=probe_only)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    spec = design()
    OUT.mkdir(parents=True, exist_ok=True)
    folder = fetch_model(spec)
    checkpoint = convert(folder, spec)
    if args.verify_only:
        print(json.dumps({"checkpoint": str(checkpoint), "verified": True}))
        return
    import torch

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("Preparation requires the rental GPU")
    h5_path = fetch_dataset(spec)
    scalers = build_scalers(h5_path)
    from helpers.safecem10_runtime import load_world_model

    model = load_world_model(device)
    probe_path = train_probe(model, h5_path, spec, device)
    result = imagination(model, h5_path, spec, device)
    free = shutil.disk_usage(ROOT).free
    receipt = {
        "utc": utc(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "scalers_sha256": digest(scalers),
        "probe_sha256": digest(probe_path),
        "probe_stats": torch.load(probe_path, map_location="cpu", weights_only=False)["stats"],
        "imagination": result,
        "free_disk_gb": free / 1e9,
        "gpu": torch.cuda.get_device_name(),
    }
    atomic_json(OUT / "prepare-complete.json", receipt)
    print(json.dumps({"status": "prepared", "quantiles": result["radial_quantiles_px"]}))


if __name__ == "__main__":
    main()
