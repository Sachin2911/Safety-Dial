"""Remote preparation (task 13, Reacher): pinned downloads, checkpoint conversion, the dataset
archive unpacked and summarized, action scalers, and the MuJoCo render backend.

Public Hub repositories only, so no credential is needed or sent to the rental. Every
artifact is verified by size and sha256 against configs/reacher13/design.json before use.
No episode runs here. The summary records what every dataset column holds and how far it
moves over LeWM's goal offset, which the planner diagnosis and the hazard design build on.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.reacher13_protocol import design, digest

DATA = ROOT / "data/reacher13"
OUT = ROOT / "outputs/reacher13"
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
    folder = DATA / "hf_model"
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

    name = spec["pins"]["checkpoint_name"]
    target = Path(os.environ["STABLEWM_HOME"]) / f"checkpoints/{name}_object.ckpt"
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


def unpack_dataset(spec):
    """Download the pinned archive, verify it, and unpack every member under processed/."""
    import tarfile

    import zstandard
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
    target = DATA / "processed"
    done = DATA / "unpacked.json"
    if done.exists():
        return json.loads(done.read_text())
    staging = DATA / "processed.part"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(target, ignore_errors=True)
    members = []
    started = time.perf_counter()
    with archive.open("rb") as fin:
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fin)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError("Unsafe member path in the dataset archive")
                if member.isdir():
                    (staging / name).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                out = staging / name
                out.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, out.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=16 << 20)
                members.append({"name": member.name, "bytes": int(member.size)})
    if not members:
        raise ValueError("The dataset archive held no files")
    staging.rename(target)
    record = {"utc": utc(), "seconds": time.perf_counter() - started, "members": members}
    atomic_json(done, record)
    return record


def dataset_file():
    files = sorted((DATA / "processed").rglob("*.h5"))
    if len(files) != 1:
        raise ValueError(f"Expected one HDF5 file in the dataset, found {len(files)}")
    return files[0]


def summarize(h5_path, offset):
    """Every column's shape and dtype, the episode layout, and how far each small numeric
    column moves over `offset` steps within an episode (LeWM's goal distance)."""
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np

    out = {"file": str(h5_path.relative_to(DATA)), "bytes": h5_path.stat().st_size, "columns": {}}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            ds = f[key]
            out["columns"][key] = {
                "shape": list(ds.shape),
                "dtype": str(ds.dtype),
                "compression": ds.compression,
            }
        lengths = np.asarray(f["ep_len"][:])
        offsets = np.asarray(f["ep_offset"][:])
        steps = int(lengths.sum())
        out["episodes"] = int(len(lengths))
        out["steps"] = steps
        out["ep_len"] = {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "max": int(lengths.max()),
        }
        pick = np.linspace(0, len(lengths) - 1, 400).round().astype(int)
        moves, examples = {}, {}
        for key in f.keys():
            ds = f[key]
            small = ds.ndim == 2 and ds.shape[1] <= 64 and ds.shape[0] >= steps
            if not small or not np.issubdtype(ds.dtype, np.number):
                continue
            first = int(offsets[0])
            examples[key] = np.asarray(ds[first : first + 3], dtype=np.float64).round(5).tolist()
            values = []
            for i in pick:
                lo, n = int(offsets[i]), int(lengths[i])
                if n <= offset:
                    continue
                seg = np.asarray(ds[lo : lo + n], dtype=np.float64)
                seg = seg[np.isfinite(seg).all(axis=1)]
                if len(seg) > offset:
                    values.append(np.linalg.norm(seg[offset:] - seg[:-offset], axis=1))
            if values:
                v = np.concatenate(values)
                moves[key] = {
                    "median": float(np.median(v)),
                    "p10": float(np.percentile(v, 10)),
                    "p90": float(np.percentile(v, 90)),
                    "max": float(v.max()),
                }
        out["offset_steps"] = int(offset)
        out["offset_displacement"] = moves
        out["example_rows"] = examples
    return out


def build_scalers(h5_path):
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np

    path = DATA / "scalers.json"
    if path.exists():
        return path
    columns = {}
    with h5py.File(h5_path, "r") as f:
        for col in ("action", "proprio"):
            if col not in f:
                continue
            data = np.asarray(f[col][:], dtype=np.float64)
            data = data[~np.isnan(data).any(axis=1)]
            scale = data.std(axis=0)
            scale[scale == 0] = 1.0
            columns[col] = {
                "mean": data.mean(axis=0).tolist(),
                "scale": scale.tolist(),
                "rows": int(len(data)),
            }
    if "action" not in columns:
        raise ValueError("The dataset has no action column")
    atomic_json(path, {"utc": utc(), "columns": columns})
    return path


def render_backend(spec):
    """The MuJoCo GL backend that renders here (EGL on the GPU, else OSMesa) and its speed."""
    env = spec["environment"]
    code = (
        "import json, time, numpy as np, gymnasium as gym, stable_worldmodel\n"
        f"env = gym.make({env['base_env_id']!r}, **{env['env_kwargs']!r})\n"
        "env.reset(seed=0)\n"
        "img = np.asarray(env.unwrapped.render())\n"
        "t = time.perf_counter()\n"
        "for _ in range(20):\n"
        "    img = np.asarray(env.unwrapped.render())\n"
        "print(json.dumps({'shape': list(img.shape), 'std': float(img.std()),\n"
        "                  'seconds_per_frame': (time.perf_counter() - t) / 20}))\n"
    )
    attempts = {}
    for backend in ("egl", "osmesa"):
        environ = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=600,
                env=environ,
                cwd=ROOT,
            )
        except subprocess.TimeoutExpired:
            attempts[backend] = {"ok": False, "error": "timeout"}
            continue
        lines = [x for x in result.stdout.splitlines() if x.startswith("{")]
        if result.returncode == 0 and lines:
            measured = json.loads(lines[-1])
            if measured["std"] > 0:
                record = {"backend": backend, **measured, "attempts": attempts}
                atomic_json(OUT / "environment/render-backend.json", record)
                return record
        attempts[backend] = {"ok": False, "error": result.stderr[-600:]}
    raise RuntimeError(f"No MuJoCo render backend works: {json.dumps(attempts)[:1500]}")


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

    if not torch.cuda.is_available():
        raise RuntimeError("Preparation requires the rental GPU")
    backend = render_backend(spec)
    print(json.dumps({"event": "render_backend", **backend}), flush=True)
    unpacked = unpack_dataset(spec)
    print(json.dumps({"event": "unpacked", "files": len(unpacked["members"])}), flush=True)
    h5_path = dataset_file()
    summary = summarize(h5_path, spec["environment"]["goal_offset_steps"])
    atomic_json(OUT / "dataset-summary.json", summary)
    scalers = build_scalers(h5_path)
    import stable_worldmodel as swm

    model = swm.policy.AutoCostModel(spec["pins"]["checkpoint_name"]).to("cuda").eval()
    parameters = sum(p.numel() for p in model.parameters())
    receipt = {
        "utc": utc(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "model_parameters": int(parameters),
        "scalers_sha256": digest(scalers),
        "scalers": json.loads(Path(scalers).read_text()),
        "render_backend": backend,
        "dataset_members": unpacked["members"],
        "dataset_summary": summary,
        "free_disk_gb": shutil.disk_usage(ROOT).free / 1e9,
        "gpu": torch.cuda.get_device_name(),
    }
    atomic_json(OUT / "prepare-complete.json", receipt)
    print(json.dumps({"status": "prepared", "episodes": summary["episodes"]}))


if __name__ == "__main__":
    main()
