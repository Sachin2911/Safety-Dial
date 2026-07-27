#Libraries and Imports
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
H5_PATH = REPO_ROOT / "data" / "processed" / "pusht_expert_train.h5"
SUMMARY_PATH = REPO_ROOT / "notes" / "pushTDataExp.md"

# Heuristic labels for state[:, :5] (Push-T convention). Remaining cols printed as extras.
STATE_LABELS = [
    "pusher_x",
    "pusher_y",
    "block_x",
    "block_y",
    "block_theta",
    "extra_5",
    "extra_6",
]

CHUNK = 200_000


def walk_tree(f: h5py.File) -> None:
    print("=== HDF5 tree ===")
    print(f"root keys: {list(f.keys())}")

    def _visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            nbytes = int(np.prod(obj.shape)) * obj.dtype.itemsize
            print(
                f"  DATASET  {name:20s}  shape={obj.shape}  dtype={obj.dtype}  "
                f"~{nbytes / 1e9:.2f} GB"
            )
            if obj.attrs:
                for k, v in obj.attrs.items():
                    print(f"           attr[{k}]={v}")
        elif isinstance(obj, h5py.Group):
            print(f"  GROUP    {name}/  keys={list(obj.keys())}")

    f.visititems(_visit)
    print()


def chunked_minmax(ds: h5py.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Min/max over axis 0 without loading the full dataset."""
    n, d = ds.shape
    mins = np.full(d, np.inf, dtype=np.float64)
    maxs = np.full(d, -np.inf, dtype=np.float64)
    for i in range(0, n, CHUNK):
        chunk = np.asarray(ds[i : i + CHUNK], dtype=np.float64)
        mins = np.minimum(mins, chunk.min(axis=0))
        maxs = np.maximum(maxs, chunk.max(axis=0))
    return mins, maxs


def describe_array(f: h5py.File, key: str, labels: list[str] | None = None) -> None:
    ds = f[key]
    print(f"=== {key} ===")
    print(f"shape={ds.shape}  dtype={ds.dtype}")
    if ds.ndim != 2:
        print("(skipping min/max for non-2D array)\n")
        return

    mins, maxs = chunked_minmax(ds)
    n_cols = ds.shape[1]
    for j in range(n_cols):
        name = labels[j] if labels and j < len(labels) else f"col_{j}"
        print(f"  {name:14s}  min={mins[j]:12.4f}  max={maxs[j]:12.4f}")

    print("first 3 rows:")
    print(np.asarray(ds[:3]))
    print()


def describe_episodes(f: h5py.File) -> None:
    ep_len = np.asarray(f["ep_len"][:])
    ep_offset = np.asarray(f["ep_offset"][:])
    n_steps = f["state"].shape[0]

    print("=== episodes ===")
    print(f"n_episodes     = {len(ep_len)}")
    print(f"n_steps        = {n_steps}")
    print(
        f"ep_len         min={ep_len.min()}  mean={ep_len.mean():.1f}  "
        f"max={ep_len.max()}"
    )
    print(f"ep_offset[:5]  = {ep_offset[:5].tolist()}")
    print(f"ep_len[:5]     = {ep_len[:5].tolist()}")
    print(f"last end       = {int(ep_offset[-1] + ep_len[-1])}  (should equal n_steps)")
    print()
    print("slicing: episode i is arrays[ep_offset[i] : ep_offset[i] + ep_len[i]]")
    print()


def write_summary(f: h5py.File) -> None:
    """Short markdown note for Stage 0 bookkeeping."""
    ep_len = np.asarray(f["ep_len"][:])
    state_min, state_max = chunked_minmax(f["state"])
    lines = [
        "# Push-T dataset exploration",
        "",
        f"Source file: `{H5_PATH}`",
        "",
        "## Layout",
        "",
        "- Flat timestep arrays (not one group per episode).",
        f"- `{f['state'].shape[0]}` steps, `{len(ep_len)}` episodes.",
        "- Episode i: `ep_offset[i] : ep_offset[i] + ep_len[i]`.",
        "",
        "## Keys",
        "",
        "| key | shape | dtype | notes |",
        "|-----|-------|-------|-------|",
        f"| action | {f['action'].shape} | {f['action'].dtype} | appears normalised / relative, not raw 512px targets |",
        f"| state | {f['state'].shape} | {f['state'].dtype} | see column guess below |",
        f"| proprio | {f['proprio'].shape} | {f['proprio'].dtype} | looks like pusher xy + 2 extras |",
        f"| pixels | {f['pixels'].shape} | {f['pixels'].dtype} | do not load whole array |",
        f"| ep_len | {f['ep_len'].shape} | {f['ep_len'].dtype} | |",
        f"| ep_offset | {f['ep_offset'].shape} | {f['ep_offset'].dtype} | |",
        f"| episode_idx | {f['episode_idx'].shape} | {f['episode_idx'].dtype} | per-step episode id |",
        f"| step_idx | {f['step_idx'].shape} | {f['step_idx'].dtype} | per-step index within episode |",
        "",
        "## State columns (heuristic)",
        "",
        "| col | label | min | max |",
        "|-----|-------|-----|-----|",
    ]
    for j, name in enumerate(STATE_LABELS):
        if j < len(state_min):
            lines.append(
                f"| {j} | {name} | {state_min[j]:.4f} | {state_max[j]:.4f} |"
            )
    lines += [
        "",
        "## Episode length",
        "",
        f"- min / mean / max: {ep_len.min()} / {ep_len.mean():.1f} / {ep_len.max()}",
        "",
        "## Arena note",
        "",
        "Pusher/block xy are roughly arena-scale but can leave `[0, 512]` "
        "(negatives and values > 512 appear). Confirm against `gym-pusht` live.",
        "",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"[wrote] {SUMMARY_PATH}")


def main() -> int:
    if not H5_PATH.exists():
        print(
            f"[error] missing {H5_PATH}\n"
            "  run: uv run python scripts/dataExp/download_data.py",
            file=sys.stderr,
        )
        return 1

    print(f"opening {H5_PATH}")
    with h5py.File(H5_PATH, "r") as f:
        walk_tree(f)
        describe_episodes(f)
        describe_array(f, "state", STATE_LABELS)
        describe_array(f, "proprio", ["pusher_x", "pusher_y", "extra_2", "extra_3"])
        describe_array(f, "action", ["action_0", "action_1"])
        print("=== pixels ===")
        print(f"shape={f['pixels'].shape}  dtype={f['pixels'].dtype}  (not loaded)")
        print()
        write_summary(f)

    print("[done] inspection complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
