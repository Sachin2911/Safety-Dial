"""Restore public pinned PushT data and freeze one metadata-selected smoke-test pair."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]


class SplitReader(io.RawIOBase):
    """Read one logical HDF5 file split between temporary and persistent storage."""

    def __init__(self, paths):
        self.files = [p.open("rb") for p in paths]
        self.sizes = [p.stat().st_size for p in paths]
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (
            self.position if whence == 1 else sum(self.sizes) if whence == 2 else 0
        )
        return self.position

    def readinto(self, buffer):
        view = memoryview(buffer)
        remaining, written, offset = len(view), 0, self.position
        for stream, size in zip(self.files, self.sizes):
            if offset >= size:
                offset -= size
                continue
            stream.seek(offset)
            take = min(remaining, size - offset)
            count = stream.readinto(view[written : written + take])
            written += count
            remaining -= count
            self.position += count
            offset = 0
            if remaining == 0:
                break
        return written

    def close(self):
        for stream in self.files:
            stream.close()
        super().close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import zstandard

    args.data.mkdir(parents=True, exist_ok=True)
    pins = json.loads((ROOT / "configs/lewm18/pusht.json").read_text())["pins"]
    started = time.time()
    archive = Path(
        hf_hub_download(
            pins["dataset_repo"],
            pins["dataset_file"],
            repo_type="dataset",
            revision=pins["dataset_revision"],
            token=False,
            local_dir=args.data / "download",
        )
    )
    print("Downloaded pinned public dataset", flush=True)
    sha = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    if archive.stat().st_size != pins["dataset_bytes"] or sha.hexdigest() != pins["dataset_sha256"]:
        raise RuntimeError("Public PushT dataset size/hash mismatch")
    args.scratch.mkdir(parents=True, exist_ok=True)
    parts = [args.scratch / "pusht.h5.0", args.data / "pusht.h5.1"]
    split_bytes = 25 * 1024**3
    expected_bytes = 46300921856
    if sum(p.stat().st_size for p in parts if p.exists()) != expected_bytes:
        with archive.open("rb") as source:
            with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                copied = 0
                for index, path in enumerate(parts):
                    with path.open("wb") as dest:
                        while index == 1 or copied < split_bytes:
                            take = (
                                8 * 1024**2
                                if index == 1
                                else min(8 * 1024**2, split_bytes - copied)
                            )
                            block = reader.read(take)
                            if not block:
                                break
                            dest.write(block)
                            copied += len(block)
                            if copied % (1024**3) == 0:
                                print(f"Decompressed {copied // 1024**3} GiB", flush=True)
        if sum(p.stat().st_size for p in parts) != expected_bytes:
            raise RuntimeError("Incomplete split scratch file")
    print("Dataset verified and decompressed", flush=True)
    with SplitReader(parts) as source, h5py.File(source, "r") as dataset:
        scalers = {"columns": {}, "source": pins}
        for key in ("action", "proprio", "state"):
            values = np.asarray(dataset[key], dtype=np.float64)
            values = values[np.isfinite(values).all(axis=1)]
            scale = values.std(axis=0)
            scale[scale == 0] = 1
            scalers["columns"][key] = {
                "mean": values.mean(axis=0).tolist(),
                "scale": scale.tolist(),
                "rows": len(values),
            }
        (args.data / "scalers.json").write_text(json.dumps(scalers, indent=2) + "\n")
        states = np.asarray(dataset["state"], dtype=float)
        episodes = np.asarray(dataset["episode_idx"]).reshape(-1)
        rows = np.arange(0, len(states) - 25, 25)
        a, b = states[rows], states[rows + 25]
        valid = episodes[rows] == episodes[rows + 25]
        valid &= np.linalg.norm(a[:, :2] - b[:, :2], axis=1) >= 100
        valid &= np.linalg.norm(a[:, 2:4] - b[:, 2:4], axis=1) >= 30
        valid &= ((a[:, :2] >= 70) & (a[:, :2] <= 442)).all(axis=1)
        valid &= ((b[:, :2] >= 70) & (b[:, :2] <= 442)).all(axis=1)
        if not valid.any():
            raise RuntimeError("No metadata-valid smoke pair")
        row = int(rows[np.flatnonzero(valid)[0]])
        pair = {
            "start_row": row,
            "goal_row": row + 25,
            "episode": int(episodes[row]),
            "initial_state": states[row].tolist(),
            "goal_state": states[row + 25].tolist(),
        }
        (args.data / "pair.json").write_text(json.dumps(pair, indent=2) + "\n")
        np.savez_compressed(
            args.data / "pair.npz",
            initial_state=states[row],
            goal_state=states[row + 25],
            start_image=np.asarray(dataset["pixels"][row]),
            goal_image=np.asarray(dataset["pixels"][row + 25]),
        )
    receipt = {
        "source": pins,
        "pair": pair,
        "preparation_seconds": time.time() - started,
        "simulator_resets": 0,
        "simulator_actions": 0,
    }
    (args.data / "prepared.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for part in parts:
        part.unlink()
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
