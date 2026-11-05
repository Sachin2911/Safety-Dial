"""Download and verify the pinned public assets on the rented host."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

from common import TASKS, assets, configuration, configure, publish, sha, write
from run_io import report_progress

CHUNK = 8 * 1024 * 1024


class Reader:
    def __init__(self, response, total):
        self.response, self.total = response, total
        self.count, self.last = 0, 0
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        data = self.response.read(size)
        self.count += len(data)
        self.hash.update(data)
        if time.monotonic() - self.last > 30:
            self.last = time.monotonic()
            report_progress(min(self.count, self.total), self.total, unit="compressed bytes")
            print(json.dumps({"downloaded_bytes": self.count, "total_bytes": self.total}),
                  flush=True)
        return data


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=180) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=CHUNK)
        if expected and sha(partial) != expected:
            raise ValueError("Downloaded checksum differs: " + path.name)
        partial.replace(path)
    actual = sha(path)
    if expected and actual != expected:
        raise ValueError("Existing checksum differs: " + path.name)
    return {"sha256": actual, "bytes": path.stat().st_size}


def main(task):
    config = configuration()
    spec = config["task_specs"][task]
    target = assets(config, task)
    root = Path(config["asset_root"])
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ap_root, _ = configure(config, task, require_overlay=False)
    # zstandard is an archive utility installed separately on the rented host.
    import zstandard

    base = f"https://huggingface.co/{spec['model_repo']}/resolve/{spec['model_revision']}"
    models = {"weights": download(base + "/weights.pt", target["checkpoint"],
                                   spec["weights_sha256"]),
              "config": download(base + "/config.json", target["checkpoint"].with_name(
                  "config.json"), spec.get("config_sha256"))}
    destination = target["dataset"]
    receipt_path = destination.with_suffix(".receipt.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["archive_sha256"] != spec["archive_sha256"]:
            raise ValueError("Cached dataset source differs")
        if sha(destination) != receipt["sha256"]:
            raise ValueError("Cached uncompressed dataset checksum differs")
    else:
        url = (f"https://huggingface.co/datasets/{spec['dataset_repo']}/resolve/"
               f"{spec['dataset_revision']}/{spec['archive']}")
        partial = destination.with_suffix(".partial")
        h5_hash, h5_bytes = hashlib.sha256(), 0
        member_name = None

        def copy(source, output):
            nonlocal h5_bytes
            while chunk := source.read(CHUNK):
                if shutil.disk_usage(root).free < len(chunk) + 5 * 10**9:
                    raise ValueError("Dataset extraction reached the reserved free-space margin")
                output.write(chunk)
                h5_hash.update(chunk)
                h5_bytes += len(chunk)

        with urllib.request.urlopen(url, timeout=180) as response:
            reader = Reader(response, spec["archive_bytes"])
            with zstandard.ZstdDecompressor().stream_reader(reader, closefd=False) as stream:
                with partial.open("wb") as output:
                    if spec["archive"].endswith(".h5.zst"):
                        copy(stream, output)
                    else:
                        found = False
                        with tarfile.open(fileobj=stream, mode="r|") as archive:
                            for member in archive:
                                if not member.isfile() or not member.name.endswith(".h5"):
                                    continue
                                if spec.get("member") and Path(member.name).name != spec["member"]:
                                    continue
                                if found:
                                    raise ValueError("Ambiguous HDF5 archive")
                                if shutil.disk_usage(root).free < member.size + 5 * 10**9:
                                    raise ValueError("Insufficient space for the original dataset")
                                member_name = member.name
                                with archive.extractfile(member) as source:
                                    copy(source, output)
                                found = True
                            if not found:
                                raise ValueError("Original HDF5 dataset absent from archive")
                # Validate the complete compressed source, including archive padding.
                while stream.read(CHUNK):
                    pass
            while reader.read(CHUNK):
                pass
        if reader.count != spec["archive_bytes"] or reader.hash.hexdigest() != spec[
                "archive_sha256"]:
            raise ValueError("Original compressed archive verification failed")
        partial.replace(destination)
        receipt = {"task": task, "dataset_revision": spec["dataset_revision"],
                   "archive_sha256": reader.hash.hexdigest(), "archive_bytes": reader.count,
                   "member": member_name, "bytes": h5_bytes, "sha256": h5_hash.hexdigest(),
                   "path": str(destination)}
        write(receipt_path, receipt)
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(destination, "r") as handle:
        expected = {"tworoom": 920809, "pusht": 2336736, "reacher": 2010000,
                    "cube": 2010000}[task]
        if handle["pixels"].shape != (expected, 224, 224, 3):
            raise ValueError("Original AP row layout differs")
        shapes = {key: {"shape": list(handle[key].shape), "dtype": str(handle[key].dtype)}
                  for key in ("pixels", "action", "ep_len", "ep_offset", target["state_key"])}
        publish("artifacts/dataset-index.json", {"task": task,
                "ep_len": handle["ep_len"][:].tolist(),
                "ep_offset": handle["ep_offset"][:].tolist()})
    publish("artifacts/preparation.json", {"task": task, "specification": spec,
            "models": models, "dataset": receipt, "datasets": shapes,
            "ap_root": str(ap_root), "seconds": time.time() - started,
            "simulator_actions": 0, "model_queries": 0})
    print(json.dumps({"prepared": task, "seconds": time.time() - started}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    main(parser.parse_args().task)
