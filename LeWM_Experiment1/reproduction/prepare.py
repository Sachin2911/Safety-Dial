"""Stage pinned public assets on the rented host, without model or simulator execution."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

from run_io import publish_artifact, report_progress

CHUNK = 8 * 1024 * 1024


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def publish(name, value):
    publish_artifact("artifacts/" + name, data=json.dumps(value, indent=2) + "\n")


class HashingReader:
    def __init__(self, response, total):
        self.response, self.total = response, total
        self.count, self.last = 0, 0
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        data = self.response.read(size)
        self.count += len(data)
        self.hash.update(data)
        if time.monotonic() - self.last > 20:
            self.last = time.monotonic()
            print(json.dumps({"downloaded_bytes": self.count, "total_bytes": self.total}),
                  flush=True)
            report_progress(self.count, self.total, unit="compressed bytes")
        return data


def fetch_small(url, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as out:
            shutil.copyfileobj(source, out, length=CHUNK)
        if expected and digest(partial) != expected:
            raise ValueError("Downloaded file checksum mismatch: " + target.name)
        partial.rename(target)
    if expected and digest(target) != expected:
        raise ValueError("Cached file checksum mismatch: " + target.name)
    return {"bytes": target.stat().st_size, "sha256": digest(target)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["tworoom", "reacher", "pusht", "cube"])
    args = parser.parse_args()
    cfg = json.loads(Path(os.environ["LEWM_CONFIG"]).read_text())
    root = Path(cfg["asset_root"])
    if not str(root).startswith("/workspace/LeWM_Experiment1/"):
        raise ValueError("Preparation must run on the rented workspace")
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # Only an archive decompressor is added; model/runtime packages are unchanged.
    try:
        import zstandard
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "zstandard==0.25.0"],
                       check=True)
        import zstandard

    source_root = root / "official-source"
    names = ["README.md", "jepa.py", "module.py", "eval.py", "utils.py"]
    names += ["config/eval/" + name + ".yaml" for name in cfg["tasks"]]
    names += ["config/eval/solver/cem.yaml"]
    source_hashes = {}
    for name in names:
        url = (f"https://raw.githubusercontent.com/{cfg['source_repo']}/"
               f"{cfg['source_revision']}/{name}")
        source_hashes[name] = fetch_small(url, source_root / name)
        publish_artifact("artifacts/official-source/" + name, source=source_root / name)
    publish("source-provenance.json", {"revision": cfg["source_revision"],
                                       "files": source_hashes})

    model_records = {}
    for task, spec in cfg["tasks"].items():
        base = f"https://huggingface.co/{spec['model_repo']}/resolve/{spec['model_revision']}"
        directory = root / "models" / task
        record = {"revision": spec["model_revision"]}
        record["config"] = fetch_small(base + "/config.json", directory / "config.json")
        record["weights"] = fetch_small(base + "/weights.pt", directory / "weights.pt",
                                         spec["weights_sha256"])
        model_records[task] = record
    publish("model-provenance.json", model_records)

    spec = cfg["tasks"][args.task]
    target = root / "datasets" / (spec["dataset_name"] + ".h5")
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.with_suffix(".receipt.json")
    if target.exists():
        receipt = json.loads(receipt_path.read_text())
        if digest(target) != receipt["sha256"]:
            raise ValueError("Existing dataset checksum mismatch")
    else:
        # Stream and discard compressed bytes, keeping only the verified HDF5.
        url = (f"https://huggingface.co/datasets/{spec['dataset_repo']}/resolve/"
               f"{spec['dataset_revision']}/{spec['archive']}")
        h5_hash, h5_bytes = hashlib.sha256(), 0
        member_name = None
        partial = target.with_suffix(".part")

        def copy_member(source, output):
            nonlocal h5_bytes
            while data := source.read(CHUNK):
                output.write(data)
                h5_hash.update(data)
                h5_bytes += len(data)

        with urllib.request.urlopen(url, timeout=180) as response:
            reader = HashingReader(response, spec["archive_bytes"])
            with zstandard.ZstdDecompressor().stream_reader(reader, closefd=False) as stream:
                with partial.open("wb") as output:
                    if spec["archive"].endswith(".h5.zst"):
                        copy_member(stream, output)
                    else:
                        found = False
                        with tarfile.open(fileobj=stream, mode="r|") as archive:
                            for member in archive:
                                if member.isfile() and member.name.endswith(".h5"):
                                    if spec["member"] and Path(member.name).name != spec["member"]:
                                        continue
                                    if found:
                                        raise ValueError("Ambiguous dataset archive")
                                    member_name = member.name
                                    print(json.dumps({"member": member.name,
                                                      "uncompressed_bytes": member.size}),
                                          flush=True)
                                    if shutil.disk_usage(root).free < member.size + 5 * 10**9:
                                        raise ValueError("Insufficient free disk for this dataset")
                                    with archive.extractfile(member) as source:
                                        copy_member(source, output)
                                    found = True
                            if not found:
                                raise ValueError("Dataset member absent")
                while stream.read(CHUNK):
                    pass
            while reader.read(CHUNK):
                pass
        if reader.count != spec["archive_bytes"]:
            raise ValueError("Compressed byte count mismatch")
        if reader.hash.hexdigest() != spec["archive_sha256"]:
            raise ValueError("Compressed checksum mismatch")
        partial.rename(target)
        receipt = {"task": args.task, "dataset_revision": spec["dataset_revision"],
                   "archive_sha256": reader.hash.hexdigest(), "archive_bytes": reader.count,
                   "member": member_name, "bytes": h5_bytes, "sha256": h5_hash.hexdigest(),
                   "path": str(target), "seconds": time.time() - started}
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    publish("dataset-provenance.json", receipt)
    packages = sorted((p.metadata["Name"], p.version) for p in importlib.metadata.distributions())
    publish("runtime-packages.json", packages)
    publish("preparation.json", {"task": args.task, "elapsed_seconds": time.time() - started,
                                 "simulator_actions": 0, "model_queries": 0,
                                 "free_disk_bytes": shutil.disk_usage(root).free})
    print(json.dumps({"prepared": args.task, "seconds": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
