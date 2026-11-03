"""Acquire the pinned public PushT and Cube data on the authorised pod, without simulation.

Streams each archive once: the archive bytes are hashed as they arrive, decompressed
with zstandard, and the HDF5 bytes are hashed as they are written. Nothing is run
against a simulator or model here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

import zstandard

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/manip-20261101"
CHUNK = 8 * 1024 * 1024

TASKS = {
    "pusht": dict(
        dataset_repo="quentinll/lewm-pusht",
        dataset_revision="655cd446b9929369d7d406001da85c15d1457850",
        archive="pusht_expert_train.h5.zst",
        archive_bytes=13136247974,
        archive_sha256="7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318",
        member=None,
        h5="pusht_expert_train.h5",
        h5_bytes=46300921856,
        model_repo="quentinll/lewm-pusht",
        model_revision="22b330c28c27ead4bfd1888615af1340e3fe9052",
        weights_bytes=72290721,
        weights_sha256="48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607",
    ),
    "cube": dict(
        dataset_repo="quentinll/lewm-cube",
        dataset_revision="02a19a67a0dc8c9d6215f89c19e0a597691e152a",
        archive="cube_single_expert.tar.zst",
        archive_bytes=46184624478,
        archive_sha256="3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600",
        member="cube_single_expert.h5",
        h5="cube_single_expert.h5",
        h5_bytes=101942558720,
        model_repo="quentinll/lewm-cube",
        model_revision="b0747c5002e86d2ce8f3cd8178004b97524c587d",
        weights_bytes=72291425,
        weights_sha256="2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89",
    ),
}


class HashingReader:
    """Wrap an HTTP response so every byte read is hashed and counted."""

    def __init__(self, response, progress):
        self.response, self.progress = response, progress
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        data = self.response.read(size)
        self.digest.update(data)
        self.count += len(data)
        self.progress(self.count)
        return data


class HashingWriter:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.count = 0

    def write(self, data):
        self.handle.write(data)
        self.digest.update(data)
        self.count += len(data)
        return len(data)


def fetch_weights(task, spec, receipt):
    folder = ROOT / "assets" / task
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "weights.pt"
    url = f"https://huggingface.co/{spec['model_repo']}/resolve/{spec['model_revision']}/weights.pt"
    if not target.exists():
        with urllib.request.urlopen(url, timeout=120) as response, target.with_suffix(".part").open(
            "wb"
        ) as handle:
            shutil.copyfileobj(response, handle, length=CHUNK)
        target.with_suffix(".part").rename(target)
    assert target.stat().st_size == spec["weights_bytes"], target.stat().st_size
    with target.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    assert digest == spec["weights_sha256"], digest
    receipt["weights"] = dict(path=str(target.relative_to(ROOT)), sha256=digest,
                              bytes=spec["weights_bytes"], revision=spec["model_revision"])


def fetch_dataset(task, spec, receipt):
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    target = data / spec["h5"]
    progress_path = OUT / f"fetch-{task}-progress.json"
    if target.exists():
        assert target.stat().st_size == spec["h5_bytes"], target.stat().st_size
        receipt["dataset"] = dict(path=str(target.relative_to(ROOT)), bytes=spec["h5_bytes"],
                                  revision=spec["dataset_revision"], status="already present")
        return
    url = (f"https://huggingface.co/datasets/{spec['dataset_repo']}/resolve/"
           f"{spec['dataset_revision']}/{spec['archive']}")
    started = time.time()
    last = [0.0]

    def progress(count):
        now = time.time()
        if now - last[0] >= 30:
            last[0] = now
            progress_path.write_text(json.dumps(dict(
                archive_bytes=count, archive_total=spec["archive_bytes"],
                seconds=now - started)) + "\n")

    partial = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response:
        reader = HashingReader(response, progress)
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(reader) as stream, partial.open("wb") as handle:
            writer = HashingWriter(handle)
            if spec["member"] is None:
                shutil.copyfileobj(stream, writer, length=CHUNK)
            else:
                with tarfile.open(fileobj=stream, mode="r|") as tar:
                    found = False
                    for member in tar:
                        if Path(member.name).name == spec["member"]:
                            assert member.isfile() and member.size == spec["h5_bytes"], member.size
                            with tar.extractfile(member) as source:
                                shutil.copyfileobj(source, writer, length=CHUNK)
                            found = True
                            break
                    assert found, "member not found in archive"
        # Drain the remaining archive bytes so the archive hash covers the whole file.
        while reader.read(CHUNK):
            pass
    assert reader.count == spec["archive_bytes"], reader.count
    assert reader.digest.hexdigest() == spec["archive_sha256"], reader.digest.hexdigest()
    assert writer.count == spec["h5_bytes"], writer.count
    partial.rename(target)
    receipt["dataset"] = dict(
        path=str(target.relative_to(ROOT)), bytes=writer.count, sha256=writer.digest.hexdigest(),
        archive=spec["archive"], archive_bytes=reader.count, archive_sha256=reader.digest.hexdigest(),
        revision=spec["dataset_revision"], elapsed_seconds=time.time() - started,
        archive_retained=False,
    )


def main():
    assert os.environ.get("RUNPOD_POD_ID") == "yvrg8ow7i9n9u6"
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(TASKS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = TASKS[args.task]
    receipt = dict(task=args.task, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    fetch_weights(args.task, spec, receipt)
    fetch_dataset(args.task, spec, receipt)
    (OUT / f"data-receipt-{args.task}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
