"""Acquire the pinned public TwoRoom data on the authorised pod, without simulation."""

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


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    assert os.environ.get("RUNPOD_POD_ID") == "yvrg8ow7i9n9u6"
    data = ROOT / "data"
    out = ROOT / "outputs/baseline-20261101"
    archive, target = data / "tworoom.tar.zst", data / "tworoom.h5"
    started = time.time()
    revision = "6903a2de048b13819d812da0b4dd661290bc01e4"
    url = f"https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/{revision}/tworoom.tar.zst"
    expected = "494b1a02f0765cd9a0d9daf1786c419ced1009977fc45d01e3158932f8d080ca"
    if not archive.exists():
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            archive.with_suffix(".part").open("wb") as f,
        ):
            shutil.copyfileobj(response, f, length=8 * 1024 * 1024)
        archive.with_suffix(".part").rename(archive)
    assert archive.stat().st_size == 3425937909
    assert digest(archive) == expected
    if not target.exists():
        with archive.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as z:
            with tarfile.open(fileobj=z, mode="r|") as tar:
                for member in tar:
                    if Path(member.name).name == "tworoom.h5":
                        assert member.isfile() and member.size == 12775849984
                        with (
                            tar.extractfile(member) as src,
                            target.with_suffix(".part").open("wb") as dst,
                        ):
                            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                        target.with_suffix(".part").rename(target)
                        break
    assert target.stat().st_size == 12775849984
    receipt = dict(
        revision=revision,
        archive_sha256=expected,
        h5_sha256=digest(target),
        h5_bytes=target.stat().st_size,
        elapsed_seconds=time.time() - started,
    )
    (out / "data-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
