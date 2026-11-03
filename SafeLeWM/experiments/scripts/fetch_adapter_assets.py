"""Public pinned baseline assets. Download and extract only on the rented host."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import tarfile
import urllib.request

import zstandard

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(relative, url, expected):
    target = ROOT / relative
    if target.exists():
        assert digest(target) == expected
        return target
    temporary = target.with_suffix(".download")
    with urllib.request.urlopen(url, timeout=180) as source, temporary.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    assert digest(temporary) == expected, relative
    temporary.replace(target)
    print(f"Verified {relative}", flush=True)
    return target


def main():
    assert str(ROOT) == "/workspace/SafeLeWM"
    with ThreadPoolExecutor(max_workers=2) as pool:
        weights = pool.submit(
            fetch,
            "assets/weights.pt",
            "https://huggingface.co/quentinll/lewm-tworooms/resolve/"
            "77adaae0bc31deab21c93740d1f8bb947cd0bdec/weights.pt",
            "566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd",
        )
        compressed = pool.submit(
            fetch,
            "data/tworoom.tar.zst",
            "https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/"
            "6903a2de048b13819d812da0b4dd661290bc01e4/tworoom.tar.zst",
            "494b1a02f0765cd9a0d9daf1786c419ced1009977fc45d01e3158932f8d080ca",
        )
        weights.result()
        archive = compressed.result()
    target = ROOT / "data/tworoom.h5"
    if not target.exists():
        with archive.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as decoded:
            with tarfile.open(fileobj=decoded, mode="r|") as bundle:
                for member in bundle:
                    if Path(member.name).name == "tworoom.h5" and member.isfile():
                        with bundle.extractfile(member) as source, target.open("wb") as destination:
                            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                        break
    assert digest(target) == "129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623"
    print("Verified extracted dataset", flush=True)


if __name__ == "__main__":
    main()
