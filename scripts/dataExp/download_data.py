#Libraries and imports

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
import zstandard as zstd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

REPO_ID = "quentinll/lewm-pusht"
FILENAME = "pusht_expert_train.h5.zst"
PROCESSED_NAME = "pusht_expert_train.h5"


# Download the data and return the path
def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / FILENAME

    if dest.exists():
        print(f"Already Downloaded")
        return dest

    print(f"Fetching download {REPO_ID}/{FILENAME} -> {RAW_DIR}")
    path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
            local_dir=RAW_DIR,
        )
    )
    print(f"Wrote download {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


# Decompressing the download
def decompress(src: Path) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dst = PROCESSED_DIR / PROCESSED_NAME

    if dst.exists():
        print(f"Decompressed version already present")
        return dst

    print(f"Decompressing from -> to: {src} -> {dst}")
    dctx = zstd.ZstdDecompressor()
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        dctx.copy_stream(fin, fout)

    print(f"Wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def main() -> int:
    try:
        zst_path = download()
        h5_path = decompress(zst_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error {exc}", file=sys.stderr)
        return 1

    print(f"Done: {h5_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
