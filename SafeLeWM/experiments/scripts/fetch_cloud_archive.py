"""Fetch pinned cloud archives to their original paths without running experiments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "SafeLeWM/archive"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path outside destination: {relative}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", help="Original manifest filename in SafeLeWM/archive/")
    group.add_argument("--archive", help="One evidence archive filename, including .tar.gz")
    group.add_argument("--all", action="store_true", help="Fetch all 16 evidence archives")
    args = parser.parse_args()
    index = json.loads((ARCHIVE / "cloud-index-20261103.json").read_text())
    receipt = json.loads((ARCHIVE / "cloud-receipt-20261103.json").read_text())
    revision = receipt.get("revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Cloud receipt must pin a full commit SHA")
    if digest(ARCHIVE / "cloud-index-20261103.json") != receipt["index_sha256"]:
        raise ValueError("Cloud index differs from the verified receipt")
    rows = [r for r in index["files"] if r["kind"] == "evidence"]
    if args.manifest:
        manifest = json.loads(contained(ARCHIVE, args.manifest).read_text())
        wanted = {"SafeLeWM/archive/" + name for name in manifest["archives"]}
        rows = [r for r in rows if r["local_path"] in wanted]
        if {r["local_path"] for r in rows} != wanted:
            raise ValueError("Manifest references an archive outside this cloud batch")
        for row in rows:
            rel = str(Path(row["local_path"]).relative_to("SafeLeWM/archive"))
            if row["sha256"] != manifest["archives"][rel]:
                raise ValueError("Original manifest and cloud index disagree")
    elif args.archive:
        rows = [r for r in rows if Path(r["local_path"]).name == args.archive]
    if not rows:
        parser.error("No matching evidence archives")

    # Existing ignored credentials are used only by the authenticated Hub client.
    if not os.environ.get("HF_TOKEN") and (ROOT / ".env").exists():
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    from huggingface_hub import hf_hub_download

    for row in rows:
        target = contained(ROOT, row["local_path"])
        if target.is_symlink():
            raise ValueError(f"Refusing symlink destination: {target}")
        if target.exists():
            if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
                raise ValueError(f"Existing file differs; refusing overwrite: {target}")
            print(f"Already verified: {row['local_path']}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".cloud-fetch-", dir=target.parent) as tmp:
            downloaded = Path(
                hf_hub_download(
                    repo_id=index["repo_id"],
                    repo_type="dataset",
                    revision=revision,
                    filename=row["remote_path"],
                    local_dir=tmp,
                    force_download=True,
                )
            )
            if downloaded.stat().st_size != row["bytes"] or digest(downloaded) != row["sha256"]:
                raise ValueError(f"Downloaded archive checksum mismatch: {row['remote_path']}")
            # Publish atomically without overwriting a file created by another process.
            os.link(downloaded, target)
        print(f"Restored and verified: {row['local_path']}", flush=True)


if __name__ == "__main__":
    main()
