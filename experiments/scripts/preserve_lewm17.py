"""Create immutable private snapshots without a workstation dependency or credentials."""

import fcntl
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm17_remote import OUT, hf_secrets, rental, utc, write_json

REPO = "Sachioster/maniguard-jar-replication-20261022"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sources():
    found = set()
    for name in ("experiments", "configs", "outputs/lewm17"):
        for path in (ROOT / name).rglob("*"):
            rel = path.relative_to(ROOT)
            if any(p in {"__pycache__", "preservation", "hf-cache", "cache", ".git"} for p in rel.parts):
                continue
            if path.is_file() and not path.is_symlink() and not path.name.endswith(".part"):
                found.add(path)
    for path in (ROOT / "archive").rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".json", ".yaml", ".yml"}:
            found.add(path)
    for name in (
        "AGENTS.md", "firstPrinciples.md", "progress/README.md", "progress/receipts/README.md",
        "progress/17-lewm-evolution-campaign.md", "progress/goal-prompt-17-lewm-evolution.md",
        "progress/17-remote-codex.md", "progress/rental-lewm17-20261026.json",
        "progress/theOdessy.tex", "progress/theOdessy.pdf",
        "docs/research/research_notes/LeWM evolution task 17.md",
    ):
        if (ROOT / name).is_file():
            found.add(ROOT / name)
    found.update(p for p in (ROOT / "progress/receipts").glob("17-*") if p.is_file())
    return sorted(found)


def main():
    from huggingface_hub import HfApi, hf_hub_download

    record = rental()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "control/preservation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        secret = hf_secrets()
        api = HfApi(token=secret["HF_TOKEN"])
        if not api.repo_info(REPO, repo_type="dataset").private:
            raise ValueError("Destination is not private")
        snapshot = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
        folder = OUT / "preservation" / snapshot
        folder.mkdir(parents=True)
        sensitive = [secret["HF_TOKEN"].encode()]
        if os.environ.get("CONTAINER_API_KEY"):
            sensitive.append(os.environ["CONTAINER_API_KEY"].encode())
        manifest = {}
        with tarfile.open(folder / "evidence.tar.gz", "w:gz") as tar:
            import io

            for path in sources():
                # Copy the exact captured bytes, so concurrent log growth cannot invalidate hashes.
                blob = path.read_bytes()
                if any(s in blob for s in sensitive):
                    raise ValueError("Credential found in snapshot input; path/output suppressed")
                name = str(path.relative_to(ROOT))
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(blob), path.stat().st_mode & 0o777
                tar.addfile(info, io.BytesIO(blob))
                manifest[name] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        write_json(folder / "files.json", manifest)
        index = dict(
            schema="lewm17-preservation-v1", utc=utc(), instance_id=record["instance_id"],
            snapshot=snapshot, files={p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                                      for p in folder.iterdir()},
            capture="Per-file immutable bytes; active attempts remain explicitly incomplete.",
        )
        write_json(folder / "index.json", index)
        prefix = f"17_lewm17_campaign/{record['instance_id']}/{snapshot}"
        commit = api.upload_folder(
            folder_path=folder, path_in_repo=prefix, repo_id=REPO, repo_type="dataset",
            commit_message=f"task17: preserve {snapshot}",
        )
        for path in sorted(folder.iterdir()):
            copied = hf_hub_download(
                REPO, prefix + "/" + path.name, repo_type="dataset", revision=commit.oid,
                token=secret["HF_TOKEN"], cache_dir=OUT / "hf-cache",
            )
            if sha(copied) != sha(path):
                raise ValueError("Private snapshot checksum mismatch")
        receipt = dict(
            utc=utc(), verified=True, private=True, repo=REPO, prefix=prefix,
            revision=commit.oid, files=len(manifest), index_sha256=sha(folder / "index.json"),
            instance_id=record["instance_id"],
        )
        write_json(OUT / "control/preservation-latest.json", receipt)
        write_json(ROOT / f"progress/receipts/17-preservation-{snapshot}.json", receipt)
        print(json.dumps(receipt))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
