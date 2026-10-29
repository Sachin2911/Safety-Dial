"""Publish only validated new campaign paths, then read back every uploaded object."""

import concurrent.futures
import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import tarfile

from dotenv import dotenv_values
from huggingface_hub import HfApi, hf_hub_download

from consolidate_lewm_media import OUT, ROOT, sha, write_json

logging.getLogger("httpx").setLevel(logging.CRITICAL)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def main():
    secret = dotenv_values(ROOT / ".env")
    token = secret.get("HF_TOKEN")
    sensitive = [v.encode() for k, v in secret.items() if v and ("TOKEN" in k or "KEY" in k)]
    api = HfApi(token=token)
    destinations = json.loads((OUT / "destinations.json").read_text())
    published = []
    for env, title in [("tworoom", "TwoRoom"), ("reacher", "Reacher")]:
        dest = OUT / "stage" / env
        info = destinations[title]
        repo = info["repo"]
        current = api.repo_info(repo, repo_type="dataset")
        if not current.private:
            raise ValueError("Expected existing private dataset")
        if current.sha != info["revision"]:
            raise ValueError("Destination changed; reconcile before upload")
        files = [p for p in sorted(dest.rglob("*")) if p.is_file()]
        manifest = json.loads((dest / "sha256-manifest.json").read_text())
        existing = set(info["files"])
        overlap = set(str(p.relative_to(dest)) for p in files) & existing
        if overlap != {"README.md", "sha256-manifest.json"}:
            raise ValueError("Unexpected existing-file overwrite")
        for p in files:
            rel = str(p.relative_to(dest))
            if p.is_symlink() or any(x in p.parts for x in [".env", ".codex", ".aws", ".ssh"]):
                raise ValueError("Disallowed upload path")
            blob = p.read_bytes()
            if any(v in blob for v in sensitive):
                raise ValueError("Credential in input; details withheld")
            if p != dest / "sha256-manifest.json":
                assert sha(p) == manifest[rel]["sha256"]
            if p.name.endswith(".tar.gz"):
                with tarfile.open(p) as archive:
                    for member in archive:
                        if member.isfile():
                            blob = archive.extractfile(member).read()
                            if any(v in blob for v in sensitive):
                                raise ValueError("Credential in archive; details withheld")
        print(
            json.dumps(
                {
                    "repo": repo,
                    "validated_files": len(files),
                    "bytes": sum(p.stat().st_size for p in files),
                    "uploading": True,
                }
            ),
            flush=True,
        )
        commit = api.upload_folder(
            folder_path=dest,
            repo_id=repo,
            repo_type="dataset",
            parent_commit=info["revision"],
            commit_message="Add tasks 16-17 data, replay videos, figures and consolidated paper framing",
        )
        revision = commit.oid
        write_json(
            OUT / f"{env}-upload-ack.json",
            {"repo": repo, "revision": revision, "files": len(files)},
        )

        def verify(p):
            relative = str(p.relative_to(dest))
            copied = Path(
                hf_hub_download(
                    repo,
                    relative,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                    cache_dir=str(OUT / "upload-verify"),
                )
            )
            if hashlib.sha256(copied.read_bytes()).hexdigest() != sha(p):
                raise ValueError("Upload checksum mismatch")
            return relative

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            verified = list(pool.map(verify, files))
        after = set(api.list_repo_files(repo, repo_type="dataset", revision=revision))
        assert after == existing | set(verified)
        result = {
            "repo": repo,
            "private": True,
            "parent_revision": info["revision"],
            "revision": revision,
            "verified_uploaded_files": len(verified),
            "bytes": sum(p.stat().st_size for p in files),
            "old_files_deleted": 0,
            "old_paths_updated": ["README.md", "sha256-manifest.json"],
            "verification": "Every uploaded file downloaded at pinned revision and SHA-256 checked",
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        published.append(result)
        print(json.dumps(result), flush=True)
        write_json(
            ROOT / "progress/receipts/lewm-datasets-consolidation-20261026.json",
            {
                "uploads": published,
                "packaging": json.loads((OUT / "packaging-validation.json").read_text()),
            },
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__, "details_withheld": True}), flush=True)
        raise SystemExit(1) from None
