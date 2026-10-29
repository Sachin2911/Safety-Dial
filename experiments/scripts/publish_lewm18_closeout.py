"""Append verified final rental/document accounting after the primary evidence upload."""

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from package_lewm18_evidence import ROOT, read, write


def encoded(obj):
    return (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode()


def item(blob):
    return {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def main():
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    closeout = ROOT / "progress/receipts/18-closeout.json"
    record = read(closeout)
    assert record["rentals_destroyed_absence_verified"]
    assert record["combined_cost_upper_usd"] < 5
    assert record["papers_rendered_and_reviewed"]
    assert len(record["verified_primary_dataset_uploads"]) == 2
    secrets = dotenv_values(ROOT / ".env")
    sensitive = [v.encode() for k, v in secrets.items() if v and ("TOKEN" in k or "KEY" in k)]
    api = HfApi(token=secrets["HF_TOKEN"])
    out = ROOT / "outputs/lewm18-publication"
    receipts = []
    for title in ("PushT", "Cube"):
        repo = f"Sachioster/SafetyDial-{title}"
        head = api.repo_info(repo, repo_type="dataset")
        assert head.private
        existing = set(api.list_repo_files(repo, repo_type="dataset", revision=head.sha))
        path = "campaigns/lewm18/protocol/final-closeout.json"

        def download(name, revision):
            return Path(
                hf_hub_download(
                    repo,
                    name,
                    repo_type="dataset",
                    revision=revision,
                    token=secrets["HF_TOKEN"],
                    cache_dir=out / "verification-cache",
                )
            ).read_bytes()

        if path in existing:
            assert download(path, head.sha) == closeout.read_bytes()
            blobs = {
                name: download(name, head.sha)
                for name in (path, "campaigns/lewm18/sha256-manifest.json", "sha256-manifest.json")
            }
            revision = head.sha
        else:
            blobs = {path: closeout.read_bytes()}
            campaign = json.loads(download("campaigns/lewm18/sha256-manifest.json", head.sha))
            campaign["protocol/final-closeout.json"] = item(blobs[path])
            blobs["campaigns/lewm18/sha256-manifest.json"] = encoded(campaign)
            manifest = json.loads(download("sha256-manifest.json", head.sha))
            for name, blob in blobs.items():
                manifest[name] = item(blob)
            blobs["sha256-manifest.json"] = encoded(manifest)
            assert not any(value in blob for value in sensitive for blob in blobs.values())
            commit = api.create_commit(
                repo_id=repo,
                repo_type="dataset",
                parent_commit=head.sha,
                commit_message="Record final task-18 preservation, paper checks and rental cleanup",
                operations=[
                    CommitOperationAdd(path_in_repo=name, path_or_fileobj=blob)
                    for name, blob in blobs.items()
                ],
            )
            revision = commit.oid
        for name, blob in blobs.items():
            assert download(name, revision) == blob
        campaign = json.loads(blobs["campaigns/lewm18/sha256-manifest.json"])
        manifest = json.loads(blobs["sha256-manifest.json"])
        assert campaign["protocol/final-closeout.json"] == item(blobs[path])
        assert manifest[path] == item(blobs[path])
        assert manifest["campaigns/lewm18/sha256-manifest.json"] == item(
            blobs["campaigns/lewm18/sha256-manifest.json"]
        )
        after = set(api.list_repo_files(repo, repo_type="dataset", revision=revision))
        assert after == existing | {path}
        receipt = {
            "repo": repo,
            "private": True,
            "parent_revision": head.sha,
            "revision": revision,
            "verified_files": {name: item(blob) for name, blob in blobs.items()},
            "deleted_files": 0,
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        receipts.append(receipt)
        write(ROOT / "progress/receipts/18-final-publication.json", {"uploads": receipts})
        print(json.dumps({"repo": repo, "revision": revision, "verified": True}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__, "details_withheld": True}), flush=True)
        raise SystemExit(1) from None
