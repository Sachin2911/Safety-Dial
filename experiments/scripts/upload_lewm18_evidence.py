"""Append task-18 evidence to existing private datasets and verify every uploaded byte."""

import concurrent.futures
import datetime as dt
import json
import logging
import os
from pathlib import Path
import shutil
import tarfile

from dotenv import dotenv_values
from huggingface_hub import HfApi, hf_hub_download

from package_lewm18_evidence import ROOT, read, sha, write

OUT = ROOT / "outputs/lewm18-publication"
logging.getLogger("httpx").setLevel(logging.CRITICAL)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def main():
    secrets = dotenv_values(ROOT / ".env")
    token = secrets["HF_TOKEN"]
    sensitive = [v.encode() for k, v in secrets.items() if v and ("TOKEN" in k or "KEY" in k)]
    api = HfApi(token=token)
    verified = []
    for kind, title in (("pusht", "PushT"), ("cube", "Cube")):
        repo = f"Sachioster/SafetyDial-{title}"
        stage = OUT / "stage" / kind
        package = stage / "campaigns/lewm18"
        assert package.exists() and (OUT / "packaging-validation.json").exists()
        head = api.repo_info(repo, repo_type="dataset")
        assert head.private, "Expected an existing private dataset"
        existing = set(api.list_repo_files(repo, repo_type="dataset", revision=head.sha))
        if any(p.startswith("campaigns/lewm18/") for p in existing):
            raise RuntimeError(
                "Task 18 already exists; reconcile its upload receipt before retrying"
            )

        def download(name, revision):
            return Path(
                hf_hub_download(
                    repo,
                    name,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                    cache_dir=OUT / "verification-cache",
                )
            )

        old_card = download("README.md", head.sha)
        old_manifest = download("sha256-manifest.json", head.sha)
        history = stage / "history/before-lewm18"
        history.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_card, history / "README.md")
        shutil.copy2(old_manifest, history / "sha256-manifest.json")
        card = old_card.read_text()
        note = (
            "## New: matched SafeCEM comparison and speed measurements\n\n"
            "[Task 18: results, four-method videos, pictures and exact data](campaigns/lewm18/README.md) "
            "compares native planning, original SafeCEM, tuned fixed settings and feedback at "
            "the same three safety margins. Failures are included in the speed score.\n\n"
        )
        if kind == "cube":
            note += (
                "Task 18 uses the original **cube object-moving goal**. The earlier Cube evidence "
                "below measured gripper reaching; keep those objectives separate.\n\n"
            )
        if card.startswith("---\n"):
            end = card.index("\n---", 4) + 4
            card = card[:end] + "\n\n" + note + card[end:].lstrip()
        else:
            card = note + card
        (stage / "README.md").write_text(card)
        root_manifest = read(old_manifest)
        assert isinstance(root_manifest, dict)
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path != stage / "sha256-manifest.json":
                root_manifest[str(path.relative_to(stage))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha(path),
                }
        write(stage / "sha256-manifest.json", root_manifest)
        files = [p for p in sorted(stage.rglob("*")) if p.is_file()]
        overlap = existing & {str(p.relative_to(stage)) for p in files}
        assert overlap == {"README.md", "sha256-manifest.json"}, overlap
        campaign_manifest = read(package / "sha256-manifest.json")
        for relative, item in campaign_manifest.items():
            assert sha(package / relative) == item["sha256"]
        for path in files:
            assert not path.is_symlink()
            assert not {".env", ".ssh", ".codex", ".aws"} & set(path.parts)
            if any(value in path.read_bytes() for value in sensitive):
                raise RuntimeError("Credential in upload input; details withheld")
            if path.name.endswith(".tar.gz"):
                with tarfile.open(path) as archive:
                    for member in archive:
                        if member.isfile():
                            blob = archive.extractfile(member).read()
                            if any(value in blob for value in sensitive):
                                raise RuntimeError("Credential in archive; details withheld")
        print(
            json.dumps({"repo": repo, "validated_files": len(files), "uploading": True}), flush=True
        )
        commit = api.upload_folder(
            folder_path=stage,
            repo_id=repo,
            repo_type="dataset",
            parent_commit=head.sha,
            commit_message="Add matched task-18 SafeCEM comparisons, speed metrics and saved videos",
        )
        write(OUT / f"{kind}-upload-ack.json", {"repo": repo, "revision": commit.oid})

        def check(path):
            relative = str(path.relative_to(stage))
            assert sha(download(relative, commit.oid)) == sha(path), relative
            return relative

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            checked = list(pool.map(check, files))
        after = set(api.list_repo_files(repo, repo_type="dataset", revision=commit.oid))
        assert after == existing | set(checked)
        verified.append(
            {
                "repo": repo,
                "private": True,
                "parent_revision": head.sha,
                "revision": commit.oid,
                "files_verified": len(checked),
                "bytes": sum(p.stat().st_size for p in files),
                "deleted_files": 0,
                "verification": "Every uploaded file read back at the commit and SHA-256 checked",
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        write(ROOT / "progress/receipts/18-datasets-preservation.json", {"uploads": verified})
        print(json.dumps(verified[-1]), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__, "details_withheld": True}), flush=True)
        raise SystemExit(1) from None
