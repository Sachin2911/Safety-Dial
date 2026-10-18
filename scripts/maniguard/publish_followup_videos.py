"""Publish already recorded matched videos to the explicitly approved private dataset."""

import argparse
import json
import os
import time

from backup_followup import REPO, RUN, SECRETS, digest


def publish(stages):
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    authorization = json.loads((RUN / "backup-authorization.json").read_text())
    if authorization.get("repo_id") != REPO or authorization.get("parts") != [2, 3]:
        raise ValueError("Explicit Parts 2 and 3 evidence authorization is required")
    api = HfApi()
    if api.dataset_info(REPO).private is not True:
        raise ValueError("Video destination must remain private")
    secrets = [os.environ[k].encode() for k in SECRETS if len(os.environ.get(k, "")) >= 8]
    files = []
    operations = []
    for stage in stages:
        folder = RUN / "matched-videos" / stage
        index = json.loads((folder / "index.json").read_text())
        paths = [RUN / item["path"] for item in index["videos"]] + [folder / "index.json"]
        for path in paths:
            if path.parent != folder or path.suffix not in (".mp4", ".json"):
                raise ValueError("Only indexed matched videos and their provenance are allowed")
            raw = path.read_bytes()
            if any(value in raw for value in secrets):
                raise ValueError("Credential scan failed")
            remote = "matched-videos-followup/" + stage + "/" + path.name
            files.append({"stage": stage, "path": remote, "bytes": len(raw),
                          "sha256": digest(path)})
            operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(path)))
    commit = api.create_commit(repo_id=REPO, repo_type="dataset", operations=operations,
                               commit_message="Preserve audited followup matched videos")
    revision = commit.oid
    for item in files:
        downloaded = hf_hub_download(REPO, item["path"], repo_type="dataset", revision=revision)
        if digest(downloaded) != item["sha256"]:
            raise ValueError("Remote video does not match its recorded SHA256")
        item["url"] = f"https://huggingface.co/datasets/{REPO}/blob/{revision}/{item['path']}"
    receipt = {"schema": "maniguard-followup-video-backup-v1", "at": time.time(),
               "repo_id": REPO, "private": True, "revision": revision,
               "credential_scan": "passed", "download_verified": True,
               "files": files, "new_physics": 0, "VLA_queries": 0}
    target = RUN / "evidence" / ("-".join(stages) + "-video-backup.json")
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", choices=("part2", "part3-recording", "part3-comparison"))
    publish(parser.parse_args().stages)
