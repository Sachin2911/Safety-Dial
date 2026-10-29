"""Publish small readable deliverables only to the already authorized private dataset."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm17_protocol import OUT, digest, write_json
from helpers.lewm17_remote import hf_secrets, rental, utc


def main():
    from huggingface_hub import HfApi, hf_hub_download

    record = rental()
    token = hf_secrets()["HF_TOKEN"]
    repo = "Sachioster/maniguard-jar-replication-20261022"
    api = HfApi(token=token)
    assert api.repo_info(repo, repo_type="dataset").private
    folder = OUT / "readable-deliverables"
    folder.mkdir(exist_ok=False)
    relative_paths = [
        "progress/receipts/17-lewm-evolution-results.md",
        "progress/receipts/17-final-statistics.json",
        "progress/receipts/17-final-audit.json",
        "progress/receipts/17-completion-audit.json",
        "progress/receipts/17-completion-status.json",
        "progress/receipts/17-optimizer-comparison.json",
        "progress/receipts/17-final-evaluation.json",
        "progress/receipts/17-final-planner-seed-amendment.json",
        "progress/receipts/17-synthetic-test-work.json",
        "progress/receipts/17-reporting-revision.json",
        "progress/17-lewm-evolution-campaign.md",
        "progress/theOdessy.pdf",
        "outputs/lewm17/analysis/safe-completion.pdf",
        "outputs/lewm17/analysis/primary-contrasts.pdf",
        "outputs/lewm17/analysis/training-resources.pdf",
    ]
    for relative in relative_paths:
        data = (ROOT / relative).read_bytes()
        assert token.encode() not in data
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    prefix = f"17_lewm17_campaign/{record['instance_id']}/final-reports"
    commit = api.upload_folder(
        folder_path=folder,
        path_in_repo=prefix,
        repo_id=repo,
        repo_type="dataset",
        commit_message="task17: readable final report and Odyssey",
    )
    checksums = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(folder))
        remote = hf_hub_download(
            repo,
            prefix + "/" + relative,
            repo_type="dataset",
            revision=commit.oid,
            token=token,
            cache_dir=OUT / "hf-cache",
        )
        checksums[relative] = digest(path)
        assert digest(remote) == checksums[relative]
    receipt = {
        "utc": utc(),
        "verified": True,
        "private": True,
        "repo": repo,
        "prefix": prefix,
        "revision": commit.oid,
        "sha256": checksums,
    }
    write_json(ROOT / "progress/receipts/17-readable-report-preservation.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
