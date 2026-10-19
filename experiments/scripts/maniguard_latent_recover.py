"""Read back immutable private stage archives and reconstruct evidence in new staging."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def unpack(archive, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("A study evidence archive contains an unsafe member")
        source.extractall(destination, members=members, filter="data")
    return len(members)


def recover(receipts, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    api = HfApi()
    merged = output / "merged"
    merged.mkdir()
    verified = []
    for index, receipt_path in enumerate(receipts):
        receipt_path = Path(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        repo, kind, revision = receipt["repo_id"], receipt["repo_type"], receipt["revision"]
        info = api.repo_info(repo, repo_type=kind, revision=revision)
        if info.private is not True or info.sha != revision or receipt["private"] is not True:
            raise ValueError("Private immutable artifact provenance could not be verified")
        prefix = receipt["prefix"]
        manifest_path = hf_hub_download(repo, prefix + "manifest.json", repo_type=kind,
                                        revision=revision)
        if digest(manifest_path) != receipt["manifest_sha256"]:
            raise ValueError("Remote stage manifest checksum differs from its receipt")
        manifest = json.loads(Path(manifest_path).read_text())
        expected = [{key: row[key] for key in ["path", "bytes", "sha256"]}
                    for row in receipt["files"]]
        if (manifest["files"] != expected or manifest["source_commit"] != receipt["source_commit"]
                or manifest["repo_id"] != repo or manifest["repo_type"] != kind):
            raise ValueError("Remote stage manifest and preservation receipt disagree")
        for row in expected:
            if not row["path"].endswith(".tar.gz"):
                continue
            archive = hf_hub_download(repo, row["path"], repo_type=kind, revision=revision)
            if Path(archive).stat().st_size != row["bytes"] or digest(archive) != row["sha256"]:
                raise ValueError("Redownloaded private archive checksum mismatch")
            stage = output / "stages" / f"{index:02d}-{kind}"
            count = unpack(archive, stage)
            for source in stage.rglob("*"):
                if not source.is_file():
                    continue
                relative = source.relative_to(stage)
                destination = merged / relative
                # Mutable run state is layered in the explicit supplied stage
                # order. Preserve each original independently under stages/.
                if (destination.exists() and digest(source) != digest(destination)
                        and relative.parts[0] not in ["run", "logs", "server", "configs", "search",
                                                      "analysis"]):
                    raise ValueError("Immutable physical evidence differs across stages: "
                                     + str(relative))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            verified.append({"repo_id": repo, "repo_type": kind, "revision": revision,
                             "path": row["path"], "sha256": row["sha256"], "members": count,
                             "preservation_receipt_sha256": digest(receipt_path)})
    result = {"schema": "maniguard-latent-recovery-v1", "archives": verified,
              "merged": str(merged.resolve()), "all_archives_redownloaded_and_sha256_checked": True,
              "scope": "Evidence staging only. No jobs, simulators or policy services were started.",
              "resume_rule": "Reconcile copied queue handles as historical, never as live processes. "
              "Restore the canonical run layout from the job output paths and preservation index; "
              "review invalid or incomplete attempts before any separately authorized physical resume."}
    (output / "recovery-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True,
                        help="Preservation receipts in original stage order, model and dataset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover(args.receipts, args.output)


if __name__ == "__main__":
    main()
