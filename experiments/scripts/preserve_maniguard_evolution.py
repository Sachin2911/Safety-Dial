"""Stage, upload and verify evolution artifacts in the existing private dataset.

This is file preservation and orchestration. It performs no experiment computation.
Live snapshots include only immutable source records and completed episode folders.
Final snapshots require the whole declared training and validation schedule to finish.
"""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

REPO = "Sachioster/maniguard-jar-replication-20261022"
BASE_PREFIX = "21_recurrent_adapter_development/20261029T204801Z"
SOURCE_FOLDERS = {"experiments", "configs", "docs", "scripts", "progress", "review-v3"}
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".cache", ".git"}
STATIC_RUN_FILES = {"config.json", "source-manifest.json", "host.json"}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def eligible(path):
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    if path.name.endswith((".tmp", ".pyc", ".partial")):
        return False
    if (path.name.startswith(".env") or path.name in
            {"id_rsa", "id_ed25519", "vast_api_key", "lease.json"}):
        raise ValueError("Unexpected credential-like file in artifact tree")
    return True


def stage(mirror, release, *, final=False, terminal=False, run_name="run-v2"):
    mirror, release = Path(mirror).resolve(), Path(release).resolve()
    if release == mirror or mirror in release.parents:
        raise ValueError("Release must be outside the live mirror")
    state = json.loads((mirror / run_name / "status.json").read_text())
    config = json.loads((mirror / run_name / "config.json").read_text())
    search, validation = config["search"], config["validation"]
    planned = search["generations"] * (search["population"] + 1) * len(search["tasks"])
    planned += 2 * len(validation["tasks"]) * len(validation["policy_seeds"])
    if "selection" in config:
        panel = config["selection"]
        planned += 3 * len(panel["tasks"]) * len(panel["policy_seeds"])
    complete = (state["status"] == "completed" and state["generation"] == search["generations"]
                and len(state["jobs"]) == planned
                and all(row.get("valid") for row in state["jobs"].values()))
    if final and not complete:
        raise ValueError("Final release requires every declared training and validation episode")
    if terminal and state["status"] in ("ready", "running"):
        raise ValueError("Terminal preservation requires an honestly terminal run")
    archival_final = final or terminal
    base_prefix = config.get("preservation", {}).get("base_prefix", BASE_PREFIX)
    if not base_prefix.startswith(("21_recurrent_adapter_development/",
                                   "22_conservative_adapter_development/")):
        raise ValueError("Unexpected experiment evidence prefix")
    release.mkdir(parents=True, exist_ok=False)
    files = {}
    for source in sorted(mirror.rglob("*")):
        if source.is_symlink():
            raise ValueError("Symlinks are not accepted in a release")
        if not source.is_file():
            continue
        relative = source.relative_to(mirror)
        if not eligible(relative):
            continue
        top = relative.parts[0]
        include = archival_final
        if top in SOURCE_FOLDERS:
            include = True
        elif top.startswith("run-"):
            run_state = json.loads((mirror / top / "status.json").read_text())
            if run_state["status"] not in ("running", "ready"):
                include = True
            elif len(relative.parts) >= 2:
                include = relative.parts[1] in STATIC_RUN_FILES | {
                    "source", "generated", "generation-snapshots"}
                if relative.parts[1] == "episodes" and len(relative.parts) >= 4:
                    include = (mirror.joinpath(*relative.parts[:3]) / "outcome.json").exists()
        elif relative.name.startswith("service-"):
            include = True
        if not include:
            continue
        before = digest(source)
        destination = release / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if before != digest(destination) or before != digest(source):
            raise ValueError("Artifact changed during release staging: " + str(relative))
        files[relative.as_posix()] = {"bytes": destination.stat().st_size, "sha256": before}
    observation = {
        "schema": "maniguard-evolution-preservation-observation-v1",
        "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_status": state["status"], "completed_generations": state["generation"],
        "completed_episodes": len(state["jobs"]), "planned_episodes": planned,
        "final": archival_final, "complete_schedule": complete,
        "active": state.get("active"), "validation": state["validation"],
        "selected_kind": state.get("selected_kind"),
    }
    atomic(release / "preservation-observation.json", observation)
    files["preservation-observation.json"] = {
        "bytes": (release / "preservation-observation.json").stat().st_size,
        "sha256": digest(release / "preservation-observation.json")}
    index = {
        "schema": "maniguard-evolution-artifact-index-v1", "repo": REPO, "private": True,
        "base_prefix": base_prefix, "created_utc": observation["observed_utc"],
        "final": archival_final, "complete_schedule": complete,
        "run": run_name, "observation": observation,
        "files": files, "file_count_excluding_index": len(files),
        "bytes_excluding_index": sum(row["bytes"] for row in files.values()),
    }
    atomic(release / "index.json", index)
    return index


def upload_and_verify(release, verification, name, receipt_path):
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    from preserve_jar_grasp_probe import token

    release, verification = Path(release).resolve(), Path(verification).resolve()
    if name != Path(name).name or name in (".", ".."):
        raise ValueError("Snapshot name must be one path component")
    index = json.loads((release / "index.json").read_text())
    expected = dict(index["files"])
    expected["index.json"] = {"bytes": (release / "index.json").stat().st_size,
                              "sha256": digest(release / "index.json")}
    for relative, value in expected.items():
        path = release / relative
        if path.stat().st_size != value["bytes"] or digest(path) != value["sha256"]:
            raise ValueError("Staged release changed before upload: " + relative)
    credential = token()
    api = HfApi(token=credential)
    info = api.repo_info(REPO, repo_type="dataset")
    if not info.private:
        raise ValueError("The evidence repository must be private")
    prefix = index["base_prefix"] + "/" + name
    disable_progress_bars()
    if Path(receipt_path).exists():
        receipt = json.loads(Path(receipt_path).read_text())
        if (receipt["repo"] != REPO or receipt["prefix"] != prefix
                or receipt["index_sha256"] != digest(release / "index.json")):
            raise ValueError("Existing preservation receipt describes a different release")
        revision = receipt["revision"]
    else:
        if any(path.startswith(prefix + "/") for path in api.list_repo_files(
                REPO, repo_type="dataset", revision=info.sha)):
            raise ValueError("Snapshot prefix already exists; refusing to overwrite evidence")
        commit = api.upload_folder(
            repo_id=REPO, repo_type="dataset", folder_path=str(release), path_in_repo=prefix,
            allow_patterns=list(expected), parent_commit=info.sha,
            commit_message="Preserve recurrent ManiGuard adapter " + name,
        )
        revision = commit.oid
        receipt = {"schema": "maniguard-evolution-preservation-v1", "repo": REPO,
                   "private": True, "prefix": prefix, "revision": revision,
                   "final": index["final"], "verification_complete": False,
                   "complete_schedule": index.get("complete_schedule", index["final"]),
                   "index_sha256": digest(release / "index.json"),
                   "expected_files": len(expected),
                   "expected_bytes": sum(row["bytes"] for row in expected.values())}
        atomic(Path(receipt_path), receipt)
        print(json.dumps({"uploaded_revision": revision, "files": len(expected)}), flush=True)
    snapshot_download(REPO, repo_type="dataset", revision=revision, token=credential,
                      allow_patterns=[prefix + "/*"], local_dir=verification, max_workers=4)
    for relative, value in expected.items():
        path = verification / prefix / relative
        if path.stat().st_size != value["bytes"] or digest(path) != value["sha256"]:
            raise ValueError("Pinned private download differs from release: " + relative)
    receipt.update(verification_complete=True, verified_files=len(expected),
                   verified_bytes=receipt["expected_bytes"],
                   verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   verification=("Every indexed file downloaded at the pinned revision "
                                 "and SHA-256 matched"))
    atomic(Path(receipt_path), receipt)
    print(json.dumps({"verified_revision": revision, "files": len(expected),
                      "bytes": receipt["verified_bytes"]}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--terminal", action="store_true",
                        help="Archive an honestly incomplete terminal run without calling it complete")
    parser.add_argument("--run", default="run-v2")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    index = stage(args.mirror, args.release, final=args.final,
                  terminal=args.terminal, run_name=args.run)
    print(json.dumps({"staged_files": len(index["files"]) + 1,
                      "staged_bytes": index["bytes_excluding_index"], "final": args.final}),
          flush=True)
    if args.upload:
        if not args.name or args.verification is None or args.receipt is None:
            parser.error("Upload requires --name, --verification and --receipt")
        upload_and_verify(args.release, args.verification, args.name, args.receipt)


if __name__ == "__main__":
    main()
