#!/usr/bin/env python3
"""Stage, review and explicitly publish curated AP presentation files.

1. Build an exact local manifest after catalog, selected clips, plots and reports
   are assembled: --content-dir DIR --write-manifest MANIFEST.json.
2. Stage and perform read-only remote checks: --content-dir DIR
   --expected-manifest MANIFEST.json --stage-dir REVIEW_DIR.
3. Review REVIEW_DIR/publication-plan.json, then repeat step 2 with --publish.

Only --publish can call create_commit. The repository must already exist and be
private. Every runs/ destination is forbidden. Original MP4 copies require pinned
source-manifest proofs, extracted from the ap_media index when building a manifest.
Derived comparison MP4s require their comparison index, generator hash and both
arms' pinned original video, case and trajectory proofs.
--verify-revision can finish read-back after an ambiguous publication response;
an interrupted publication is never blindly retried.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import quote
from urllib.request import Request, build_opener

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "LeWM_Experiment1/computeConfigs/scripts"))
from compute_common import (  # noqa: E402
    ComputeError, atomic_json, load_secrets, reject_secrets, relative_path, require_secret,
    safe_path, sha256, valid_id,
)
from fetch_ap_evidence import (  # noqa: E402
    SafeRedirect, error_message, install_file, validate_manifest as validate_run_manifest,
    verified_existing, write_new,
)

REPOSITORY = "Sachioster/LeWMExp1_AP_Reproduction"
ROOT_FILES = {"README.md", "catalog-manifest.json"}
PREFIXES = {"reports", "media"}
EXTENSIONS = {".md", ".json", ".jsonl", ".gz", ".pdf", ".tex", ".mp4", ".png", ".csv", ".txt"}
CHUNK = 1024 * 1024
MAX_MANIFEST_BYTES = 8 * CHUNK


def read(path):
    return json.loads(Path(path).read_text())


def checksum(value, length=64):
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ComputeError("Expected a full pinned checksum or commit revision")
    return value


def safe_error(error, secrets):
    status = getattr(getattr(error, "response", None), "status_code", None)
    if type(status) is int:
        return "Hugging Face request failed with HTTP " + str(status)
    if type(error).__module__.startswith(("huggingface_hub", "requests", "httpx")):
        return "Hugging Face request failed: " + type(error).__name__
    return error_message(error, secrets)


def destination(name):
    relative_path(name)
    if name not in ROOT_FILES and name.split("/", 1)[0] not in PREFIXES:
        raise ComputeError("Publication destination is outside README/catalog/reports/media")
    if Path(name).suffix.lower() not in EXTENSIONS:
        raise ComputeError("Unsupported curated publication file type")
    return name


def inventory(root):
    if root.is_symlink() or not root.is_dir():
        raise ComputeError("Content must be an existing real directory")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ComputeError("Symlinks are forbidden in curated content")
        if path.is_dir():
            continue
        name = destination(path.relative_to(root).as_posix())
        path = safe_path(root, name)
        if not path.is_file():
            raise ComputeError("Only regular curated files may be published")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if not files or len(files) > 1000:
        raise ComputeError("Curated publication requires between one and 1000 files")
    return files


def copy_proofs(root, files):
    proofs = {}
    for name in files:
        if not re.fullmatch(r"media/[^/]+/index\.json", name):
            continue
        index = read(root / name)
        if index.get("repo_id") != REPOSITORY or index.get("selection_verified") is not True:
            raise ComputeError("Media index lacks verified AP selection provenance")
        for example in index["examples"]:
            for item in example.get("files", []):
                target = item["path"]
                if not target.endswith(".mp4"):
                    continue
                if target not in files or files[target] != {
                        "bytes": item["bytes"], "sha256": item["sha256"]}:
                    raise ComputeError(
                        "Selected original MP4 is missing or differs from its media index")
                proof = {"source_path": item["source_path"],
                         "source_revision": example["source_revision"],
                         "source_manifest_sha256": example["source_manifest_sha256"]}
                if target in proofs and proofs[target] != proof:
                    raise ComputeError("Conflicting pinned provenance for a selected MP4")
                proofs[target] = proof
    return proofs


def validate_comparison_sources(entry):
    task, ordinal = entry.get("environment"), entry.get("query_ordinal")
    if task not in ("tworoom", "pusht", "reacher", "cube") or type(ordinal) is not int:
        raise ComputeError("Derived comparison has an invalid query identity")
    if not 0 <= ordinal < 128 or entry.get("outcome") not in ("success", "failure"):
        raise ComputeError("Derived comparison is outside the original query cohort")
    if entry.get("comparison_checks") != {
            "same_seed": True, "same_initial_state": True, "same_goal": True}:
        raise ComputeError("Derived comparison lacks the matched-pair checks")
    if entry.get("simulator_actions") != 0 or entry.get("model_queries") != 0:
        raise ComputeError("Derived comparison must use saved evidence only")
    sources = entry.get("sources")
    if not isinstance(sources, list) or len(sources) != 6:
        raise ComputeError("Comparison needs both original videos, cases and trajectories")
    expected, source_arms = set(), {}
    for arm in ("gaussian_final", "gaussian_observed"):
        stem = f"ap-fresh-{task}-{ordinal:03d}_clean_{arm}"
        names = (f"artifacts/videos/{stem}.mp4", f"metrics/cases/{stem}.json",
                 f"metrics/trajectories/{stem}.json.gz")
        expected.update(names)
        source_arms.update({name: arm for name in names})
    observed = set()
    envelopes = {arm: set() for arm in ("gaussian_final", "gaussian_observed")}
    for source in sources:
        if set(source) != {"source_path", "source_revision", "source_manifest_sha256",
                           "bytes", "sha256"}:
            raise ComputeError("Invalid comparison input proof")
        parts = relative_path(source["source_path"]).parts
        if len(parts) < 4 or parts[0] != "runs":
            raise ComputeError("Comparison inputs must come from pinned original runs")
        valid_id(parts[1])
        relative = "/".join(parts[2:])
        if relative in observed or relative not in source_arms:
            raise ComputeError("Duplicate or unexpected comparison input")
        observed.add(relative)
        checksum(source["source_revision"], 40)
        checksum(source["source_manifest_sha256"])
        checksum(source["sha256"])
        if type(source["bytes"]) is not int or source["bytes"] <= 0:
            raise ComputeError("Invalid comparison input size")
        envelopes[source_arms[relative]].add(
            (parts[1], source["source_revision"], source["source_manifest_sha256"]))
    if observed != expected:
        raise ComputeError("Comparison inputs do not identify the same query and both planners")
    if any(len(values) != 1 for values in envelopes.values()):
        raise ComputeError("One planner's case, trajectory and video must share one original run")


def derived_proofs(root, files):
    proofs = {}
    for name in files:
        if not re.fullmatch(r"media/[^/]+/comparisons\.json", name):
            continue
        index = read(root / name)
        if index.get("repo_id") != REPOSITORY or index.get("derived_comparisons_verified") is not True:
            raise ComputeError("Comparison index lacks verified derivation provenance")
        generator = checksum(index["generator"]["source_sha256"])
        if sha256(Path(__file__).with_name("ap_ghost_videos.py")) != generator:
            raise ComputeError("Comparison generator changed after the videos were produced")
        helper = checksum(index["generator"]["coordinate_helper_sha256"])
        if sha256(Path(__file__).with_name("ap_media.py")) != helper:
            raise ComputeError("Comparison coordinate helper changed after video generation")
        for entry in index["comparisons"]:
            validate_comparison_sources(entry)
            output = entry["output"]
            target = output["path"]
            if (not target.endswith(".mp4") or target not in files
                    or files[target] != {"bytes": output["bytes"], "sha256": output["sha256"]}):
                raise ComputeError("Derived comparison differs from its exact output hash")
            if target in proofs:
                raise ComputeError("Duplicate derived comparison destination")
            poster = entry.get("poster")
            if poster is not None:
                if (not poster["path"].endswith(".png") or poster["path"] not in files
                        or files[poster["path"]] != {
                            "bytes": poster["bytes"], "sha256": poster["sha256"]}):
                    raise ComputeError("Comparison poster differs from its indexed bytes")
            proofs[target] = {"index_path": name, "index_sha256": files[name]["sha256"],
                              "generator_sha256": generator, "coordinate_helper_sha256": helper,
                              "sources": entry["sources"]}
    return proofs


def validate_expected(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("repo_id") != REPOSITORY:
        raise ComputeError("Expected manifest must identify the private AP dataset")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > 1000:
        raise ComputeError("Expected manifest file inventory is invalid")
    for name, expected in files.items():
        destination(name)
        if (not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}
                or type(expected["bytes"]) is not int or expected["bytes"] < 0):
            raise ComputeError("Expected manifest contains invalid file metadata")
        checksum(expected["sha256"])
    proofs = manifest.get("source_copies", {})
    if not isinstance(proofs, dict) or set(proofs) - set(files):
        raise ComputeError("Copy proofs must refer to selected local publication files")
    derived = manifest.get("derived_media", {})
    if not isinstance(derived, dict) or set(derived) - set(files) or set(derived) & set(proofs):
        raise ComputeError("Derived media proofs conflict with the selected original copies")
    if {name for name in files if name.endswith(".mp4")} - (set(proofs) | set(derived)):
        raise ComputeError("Every MP4 requires pinned original or derived comparison provenance")
    for name, proof in proofs.items():
        if not isinstance(proof, dict) or set(proof) != {
                "source_path", "source_revision", "source_manifest_sha256"}:
            raise ComputeError("Invalid original-source copy proof")
        parts = relative_path(proof["source_path"]).parts
        if len(parts) < 4 or parts[0] != "runs":
            raise ComputeError("Original copies must come from a pinned original run")
        valid_id(parts[1])
        if name.endswith(".mp4") and not proof["source_path"].endswith(".mp4"):
            raise ComputeError("An original MP4 copy must retain its source media type")
        checksum(proof["source_revision"], 40)
        checksum(proof["source_manifest_sha256"])
    for name, proof in derived.items():
        if not name.endswith(".mp4") or set(proof) != {
                "index_path", "index_sha256", "generator_sha256", "coordinate_helper_sha256",
                "sources"}:
            raise ComputeError("Invalid derived media proof")
        if (not re.fullmatch(r"media/[^/]+/comparisons\.json", proof["index_path"])
                or proof["index_path"] not in files
                or files[proof["index_path"]]["sha256"] != proof["index_sha256"]):
            raise ComputeError("Derived media index is absent or changed")
        checksum(proof["generator_sha256"])
        checksum(proof["coordinate_helper_sha256"])
    return files


def stage_payload(content, stage, files, secrets):
    if stage.is_symlink() or stage.resolve().is_relative_to(content.resolve()):
        raise ComputeError("Review staging must be separate from the content directory")
    stage.mkdir(parents=True, exist_ok=True)
    payload = stage / "payload"
    if payload.is_symlink():
        raise ComputeError("Staged payload cannot be a symlink")
    payload.mkdir(exist_ok=True)
    known_secrets = list(secrets.values())
    carry_size = max([128, *[len(value) for value in known_secrets]])
    for name, expected in files.items():
        source = safe_path(content, name)
        target = safe_path(payload, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Scan each reviewed source even when an identical staged copy already exists.
        with source.open("rb") as incoming:
            carry = b""
            for block in iter(lambda: incoming.read(CHUNK), b""):
                reject_secrets(carry + block, known_secrets)
                carry = block[-carry_size:]
        if verified_existing(target, expected):
            continue
        fd, temporary = tempfile.mkstemp(prefix=".stage-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as outgoing, source.open("rb") as incoming:
                shutil.copyfileobj(incoming, outgoing, length=CHUNK)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if Path(temporary).stat().st_size != expected["bytes"] or sha256(temporary) != expected[
                    "sha256"]:
                raise ComputeError("Content changed while staging")
            install_file(temporary, payload, name, expected)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if inventory(payload) != files:
        raise ComputeError(
            "Staging contains unexpected or changed files; use a clean staging directory")
    return payload


def open_remote(token, revision, name):
    relative_path(name)
    checksum(revision, 40)
    url = (f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/"
           + quote(name, safe="/"))
    return build_opener(SafeRedirect()).open(Request(url, headers={
        "Authorization": "Bearer " + token, "Cache-Control": "no-cache",
        "Accept-Encoding": "identity"}), timeout=120)


class Publisher:
    def __init__(self, token):
        from huggingface_hub import HfApi

        self.token = token
        self.api = HfApi(endpoint="https://huggingface.co", token=token)

    def revision(self, revision="main"):
        info = self.api.repo_info(REPOSITORY, repo_type="dataset", revision=revision)
        if info.private is not True or info.id != REPOSITORY:
            raise ComputeError("Publication requires the existing private AP dataset")
        head = checksum(info.sha, 40)
        if revision != "main" and head != revision:
            raise ComputeError("Pinned publication revision differs from the requested commit")
        return head

    def runs_tree(self, revision):
        rows = self.api.get_paths_info(REPOSITORY, ["runs"], repo_type="dataset", revision=revision)
        if len(rows) != 1 or rows[0].path != "runs" or not hasattr(rows[0], "tree_id"):
            raise ComputeError("Original run tree is missing or is not a directory")
        return checksum(rows[0].tree_id, 40)

    def source_proofs(self, manifest):
        checked = []
        cached = {}
        inputs = [(name, proof, manifest["files"][name], "original_copy")
                  for name, proof in sorted(manifest.get("source_copies", {}).items())]
        for name, derivation in sorted(manifest.get("derived_media", {}).items()):
            inputs.extend((name, source, {"bytes": source["bytes"], "sha256": source["sha256"]},
                           "derived_input") for source in derivation["sources"])
        for name, proof, expected, kind in inputs:
            parts = relative_path(proof["source_path"]).parts
            identity = parts[1]
            key = (identity, proof["source_revision"], proof["source_manifest_sha256"])
            if key not in cached:
                self.revision(proof["source_revision"])
                with open_remote(self.token, proof["source_revision"],
                                 f"runs/{identity}/manifest.json") as response:
                    data = response.read(MAX_MANIFEST_BYTES + 1)
                if len(data) > MAX_MANIFEST_BYTES or hashlib.sha256(data).hexdigest() != proof[
                        "source_manifest_sha256"]:
                    raise ComputeError("Fresh original-source manifest differs from its proof")
                cached[key] = validate_run_manifest(data, identity)
            source_relative = "/".join(parts[2:])
            if cached[key]["files"].get(source_relative) != expected:
                raise ComputeError("Curated media input differs from its pinned original file")
            checked.append({"destination": name, "kind": kind, **proof, **expected,
                            "verified": True})
        return checked

    def changes(self, revision, files, payload):
        remote = {}
        names = list(files)
        for offset in range(0, len(names), 100):
            rows = self.api.get_paths_info(REPOSITORY, names[offset:offset + 100],
                                           repo_type="dataset", revision=revision)
            remote.update({row.path: row for row in rows})
        changes = []
        for name, expected in files.items():
            row = remote.get(name)
            previous = None
            action = "add"
            if row:
                if not hasattr(row, "blob_id"):
                    raise ComputeError("A publication file would replace an existing directory")
                previous = {"bytes": row.size, "git_blob_id": row.blob_id}
                lfs = getattr(row, "lfs", None)
                if lfs:
                    old_sha = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
                    previous["sha256"] = old_sha
                    same = row.size == expected["bytes"] and old_sha == expected["sha256"]
                else:
                    digest = hashlib.sha1(usedforsecurity=False)
                    digest.update(f"blob {expected['bytes']}\0".encode())
                    with (payload / name).open("rb") as stream:
                        for block in iter(lambda: stream.read(CHUNK), b""):
                            digest.update(block)
                    same = row.size == expected["bytes"] and row.blob_id == digest.hexdigest()
                action = "unchanged" if same else "update"
            changes.append({"path": name, "action": action, "previous": previous, **expected})
        return changes

    def commit(self, plan, payload):
        from huggingface_hub import CommitOperationAdd

        operations = [CommitOperationAdd(path_in_repo=row["path"],
                                         path_or_fileobj=str(payload / row["path"]))
                      for row in plan["changes"] if row["action"] != "unchanged"]
        if not operations:
            return plan["previous_revision"]
        commit = self.api.create_commit(
            repo_id=REPOSITORY, repo_type="dataset", revision="main",
            parent_commit=plan["previous_revision"], operations=operations,
            commit_message="Publish AP reproduction results, original clips and matched ghost comparisons")
        return checksum(commit.oid, 40)

    def verify_files(self, revision, files, workers):
        def verify_one(name, expected):
            count, digest = 0, hashlib.sha256()
            with open_remote(self.token, revision, name) as response:
                while block := response.read(CHUNK):
                    count += len(block)
                    if count > expected["bytes"]:
                        raise ComputeError("Published download exceeds its declared size")
                    digest.update(block)
            if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                raise ComputeError("Fresh published download differs in size or SHA-256")
            return {"path": name, **expected, "verified": True}

        verified = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(verify_one, name, expected)
                       for name, expected in files.items()]
            for future in as_completed(futures):
                verified.append(future.result())
        return sorted(verified, key=lambda row: row["path"])


def write_manifest(args):
    if args.publish or args.verify_revision:
        raise ComputeError("Manifest generation cannot publish or perform publication recovery")
    content = args.content_dir.resolve()
    output = args.write_manifest.resolve()
    if output.is_relative_to(content):
        raise ComputeError("Write the expected manifest outside the content directory")
    files = inventory(content)
    manifest = {"schema_version": 1, "repo_id": REPOSITORY, "files": files,
                "source_copies": copy_proofs(content, files),
                "derived_media": derived_proofs(content, files),
                "original_runs_must_remain_unchanged": True}
    validate_expected(manifest)
    write_new(output.parent, output.name, (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({"manifest": str(output), "manifest_sha256": sha256(output),
                      "files": len(files), "bytes": sum(row["bytes"] for row in files.values()),
                      "published": False}))


def main(args, secrets):
    if args.content_dir.is_symlink() or (args.stage_dir and args.stage_dir.is_symlink()):
        raise ComputeError("Content and staging roots cannot be symlinks")
    if args.write_manifest:
        write_manifest(args)
        return
    if not args.stage_dir:
        raise ComputeError("A separate --stage-dir is required for review and publication")
    if not 1 <= args.workers <= 8 or args.max_total_bytes < 1:
        raise ComputeError("Use one to eight verification workers and a positive total byte limit")
    expected_path = args.expected_manifest.resolve()
    expected_sha = sha256(expected_path)
    manifest = read(expected_path)
    files = validate_expected(manifest)
    if sum(row["bytes"] for row in files.values()) > args.max_total_bytes:
        raise ComputeError("Curated publication exceeds its declared total byte limit")
    content, stage = args.content_dir.resolve(), args.stage_dir.resolve()
    if inventory(content) != files:
        raise ComputeError("Local content differs from the exact reviewed manifest")
    payload = stage_payload(content, stage, files, secrets)
    if derived_proofs(payload, files) != manifest.get("derived_media", {}):
        raise ComputeError("Staged comparison provenance differs from the reviewed manifest")
    token = require_secret(secrets, "HF_TOKEN")
    publisher = Publisher(token)
    current_head = publisher.revision()
    source_proofs = publisher.source_proofs(manifest)
    plan_path = stage / "publication-plan.json"
    intent_path = stage / "publication-intent.json"
    receipt_path = stage / "publication-receipt.json"
    if not args.publish and not args.verify_revision:
        if intent_path.exists():
            raise ComputeError(
                "A publication was attempted; use --verify-revision instead of resetting it")
        plan = {"schema_version": 1, "repo_id": REPOSITORY, "private": True,
                "expected_manifest_sha256": expected_sha, "previous_revision": current_head,
                "original_runs_tree_id": publisher.runs_tree(current_head),
                "source_proofs": source_proofs,
                "changes": publisher.changes(current_head, files, payload),
                "created_utc": datetime.now(timezone.utc).isoformat(), "published": False}
        atomic_json(plan_path, plan)
        print(json.dumps({"published": False, "plan": str(plan_path),
                          "plan_sha256": sha256(plan_path), "previous_revision": current_head,
                          "files": len(files), "changes": sum(row["action"] != "unchanged"
                                                               for row in plan["changes"])}))
        return
    if not plan_path.is_file() or plan_path.is_symlink():
        raise ComputeError("Run and review the dry staging step before publication")
    plan = read(plan_path)
    if (plan.get("repo_id") != REPOSITORY or plan.get("private") is not True
            or plan.get("expected_manifest_sha256") != expected_sha):
        raise ComputeError("Reviewed publication plan differs from the expected manifest")
    checksum(plan["previous_revision"], 40)
    checksum(plan["original_runs_tree_id"], 40)
    if {row["path"]: {key: row[key] for key in ("bytes", "sha256")}
            for row in plan["changes"]} != files:
        raise ComputeError("Reviewed publication operations differ from the expected manifest")
    if args.publish:
        if intent_path.exists() or receipt_path.exists():
            raise ComputeError(
                "A publication was already attempted; verify it instead of resubmitting")
        if current_head != plan["previous_revision"]:
            raise ComputeError("Repository head changed after review; refresh the dry-run plan")
        if publisher.runs_tree(current_head) != plan["original_runs_tree_id"]:
            raise ComputeError("Original run tree changed after review")
        if publisher.changes(current_head, files, payload) != plan["changes"]:
            raise ComputeError("Remote destinations changed after review")
        if inventory(payload) != files or sha256(expected_path) != expected_sha:
            raise ComputeError("Reviewed staging or manifest changed before publication")
        atomic_json(intent_path, {"repo_id": REPOSITORY, "previous_revision": current_head,
                    "expected_manifest_sha256": expected_sha, "plan_sha256": sha256(plan_path),
                    "started_utc": datetime.now(timezone.utc).isoformat()})
        # One mutation attempt only. If the response is lost, the intent remains.
        revision = publisher.commit(plan, payload)
        atomic_json(stage / "publication-commit.json", {"repo_id": REPOSITORY,
                    "revision": revision, "previous_revision": current_head})
    else:
        revision = checksum(args.verify_revision, 40)
    publisher.revision(revision)
    if publisher.runs_tree(revision) != plan["original_runs_tree_id"]:
        raise ComputeError("Original run tree differs after publication; verification refused")
    verified = publisher.verify_files(revision, files, args.workers)
    receipt = {"schema_version": 1, "repository": REPOSITORY, "private": True,
               "previous_revision": plan["previous_revision"], "revision": revision,
               "mode": "published_and_verified" if args.publish else "verified_existing_revision",
               "expected_manifest_sha256": expected_sha, "plan_sha256": sha256(plan_path),
               "verification": "Fresh independent HTTP downloads of every selected final file; "
                               "size and SHA-256",
               "verified_files": verified, "source_proofs": source_proofs,
               "original_runs_tree_id_before": plan["original_runs_tree_id"],
               "original_runs_tree_id_after": publisher.runs_tree(revision),
               "original_runs_unchanged": True, "deletion_operations": 0,
               "verified_utc": datetime.now(timezone.utc).isoformat(),
               "simulator_actions": 0, "model_queries": 0}
    atomic_json(receipt_path, receipt)
    print(json.dumps({"verified": True, "revision": revision, "files": len(verified),
                      "receipt": str(receipt_path), "original_runs_unchanged": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content-dir", type=Path, required=True)
    manifests = parser.add_mutually_exclusive_group(required=True)
    manifests.add_argument("--write-manifest", type=Path)
    manifests.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--stage-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--verify-revision")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-total-bytes", type=int, default=1_000_000_000)
    arguments = parser.parse_args()
    credentials = {}
    try:
        if not arguments.write_manifest:
            credentials = load_secrets(PROJECT)
        main(arguments, credentials)
    except Exception as error:
        print("AP publication stopped: " + safe_error(error, credentials), file=sys.stderr)
        if arguments.publish:
            print("If a commit response was lost, inspect the saved intent and use "
                  "--verify-revision; "
                  "do not resubmit the publication.", file=sys.stderr)
        sys.exit(1)
