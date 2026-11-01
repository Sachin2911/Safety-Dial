"""Safely restore one verified private archive outside the active project tree."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import traceback

PRESERVER_SHA = "d8f393d9adea59e721e7ce2a63a66654a82f99bd4fc7c6db8318af424d482568"
ORIGINAL_PODS = {"wq4ar2qx2ff6r4", "vukovc3t4o1dui", "nj2fqkp2prgk8x"}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity():
    inherited = os.environ.get("RUNPOD_POD_ID")
    found = [
        value.decode().split("=", 1)[1]
        for value in Path("/proc/1/environ").read_bytes().split(b"\0")
        if value.startswith(b"RUNPOD_POD_ID=")
    ]
    require(len(found) == 1, "Actual provider pod identity is required")
    require(inherited is None or inherited == found[0], "Provider identity mismatch")
    return found[0]


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def extract(args):
    require(identity() == args.execution_pod_id, "Use only the exact declared rented host")
    require(args.owner_pod_id in ORIGINAL_PODS, "Unknown original archive owner")
    destination = args.destination.resolve()
    require(
        destination.parent in {Path("/workspace"), Path("/tmp")}
        and destination.name.startswith("origin-")
        and destination != Path("/workspace/Safety-Dial")
        and not destination.exists()
        and not args.out.exists(),
        "Use a fresh origin directory outside the canonical project and a fresh receipt",
    )
    archive = args.archive.resolve()
    require(digest(archive) == args.archive_sha256, "Transferred archive SHA differs")
    receipt = json.loads(args.receipt.read_text())
    external = json.loads(args.manifest.read_text())
    verification = json.loads(args.local_verification.read_text())
    require(
        receipt["archive_sha256"] == verification["archive_sha256"] == args.archive_sha256
        and receipt["source_builder_sha256"] == PRESERVER_SHA
        and receipt["every_member_streamed_and_hashed"] is True
        and verification["verified"] is True
        and verification["all_archive_members_verified"] is True
        and verification["pod_id"] == args.owner_pod_id
        and external["pod_id"] == args.owner_pod_id
        and external["source_builder_sha256"] == PRESERVER_SHA
        and receipt["archive_bytes"] == archive.stat().st_size,
        "Original all-member preservation provenance differs",
    )
    expected = external["files"]
    require(len(expected) == receipt["source_files"], "Original member count differs")
    destination.mkdir()
    seen = set()
    copied_bytes = 0
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = member.name
            path = PurePosixPath(name)
            require(
                name
                and not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in name
                and str(path) == name
                and name not in seen
                and member.isfile()
                and not member.issym()
                and not member.islnk(),
                "Unsafe or duplicate archive member",
            )
            is_manifest = name == "MANIFEST-20261031.json"
            require(is_manifest or name in expected, "Archive member is outside the manifest")
            target = destination / name
            require(target.resolve().is_relative_to(destination), "Member escaped origin tree")
            target.parent.mkdir(parents=True, exist_ok=True)
            incoming = stream.extractfile(member)
            require(incoming is not None, "Archive member bytes absent")
            hasher, size = hashlib.sha256(), 0
            with target.open("xb") as outgoing:
                for block in iter(lambda: incoming.read(1024 * 1024), b""):
                    outgoing.write(block)
                    hasher.update(block)
                    size += len(block)
            expected_sha = receipt["manifest_sha256"] if is_manifest else expected[name]["sha256"]
            require(
                size == member.size
                and (is_manifest or size == expected[name]["size"])
                and hasher.hexdigest() == expected_sha
                and digest(target) == expected_sha,
                "Extracted member SHA/readback differs: " + name,
            )
            target.chmod(0o400)
            seen.add(name)
            copied_bytes += size
    require(seen == set(expected) | {"MANIFEST-20261031.json"}, "Extraction member set differs")
    require(
        json.loads((destination / "MANIFEST-20261031.json").read_text()) == external,
        "Embedded/external preservation manifests differ",
    )
    result = {
        "schema": "lewm-preserved-origin-extraction-v1",
        "verified": True,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "collection_execution_pod_id": args.execution_pod_id,
        "original_owner_pod_id": args.owner_pod_id,
        "archive": {"path": str(archive), "sha256": args.archive_sha256},
        "original_preserver_receipt_sha256": digest(args.receipt),
        "local_all_member_verification_sha256": digest(args.local_verification),
        "external_manifest_sha256": digest(args.manifest),
        "embedded_manifest_sha256": receipt["manifest_sha256"],
        "destination": str(destination),
        "origin_storage": "container_local_tmp"
        if destination.parent == Path("/tmp")
        else "network_workspace",
        "members": len(seen),
        "copied_bytes": copied_bytes,
        "every_extracted_member_hashed_and_readback_verified": True,
        "origin_files_read_only": True,
        "canonical_project_imports": 0,
        "source_script_sha256": digest(__file__),
        "new_model_queries": 0,
        "new_simulator_actions": 0,
    }
    write_new(args.out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-pod-id", required=True)
    parser.add_argument("--owner-pod-id", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-verification", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = extract(args)
    except BaseException:
        failure = args.out.with_suffix(".failure.json")
        if not failure.exists():
            write_new(failure, {"error": traceback.format_exc(), "new_model_simulator_calls": 0})
        raise
    print(json.dumps(result))


if __name__ == "__main__":
    main()
