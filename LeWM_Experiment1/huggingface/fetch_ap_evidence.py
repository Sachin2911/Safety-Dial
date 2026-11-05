#!/usr/bin/env python3
"""Download selected AP run evidence from a pinned private Hugging Face revision.

The controller's verified manifest hash is required. This downloads saved files
only; it does not import a model, import a simulator, or make a full-backup claim.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "LeWM_Experiment1/computeConfigs/scripts"))
from compute_common import (  # noqa: E402
    ComputeError, load_secrets, redact, relative_path, require_secret, safe_path, sha256, valid_id,
)

REPOSITORY = "Sachioster/LeWMExp1_AP_Reproduction"
ROOT_FILES = {"run.json", "config.json", "protocol.md", "source.tar.gz", "result.json"}
OUTPUT_DIRS = {"metrics", "logs", "artifacts"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif"}
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
CHUNK = 1024 * 1024


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ComputeError("Refusing a non-HTTPS download redirect")
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if urlsplit(request.full_url).netloc != urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def response(token, revision, run_id, relative):
    remote = quote(f"runs/{run_id}/{relative}", safe="/")
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{remote}"
    request = Request(url, headers={"Authorization": "Bearer " + token,
                                   "Cache-Control": "no-cache", "Accept-Encoding": "identity"})
    return build_opener(SafeRedirect()).open(request, timeout=120)


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ComputeError("Manifest contains a duplicate JSON key")
        value[key] = item
    return value


def verified_existing(path, expected):
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise ComputeError("Existing local file differs; refusing to overwrite: " + path.name)
    if sha256(path) != expected["sha256"]:
        raise ComputeError("Existing local checksum differs; refusing to overwrite: " + path.name)
    return True


def install_file(temporary, root, relative, expected):
    """Atomic, exclusive installation; a concurrent identical download can be reused."""
    destination = safe_path(root, relative)
    try:
        os.link(temporary, destination)
        return "downloaded"
    except FileExistsError:
        verified_existing(safe_path(root, relative), expected)
        return "reused_verified"


def write_new(root, relative, data):
    destination = safe_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if verified_existing(destination, expected):
        return "reused_verified"
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return install_file(temporary, root, relative, expected)
    finally:
        Path(temporary).unlink(missing_ok=True)


def fetch_file(token, revision, run_id, root, relative, expected):
    destination = safe_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if verified_existing(destination, expected):
        return {"path": relative, **expected, "status": "reused_verified"}
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
    try:
        count, digest = 0, hashlib.sha256()
        with os.fdopen(fd, "wb") as stream, response(token, revision, run_id, relative) as source:
            while chunk := source.read(CHUNK):
                count += len(chunk)
                if count > expected["bytes"]:
                    raise ComputeError("Downloaded file exceeds its manifest byte count")
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            raise ComputeError("Downloaded file differs from its manifest size or checksum")
        status = install_file(temporary, root, relative, expected)
        return {"path": relative, **expected, "status": status}
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_manifest(data, run_id):
    try:
        manifest = json.loads(data, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError) as error:
        raise ComputeError("Manifest is not valid UTF-8 JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ComputeError("Unsupported manifest schema")
    if manifest.get("run_id") != run_id or not isinstance(manifest.get("files"), dict):
        raise ComputeError("Manifest run identity or file inventory differs")
    files = manifest["files"]
    if not files or len(files) > 100000:
        raise ComputeError("Manifest has an invalid file count")
    for relative, expected in files.items():
        relative_path(relative)
        if relative not in ROOT_FILES and relative.split("/", 1)[0] not in OUTPUT_DIRS:
            raise ComputeError("Manifest contains an unexpected artifact location")
        if not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}:
            raise ComputeError("Manifest file metadata is invalid")
        if type(expected["bytes"]) is not int or expected["bytes"] < 0:
            raise ComputeError("Manifest contains an invalid byte count")
        if not isinstance(expected["sha256"], str) or not re.fullmatch(
                r"[0-9a-f]{64}", expected["sha256"]):
            raise ComputeError("Manifest contains an invalid file checksum")
    if not ROOT_FILES.issubset(files):
        raise ComputeError("Completed run manifest lacks required source or run metadata")
    return manifest


def exclusion(relative, expected, args):
    if relative.startswith("artifacts/retrieval-cache/"):
        return None if args.include_retrieval_cache else "retrieval cache not requested"
    if relative in args.include_file:
        return None
    if Path(relative).suffix.lower() in VIDEO_SUFFIXES:
        return None if args.include_media else "video or animation not requested"
    if expected["bytes"] > args.max_small_file_bytes:
        return "file exceeds the declared small-evidence size limit"
    return None


def error_message(error, secrets):
    if isinstance(error, HTTPError):
        return "Hugging Face download failed with HTTP " + str(error.code)
    if isinstance(error, (URLError, TimeoutError)):
        return "Download connection failed or timed out"
    if isinstance(error, (ComputeError, OSError, ValueError)):
        return redact(str(error), secrets.values())
    return "Download failed: " + type(error).__name__


def main(args, secrets):
    run_id = valid_id(args.run_id)
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ComputeError("A full, pinned 40-character Hugging Face commit is required")
    if not re.fullmatch(r"[0-9a-f]{64}", args.manifest_sha256):
        raise ComputeError("The verified controller's full manifest SHA-256 is required")
    if not 1 <= args.workers <= 8:
        raise ComputeError("Download concurrency must be between one and eight")
    if args.max_small_file_bytes <= 0 or args.max_total_bytes <= 0:
        raise ComputeError("Download byte limits must be positive")
    token = require_secret(secrets, "HF_TOKEN")
    output_root = Path(args.output_root).expanduser().resolve()
    root = safe_path(output_root, run_id)
    root.mkdir(parents=True, exist_ok=True)
    # This HTTP request always runs, even when a matching local manifest exists.
    with response(token, args.revision, run_id, "manifest.json") as source:
        data = source.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ComputeError("Manifest exceeds the bounded download size")
    if hashlib.sha256(data).hexdigest() != args.manifest_sha256:
        raise ComputeError("Fresh manifest does not match the verified controller hash")
    manifest = validate_manifest(data, run_id)
    for relative in args.include_file:
        relative_path(relative)
        if relative not in manifest["files"]:
            raise ComputeError("Explicitly requested file is absent from the pinned manifest")
        if relative.startswith("artifacts/retrieval-cache/") and not args.include_retrieval_cache:
            raise ComputeError("Explicit retrieval-cache requests require --include-retrieval-cache")
    write_new(root, "manifest.json", data)
    selected, excluded = {}, []
    for relative, expected in sorted(manifest["files"].items()):
        safe_path(root, relative)
        reason = exclusion(relative, expected, args)
        if reason:
            excluded.append({"path": relative, **expected, "reason": reason})
        else:
            selected[relative] = expected
    total = sum(item["bytes"] for item in selected.values())
    if total > args.max_total_bytes:
        raise ComputeError("Selected evidence exceeds the declared total download byte limit")
    verified, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(fetch_file, token, args.revision, run_id, root, name, expected):
                   name for name, expected in selected.items()}
        for future in as_completed(pending):
            name = pending[future]
            try:
                verified.append(future.result())
            except Exception as error:
                failures.append({"path": name, "error": error_message(error, secrets)})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt_name = f"download-receipt-{stamp}-{uuid.uuid4().hex[:8]}.json"
    receipt = {"schema_version": 1, "repository": REPOSITORY, "run_id": run_id,
               "revision": args.revision, "manifest_sha256": args.manifest_sha256,
               "manifest_verification": "fresh HTTP download and exact SHA-256",
               "completed_utc": stamp, "local_run_directory": str(root),
               "selected_files": [{"path": name, **expected}
                                  for name, expected in selected.items()],
               "verified_files": sorted(verified, key=lambda item: item["path"]),
               "excluded_files": excluded, "failed_files": sorted(failures,
                                                                      key=lambda r: r["path"]),
               "selected_bytes": total, "verified_bytes": sum(f["bytes"] for f in verified),
               "selected_evidence_verified": not failures,
               "scope": "Selected evidence download only; this receipt makes no full-backup claim",
               "options": {"include_media": args.include_media,
                           "include_file": sorted(set(args.include_file)),
                           "include_retrieval_cache": args.include_retrieval_cache,
                           "max_small_file_bytes": args.max_small_file_bytes,
                           "max_total_bytes": args.max_total_bytes, "workers": args.workers}}
    write_new(root, receipt_name, (json.dumps(receipt, indent=2) + "\n").encode())
    print(json.dumps({"selected_evidence_verified": not failures, "run_id": run_id,
                      "verified_files": len(verified), "excluded_files": len(excluded),
                      "failed_files": len(failures), "receipt": str(root / receipt_name)}))
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT / "LeWM_Experiment1/runs/AP")
    parser.add_argument("--include-media", action="store_true",
                        help="Also download videos and animations")
    parser.add_argument("--include-file", action="append", default=[], metavar="PATH",
                        help="Also download this exact manifest path; repeat for selected clips")
    parser.add_argument("--include-retrieval-cache", action="store_true",
                        help="Explicitly download the full derived retrieval cache")
    parser.add_argument("--max-small-file-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-total-bytes", type=int, default=5_000_000_000)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    credentials = {}
    try:
        credentials = load_secrets(PROJECT)
        sys.exit(main(arguments, credentials))
    except Exception as exception:
        print("Evidence download failed: " + error_message(exception, credentials), file=sys.stderr)
        sys.exit(1)
