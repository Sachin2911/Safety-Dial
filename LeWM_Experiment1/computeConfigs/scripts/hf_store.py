"""Immutable run uploads and full read-back verification on the rented host."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil
import tempfile
import time

from compute_common import (
    ComputeError,
    atomic_json,
    locked,
    read_json,
    reject_secrets,
    relative_path,
    safe_path,
    sha256,
    validate_storage,
)


ROOT_FILES = {"run.json", "config.json", "protocol.md", "source.tar.gz", "result.json"}
OUTPUT_DIRS = {"metrics", "logs", "artifacts"}
READBACK_WORKERS = 8


def inventory(root, final):
    root = Path(root)
    if final:
        names = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if relative.split("/", 1)[0] == ".control" or relative == "manifest.json":
                continue
            if path.is_symlink():
                raise ComputeError("Run output contains a symlink; finalization refused")
            if path.is_file():
                names.append(relative)
    else:
        names = [
            read_json(path)["path"] for path in sorted((root / ".control/ready").glob("*.json"))
        ]
    result = {}
    for name in names:
        relative_path(name)
        if name not in ROOT_FILES and name.split("/", 1)[0] not in OUTPUT_DIRS:
            raise ComputeError(f"Unexpected output location: {name}")
        if Path(name).suffix.lower() in (".pem", ".key", ".env"):
            raise ComputeError("Credential-like output filename refused")
        path = safe_path(root, name)
        if not path.is_file():
            raise ComputeError(f"Published output disappeared: {name}")
        result[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if not final:
        for marker in (root / ".control/ready").glob("*.json"):
            expected = read_json(marker)
            if result[expected["path"]] != {key: expected[key] for key in ("bytes", "sha256")}:
                raise ComputeError(f"Published output changed: {expected['path']}")
    return result


class HubStore:
    """SDK imports are lazy so local controller tests need no ML environment."""

    def __init__(self, storage, token):
        from huggingface_hub import HfApi

        storage = validate_storage(storage)
        self.api = HfApi(token=token)
        self.token, self.storage = token, storage
        self.repo = storage["repo_id"]
        self.branch = storage["branch"]

    def head(self):
        info = self.api.repo_info(self.repo, repo_type="dataset", revision=self.branch)
        if not info.private:
            raise ComputeError("Destination is public; upload refused")
        return info.sha

    def check_revision(self, revision):
        info = self.api.repo_info(self.repo, repo_type="dataset", revision=revision)
        if not info.private or info.sha != revision:
            raise ComputeError("Previously verified private revision is unavailable")

    def existing(self, names, revision):
        result = {}
        for offset in range(0, len(names), 100):
            for item in self.api.get_paths_info(
                self.repo, names[offset : offset + 100], repo_type="dataset", revision=revision
            ):
                result[item.path] = item
        return result

    @staticmethod
    def matches(item, path, expected):
        if getattr(item, "size", None) != expected["bytes"]:
            return False
        lfs = getattr(item, "lfs", None)
        if lfs:
            digest = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
            return digest == expected["sha256"]
        hasher = hashlib.sha1(usedforsecurity=False)
        hasher.update(f"blob {expected['bytes']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        return getattr(item, "blob_id", None) == hasher.hexdigest()

    def commit(self, prefix, files, directory):
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.errors import HfHubHTTPError

        # Re-read on conflict or ambiguous response. Never blindly retry a mutation.
        for attempt in range(3):
            revision = self.head()
            remote = self.existing([f"{prefix}/{name}" for name in files], revision)
            operations = []
            for name, expected in files.items():
                destination = f"{prefix}/{name}"
                local = directory / name
                if destination in remote:
                    if not self.matches(remote[destination], local, expected):
                        raise ComputeError(f"Different bytes already exist in HF: {destination}")
                else:
                    operations.append(CommitOperationAdd(destination, local))
            if not operations:
                return revision
            try:
                commit = self.api.create_commit(
                    repo_id=self.repo,
                    repo_type="dataset",
                    revision=self.branch,
                    parent_commit=revision,
                    operations=operations,
                    commit_message=f"Save {prefix}: {len(operations)} completed files",
                )
                return commit.oid
            except HfHubHTTPError as error:
                if error.response is not None and error.response.status_code not in (
                    409,
                    412,
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    raise
                if attempt == 2:
                    raise
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(attempt + 1)
        raise ComputeError("Unable to commit completed files")

    def verify(self, prefix, files, revision):
        import requests
        from huggingface_hub import hf_hub_url

        def verify_one(name, expected):
            # requests.get creates an independent session for each streamed response.
            url = hf_hub_url(self.repo, f"{prefix}/{name}", repo_type="dataset", revision=revision)
            hasher, count = hashlib.sha256(), 0
            with requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Cache-Control": "no-cache",
                    "Accept-Encoding": "identity",
                },
                stream=True,
                timeout=(15, 60),
            ) as response:
                response.raise_for_status()
                for block in response.iter_content(1024 * 1024):
                    count += len(block)
                    if count > expected["bytes"]:
                        raise ComputeError(f"HF download is larger than its manifest: {name}")
                    hasher.update(block)
            if count != expected["bytes"] or hasher.hexdigest() != expected["sha256"]:
                raise ComputeError(f"HF read-back differs from local output: {name}")

        # Every file still receives a fresh full HTTP read-back at the pinned revision.
        # A failure cancels queued work and propagates only after active streams close.
        if not files:
            return
        with ThreadPoolExecutor(max_workers=min(READBACK_WORKERS, len(files)),
                                thread_name_prefix="hf-readback") as executor:
            pending = [executor.submit(verify_one, name, expected) for name, expected in files.items()]
            try:
                for future in as_completed(pending):
                    future.result()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise


def sync_run(root, token, *, final=False, store_factory=HubStore):
    root = Path(root)
    with locked(root / ".control/upload.lock"):
        run = read_json(root / "run.json")
        if not run:
            raise ComputeError("Run metadata is missing")
        if final and read_json(root / "result.json") is None:
            raise ComputeError("Final upload requires a recorded process exit")
        files = inventory(root, final)
        journal_path = root / ".control/upload.json"
        journal = read_json(journal_path, {"uploaded": {}})
        for name, expected in journal.get("uploaded", {}).items():
            if name == "manifest.json":
                continue
            if files.get(name) != expected:
                raise ComputeError(f"Previously uploaded output changed or disappeared: {name}")
        if sum(item["bytes"] for item in files.values()) > run["max_artifact_bytes"]:
            raise ComputeError("Run exceeds its declared artifact transfer allowance")
        if final:
            manifest = {
                "schema_version": 1,
                "run_id": run["run_id"],
                "files": files,
                "result": read_json(root / "result.json"),
            }
            manifest_path = root / "manifest.json"
            existing = read_json(manifest_path)
            if existing is not None and existing != manifest:
                raise ComputeError("Finalized run changed; use a new run ID")
            atomic_json(manifest_path, manifest)
            files = dict(
                files,
                **{
                    "manifest.json": {
                        "bytes": manifest_path.stat().st_size,
                        "sha256": sha256(manifest_path),
                    }
                },
            )
        storage = run["storage"]
        prefix = f"{storage['path_prefix']}/{run['run_id']}"
        relative_path(prefix)
        store = store_factory(storage, token)
        if final and journal.get("verified") and journal.get("uploaded") == files:
            store.check_revision(journal["revision"])
            return {key: value for key, value in journal.items() if key != "uploaded"}
        with tempfile.TemporaryDirectory(prefix="upload-", dir=root / ".control") as temporary:
            stage = Path(temporary)
            for name, expected in files.items():
                path = safe_path(root, name)
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if (
                    target.stat().st_size != expected["bytes"]
                    or sha256(target) != expected["sha256"]
                ):
                    raise ComputeError(f"Output changed while snapshotting: {name}")
                if target.suffix in (".json", ".jsonl", ".txt", ".md", ".log"):
                    with target.open("rb") as stream:
                        carry = b""
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            reject_secrets(carry + block, [token])
                            carry = block[-max(len(token), 128) :]
            revision = store.commit(prefix, files, stage)
            journal.update(
                uploaded=files,
                revision=revision,
                last_sync_epoch=time.time(),
                verified=False,
                error=None,
                files=len(files),
                bytes=sum(item["bytes"] for item in files.values()),
            )
            atomic_json(journal_path, journal)
            if final:
                store.verify(prefix, files, revision)
                # Also catch modifications made locally during the final upload/read-back.
                if inventory(root, True) != manifest["files"]:
                    raise ComputeError("Local run changed during final verification")
                journal.update(
                    verified=True,
                    verified_epoch=time.time(),
                    manifest_sha256=files["manifest.json"]["sha256"],
                    verification="fresh full HTTP downloads, SHA-256 and size",
                )
                atomic_json(journal_path, journal)
        return {key: value for key, value in journal.items() if key != "uploaded"} | {
            "files": len(files),
            "bytes": sum(item["bytes"] for item in files.values()),
        }
