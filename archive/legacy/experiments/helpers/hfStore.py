"""Hugging Face run storage: upload/download run folders with manifests, pinned revisions.

Every model checkpoint and every generated bank lives in a private Hugging Face
repository, one folder per run (`<kind>/<run_id>/`), tagged with the run ID
([infrastructure.md]). Consumers pin `revision=` to a tag or commit and record it.

    from helpers.hfStore import HFStore
    store = HFStore()                       # namespace from HF_NAMESPACE or the token owner
    sha = store.upload_run("pusht", "assets", run_dir)      # -> commit sha, tag = run_id
    local = store.download_run("pusht", "assets/<run_id>", revision=sha)

Never print token values. Never load "latest": `download_run` requires a revision.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ROOT = Path(__file__).resolve().parents[2]

# repo key -> (suffix, repo_type)
REPOS = {
    "pusht": ("safetydial-pusht", "model"),
    "walker2d": ("safetydial-walker2d", "model"),
    "walker2d-data": ("safetydial-walker2d-data", "dataset"),
    "pusht-banks": ("safetydial-pusht-banks", "dataset"),
}


def _load_dotenv() -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


class HFStore:
    def __init__(self, namespace: str | None = None, token: str | None = None):
        _load_dotenv()
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise RuntimeError("HF_TOKEN is not set (env or .env)")
        self.api = HfApi(token=self.token)
        self.namespace = namespace or os.environ.get("HF_NAMESPACE") or self.api.whoami()["name"]

    # ---- repos -------------------------------------------------------------------
    def repo_id(self, key: str) -> str:
        suffix, _ = REPOS[key]
        return f"{self.namespace}/{suffix}"

    def repo_type(self, key: str) -> str:
        return REPOS[key][1]

    def ensure_repo(self, key: str) -> str:
        rid, rtype = self.repo_id(key), self.repo_type(key)
        self.api.create_repo(rid, repo_type=rtype, private=True, exist_ok=True)
        if self.api.repo_info(rid, repo_type=rtype).private is not True:
            raise RuntimeError(f"Refusing checkpoint upload: {rid} is not a private repository")
        return rid

    # ---- uploads -----------------------------------------------------------------
    def upload_run(
        self,
        key: str,
        kind: str,
        run_dir: str | Path,
        *,
        run_id: str | None = None,
        message: str | None = None,
        tag: bool = True,
    ) -> str:
        """Upload `run_dir` to `<kind>/<run_id>/` and tag the commit. Returns the sha."""
        run_dir = Path(run_dir)
        run_id = run_id or run_dir.name
        if not (run_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"{run_dir} has no manifest.json; write one first")
        rid, rtype = self.ensure_repo(key), self.repo_type(key)
        info = self.api.upload_folder(
            repo_id=rid,
            repo_type=rtype,
            folder_path=str(run_dir),
            path_in_repo=f"{kind}/{run_id}",
            commit_message=message or f"{kind}/{run_id}",
            ignore_patterns=["hf_upload.json", "hf_upload.json.tmp"],
        )
        sha = getattr(info, "oid", None) or self.api.repo_info(rid, repo_type=rtype).sha
        record = {"repo_id": rid, "repo_type": rtype, "path": f"{kind}/{run_id}",
                  "revision": sha, "run_id": run_id}
        record_path = run_dir / "hf_upload.json"
        temporary = record_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(record_path)
        if tag:
            try:
                self.api.create_tag(rid, tag=run_id, repo_type=rtype, revision=sha)
            except HfHubHTTPError:
                # A tag is an immutable run identity. A different existing target is
                # not a successful tagged upload, even though the commit is durable.
                target = self.api.repo_info(rid, repo_type=rtype, revision=run_id).sha
                if target != sha:
                    raise RuntimeError(
                        f"Run tag {run_id} already refers to {target}; uploaded commit is {sha}. "
                        "Use a new run ID. Upload provenance is saved in hf_upload.json."
                    ) from None
        print(f"[hfStore] uploaded {run_dir} -> {rid}:{kind}/{run_id} @ {sha}")
        return sha

    def upload_file(self, key: str, local: str | Path, path_in_repo: str, message: str = "") -> str:
        rid, rtype = self.ensure_repo(key), self.repo_type(key)
        info = self.api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=rid,
            repo_type=rtype,
            commit_message=message or f"add {path_in_repo}",
        )
        return getattr(info, "oid", None) or self.api.repo_info(rid, repo_type=rtype).sha

    # ---- downloads ---------------------------------------------------------------
    def download_run(
        self,
        key: str,
        path_in_repo: str,
        *,
        revision: str,
        local_root: str | Path | None = None,
    ) -> Path:
        """Fetch `<path_in_repo>/**` at a pinned `revision` into `local_root/<path_in_repo>`."""
        if not revision or revision in {"main", "master", "latest", "HEAD"}:
            raise ValueError("revision must pin a run tag or full commit, never a moving branch")
        rid, rtype = self.repo_id(key), self.repo_type(key)
        if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision):
            tags = self.api.list_repo_refs(rid, repo_type=rtype).tags
            if revision not in {t.name for t in tags}:
                raise ValueError(f"Revision {revision!r} is not a run tag or full commit")
        resolved = self.api.repo_info(rid, repo_type=rtype, revision=revision).sha
        local_root = Path(local_root or REPO_ROOT / "data" / "hf" / REPOS[key][0])
        snapshot_download(
            repo_id=rid,
            repo_type=rtype,
            revision=resolved,
            allow_patterns=[f"{path_in_repo}/**", f"{path_in_repo}/*"],
            local_dir=str(local_root),
            token=self.token,
        )
        out = local_root / path_in_repo
        if not out.exists():
            raise FileNotFoundError(f"{rid}:{path_in_repo}@{revision} downloaded nothing")
        (out / "hf_download.json").write_text(json.dumps({
            "repo_id": rid, "repo_type": rtype, "path": path_in_repo,
            "requested_revision": revision, "revision": resolved,
        }, indent=2) + "\n")
        return out

    def reference_run(self, key: str, kind: str, run_dir: str | Path) -> dict:
        """Pin an existing local bundle by its saved upload receipt or immutable run tag.

        Local file hashes make a consuming manifest explicit about the bytes it used.
        Older bundles without receipts resolve their original run-ID tag once.
        """
        from helpers.runManifest import file_sha256

        run_dir = Path(run_dir)
        reference = None
        for name in ("hf_upload.json", "hf_download.json"):
            path = run_dir / name
            if path.is_file():
                reference = json.loads(path.read_text())
                break
        legacy = run_dir / "hf_revision.txt"
        if reference is None and legacy.is_file():
            repo, path, revision = legacy.read_text().split()[:3]
            reference = {"repo_id": repo, "path": path, "revision": revision}
        if reference is None:
            reference = {"repo_id": self.repo_id(key), "path": f"{kind}/{run_dir.name}",
                         "revision": self.resolve_revision(key, run_dir.name)}
        if reference["repo_id"] != self.repo_id(key):
            raise ValueError(f"Bundle repository does not match {self.repo_id(key)}")
        sha = reference["revision"]
        if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", sha):
            raise ValueError("Bundle reference must contain a full immutable commit")
        reference = {"repo_id": reference["repo_id"], "repo_type": self.repo_type(key),
                     "path": reference["path"], "revision": sha,
                     "local_path": str(run_dir), "files_sha256": {}}
        for path in sorted(run_dir.iterdir()):
            if path.is_file() and path.suffix in {".pt", ".ckpt", ".npz", ".yaml"}:
                reference["files_sha256"][path.name] = file_sha256(path)
        reference["manifest_sha256"] = file_sha256(run_dir / "manifest.json")
        return reference

    def resolve_revision(self, key: str, ref: str) -> str:
        """Commit sha for a tag/branch/sha, from the Hub."""
        rid, rtype = self.repo_id(key), self.repo_type(key)
        return self.api.repo_info(rid, repo_type=rtype, revision=ref).sha

    def list_runs(self, key: str, kind: str) -> list[str]:
        rid, rtype = self.repo_id(key), self.repo_type(key)
        files = self.api.list_repo_files(rid, repo_type=rtype)
        runs = sorted({f.split("/")[1] for f in files if f.startswith(f"{kind}/") and f.count("/") >= 2})
        return runs


def public_model_revision(repo_id: str = "quentinll/lewm-pusht") -> str:
    """Commit sha of the released Push-T weights (public repo)."""
    return HfApi().repo_info(repo_id, repo_type="model").sha


def public_dataset_revision(repo_id: str = "quentinll/lewm-pusht") -> str:
    return HfApi().repo_info(repo_id, repo_type="dataset").sha
