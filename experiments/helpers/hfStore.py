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

import os
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
        )
        sha = getattr(info, "oid", None) or self.api.repo_info(rid, repo_type=rtype).sha
        if tag:
            try:
                self.api.create_tag(rid, tag=run_id, repo_type=rtype, revision=sha)
            except HfHubHTTPError as exc:  # tag exists: keep the first, warn
                print(f"[hfStore] tag {run_id} not created ({exc.__class__.__name__}); sha={sha}")
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
        if not revision:
            raise ValueError("revision is required: pin a tag or commit, never 'latest'")
        rid, rtype = self.repo_id(key), self.repo_type(key)
        local_root = Path(local_root or REPO_ROOT / "data" / "hf" / REPOS[key][0])
        snapshot_download(
            repo_id=rid,
            repo_type=rtype,
            revision=revision,
            allow_patterns=[f"{path_in_repo}/**", f"{path_in_repo}/*"],
            local_dir=str(local_root),
            token=self.token,
        )
        out = local_root / path_in_repo
        if not out.exists():
            raise FileNotFoundError(f"{rid}:{path_in_repo}@{revision} downloaded nothing")
        return out

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
