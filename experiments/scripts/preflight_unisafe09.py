"""Read-only credential, checkpoint and upstream-source checks before rental."""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_source import prepare_source


def main():
    load_dotenv(ROOT / ".env", override=False)
    out = ROOT / "data/unisafe09/preflight"
    out.mkdir(parents=True, exist_ok=True)
    design = json.loads((ROOT / "configs/unisafe09/design.json").read_text())
    pins = design["pins"]
    api = HfApi()
    info = api.model_info(pins["checkpoint_repo"], revision=pins["checkpoint_revision"])
    path = hf_hub_download(info.id, "manifest.json", revision=info.sha, local_dir=out)
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == pins["checkpoint_manifest_sha256"]
    manifest = json.loads(raw)
    dataset = api.repo_info(design["paths"]["private_dataset"], repo_type="dataset")
    assert info.private and dataset.private
    records = []
    sources = [
        ("CMU-IntentLab/UNISafe", pins["unisafe_revision"], "takeoff/mdp/rewards.py"),
        ("CMU-IntentLab/UNISafe", pins["unisafe_revision"], "takeoff/mdp/terminations.py"),
        ("CMU-IntentLab/UNISafe", pins["unisafe_revision"], "takeoff/mdp/failure.py"),
        ("CMU-IntentLab/UNISafe", pins["unisafe_revision"], "takeoff/takeoff_env_cfg.py"),
        (
            "isaac-sim/IsaacLab",
            pins["isaaclab_revision"],
            "source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
        ),
    ]
    for repo, revision, name in sources:
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{name}"
        with urllib.request.urlopen(url, timeout=40) as response:
            content = response.read()
        destination = out / repo.split("/")[-1] / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        records.append({"url": url, "sha256": hashlib.sha256(content).hexdigest()})
    # Docker Hub issues a public pull token. It stays in memory and is never logged.
    token_url = (
        "https://auth.docker.io/token?service=registry.docker.io"
        "&scope=repository:vastai/base-image:pull"
    )
    with urllib.request.urlopen(token_url, timeout=30) as response:
        pull_token = json.load(response)["token"]
    image_digest = "sha256:c1d2b5326fae806b04d2c2d97a2b3948d0ccb0026dd3085e9a2ad55e304193f8"
    request = urllib.request.Request(
        "https://registry-1.docker.io/v2/vastai/base-image/manifests/" + image_digest,
        headers={
            "Authorization": "Bearer " + pull_token,
            "Accept": "application/vnd.oci.image.index.v1+json, "
            "application/vnd.docker.distribution.manifest.list.v2+json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        image_manifest = response.read()
    assert "sha256:" + hashlib.sha256(image_manifest).hexdigest() == image_digest
    (out / "image-manifest.json").write_bytes(image_manifest)
    runtime = prepare_source()
    result = {
        "utc": utc(),
        "checkpoint_revision": info.sha,
        "checkpoint_private": info.private,
        "checkpoint_manifest_sha256": digest,
        "checkpoint_files": len(manifest["files"]),
        "checkpoint_bytes": sum(x["bytes"] for x in manifest["files"]),
        "destination_private": dataset.private,
        "sources": records,
        "image_digest_verified": image_digest,
        "compatibility_manifest_sha256": hashlib.sha256(
            (runtime / "compatibility.json").read_bytes()
        ).hexdigest(),
    }
    atomic_json(ROOT / "progress/receipts/09-unisafe-access.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Third-party network exceptions can embed authentication in URLs.
        print(json.dumps({"error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
