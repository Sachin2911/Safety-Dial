"""Prepare or privately preserve the pilot's own trained artifacts, never asset bundles."""

import argparse
import json
import os
import shutil
import time

from backup_followup import ROOT, RUN, SECRETS, digest

MODEL = ROOT / "data/maniguard/models/phase3-parts2-3-20261018"
REPO = "Sachioster/maniguard-phase3-part3-20261018"
NAMES = (
    "artifacts.npz",
    "ensemble.pt",
    "training-ledger.json",
    "trained-artifact-receipt.json",
    "calibration.json",
    "calibration-windows.npz",
    "forecast-freeze.json",
    "context-bank.json",
    "context-bank.npz",
    "forecast-audit.json",
    "forecast-windows.npz",
    "acquisition-engineering.json",
)


def preserve(upload=False):
    trained = json.loads((MODEL / "trained-artifact-receipt.json").read_text())
    for name, expected in trained["files"].items():
        if digest(MODEL / name) != expected:
            raise ValueError("A frozen trained artifact changed")
    audit = json.loads((MODEL / "forecast-audit.json").read_text())
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    export = ROOT / "data/maniguard/evidence-backups" / ("part3-trained-model-" + stamp)
    export.mkdir(parents=True, exist_ok=False)
    for name in NAMES:
        shutil.copyfile(MODEL / name, export / name)
    shutil.copyfile(RUN / "part3/model-source-freeze.json", export / "source-freeze.json")
    shutil.copyfile(RUN / "part3/data-decision.json", export / "data-decision.json")
    (export / "README.md").write_text(
        "---\ntags:\n- robotics\n- research\n---\n"
        "# ManiGuard Part 3 frozen pilot artifacts\n\n"
        "Private research preservation of the five small residual dynamics networks, "
        "fit-only preprocessing/readouts, calibration, context bank and forecast audit.\n\n"
        "The frozen forecast gate is **" + audit["gate"]["status"] + "**. "
        "This is not a validated safety filter or an optimizer superiority result. "
        "Read the complete audit and source freeze before reuse.\n\n"
        "Fitting tasks: 0039/0010. Calibration: 0046. Forecast audit: 0006. "
        "Executed environment-input actions train the dynamics. "
        "Evaluation-only numerical targets train readouts, not dynamics inputs.\n\n"
        "The external encoder facebook/dinov2-small is pinned at "
        "ed25f3a31f01632728cabb09d1542f84ab7b0056; its weights are not redistributed here. "
        "No BEHAVIOR asset bundle, VLA checkpoint, credential or raw image dataset is included.\n"
    )
    secrets = [os.environ[k].encode() for k in SECRETS if len(os.environ.get(k, "")) >= 8]
    inventory = []
    for path in sorted(export.iterdir()):
        raw = path.read_bytes()
        if any(value in raw for value in secrets):
            raise ValueError("Credential detected in prepared model payload")
        inventory.append({"path": path.name, "bytes": len(raw), "sha256": digest(path)})
    receipt = {
        "schema": "maniguard-followup-model-backup-v1",
        "at": time.time(),
        "repo_id": REPO,
        "repo_type": "model",
        "export": str(export.relative_to(ROOT)),
        "files": inventory,
        "bytes": sum(f["bytes"] for f in inventory),
        "credential_scan": "passed",
        "forecast_gate": audit["gate"],
        "scope": "Own trained pilot artifacts in a separate private model repository; no credentials, assets or external pretrained weights",
    }
    receipt_path = RUN / "part3/trained-model-backup.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if upload:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        api = HfApi()
        api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
        if api.model_info(REPO).private is not True:
            raise ValueError("Trained model destination must be private")
        commit = api.create_commit(
            repo_id=REPO,
            repo_type="model",
            operations=[
                CommitOperationAdd(path_in_repo=f["path"], path_or_fileobj=str(export / f["path"]))
                for f in inventory
            ],
            commit_message="Preserve frozen ManiGuard Part 3 pilot and forecast audit",
        )
        for item in inventory:
            downloaded = hf_hub_download(REPO, item["path"], repo_type="model", revision=commit.oid)
            if digest(downloaded) != item["sha256"]:
                raise ValueError("Remote model integrity verification failed")
        receipt.update(private=True, revision=commit.oid, download_verified=True, off_instance=True)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    preserve(parser.parse_args().upload)
