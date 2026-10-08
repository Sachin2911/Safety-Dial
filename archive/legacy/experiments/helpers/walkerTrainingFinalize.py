"""Verify and publish completed local Walker checkpoints without any optimization."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from helpers.walkerProtocol import file_sha256
from helpers.walkerTrainingState import upload_checkpoint


def _same_state(actual, expected, label):
    if set(actual) != set(expected):
        raise ValueError(f"{label} state keys differ from the recovery bundle")
    for key in expected:
        a, b = actual[key], expected[key]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
            raise ValueError(f"{label} state differs from the recovery bundle: {key}")
        if (b.is_floating_point() or b.is_complex()) and not bool(torch.isfinite(b).all()):
            raise ValueError(f"Nonfinite completed checkpoint tensor: {key}")


def verify_completed_exports(run_dir):
    """Fail closed on mixed exports; the bundle is the recovery authority.

    All tensor loads map to CPU. The serialized object is a trusted local training
    export, and may require the original source snapshot/package versions to import.
    """
    run_dir = Path(run_dir)
    filenames = ("optimizer.pt", "weights.pt", "lewm_object.ckpt", "config.json", "config.yaml",
                 "scalers.npz", "splits.json", "train_state.json", "manifest.json")
    before = {name: file_sha256(run_dir / name) for name in filenames}
    bundle = torch.load(run_dir / "optimizer.pt", map_location="cpu", weights_only=False)
    identity, step = bundle["identity"], bundle["step"]
    if bundle.get("protocol_version") != 1 or type(step) is not int or step < 1 or step != identity["total_steps"]:
        raise ValueError("Finalization requires a completed atomic recovery bundle")
    config = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"))
    state = json.loads((run_dir / "train_state.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if any(value != step for value in (config["total_steps"], state["step"], manifest["metrics"]["step"], bundle["scheduler"]["last_epoch"])):
        raise ValueError("Completed bundle, recipe, exports, scheduler and manifest step disagree")
    for value in bundle["optimizer"]["state"].values():
        if "step" in value and float(value["step"]) != step:
            raise ValueError("Optimizer parameter step differs from the completed bundle")
    if state.get("recovery_bundle") != "optimizer.pt" or state.get("recovery_sha256") != before["optimizer.pt"]:
        raise ValueError("train_state does not identify this exact atomic recovery bundle")
    if manifest["run_id"] != run_dir.name or config["name"] != manifest["kind"]:
        raise ValueError("Export run identity differs from its manifest")
    if manifest["data"]["sha256"] != identity["data_sha256"]:
        raise ValueError("Manifest training source differs from the completed bundle")
    if {key: config[key] for key in identity["recipe"]} != identity["recipe"]:
        raise ValueError("Exported optimizer recipe differs from the completed bundle")
    if config["model"] != identity["model"] or json.loads((run_dir / "config.json").read_text()) != identity["model"]:
        raise ValueError("Exported architecture differs from the completed bundle")
    roles = identity["episode_splits"]
    if any(value != roles for value in (config["episode_splits"], manifest["data"]["episode_splits"], json.loads((run_dir / "splits.json").read_text()))):
        raise ValueError("Exported episode splits differ from the completed bundle")
    with np.load(run_dir / "scalers.npz") as scalers:
        for key, expected in zip(("action_mean", "action_std"), identity["scaler"], strict=True):
            np.testing.assert_array_equal(scalers[key], np.asarray(expected), err_msg="Exported normalizers differ")
    weights = torch.load(run_dir / "weights.pt", map_location="cpu", weights_only=True)
    _same_state(weights, bundle["model"], "weights.pt")
    try:
        model = torch.load(run_dir / "lewm_object.ckpt", map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError("Cannot load trusted model object; use the original source snapshot and package versions to verify or regenerate exports") from exc
    _same_state(model.state_dict(), bundle["model"], "lewm_object.ckpt")
    n_params = sum(p.numel() for p in model.parameters())
    val_loss = state["val_pred_loss"]
    if not math.isfinite(val_loss) or manifest["metrics"]["val_pred_loss"] != val_loss:
        raise ValueError("Exported validation metrics disagree or are nonfinite")
    expected_train_loss = bundle["history"][-1]["loss"] if bundle["history"] else None
    if manifest["metrics"].get("train_loss") != expected_train_loss:
        raise ValueError("Manifest training loss differs from the recovery history")
    if state["history"] != bundle["history"][-50:]:
        raise ValueError("Exported training history differs from the recovery bundle")
    source_hashes = config.get("source_hashes", {})
    if source_hashes != manifest["data"].get("source_hashes", {}):
        raise ValueError("Source snapshot identities disagree")
    for name, expected in source_hashes.items():
        source = (run_dir / "source_snapshot" / name).resolve()
        if not source.is_relative_to((run_dir / "source_snapshot").resolve()) or file_sha256(source) != expected:
            raise ValueError("Recorded source snapshot changed")
    if {name: file_sha256(run_dir / name) for name in filenames} != before:
        raise ValueError("Checkpoint changed during finalization verification; stop its writer first")
    return {"bundle": bundle, "config": config, "manifest": manifest, "n_params": n_params,
            "export_sha256": before}


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def finalize_completed_run(run_dir, report_path, store):
    """Publish consistent final exports, then write the missing receipt and result."""
    run_dir, report_path = Path(run_dir), Path(report_path)
    if report_path.exists():
        raise FileExistsError("Final report already exists; finalization will not overwrite it")
    verified = verify_completed_exports(run_dir)
    bundle, manifest = verified["bundle"], verified["manifest"]
    revision = upload_checkpoint(store, "walker2d", manifest["kind"], run_dir,
                                 run_id=run_dir.name, final=True)
    receipt = {"repo_id": store.repo_id("walker2d"), "revision": revision,
               "path": f"{manifest['kind']}/{run_dir.name}", "step": bundle["step"]}
    _atomic_json(run_dir / "upload_receipt.json", receipt)
    # Original live bundles did not retain initial render/timing telemetry. Do not
    # invent measurements or rerender/reoptimize merely to fill report fields.
    storage_path = run_dir / "storage_budget.json"
    report = {"run_id": run_dir.name, "steps": bundle["step"], "history": bundle["history"],
        "wall_clock_s": None, "n_clips": manifest["data"]["n_clips"], "n_params": verified["n_params"],
        "episode_splits": bundle["identity"]["episode_splits"], "checkpoint": receipt,
        "render_validation": {"passed": None, "reason": "original first-batch telemetry was not saved in the recovery bundle"},
        "steps_per_epoch": None,
        "storage_budget": json.loads(storage_path.read_text()) if storage_path.is_file() else None,
        "recovery_completion": {"optimizer_steps_executed": 0, "export_sha256": verified["export_sha256"],
            "unavailable_original_telemetry": ["wall_clock_s", "steps_per_epoch", "render_validation"]}}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    return report
