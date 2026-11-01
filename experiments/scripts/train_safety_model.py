"""Fit a clearance head offline; keep training, calibration and test sources separate."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.safety_data import load_examples, write_json
from helpers.safety_model import (
    ClearanceModel,
    calibrate,
    file_digest,
    prediction_metrics,
    save_model,
)


def fit_model(arrays, spec, config, device="cpu"):
    torch.manual_seed(config["seed"])
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    model = ClearanceModel(spec, config["hidden"]).to(device)
    train = arrays["splits"] == "train"
    if not train.any():
        raise ValueError("No training examples")
    features = torch.as_tensor(arrays["features"][train], device=device)
    targets = torch.as_tensor(arrays["targets"][train], device=device)
    model.fit_normalization(features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("Invalid training allocation")
    history = []
    for epoch in range(config["epochs"]):
        total = 0.0
        order = torch.randperm(len(targets), generator=generator)
        for batch in order.split(config["batch_size"]):
            batch = batch.to(device)
            prediction = model(features[batch])
            target = targets[batch] / spec.length_scale
            loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.02)
            boundary = 1 + config["boundary_weight"] * torch.exp(
                -(targets[batch] - config["margin"]).abs() / config["boundary_band"]
            )
            asymmetric = torch.where(prediction > target, config["overprediction_weight"], 1.0)
            loss = (loss * boundary * asymmetric).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        history.append({"epoch": epoch + 1, "loss": total / len(targets)})
    model.eval()
    calibration = arrays["splits"] == "calibration"
    if calibration.any():
        calibrate(
            model,
            torch.as_tensor(arrays["features"][calibration], device=device),
            torch.as_tensor(arrays["targets"][calibration], device=device),
            arrays["groups"][calibration],
            config["alpha"],
        )
    else:
        model.calibration = {
            "ready": False,
            "source_groups": 0,
            "allowance": None,
            "scope": "Training-only engineering prototype; learned execution unavailable",
        }
    metrics = {}
    for name in ("train", "calibration", "test"):
        selected = arrays["splits"] == name
        if selected.any():
            metrics[name] = prediction_metrics(
                model,
                torch.as_tensor(arrays["features"][selected], device=device),
                torch.as_tensor(arrays["targets"][selected], device=device),
                arrays["groups"][selected],
                config["margin"],
            )
    return model, {"history": history, "metrics": metrics, "calibration": model.calibration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    arrays, spec, metadata = load_examples(args.data)
    config = json.loads(args.config.read_text())
    if not config.get("engineering_train_only", False):
        if not all(np.any(arrays["splits"] == name) for name in ("calibration", "test")):
            raise ValueError("Independent calibration and test sources are required")
    if config["boundary_band"] <= 0 or not 0 < config["alpha"] < 1:
        raise ValueError("Invalid physical boundary band or calibration level")
    if min(config["boundary_weight"], config["overprediction_weight"]) < 0:
        raise ValueError("Loss weights must be nonnegative")
    args.out.mkdir(parents=True, exist_ok=False)
    freeze = {
        "config": config,
        "config_sha256": file_digest(args.config),
        "dataset_sha256": file_digest(args.data),
        "dataset_metadata": metadata,
        "source_sha256": {
            name: file_digest(ROOT / name)
            for name in (
                "experiments/scripts/train_safety_model.py",
                "experiments/helpers/safety_model.py",
                "experiments/helpers/safety_data.py",
            )
        },
        "device": args.device,
        "threads": args.threads,
        "simulator_interactions": 0,
    }
    write_json(args.out / "freeze.json", freeze)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    model, report = fit_model(arrays, spec, config, args.device)
    report.update(
        {
            "elapsed_seconds": time.perf_counter() - started,
            "simulator_interactions": 0,
            "optimizer_updates": config["epochs"]
            * int(np.ceil(np.count_nonzero(arrays["splits"] == "train") / config["batch_size"])),
            "status": "Engineering implementation; closed-loop efficacy untested",
        }
    )
    save_model(args.out / "model.pt", model, {"freeze": freeze, "report": report})
    write_json(args.out / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "history"}))


if __name__ == "__main__":
    main()
