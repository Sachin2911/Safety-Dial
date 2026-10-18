"""Exact recorded inputs, disjoint whole-task partitions and frozen visual features."""

import fnmatch
import json
import time
from pathlib import Path

import numpy as np

from helpers.maniguard_followup_budget import atomic, clock
from helpers.maniguard_part2_protocol import RUN
from helpers.maniguard_part3_protocol import ENCODER, PARTITIONS
from helpers.maniguard_phase2_families import ROOT
from helpers.maniguard_phase3_protocol import sha


def partition_for(scene):
    groups = [name for name, scenes in PARTITIONS.items() if scene in scenes]
    if len(groups) != 1:
        raise ValueError("Episode has no unique whole-task model partition")
    return groups[0]


def episode_arrays(output):
    output = Path(output)
    trace = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    proposals = {row["step"]: row for row in trace if row["type"] == "proposal"}
    executed = {row["step"]: row for row in trace if row["type"] == "executed"}
    measurements = {row["step"]: row for row in trace if row["type"] == "measurement"}
    monitors = {row["step"]: row["monitor"]["observation"]["ap"]
                for row in trace if row["type"] == "observed"}
    corrections = {row["step"]: row for row in trace if row["type"] == "correction"}
    queries = {row["step"] for row in trace if row["type"] == "query"}
    job = json.loads((output / "job.json").read_text())
    diagnostic = ROOT / "data/maniguard/benchmark/clutter_pickup" / job["scene"] / "diagnostics.jsonl"
    diag = json.loads(diagnostic.read_text().splitlines()[0])
    props = diag["ltl_safety"]["propositions"]
    # Final observation lacks a subsequent proposal/finger frame. Do not fabricate it.
    n = len(proposals)
    if sorted(proposals) != list(range(n)) or set(proposals) != set(executed):
        raise ValueError("Incomplete proposal/executed pairs")
    robot, action, proposed, correction, target, query = [], [], [], [], [], []
    for t in range(n):
        objects = measurements[t]["protected_objects"]
        for name, prop in props.items():
            if prop.get("state") != "upright":
                continue
            matched = [o for o in objects if any(fnmatch.fnmatch(o["name"], pattern)
                       for pattern in prop["over"])]
            if not matched:
                raise ValueError("Missing numerical uprightness scope: " + name)
            if any(o["threshold_deg"] != prop["params"]["max_tilt_deg"] for o in matched):
                raise ValueError("Numerical threshold differs from native proposition")
            if all(o["margin_deg"] >= 0 for o in matched) != monitors[t][name]:
                raise ValueError("Numerical margin and native Boolean label disagree")
        frame = proposals[t]["frame"]
        robot.append(frame["joints"] + frame["fingers"])
        action.append(executed[t]["action"])
        proposed.append(proposals[t]["original"])
        correction.append(corrections[t]["normalized"])
        progress = measurements[t]["progress_negative_distance_m"]
        if progress is None or not objects:
            raise ValueError("Missing evaluation-only model targets")
        target.append([min(o["margin_deg"] for o in objects), progress])
        query.append(t in queries)
    return {"robot": np.asarray(robot, np.float32), "action": np.asarray(action, np.float32),
            "proposed": np.asarray(proposed, np.float32),
            "correction": np.asarray(correction, np.float32), "target": np.asarray(target, np.float32),
            "query": np.asarray(query, bool), "n": n, "job": job}


def corpus_inventory():
    items = []
    locations = [RUN / "part2/attempts", RUN / "part3/recording/attempts"]
    for location in locations:
        for audit in sorted(location.glob("*/audit.json")):
            value = json.loads(audit.read_text())
            if not value.get("full_episode") or value.get("purpose") != "train":
                continue
            if not value.get("valid"):
                raise ValueError("Invalid donor episode: " + str(audit))
            output = audit.parent / "data"
            items.append({"episode_id": value["episode_id"], "scene": value["scene"],
                          "partition": partition_for(value["scene"]), "output": str(output),
                          "audit_sha256": sha(audit), "trace_sha256": sha(output / "trace.jsonl"),
                          "raw_observations": len(list((output / "observations").glob("*.npz"))),
                          "steps": value["steps"]})
    return items


def extract(directory, batch_size=32, device="cuda"):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    items = corpus_inventory()
    expected = {"fit": 16, "calibration": 4, "audit": 4}
    actual = {key: sum(item["partition"] == key for item in items) for key in expected}
    if actual != expected:
        raise ValueError("Require complete prospective corpus: " + repr(actual))
    atomic(directory / "corpus.json", {"episodes": items, "counts": actual,
                                       "source": "exact evaluator RGB, no video decoding"})
    encoder_dir = ROOT / "data/maniguard/models/dinov2-small"
    model = AutoModel.from_pretrained(str(encoder_dir), local_files_only=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(str(encoder_dir), local_files_only=True,
                                                  use_fast=False)
    receipt = {"encoder": ENCODER, "files": {p.name: sha(p) for p in encoder_dir.glob("*.json")},
               "weights_sha256": sha(encoder_dir / "model.safetensors"), "device": device,
               "torch": torch.__version__, "batch_size_observations": batch_size,
               "images": 0, "forward_calls": 0, "started": time.time(), "episodes": [],
               "pixel_preprocessing": processor.to_dict(),
               "pool": "discard CLS; 16x16 patches, adaptive spatial 2x2 means, C-major, overview then wrist"}
    for item in items:
        began = time.monotonic()
        output = Path(item["output"])
        destination = directory / (item["episode_id"] + ".npz")
        if destination.exists():
            raise FileExistsError("Feature extraction is not silently repeated")
        arrays = episode_arrays(output)
        values = []
        for t in range(0, arrays["n"], batch_size):
            if clock()["physical_remaining"] < 45:
                raise RuntimeError("Feature extraction reached cleanup reserve")
            images = []
            for index in range(t, min(t + batch_size, arrays["n"])):
                with np.load(output / "observations" / f"{index:04d}.npz") as raw:
                    images += [Image.fromarray(raw[key]) for key in ("overview", "wrist")]
            inputs = processor(images=images, return_tensors="pt")
            with torch.inference_mode():
                latent = model(pixel_values=inputs["pixel_values"].to(device)).last_hidden_state[:, 1:]
                grid = latent.reshape(len(images), 16, 16, 384).permute(0, 3, 1, 2)
                pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (2, 2)).flatten(1)
                values.append(pooled.reshape(-1, ENCODER["features"]).cpu().numpy())
            receipt["images"] += len(images)
            receipt["forward_calls"] += 1
            atomic(directory / "extraction-ledger.json", receipt)
        features = np.concatenate(values).astype(np.float32)
        np.savez_compressed(destination, visual=features,
                            **{k: v for k, v in arrays.items() if k not in ("n", "job")})
        receipt["episodes"].append({**item, "feature_file": destination.name,
                                    "feature_sha256": sha(destination),
                                    "seconds": time.monotonic() - began,
                                    "last_observation_excluded": "no recorded final two-finger proposal frame"})
        atomic(directory / "extraction-ledger.json", receipt)
        print(json.dumps({"extracted": item["episode_id"], "frames": len(features)}), flush=True)
    receipt.update(finished=time.time(), seconds=time.time() - receipt["started"],
                   peak_gpu_bytes=int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0)
    atomic(directory / "extraction-ledger.json", receipt)
    return receipt


def load_group(directory, group):
    directory = Path(directory)
    manifest = json.loads((directory / "corpus.json").read_text())
    episodes = []
    for item in manifest["episodes"]:
        if item["partition"] != group:
            continue
        with np.load(directory / (item["episode_id"] + ".npz")) as value:
            episodes.append({"episode_id": item["episode_id"], "scene": item["scene"],
                             **{key: value[key].copy() for key in value.files}})
    return episodes
