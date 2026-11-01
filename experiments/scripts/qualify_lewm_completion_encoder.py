"""One frozen64-frame cached-training encoder contract, never an environment run."""

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def write(path, value):
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    require(protocol["schema"] == "lewm-completion-encoder-contract-v1", "Unknown probe")
    require(
        os.environ.get("RUNPOD_POD_ID") == protocol["execution_pod_id"] == "uddcaj5qijcl3j",
        "Encoder qualification belongs only on the actual replacement rented GPU",
    )
    require(
        protocol["first_frame"] == 0
        and protocol["frame_count"] == 64
        and protocol["atol"] == 1e-5
        and protocol["rtol"] == 1e-5,
        "Fixed first-batch selection and comparison tolerance must not change",
    )
    require(not args.out.exists(), "No retry/overwrite of an encoder-contract attempt")
    deadline = dt.datetime.fromisoformat(protocol["deadline_utc"])
    require(dt.datetime.now(dt.timezone.utc) < deadline, "Continuation window ended")
    for field in ("source_sha256", "asset_sha256"):
        for name, expected in protocol[field].items():
            require(digest(ROOT / name) == expected, "Probe dependency changed: " + name)
    require(
        protocol["source_sha256"][str(Path(__file__).relative_to(ROOT))] == digest(__file__),
        "Probe source differs",
    )
    args.out.mkdir(parents=True)
    resources = {
        "environment_actions": 0,
        "environment_resets": 0,
        "simulator_renders": 0,
        "predictor_queries": 0,
        "model_train_updates": 0,
        "world_model_constructor_attempts": 0,
        "world_model_constructor_successes": 0,
        "encoder_attempted_batches": 0,
        "encoder_attempted_frames": 0,
        "encoder_completed_batches": 0,
        "encoder_completed_frames": 0,
        "cached_observation_rows_materialized": 0,
    }
    write(
        args.out / "source-freeze.json",
        {
            "protocol_sha256": digest(args.protocol),
            "protocol": str(args.protocol),
            "pod_id": protocol["execution_pod_id"],
            "sources": protocol["source_sha256"],
            "assets": protocol["asset_sha256"],
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    write(args.out / "resources.json", resources)
    started = time.perf_counter()
    result = {"schema": protocol["schema"], "qualified": False}
    try:
        import numpy as np
        import torch
        from stable_worldmodel.policy import AutoCostModel

        sys.path.insert(0, str(ROOT / "experiments"))
        from helpers.safecem10_planner import IMAGENET_MEAN, IMAGENET_STD, image_transform

        require(torch.cuda.is_available(), "The rented GPU must be available")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require(not torch.is_autocast_enabled(), "Autocast is prohibited")
        preprocessing = {
            "function_source": inspect.getsource(image_transform),
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "input": "uint8 RGB224",
            "layout": "B,T,C,H,W",
            "encoding": "FP32 without autocast or TF32; exact frozen AutoCostModel.encode last frame",
        }
        pre_sha = hashlib.sha256(json.dumps(preprocessing, sort_keys=True).encode()).hexdigest()
        metadata = json.loads((ROOT / protocol["memory_metadata"]).read_text())
        require(
            pre_sha == metadata["preprocessing_sha256"] == protocol["preprocessing_sha256"],
            "Preprocessing differs",
        )
        with np.load(ROOT / protocol["observations"], allow_pickle=False) as observed:
            all_pixels = observed["pixels"]
            raw = all_pixels[:64]
            resources["cached_observation_rows_materialized"] = len(all_pixels)
            episodes = observed["episode"][:64].tolist()
            steps = observed["step"][:64].tolist()
        require(
            raw.shape == (64, 224, 224, 3)
            and raw.dtype == np.uint8
            and set(episodes) <= set(metadata["memory_episode_ids"]),
            "Only the frozen first64 training-memory observations may be encoded",
        )
        index = json.loads((ROOT / protocol["frame_index"]).read_text())
        frames = index["frames"]
        for i, pixels in enumerate(raw):
            require(
                frames[i]["index"] == i
                and frames[i]["episode"] == episodes[i]
                and frames[i]["step"] == steps[i]
                and hashlib.sha256(pixels.tobytes()).hexdigest() == frames[i]["sha256"],
                "Cached frame order/pixel provenance differs",
            )
        cached = np.load(ROOT / protocol["cached_embeddings"], allow_pickle=False, mmap_mode="r")[
            :64
        ]
        with np.load(ROOT / protocol["memory"], allow_pickle=False) as bank:
            tail = bank["embedding_tail_shape"].tolist()
        require(
            cached.dtype == np.float32 and cached.shape == (64, 192) and tail == [192],
            "Cached feature contract differs",
        )
        checkpoint = ROOT / protocol["world_model"]
        resources["world_model_checkpoint_bytes"] = checkpoint.stat().st_size
        resources["world_model_constructor_attempts"] += 1
        write(args.out / "resources.json", resources)
        construction = time.perf_counter()
        model = AutoCostModel(str(checkpoint)[: -len("_object.ckpt")], cache_dir=checkpoint.parent)
        model = model.cuda().eval().requires_grad_(False)
        torch.cuda.synchronize()
        resources["world_model_constructor_seconds"] = time.perf_counter() - construction
        resources["world_model_constructor_successes"] += 1
        resources["world_model_cuda_allocated_bytes"] = torch.cuda.memory_allocated()
        write(args.out / "resources.json", resources)
        require(next(model.parameters()).dtype == torch.float32, "Encoder must remain FP32")
        tensor = image_transform()(torch.from_numpy(raw).permute(0, 3, 1, 2)).unsqueeze(1).cuda()
        require(dt.datetime.now(dt.timezone.utc) < deadline, "Deadline before encoder attempt")
        resources["encoder_attempted_frames"] += 64
        resources["encoder_attempted_batches"] += 1
        write(args.out / "resources.json", resources)  # Durable before the actual NN call.
        encoding = time.perf_counter()
        with torch.inference_mode():
            embedding = model.encode({"pixels": tensor})["emb"][:, -1]
            torch.cuda.synchronize()
        require(torch.isfinite(embedding).all().item(), "Encoder returned nonfinite features")
        actual = embedding.flatten(1).float().cpu().numpy()
        resources["encoder_seconds"] = time.perf_counter() - encoding
        resources["encoder_completed_frames"] += 64
        resources["encoder_completed_batches"] += 1
        write(args.out / "resources.json", resources)
        require(actual.shape == cached.shape, "Feature shape differs")
        delta = actual.astype(np.float64) - cached.astype(np.float64)
        qualified = bool(
            torch.allclose(
                torch.from_numpy(actual), torch.from_numpy(cached.copy()), atol=1e-5, rtol=1e-5
            )
        )
        result.update(
            qualified=qualified,
            atol=1e-5,
            rtol=1e-5,
            max_absolute_error=float(np.abs(delta).max()),
            rmse=float(np.sqrt((delta * delta).mean())),
            bitwise_equal=bool(np.array_equal(actual, cached)),
            frames=64,
            source_episodes=episodes,
            source_steps=steps,
            embedding_shape=list(actual.shape),
            torch_version=torch.__version__,
            preprocessing_sha256=pre_sha,
            cuda_matmul_allow_tf32=False,
            cudnn_allow_tf32=False,
            autocast=False,
            cached_memory_only=True,
            heldout_images_read=0,
        )
        if not qualified:
            result["failure"] = (
                "Predeclared encoder tolerance failed; final episodes must not start"
            )
    except BaseException:
        result["error"] = traceback.format_exc()
    resources["wall_seconds"] = time.perf_counter() - started
    write(args.out / "resources.json", resources)
    result.update(
        resources=resources,
        protocol_sha256=digest(args.protocol),
        completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    write(args.out / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("qualified", "resources", "max_absolute_error", "rmse", "bitwise_equal")
                if k in result
            }
        )
    )
    if not result["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
