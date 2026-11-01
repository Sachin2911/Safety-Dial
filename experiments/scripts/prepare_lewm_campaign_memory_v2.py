"""Acquire a small observed TwoRoom memory on an explicitly declared rented pod.

The protocol pins the public archive and frozen model, enumerates memory/query episode
IDs, and fixes the source-stride/successor/endpoint grid before any controller outcomes.
``--mode fetch`` only downloads/extracts; ``--mode all`` also creates the thin bank and
FP32 encoder cache. No simulator or predictor is called. Keep outputs in a fresh path.
"""

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import traceback
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_time(protocol, reserve_seconds=600):
    deadline = dt.datetime.fromisoformat(protocol["deadline_utc"])
    if dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=reserve_seconds) >= deadline:
        raise RuntimeError("Memory preparation reached the campaign preservation reserve")


def event(out, name, resources, **fields):
    value = {"utc": now(), "event": name, "resources": dict(resources), **fields}
    with (out / "events.jsonl").open("a", buffering=1) as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
    write_json(out / "live.json", value)


def validate_protocol(protocol):
    if protocol.get("schema") != "lewm-campaign-memory-v1":
        raise ValueError("Explicit memory protocol schema required")
    if os.environ.get("RUNPOD_POD_ID") not in protocol["allowed_pod_ids"]:
        raise RuntimeError("Memory preparation is restricted to the declared rented pods")
    check_time(protocol)
    members = protocol["memory_episode_ids"]
    blocked = protocol["query_episode_ids"] + protocol["excluded_episode_ids"]
    if not members or len(set(members)) != len(members):
        raise ValueError("Memory membership must be nonempty and unique")
    if set(members) & set(blocked):
        raise ValueError("Memory episodes overlap queries or excluded historical episodes")
    if min(members) < 0 or protocol["source_stride"] < 1:
        raise ValueError("Invalid source episode/stride")
    successors, endpoints = protocol["successor_offsets"], protocol["endpoint_offsets"]
    if not successors or not endpoints or min(successors + endpoints) <= 0:
        raise ValueError("Explicit positive temporal offsets required")
    if protocol["pixel_shape"] != [224, 224, 3]:
        raise ValueError("The current frozen LeWM requires 224-pixel RGB observations")
    if not protocol["source_split"] or not protocol["query_source_description"]:
        raise ValueError("Memory and query information provenance must be explicit")


class LimitedProbe:
    """Bound compressed bytes read even when a server ignores HTTP Range."""

    def __init__(self, stream, limit, resources):
        self.stream, self.remaining, self.resources = stream, limit, resources

    def read(self, size=-1):
        if self.remaining <= 0:
            raise RuntimeError("Compressed-header probe exhausted its frozen byte cap")
        amount = self.remaining if size < 0 else min(size, self.remaining)
        block = self.stream.read(amount)
        self.remaining -= len(block)
        self.resources["probe_compressed_bytes_read"] += len(block)
        return block

    def readinto(self, buffer):
        block = self.read(len(buffer))
        buffer[: len(block)] = block
        return len(block)


def probe_archive(protocol, data_root, out, resources):
    """Read the tar header before acquiring a multi-GB compressed body."""
    import zstandard

    spec = protocol["dataset"]
    archive = data_root / "raw" / spec["archive"]
    if archive.is_file():
        stream = archive.open("rb")
        response_status, content_range = None, "local cached archive"
    else:
        check_time(protocol)
        url = (
            "https://huggingface.co/datasets/"
            + spec["repo_id"]
            + "/resolve/"
            + spec["revision"]
            + "/"
            + urllib.parse.quote(spec["archive"])
        )
        request = urllib.request.Request(
            url,
            headers={
                "Range": "bytes=0-" + str(protocol["maximum_probe_bytes"] - 1),
                "User-Agent": "SafetyDial-observed-memory-header-probe",
            },
        )
        stream = urllib.request.urlopen(request, timeout=60)
        response_status = stream.status
        content_range = stream.headers.get("Content-Range")
    before = resources["probe_compressed_bytes_read"]
    try:
        limited = LimitedProbe(stream, protocol["maximum_probe_bytes"], resources)
        with zstandard.ZstdDecompressor().stream_reader(limited, closefd=False) as decoded:
            with tarfile.open(fileobj=decoded, mode="r|") as tar:
                member_bytes = None
                for member in tar:
                    if member.isfile() and Path(member.name).name == spec["member"]:
                        member_bytes = member.size
                        break
                    if member.isfile() and member.size > 65536:
                        raise RuntimeError("Unexpected large member before HDF5 in header probe")
    finally:
        stream.close()
    if member_bytes is None:
        raise RuntimeError("HDF5 header was not found within the bounded probe")
    if not archive.is_file():
        resources["public_download_bytes"] += resources["probe_compressed_bytes_read"] - before
    free = shutil.disk_usage(data_root).free
    h5 = data_root / spec["member"]
    receipt = h5.with_suffix(h5.suffix + ".receipt.json")
    cached_h5 = False
    if h5.is_file() and receipt.is_file():
        prior = json.loads(receipt.read_text())
        cached_h5 = (
            prior.get("archive_sha256") == spec["archive_sha256"]
            and prior.get("h5_bytes") == member_bytes
            and h5.stat().st_size == member_bytes
        )
    # A cached member needs no second full allocation. acquire_h5 verifies its
    # complete SHA before any rows are read or any model encoding is attempted.
    required = protocol["minimum_remaining_disk_bytes"]
    if not cached_h5:
        required += member_bytes
    if not archive.is_file():
        required += spec["archive_bytes"]
    fits = free >= required and member_bytes <= protocol["maximum_extracted_bytes"]
    value = {
        "h5_member_bytes": member_bytes,
        "existing_h5_receipt_present": cached_h5,
        "archive_bytes": spec["archive_bytes"],
        "disk_free_bytes": free,
        "required_free_bytes": required,
        "fits_frozen_storage_limits": fits,
        "http_status": response_status,
        "content_range": content_range,
        "compressed_probe_bytes_read": resources["probe_compressed_bytes_read"] - before,
        "data_root": str(data_root),
        "full_archive_download_started": False,
    }
    write_json(out / "storage-probe.json", value)
    event(out, "compressed_header_probed", resources, **value)
    return value


def acquire_h5(protocol, data_root, out, resources):
    import zstandard
    from huggingface_hub import hf_hub_download

    spec = protocol["dataset"]
    raw = data_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / spec["archive"]
    cached = archive.is_file()
    started = time.perf_counter()
    event(out, "download_started", resources, archive_cache_present=cached)
    if not cached:
        check_time(protocol, 1200)
        archive = Path(
            hf_hub_download(
                spec["repo_id"],
                spec["archive"],
                repo_type="dataset",
                revision=spec["revision"],
                local_dir=str(raw),
                token=False,
            )
        )
        resources["public_download_bytes"] += archive.stat().st_size
    if archive.stat().st_size != spec["archive_bytes"] or digest(archive) != spec["archive_sha256"]:
        raise RuntimeError("Pinned public archive size or SHA256 differs")
    resources["download_wall_seconds"] += time.perf_counter() - started
    event(out, "archive_verified", resources, archive_sha256=spec["archive_sha256"])
    target = data_root / spec["member"]
    receipt = target.with_suffix(target.suffix + ".receipt.json")
    if target.exists():
        prior = json.loads(receipt.read_text())
        if prior["archive_sha256"] != spec["archive_sha256"]:
            raise RuntimeError("Existing extracted dataset has different provenance")
        if target.stat().st_size != prior["h5_bytes"] or digest(target) != prior["h5_sha256"]:
            raise RuntimeError("Existing extracted dataset was changed")
        event(out, "existing_h5_verified", resources, **prior)
        return target, prior
    check_time(protocol, 900)
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.exists():
        raise FileExistsError("Retain incomplete extraction; use a new data namespace")
    started = time.perf_counter()
    member_bytes = None
    checksum = hashlib.sha256()
    with archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as decompressed:
            with tarfile.open(fileobj=decompressed, mode="r|") as tar:
                for member in tar:
                    if Path(member.name).name != spec["member"] or not member.isfile():
                        continue
                    member_bytes = member.size
                    free = shutil.disk_usage(data_root).free
                    if member.size > protocol["maximum_extracted_bytes"]:
                        raise RuntimeError("HDF5 member exceeds the frozen extraction limit")
                    if free < member.size + protocol["minimum_remaining_disk_bytes"]:
                        raise RuntimeError("Insufficient disk for archive plus HDF5 and reserve")
                    event(
                        out,
                        "extraction_started",
                        resources,
                        member_bytes=member.size,
                        disk_free=free,
                    )
                    source = tar.extractfile(member)
                    with temporary.open("wb") as destination:
                        while block := source.read(16 << 20):
                            check_time(protocol)
                            checksum.update(block)
                            destination.write(block)
                            resources["extracted_bytes"] += len(block)
                    break
    if member_bytes is None or not temporary.exists() or temporary.stat().st_size != member_bytes:
        raise RuntimeError("The pinned archive did not contain the complete HDF5 member")
    temporary.replace(target)
    resources["extraction_wall_seconds"] += time.perf_counter() - started
    value = {
        "archive_sha256": spec["archive_sha256"],
        "h5_sha256": checksum.hexdigest(),
        "h5_bytes": member_bytes,
        "dataset_revision": spec["revision"],
    }
    write_json(receipt, value)
    event(out, "extraction_complete", resources, **value)
    return target, value


def observed_bank(protocol, h5_path, out, resources):
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np

    started = time.perf_counter()
    triples = []
    frame_meta = {}
    with h5py.File(h5_path, "r") as data:
        lengths = np.asarray(data["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(data["ep_offset"][:], dtype=np.int64)
        if len(lengths) != len(offsets) or data["pixels"].dtype != np.uint8:
            raise RuntimeError("Unexpected dataset episode metadata or pixel dtype")
        if list(data["pixels"].shape[1:]) != protocol["pixel_shape"]:
            raise RuntimeError("Source image dimensions differ from the frozen model")
        xy_key = protocol["xy_key"]
        if data[xy_key].shape[-1] != 2:
            raise RuntimeError("Memory body metadata must contain native XY only")
        for episode in protocol["memory_episode_ids"]:
            if episode >= len(lengths):
                raise IndexError("Frozen memory episode exceeds the public dataset")
            length, offset = int(lengths[episode]), int(offsets[episode])
            for source in range(0, length, protocol["source_stride"]):
                for endpoint_offset in protocol["endpoint_offsets"]:
                    endpoint = source + endpoint_offset
                    if endpoint >= length:
                        continue
                    for successor_offset in protocol["successor_offsets"]:
                        successor = source + successor_offset
                        if successor > endpoint:
                            continue
                        triple = (episode, source, successor, endpoint, offset)
                        triples.append(triple)
                        for step in (source, successor, endpoint):
                            frame_meta[offset + step] = (episode, step)
        if not triples or len(triples) > protocol["maximum_memory_triples"]:
            raise RuntimeError("Memory triple count is empty or exceeds the frozen cap")
        rows = sorted(frame_meta)
        if len(rows) > protocol["maximum_unique_frames"]:
            raise RuntimeError("Selected observation count exceeds the frozen encoder cap")
        row_index = {row: i for i, row in enumerate(rows)}
        pixels = np.empty((len(rows), *protocol["pixel_shape"]), dtype=np.uint8)
        xy = np.empty((len(rows), 2), dtype=np.float64)
        for episode in protocol["memory_episode_ids"]:
            check_time(protocol)
            offset, length = int(offsets[episode]), int(lengths[episode])
            selected = [row for row in rows if offset <= row < offset + length]
            if not selected:
                continue
            first, last = selected[0], selected[-1] + 1
            observed = np.asarray(data["pixels"][first:last])
            positions = np.asarray(data[xy_key][first:last], dtype=np.float64)
            resources["dataset_pixel_rows_read"] += last - first
            for row in selected:
                i = row_index[row]
                pixels[i], xy[i] = observed[row - first], positions[row - first]
        if not np.isfinite(xy).all():
            raise RuntimeError("Nonfinite observed body metadata")
        episodes = np.array([frame_meta[row][0] for row in rows], dtype=np.int64)
        steps = np.array([frame_meta[row][1] for row in rows], dtype=np.int64)
        np.savez_compressed(
            out / "observations.npz",
            pixels=pixels,
            xy=xy,
            global_row=rows,
            episode=episodes,
            step=steps,
        )
        frame_indices = np.array(
            [
                [row_index[offset + step] for step in (source, successor, endpoint)]
                for _, source, successor, endpoint, offset in triples
            ],
            dtype=np.int64,
        )
        triple_array = np.asarray(triples, dtype=np.int64)
        np.savez_compressed(
            out / "triples.npz",
            frame_indices=frame_indices,
            source_episode=triple_array[:, 0],
            source_step=triple_array[:, 1],
            successor_step=triple_array[:, 2],
            endpoint_step=triple_array[:, 3],
        )
        frames = [
            {
                "index": i,
                "episode": int(episodes[i]),
                "step": int(steps[i]),
                "global_row": int(row),
                "sha256": hashlib.sha256(pixels[i].tobytes()).hexdigest(),
            }
            for i, row in enumerate(rows)
        ]
        write_json(
            out / "frame-index.json",
            {
                "artifact": "observations.npz",
                "pixel_key": "pixels",
                "frames": frames,
                "source_pixel_dtype": "uint8",
                "source_pixel_shape": protocol["pixel_shape"],
            },
        )
    resources["memory_unique_observations"] = len(rows)
    resources["memory_triples"] = len(triples)
    resources["observation_selection_seconds"] += time.perf_counter() - started
    event(out, "observations_preserved", resources)


def encode_bank(protocol, out, resources):
    import numpy as np
    import torch
    from stable_worldmodel.policy import AutoCostModel
    from helpers.safecem10_planner import IMAGENET_MEAN, IMAGENET_STD, image_transform

    check_time(protocol)
    checkpoint = ROOT / protocol["world_model"]
    if digest(checkpoint) != protocol["world_model_sha256"]:
        raise RuntimeError("Frozen world model identity differs")
    if not checkpoint.name.endswith("_object.ckpt"):
        raise ValueError("Expected a converted AutoCostModel checkpoint")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoCostModel(str(checkpoint)[: -len("_object.ckpt")], cache_dir=checkpoint.parent)
    model = model.cuda().eval().requires_grad_(False)
    transform = image_transform()
    preprocessing = {
        "function_source": inspect.getsource(image_transform),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "input": "uint8 RGB224",
        "layout": "B,T,C,H,W",
        "encoding": "FP32 without autocast or TF32; exact frozen AutoCostModel.encode last frame",
    }
    preprocessing_sha = hashlib.sha256(
        json.dumps(preprocessing, sort_keys=True).encode()
    ).hexdigest()
    with np.load(out / "observations.npz", allow_pickle=False) as data:
        observed = dict(data)
    count = len(observed["pixels"])
    batches = []
    tail_shape = None
    started = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, count, protocol["encoder_batch_size"]):
            check_time(protocol)
            raw = observed["pixels"][first : first + protocol["encoder_batch_size"]]
            tensor = transform(torch.from_numpy(raw).permute(0, 3, 1, 2)).unsqueeze(1).cuda()
            resources["encoder_attempted_frames"] += len(raw)
            resources["encoder_attempted_batches"] += 1
            event(out, "encoder_batch_started", resources, first_frame=first, frame_count=len(raw))
            embedding = model.encode({"pixels": tensor})["emb"][:, -1]
            torch.cuda.synchronize()
            if not torch.isfinite(embedding).all():
                raise RuntimeError("Frozen encoder returned nonfinite memory features")
            shape = tuple(embedding.shape[1:])
            if tail_shape is not None and shape != tail_shape:
                raise RuntimeError("Encoder feature shape changed between batches")
            tail_shape = shape
            batches.append(embedding.flatten(1).float().cpu().numpy())
            resources["encoder_completed_frames"] += len(raw)
            resources["encoder_completed_batches"] += 1
            event(out, "encoder_batch_completed", resources, first_frame=first)
    embeddings = np.concatenate(batches)
    np.save(out / "observation-embeddings.npy", embeddings, allow_pickle=False)
    with np.load(out / "triples.npz", allow_pickle=False) as data:
        triples = dict(data)
    indices = triples["frame_indices"]
    source, successor, endpoint = [embeddings[indices[:, i]] for i in range(3)]
    latent_scale = np.concatenate([source, endpoint]).astype(np.float64).std(0)
    latent_scale = np.maximum(latent_scale, 1e-3).astype(np.float32)
    memory = {
        "source_embeddings": source,
        "successor_embeddings": successor,
        "endpoint_embeddings": endpoint,
        "embedding_tail_shape": np.asarray(tail_shape),
        "latent_scale": latent_scale,
    }
    for i, name in enumerate(("source", "successor", "endpoint")):
        memory[name + "_xy"] = observed["xy"][indices[:, i]]
        memory[name + "_frame_index"] = indices[:, i]
        memory[name + "_global_row"] = observed["global_row"][indices[:, i]]
    memory.update({k: v for k, v in triples.items() if k != "frame_indices"})
    np.savez_compressed(out / "memory.npz", **memory)
    resources["encoder_wall_seconds"] += time.perf_counter() - started
    metadata = {
        "schema": "lewm-observed-memory-v1",
        "created_utc": now(),
        "dataset": protocol["dataset"],
        "source_split": protocol["source_split"],
        "memory_episode_ids": protocol["memory_episode_ids"],
        "query_episode_ids": protocol["query_episode_ids"],
        "excluded_episode_ids": protocol["excluded_episode_ids"],
        "query_source_description": protocol["query_source_description"],
        "frozen_world_model_sha256": protocol["world_model_sha256"],
        "preprocessing_sha256": preprocessing_sha,
        "preprocessing": preprocessing,
        "embedding_tail_shape": list(tail_shape),
        "embedding_dimension": embeddings.shape[1],
        "memory_triples": len(source),
        "unique_observations": count,
        "resources": resources,
        "model_train_updates": 0,
        "predictor_queries": 0,
        "information": "Actual public observation RGB and causal observed body XY with episode/time metadata. No action labels or final-query goal XY are supplied to the retrieval controller. Successor/endpoint images belong to training memory, never the live query's future.",
        "safety": "Observed targets do not establish reachability or path safety; unchanged learned safety checks and actual episode monitors remain necessary.",
    }
    write_json(out / "memory-metadata.json", metadata)
    event(out, "encoder_cache_complete", resources, memory_sha256=digest(out / "memory.npz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("probe", "fetch", "all"), default="all")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    if digest(Path(__file__)) != protocol["script_sha256"]:
        raise RuntimeError("Frozen memory-preparation source differs")
    if args.out.exists():
        raise FileExistsError("Use a fresh preparation namespace; preserve every previous attempt")
    args.out.mkdir(parents=True)
    args.data_root.mkdir(parents=True, exist_ok=True)
    resources = {
        "environment_resets": 0,
        "environment_actions": 0,
        "simulator_physics_steps": 0,
        "predictor_steps": 0,
        "probe_compressed_bytes_read": 0,
        "public_download_bytes": 0,
        "extracted_bytes": 0,
        "dataset_pixel_rows_read": 0,
        "memory_unique_observations": 0,
        "memory_triples": 0,
        "encoder_attempted_frames": 0,
        "encoder_attempted_batches": 0,
        "encoder_completed_frames": 0,
        "encoder_completed_batches": 0,
        "download_wall_seconds": 0.0,
        "extraction_wall_seconds": 0.0,
        "observation_selection_seconds": 0.0,
        "encoder_wall_seconds": 0.0,
    }
    write_json(
        args.out / "source-freeze.json",
        {
            "protocol_sha256": digest(args.protocol),
            "script_sha256": digest(Path(__file__)),
            "pod_id": os.environ["RUNPOD_POD_ID"],
            "started_utc": now(),
        },
    )
    started = time.perf_counter()
    try:
        storage = probe_archive(protocol, args.data_root, args.out, resources)
        acquisition = {"storage_probe": storage}
        if args.mode != "probe":
            if not storage["fits_frozen_storage_limits"]:
                raise RuntimeError("HDF5 cannot fit: full archive was not downloaded")
            h5_path, acquisition = acquire_h5(protocol, args.data_root, args.out, resources)
        if args.mode == "all":
            observed_bank(protocol, h5_path, args.out, resources)
            encode_bank(protocol, args.out, resources)
        write_json(
            args.out / "complete.json",
            {
                "verified": True,
                "mode": args.mode,
                "completed_utc": now(),
                "acquisition": acquisition,
                "resources": resources,
                "total_wall_seconds": time.perf_counter() - started,
                "artifacts": {
                    str(p.relative_to(args.out)): digest(p)
                    for p in args.out.iterdir()
                    if p.is_file()
                },
            },
        )
        print(json.dumps({"complete": True, "mode": args.mode, "resources": resources}), flush=True)
    except BaseException:
        write_json(
            args.out / "failure.json",
            {
                "error": traceback.format_exc(),
                "failed_utc": now(),
                "resources": resources,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
