"""Prepare held-out TwoRoom cases only after all three selection locks are sealed.

Exact v3 candidate admission/gallery routines are reused. This entry point is new;
no final metadata, pixels, resets or model inputs are accessed until lock validation.
"""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_proposal_pilot import write_json
from helpers.lewm_screen import image_stats
from helpers.portable_env import NativeEnv


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


from helpers.lewm_navigation_final_admission import validate_final_opening
from prepare_lewm_navigation_cases import candidate_pair, gallery


def prepare(protocol, out, protocol_path):
    opening = validate_final_opening(protocol, ROOT)
    if os.environ.get("RUNPOD_POD_ID") not in protocol["allowed_pod_ids"]:
        raise RuntimeError("Only the declared rented navigation host may prepare cases")
    started = time.perf_counter()
    if out.exists():
        raise FileExistsError("Retain prior preparations; use a fresh output namespace")
    out.mkdir(parents=True)
    (out / "inputs").mkdir()
    (out / "quality").mkdir()
    sources = protocol["source_files"]
    actual_sources = {name: digest(ROOT / name) for name in sources}
    if protocol.get("source_sha256") and actual_sources != protocol["source_sha256"]:
        raise RuntimeError("Source differs from the prospectively frozen preparation")
    write_json(
        out / "source-freeze.json",
        {
            "protocol_sha256": digest(protocol_path),
            "sources": actual_sources,
            "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "final_opening": opening,
        },
    )
    data = Path(protocol["dataset_path"])
    receipt_path = data.with_suffix(data.suffix + ".receipt.json")
    acquisition = json.loads(receipt_path.read_text())
    if (
        acquisition["archive_sha256"] != protocol["dataset"]["archive_sha256"]
        or acquisition["dataset_revision"] != protocol["dataset"]["revision"]
    ):
        raise RuntimeError("Shared public data acquisition differs from the frozen source")
    if data.stat().st_size != acquisition["h5_bytes"]:
        raise RuntimeError("Shared HDF5 size differs from its verified extraction")
    resources = collections.Counter()
    resources["dataset_hash_bytes_read"] += data.stat().st_size
    write_json(out / "resources.json", dict(resources))
    hash_started = time.perf_counter()
    actual_data_hash = digest(data)
    resources["dataset_hash_wall_seconds"] += time.perf_counter() - hash_started
    write_json(out / "resources.json", dict(resources))
    if actual_data_hash != acquisition["h5_sha256"]:
        raise RuntimeError("Shared HDF5 changed after acquisition")
    if digest(ROOT / protocol["motion_model"]) != protocol["motion_model_sha256"]:
        raise RuntimeError("Frozen learned motion model changed")
    head = torch.load(ROOT / protocol["motion_model"], map_location="cpu", weights_only=False)
    if (
        not head["calibration"]["ready"]
        or head["calibration"]["allowance"] != protocol["geometry"]["allowance"]
    ):
        raise RuntimeError("Case admission allowance differs from the released head")
    final = set(protocol["opened_final_episode_ids"])
    memory = set(protocol["memory_episode_ids"])
    pools = protocol["prepare_pools"]
    selected = [ep for group in pools.values() for ep in group]
    if len(set(selected)) != len(selected) or set(selected) != final or set(selected) & memory:
        raise RuntimeError(
            "Final preparation pool differs from its locked opening or overlaps memory"
        )
    metadata, cases, quality = {}, [], {}
    env = None
    failure = None
    phase = "preparing"
    try:
        resources["native_environment_constructions"] += 1
        env = NativeEnv("tworoom", resources, out / "interactions.jsonl")
        native_render = env.env._render_frame

        def counted_render(*args, **kwargs):
            resources["native_render_frame_calls"] += 1
            return native_render(*args, **kwargs)

        env.env._render_frame = counted_render
        with h5py.File(data, "r") as source:
            lengths = np.asarray(source["ep_len"][:], dtype=np.int64)
            offsets = np.asarray(source["ep_offset"][:], dtype=np.int64)
            if source["pixels"].dtype != np.uint8 or list(source["pixels"].shape[1:]) != [
                224,
                224,
                3,
            ]:
                raise RuntimeError("Public pixel contract differs from the frozen model")
            for split, episodes in pools.items():
                frames, labels = [], []
                quota = protocol.get("eligible_quota_per_condition")
                condition_counts = collections.Counter()
                for episode_index, episode in enumerate(episodes):
                    if quota and all(
                        condition_counts[name] >= count for name, count in quota.items()
                    ):
                        for untouched in episodes[episode_index:]:
                            metadata[str(untouched)] = {
                                "split": split,
                                "condition": protocol["condition_cycle"][
                                    untouched % len(protocol["condition_cycle"])
                                ],
                                "status": "not_read_role_quota_filled",
                                "selected": None,
                                "rejected_candidates": [],
                            }
                        break
                    condition = protocol["condition_cycle"][
                        episode % len(protocol["condition_cycle"])
                    ]
                    if quota and condition_counts[condition] >= quota[condition]:
                        metadata[str(episode)] = {
                            "split": split,
                            "condition": condition,
                            "status": "not_read_condition_quota_filled",
                            "selected": None,
                            "rejected_candidates": [],
                        }
                        continue
                    if episode not in final:
                        raise RuntimeError("Episode is outside the lock-authorized final pool")
                    if dt.datetime.now(dt.timezone.utc) + dt.timedelta(
                        minutes=15
                    ) >= dt.datetime.fromisoformat(protocol["deadline_utc"]):
                        raise RuntimeError("Global preservation reserve reached")
                    if episode >= len(lengths):
                        raise RuntimeError("Frozen source episode is absent from the dataset")
                    offset, length = int(offsets[episode]), int(lengths[episode])
                    xy = np.asarray(
                        source[protocol["xy_key"]][offset : offset + length], dtype=np.float64
                    )
                    resources["dataset_xy_rows_read"] += length
                    if xy.shape != (length, 2) or not np.isfinite(xy).all():
                        raise RuntimeError("Public native XY metadata is invalid")
                    chosen, rejected = candidate_pair(xy, condition, protocol["geometry"])
                    metadata[str(episode)] = {
                        "split": split,
                        "condition": condition,
                        "episode_length": length,
                        "rejected_candidates": rejected,
                        "selected": chosen,
                        "status": "inadmissible" if chosen is None else "selected",
                    }
                    write_json(out / "admission-record.json", metadata)
                    if chosen is None:
                        continue
                    first, last = offset + chosen["source_step"], offset + chosen["goal_step"]
                    initial, goal = xy[chosen["source_step"]], xy[chosen["goal_step"]]
                    dataset_start = np.asarray(source["pixels"][first]).copy()
                    goal_image = np.asarray(source["pixels"][last]).copy()
                    resources["dataset_pixel_rows_read"] += 2
                    image_stats(dataset_start)
                    image_stats(goal_image)
                    seed = protocol["preparation_seed_base"] + episode
                    if resources["environment_resets"] >= protocol["maximum_preparation_resets"]:
                        raise RuntimeError("Frozen preparation-reset cap reached")
                    actual = env.reset(seed, initial, goal, goal_image)
                    np.testing.assert_allclose(actual, initial, atol=1e-5, rtol=0)
                    if env.success():
                        raise RuntimeError(
                            "Metadata-admitted source is already successful natively"
                        )
                    start_image = env.render()
                    image_stats(start_image)
                    difference = start_image.astype(float) - dataset_start.astype(float)
                    mae = float(np.abs(difference).mean())
                    rmse = float(np.sqrt(np.square(difference).mean()))
                    style_flag = (
                        mae > protocol["image_review"]["maximum_start_mae"]
                        or rmse > protocol["image_review"]["maximum_start_rmse"]
                    )
                    case_id = f"tworoom-final-{episode:05}-{condition}"
                    pair_path = out / "inputs" / (case_id + ".npz")
                    np.savez_compressed(
                        pair_path,
                        initial=initial,
                        goal=goal,
                        goal_image=goal_image,
                        start_image=start_image,
                        dataset_start_image=dataset_start,
                        reference_route=xy[chosen["source_step"] : chosen["goal_step"] + 1],
                    )
                    case = {
                        "case_id": case_id,
                        "split": split,
                        "source_episode": episode,
                        "source_step": chosen["source_step"],
                        "goal_step": chosen["goal_step"],
                        "source_global_row": first,
                        "goal_global_row": last,
                        "input_path": str(pair_path),
                        "input_sha256": digest(pair_path),
                        "shape": chosen["shape"],
                        "condition": condition,
                        "seed": protocol["planning_seed_base"] + episode,
                        "initial_xy": initial.tolist(),
                        "goal_xy": goal.tolist(),
                        "endpoint_clearance": chosen["endpoint_clearance"],
                        "image_start_mae": mae,
                        "image_start_rmse": rmse,
                        "image_style_flag": style_flag,
                        "information": "Goal XY is supplied only to reset, admission, original-task monitor and common optimizer fitness. The retrieval actor receives originalgoal RGB/embedding, not its XY.",
                    }
                    cases.append(case)
                    condition_counts[condition] += 1
                    quality[case_id] = {
                        "dataset_start": image_stats(dataset_start),
                        "native_start": image_stats(start_image),
                        "goal": image_stats(goal_image),
                        "start_mae": mae,
                        "start_rmse": rmse,
                        "style_flag": style_flag,
                    }
                    frames.extend([dataset_start, start_image, goal_image])
                    labels.extend(
                        [
                            f"e{episode} source data",
                            f"e{episode} native reset",
                            f"e{episode} goal data",
                        ]
                    )
                    write_json(out / "panel.json", {"cases": cases, "phase": "preparing"})
                    write_json(out / "quality" / "image-stats.json", quality)
                    write_json(out / "resources.json", dict(resources))
                    gallery(frames, labels, out / "quality" / (split + ".png"))
                    if style_flag:
                        raise RuntimeError(
                            "Dataset/native visual mismatch requires diagnosis before controller episodes"
                        )
                if quota and any(condition_counts[name] < count for name, count in quota.items()):
                    raise RuntimeError(
                        "Frozen source pool exhausted before balanced eligibility quota: " + split
                    )
        pilot = []
        phase = "awaiting_visual_review"
        write_json(
            out / "panel.json",
            {
                "phase": phase,
                "cases": cases,
                "pilot_case_ids": [c["case_id"] for c in pilot],
                "final_opening": opening,
                "inadmissible_episodes": [
                    int(k) for k, v in metadata.items() if v["status"] == "inadmissible"
                ],
                "quota_unread_episodes": [
                    int(k) for k, v in metadata.items() if v["status"].startswith("not_read_")
                ],
                "condition_counts": {
                    split: dict(
                        collections.Counter(c["condition"] for c in cases if c["split"] == split)
                    )
                    for split in pools
                },
                "opened_final_episode_ids": protocol["opened_final_episode_ids"],
                "final_pixels_read": int(resources.get("dataset_pixel_rows_read", 0)),
                "memory_episode_overlap": [],
                "dataset_sha256": acquisition["h5_sha256"],
                "dataset_revision": acquisition["dataset_revision"],
            },
        )
    except Exception:
        phase = "technical_failure"
        failure = traceback.format_exc()
    finally:
        if env is not None:
            env.close()
        write_json(out / "resources.json", dict(resources))
        write_json(out / "admission-record.json", metadata)
        write_json(
            out / "status.json",
            {
                "phase": phase,
                "error": failure,
                "eligible_cases": len(cases),
                "resources": dict(resources),
                "elapsed_seconds": time.perf_counter() - started,
                "final_pixels_read": int(resources.get("dataset_pixel_rows_read", 0)),
                "controller_episodes": 0,
                "new_model_queries": 0,
            },
        )
    if failure:
        raise RuntimeError("Case preparation failed; retained evidence at " + str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    prepare(protocol, args.out, args.protocol)


if __name__ == "__main__":
    main()
