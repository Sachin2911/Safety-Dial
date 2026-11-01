"""Qualify a linear XY readout using only the frozen observed-memory feature bank.

This diagnostic imports no environment or neural-model code. It never reads a
development/final query label, samples a controller, or modifies the cached bank.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def metrics(error):
    """Euclidean errors in native image pixels; RMSE is sqrt(mean(dx²+dy²))."""
    distance = np.linalg.norm(error, axis=1)
    return {
        "observations": len(distance),
        "euclidean_rmse_px": float(np.sqrt(np.mean(distance**2))),
        "euclidean_mean_px": float(np.mean(distance)),
        "euclidean_median_px": float(np.median(distance)),
        "euclidean_p90_px": float(np.quantile(distance, 0.90)),
        "euclidean_p95_px": float(np.quantile(distance, 0.95)),
        "euclidean_max_px": float(np.max(distance)),
        "axis_rmse_px": np.sqrt(np.mean(error**2, axis=0)).tolist(),
    }


def zero_ledger():
    return {
        "environment_starts": 0,
        "environment_resets": 0,
        "environment_actions": 0,
        "simulator_physics_steps": 0,
        "encoder_queries": 0,
        "encoder_frames": 0,
        "predictor_queries": 0,
        "predictor_steps": 0,
        "learned_safety_queries": 0,
        "controller_episodes": 0,
        "new_public_download_bytes": 0,
        "new_dataset_pixel_rows_read": 0,
    }


def qualify(protocol_path, out):
    protocol = json.loads(protocol_path.read_text())
    require(protocol["schema"] == "lewm-goal-readout-qualification-v1", "Unknown protocol")
    require(
        os.environ.get("RUNPOD_POD_ID") == protocol["execution_pod_id"],
        "Cached-feature fitting is restricted to the declared rented pod",
    )
    require(
        dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(protocol["deadline_utc"]),
        "The campaign deadline has passed",
    )
    require(not out.exists(), "Use a fresh diagnostic output namespace")
    require(
        digest(Path(__file__)) == protocol["script_sha256"],
        "Readout diagnostic source differs from its prospective freeze",
    )
    for name, expected in protocol["input_sha256"].items():
        require(digest(ROOT / name) == expected, "Frozen input changed: " + name)
    out.mkdir(parents=True)
    started = time.monotonic()
    source = {
        "protocol_sha256": digest(protocol_path),
        "protocol": protocol,
        "script_sha256": digest(Path(__file__)),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "execution_pod_id": os.environ["RUNPOD_POD_ID"],
    }
    write_json(out / "source-freeze.json", source)
    (out / "qualify_lewm_goal_readout.source.py").write_bytes(Path(__file__).read_bytes())
    try:
        run(protocol, out, started)
    except BaseException as error:
        write_json(
            out / "failure.json",
            {
                "exception": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "resources": zero_ledger(),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def run(protocol, out, started):
    bank = protocol["bank"]
    metadata = json.loads((ROOT / bank["metadata"]).read_text())
    require(metadata["embedding_dimension"] == 192, "Expected frozen 192-dimensional features")
    require(
        metadata["frozen_world_model_sha256"]
        == protocol["input_sha256"][protocol["frozen_world_model"]],
        "Bank and declared world model differ",
    )
    with np.load(ROOT / bank["observations"], allow_pickle=False) as data:
        xy = np.asarray(data["xy"], dtype=np.float64)
        episodes = np.asarray(data["episode"], dtype=np.int64)
        steps = np.asarray(data["step"], dtype=np.int64)
        global_rows = np.asarray(data["global_row"], dtype=np.int64)
        pixels = data["pixels"]
    feature = np.load(ROOT / bank["embeddings"], allow_pickle=False).astype(np.float64)
    count = protocol["expected_observations"]
    require(feature.shape == (count, 192) and xy.shape == (count, 2), "Bank shape differs")
    require(np.isfinite(feature).all() and np.isfinite(xy).all(), "Nonfinite cached values")
    require(len(set(global_rows.tolist())) == count, "Duplicate original dataset row")
    require(pixels.shape == (count, 224, 224, 3), "Frozen pixel layout changed")
    require(pixels.dtype == np.uint8, "Original pixels are no longer uint8")
    frame_index = json.loads((ROOT / bank["frame_index"]).read_text())
    frames = frame_index["frames"]
    require(len(frames) == count, "Frame index length differs")
    pixel_hashes = []
    for index, frame in enumerate(frames):
        require(
            frame["index"] == index
            and frame["episode"] == episodes[index]
            and frame["step"] == steps[index]
            and frame["global_row"] == global_rows[index],
            "Cached observation/frame metadata alignment changed",
        )
        pixel_hash = hashlib.sha256(pixels[index].tobytes()).hexdigest()
        require(pixel_hash == frame["sha256"], "Cached original frame pixels changed")
        pixel_hashes.append(pixel_hash)
    del pixels
    split_indices = {}
    split_names = np.full(count, "", dtype="U11")
    expected_episodes = set()
    for name in ("train", "calibration", "test"):
        members = protocol["split_episode_ids"][name]
        require(not (expected_episodes & set(members)), "Readout split episodes overlap")
        expected_episodes.update(members)
        require(set(members) <= set(episodes.tolist()), "A frozen split episode has no observations")
        selected = np.flatnonzero(np.isin(episodes, members))
        split_indices[name] = selected
        split_names[selected] = name
    require(set(episodes.tolist()) == expected_episodes, "Unexpected memory or query episode present")
    require(np.all(split_names != ""), "Unassigned cached observation")
    require(
        not (set(episodes.tolist()) & set(metadata["query_episode_ids"])),
        "Controller development/final query labels entered this diagnostic",
    )
    train, calibration, test = [split_indices[k] for k in ("train", "calibration", "test")]
    feature_mean = feature[train].mean(axis=0)
    observed_std = feature[train].std(axis=0, ddof=0)
    feature_scale = np.maximum(observed_std, protocol["feature_std_floor"])
    label_mean = xy[train].mean(axis=0)
    standardized_train = (feature[train] - feature_mean) / feature_scale
    centered_labels = xy[train] - label_mean
    gram = standardized_train.T @ standardized_train
    rhs = standardized_train.T @ centered_labels
    calibration_feature = (feature[calibration] - feature_mean) / feature_scale
    fits, calibration_grid = [], []
    for regularization in protocol["ridge_lambda_grid"]:
        coefficient = np.linalg.solve(gram + regularization * np.eye(192), rhs)
        calibration_prediction = calibration_feature @ coefficient + label_mean
        score = metrics(calibration_prediction - xy[calibration])
        fits.append(coefficient)
        calibration_grid.append({"lambda": regularization, **score})
    selected_index = min(
        range(len(calibration_grid)),
        key=lambda index: (
            calibration_grid[index]["euclidean_rmse_px"],
            calibration_grid[index]["lambda"],
        ),
    )
    coefficient = fits[selected_index]
    selected_lambda = protocol["ridge_lambda_grid"][selected_index]
    # Lock the calibration choice before reading any held-test prediction/error.
    write_json(
        out / "calibration-selection.json",
        {
            "chosen_lambda": selected_lambda,
            "chosen_index": selected_index,
            "grid": calibration_grid,
            "criterion": protocol["calibration_selection"],
            "held_test_evaluations_so_far": 0,
            "selected_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    prediction = np.empty_like(xy)
    for name in ("train", "calibration", "test"):
        index = split_indices[name]
        prediction[index] = ((feature[index] - feature_mean) / feature_scale) @ coefficient + label_mean
    raw_coefficient = coefficient / feature_scale[:, None]
    raw_intercept = label_mean - feature_mean @ raw_coefficient
    require(
        np.allclose(feature @ raw_coefficient + raw_intercept, prediction, atol=1e-9, rtol=0),
        "Saved raw-feature linear readout does not reconstruct its predictions",
    )
    error = prediction - xy
    by_split = {name: metrics(error[index]) for name, index in split_indices.items()}
    by_episode = {
        str(episode): {
            "split": str(split_names[episodes == episode][0]),
            **metrics(error[episodes == episode]),
        }
        for episode in sorted(expected_episodes)
    }
    np.savez_compressed(
        out / "readout.npz",
        standardized_coefficient=coefficient,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        observed_train_feature_std=observed_std,
        label_mean=label_mean,
        raw_feature_coefficient=raw_coefficient,
        raw_feature_intercept=raw_intercept,
        selected_lambda=np.asarray(selected_lambda),
    )
    write_json(
        out / "readout-coefficients.json",
        {
            "formula": "predicted_xy = frozen_embedding @ raw_feature_coefficient + raw_feature_intercept",
            "raw_feature_coefficient": raw_coefficient.tolist(),
            "raw_feature_intercept": raw_intercept.tolist(),
            "standardized_coefficient": coefficient.tolist(),
            "feature_mean_train_only": feature_mean.tolist(),
            "feature_scale_train_only": feature_scale.tolist(),
            "label_mean_train_only": label_mean.tolist(),
            "chosen_lambda": selected_lambda,
        },
    )
    np.savez_compressed(
        out / "predictions.npz",
        observed_xy=xy,
        predicted_xy=prediction,
        euclidean_error=np.linalg.norm(error, axis=1),
        episode=episodes,
        step=steps,
        global_row=global_rows,
        split=split_names,
        pixel_sha256=np.asarray(pixel_hashes, dtype="U64"),
    )
    write_json(
        out / "prediction-index.json",
        {
            "source_frame_index": bank["frame_index"],
            "rows": [
                {
                    **frame,
                    "split": str(split_names[index]),
                    "observed_xy": xy[index].tolist(),
                    "predicted_xy": prediction[index].tolist(),
                    "euclidean_error_px": float(np.linalg.norm(error[index])),
                }
                for index, frame in enumerate(frames)
            ],
        },
    )
    resources = zero_ledger()
    resources.update(
        cached_feature_rows_read=count,
        cached_label_rows_read=count,
        cached_frame_identity_rows_verified=count,
        ridge_fits=len(protocol["ridge_lambda_grid"]),
        held_test_evaluations=1,
    )
    report = {
        "verified": True,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selected_lambda": selected_lambda,
        "split_statistics": by_split,
        "episode_statistics": by_episode,
        "resources": resources,
        "cached_preparation_resources_reference": bank["complete_record"],
        "preparation_costs_are_not_repeated_or_hidden": True,
        "elapsed_seconds": time.monotonic() - started,
        "numpy_version": np.__version__,
        "fitting_dtype": "float64",
        "native_xy_units": "pixels in the original 224 by 224 TwoRoom observation arena",
        "task_success_radius_px": protocol["task_success_radius_px"],
        "prospective_promising_p90_threshold_px": protocol["promising_test_p90_px"],
        "test_p90_below_prospective_threshold": by_split["test"]["euclidean_p90_px"]
        < protocol["promising_test_p90_px"],
        "train_episode_only_fit": True,
        "train_episode_only_standardization": True,
        "calibration_chosen_before_held_test": True,
        "refit_on_calibration": False,
        "scope": protocol["interpretation"],
        "limitations": [
            "Memory episode generalization does not establish accuracy on new controller query scenes.",
            "A visual XY readout does not establish learned-model rollout accuracy or reachable safe paths.",
            "This is a common supervised diagnostic, with no evolutionary training or controller efficacy result.",
        ],
    }
    report["artifacts"] = {
        path.name: digest(path) for path in sorted(out.iterdir()) if path.is_file()
    }
    write_json(out / "report.json", report)
    print(json.dumps({"selected_lambda": selected_lambda, "split_statistics": by_split}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    out = args.out.resolve() if args.out else ROOT / protocol["output_namespace"]
    qualify(protocol_path, out)


if __name__ == "__main__":
    main()
