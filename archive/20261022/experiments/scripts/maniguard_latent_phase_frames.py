"""Show saved Jar query frames with factual phase labels; no rendering or inference."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New PNG path")
    args = parser.parse_args()
    receipt_path = args.output.with_suffix(".json")
    if args.output.suffix != ".png" or args.output.exists() or receipt_path.exists():
        parser.error("Choose a new PNG path with no existing companion receipt")
    if sorted(set(args.steps)) != args.steps or any(step < 0 for step in args.steps):
        parser.error("Frame steps must be distinct, increasing and nonnegative")
    episode = args.episode.resolve()
    job_path, verified_path = episode / "job.json", episode / "verified-receipt.json"
    job, verified = json.loads(job_path.read_text()), json.loads(verified_path.read_text())
    if job["family"] != "jar" or not verified["valid"] or not verified["full_episode"]:
        raise ValueError("Require a previously verified full Jar episode")
    trace_path = episode / "trace.jsonl"
    observations = {}
    with trace_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["type"] == "observed":
                if row["step"] in observations or row["monitor"]["valid"] is not True:
                    raise ValueError("Invalid or duplicate factual observation")
                observations[row["step"]] = row["monitor"]["observation"]
    if sorted(observations) != list(range(verified["physical"]["steps"] + 1)):
        raise ValueError("Incomplete factual phase history")
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, len(args.steps), figsize=(3 * len(args.steps), 6.2),
                             squeeze=False, layout="constrained")
    sources = []
    for column, step in enumerate(args.steps):
        state = observations[step]
        path = episode / "query_observations" / f"{step:06d}.npz"
        image_hashes = {}
        with np.load(path, allow_pickle=False) as saved:
            for row, camera in enumerate(("overview", "wrist")):
                pixels = saved[camera]
                if pixels.shape != (640, 640, 3) or pixels.dtype != np.uint8:
                    raise ValueError("Expected saved native 640-pixel RGB")
                image_hashes[camera] = hashlib.sha256(
                    np.ascontiguousarray(pixels).tobytes()).hexdigest()
                axes[row, column].imshow(pixels, interpolation="nearest")
                axes[row, column].set_axis_off()
                axes[row, column].set_title(f"{camera}, control {step}")
        closed = state["ap"]["jar_closed"]
        supported = state["ap"]["jar_on_support"]
        axes[1, column].text(.5, -.04, f"lid closed: {closed}; supported: {supported}",
                             transform=axes[1, column].transAxes, ha="center", va="top")
        sources.append({"step": step, "query_sha256": digest(path),
                        "rgb_sha256": image_hashes, "closed": closed,
                        "supported": supported, "doomed": state["doomed"]})
    fig.suptitle(episode.name + ": saved factual camera observations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    receipt = {"schema": "maniguard-jar-saved-phase-frames-v1", "episode": episode.name,
               "source_sha256": digest(Path(__file__)), "job_sha256": digest(job_path),
               "verified_receipt_sha256": digest(verified_path),
               "trace_sha256": digest(trace_path), "sources": sources,
               "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
               "output_sha256": digest(args.output), "new_simulator_controls": 0,
               "new_vla_requests": 0, "new_encoder_calls": 0,
               "scope": "Post hoc illustrative saved training frames and factual labels. "
                        "No new camera render, simulation, encoder or policy inference. "
                        "Display resampling does not alter the preserved source arrays."}
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "receipt": str(receipt_path)}))


if __name__ == "__main__":
    main()
