"""Summarize existing Part 1 traces without running a policy or simulator."""

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_part1_diagnostics import build, read, sha


def event_frames(data, run, output):
    """Decode both saved cameras for every first violation, with no physics replay."""
    from PIL import Image, ImageDraw, ImageFont

    output.mkdir(parents=True, exist_ok=False)
    ledger = read(run / "audit-final/attempt-ledger.json")["attempts"]
    paths = {r.get("episode_id"): Path(r["output"]) for r in ledger}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    results = []
    for row in data["episodes"]:
        if not row["first_violation"]:
            continue
        step = row["first_violation"]["observed_step"]
        canvas = Image.new("RGB", (768, 614), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 5),
            f"{row['episode_id']}: first monitor violation at step {step}",
            font=font,
            fill="black",
        )
        sources = []
        for camera_index, camera in enumerate(("main", "wrist")):
            source = paths[row["episode_id"]] / "episode" / f"{row['scene']}_{camera}.mp4"
            sources.append(
                {
                    "camera": camera,
                    "sha256": sha(source),
                    "path": str(source.relative_to(run.resolve())),
                }
            )
            for i, offset in enumerate((-20, 0, 20)):
                frame_index = step + offset
                raw = subprocess.check_output(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-threads",
                        "2",
                        "-i",
                        str(source),
                        "-vf",
                        f"select=eq(n\\,{frame_index})",
                        "-frames:v",
                        "1",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "png",
                        "-threads",
                        "2",
                        "-",
                    ],
                    timeout=30,
                )
                image = Image.open(io.BytesIO(raw)).convert("RGB")
                if image.size != (256, 256):
                    raise ValueError("Unexpected native video dimensions")
                y = 42 + camera_index * 286
                draw.text(
                    (i * 256 + 5, y),
                    f"{camera}, step {frame_index}, {offset / 20:+.0f}s",
                    font=font,
                    fill="black",
                )
                canvas.paste(image, (i * 256, y + 22))
        target = output / f"{row['episode_id']}.png"
        canvas.save(target)
        results.append(
            {
                "episode_id": row["episode_id"],
                "image": target.name,
                "sha256": sha(target),
                "sources": sources,
                "frame_indices": [step - 20, step, step + 20],
                "alignment": "Native initial frame 0, then one frame per action, 20 fps",
            }
        )
    return results


def plot(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = data["comparison_totals"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    held = [r["counts"]["held_closing_actions"] for r in rows]
    recognized = [r["counts"]["held_closing_heuristic_actions"] for r in rows]
    axes[0].bar(range(3), held, label="Held and closing", color="#88ccee")
    axes[0].bar(range(3), recognized, label="Recognized by heuristic", color="#4477aa")
    for i, (a, b) in enumerate(zip(recognized, held, strict=True)):
        axes[0].text(i, b + 10, f"{a}/{b}", ha="center")
    axes[0].set(
        xticks=range(3),
        xticklabels=[r["dial"] for r in rows],
        xlabel="Dial",
        ylabel="Actions, pooled across 8 episodes",
        title="Grasp heuristic coverage",
        ylim=(0, max(held) * 1.3),
    )
    axes[0].legend(fontsize=8)
    violations = [r for r in data["episodes"] if r["first_violation"]]
    for i, row in enumerate(violations):
        event = row["first_violation"]
        axes[1].barh(i, event["effective_horizon"], color="#dddddd")
        axes[1].scatter(event["action_number_in_prefix"], i, color="#cc6677", zorder=3)
    axes[1].set(
        yticks=range(len(violations)),
        yticklabels=[r["episode_id"] for r in violations],
        xlabel="Action number since latest policy query (1-based)",
        title="All 5 first violations; gray = query interval",
        xlim=(0, 17),
    )
    axes[1].invert_yaxis()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--event-frames-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.figure.exists():
        raise FileExistsError("Use fresh diagnostic output paths")
    started = time.time()
    data = build(args.run)
    data["analysis_started_at"] = started
    data["analysis_elapsed_seconds"] = time.time() - started
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plot(data, args.figure)
    data["figure_sha256"] = sha(args.figure)
    if args.event_frames_dir:
        data["event_frames"] = event_frames(data, args.run, args.event_frames_dir)
    data["total_analysis_elapsed_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: data[k]
                for k in (
                    "comparison_totals",
                    "within_generation_single_coordinate_pairs",
                    "analysis_elapsed_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
