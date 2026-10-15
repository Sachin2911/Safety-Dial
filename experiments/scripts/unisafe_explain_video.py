"""Render recorded sensor frames with synchronized physical-state and action labels."""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-episode", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (args.episode / "steps.jsonl").read_text().splitlines()]
    cameras = np.load(args.episode / "observations.npz")["cameras"]
    first_filter = next((i for i, row in enumerate(rows) if row["intervened"]), None)
    openings = [
        i
        for i in range(1, len(rows))
        if rows[i]["proposed_action"][-1] >= 0
        and rows[i - 1]["proposed_action"][-1] < 0
        and (first_filter is None or i <= first_filter)
    ]
    opening = openings[-1] if openings else None
    block_positions = np.asarray([row["after"]["cube_positions"][1] for row in rows])
    stationary_after = next(
        (i for i in range(len(rows)) if np.linalg.norm(np.ptp(block_positions[i:], axis=0)) < 1e-5),
        len(rows),
    )
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font, small = ImageFont.truetype(font_path, 22), ImageFont.truetype(font_path, 17)
    movie = args.output / "explained_rollout.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "768x576",
        "-r",
        "20",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "medium",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(movie),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for i, views in enumerate(cameras):
            canvas = Image.new("RGB", (768, 576), "#101722")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (14, 8),
                f"Recorded filtered episode | simulation time {i * 0.05:5.2f} s",
                font=font,
                fill="white",
            )
            draw.text((14, 42), "Fixed front camera", font=small, fill="#cbd5e1")
            draw.text((398, 42), "Wrist camera moves with the robot", font=small, fill="#cbd5e1")
            for j in range(2):
                canvas.paste(
                    Image.fromarray(views[j]).resize((384, 384), Image.Resampling.NEAREST),
                    (384 * j, 70),
                )
            if i:
                row = rows[i - 1]
                intervention = row["intervened"]
                state = row["after"]
                action = row["executed_normalized_action"]
                color = "#ffb454" if intervention else "#8cc7ff"
                draw.text(
                    (14, 466),
                    "Action: "
                    + ("SAFETY BACKUP (filter=1)" if intervention else "TASK POLICY (filter=0)"),
                    font=font,
                    fill=color,
                )
                draw.text(
                    (14, 501),
                    f"Gripper command: {'OPEN' if action[-1] >= 0 else 'CLOSE'}",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (398, 467),
                    f"Yellow block height: {100 * state['cube_positions'][1][2]:.2f} cm",
                    font=small,
                    fill="white",
                )
                draw.text(
                    (398, 501),
                    f"Block/base XY distance: {100 * state['middle_base_xy_distance_m']:.2f} cm",
                    font=small,
                    fill="white",
                )
                caption = (
                    "The block is now stationary; apparent motion in the wrist view is camera motion."
                    if i > stationary_after
                    else "Labels show the action that produced this recorded frame."
                )
                draw.text((14, 546), caption, font=small, fill="#cbd5e1")
            else:
                draw.text(
                    (14, 480),
                    "Reset observation. One recorded episode; no additional simulation.",
                    font=small,
                    fill="white",
                )
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Video encoding failed")
    t = np.arange(len(rows)) * 0.05
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout="constrained")
    axes[0].step(
        t, [r["proposed_action"][-1] for r in rows], where="post", label="Task gripper proposal"
    )
    axes[0].axhline(0, color="grey", linewidth=0.7)
    axes[0].set_ylabel("Gripper command\nnegative = close")
    axes[1].plot(t, [100 * r["before"]["cube_positions"][1][2] for r in rows], label="Block height")
    axes[1].axhline(13, color="red", linestyle="--", label="Task failure threshold")
    axes[1].set_ylabel("Block height (cm)")
    axes[1].legend()
    axes[2].plot(t, [r["next_min_q"] for r in rows], label="Sampled next safety Q")
    axes[2].axhline(0.1, color="red", linestyle="--", label="Value trigger")
    axes[2].set_ylabel("Safety value")
    axes[2].legend()
    axes[3].step(t, [r["intervened"] for r in rows], where="post", label="Safety replacement")
    axes[3].set_ylabel("Filter replacement")
    axes[3].set_xlabel("Time before command (s)")
    for ax in axes:
        if opening is not None:
            ax.axvline(opening * 0.05, color="purple", linestyle=":")
        if first_filter is not None:
            ax.axvline(first_filter * 0.05, color="orange", linestyle=":")
    fig.suptitle("Purple: sustained opening command. Orange: first safety replacement.")
    fig.savefig(args.output / "timeline.png", dpi=150)
    plt.close(fig)
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "episode": str(args.episode),
                "sensor_frames": len(cameras),
                "fps": 20,
                "source": "Saved RGB observations; nearest-neighbor display enlargement; no new simulation",
                "environment_steps": 0,
                "model_queries": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print(movie)
    if args.comparison_episode:
        comparison = np.load(args.comparison_episode / "observations.npz")["cameras"]
        comparison_rows = [
            json.loads(line)
            for line in (args.comparison_episode / "steps.jsonl").read_text().splitlines()
        ]
        comparison_summary = json.loads((args.comparison_episode / "summary.json").read_text())
        comparison_movie = args.output / "gripper_comparison.mp4"
        comparison_command = command[:-1] + [str(comparison_movie)]
        with subprocess.Popen(comparison_command, stdin=subprocess.PIPE) as encoder:
            for i in range(len(cameras)):
                canvas = Image.new("RGB", (768, 576), "#101722")
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (14, 8),
                    "Matched replay: original policy / gripper-hold diagnostic",
                    font=font,
                    fill="white",
                )
                draw.text((14, 42), f"Original: {i * 0.05:.2f} s", font=small, fill="#cbd5e1")
                j = min(i, len(comparison) - 1)
                ended = i >= len(comparison) - 1
                label = (
                    f"{comparison_summary['outcome'].upper()} at {j * 0.05:.2f} s; terminal frame held"
                    if ended
                    else f"Gripper-hold probe: {j * 0.05:.2f} s"
                )
                draw.text((398, 42), label, font=small, fill="#8fe3ad" if ended else "#cbd5e1")
                for x, frame in ((0, cameras[i, 0]), (384, comparison[j, 0])):
                    canvas.paste(
                        Image.fromarray(frame).resize((384, 384), Image.Resampling.NEAREST), (x, 70)
                    )
                if i:
                    for x, row in ((14, rows[i - 1]), (398, comparison_rows[max(j - 1, 0)])):
                        grip = "OPEN" if row["executed_normalized_action"][-1] >= 0 else "CLOSE"
                        draw.text(
                            (x, 470),
                            f"Gripper: {grip}; filter={int(row['intervened'])}",
                            font=small,
                            fill="white",
                        )
                        draw.text(
                            (x, 505),
                            f"Block/base: {100 * row['after']['middle_base_xy_distance_m']:.2f} cm",
                            font=small,
                            fill="white",
                        )
                draw.text(
                    (14, 546),
                    "Right frame is held after task termination; no later stability is implied.",
                    font=small,
                    fill="#cbd5e1",
                )
                encoder.stdin.write(canvas.tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("Comparison encoding failed")
        (args.output / "comparison_provenance.json").write_text(
            json.dumps(
                {
                    "reference": str(args.episode),
                    "candidate": str(args.comparison_episode),
                    "reference_frames": len(cameras),
                    "candidate_frames": len(comparison),
                    "presentation": "Fixed front views; candidate terminal frame held and labelled",
                    "environment_steps": 0,
                    "model_queries": 0,
                },
                indent=2,
            )
            + "\n"
        )
        print(comparison_movie)


if __name__ == "__main__":
    main()
