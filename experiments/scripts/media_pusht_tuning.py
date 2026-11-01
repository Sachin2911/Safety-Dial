"""Replay every reserved case in the audited preset-versus-selected comparison."""

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from scripts.media_pusht_smoke import overlay, panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "evaluation"), default="evaluation")
    args = parser.parse_args()
    report = json.loads((args.root / "report.json").read_text())
    if not report["audited"]:
        raise ValueError("Audit all traces before rendering comparison videos")
    selected = report["selected_configuration"]["id"]
    inputs = json.loads((args.root / "prepared-panel.json").read_text())
    cases = [c for c in inputs["cases"] if c["split"] == args.split]
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    title, text = ImageFont.truetype(str(font_path), 18), ImageFont.truetype(str(font_path), 15)
    destination = args.root / ("media" if args.split == "evaluation" else "media-development")
    destination.mkdir(exist_ok=False)
    gallery = Image.new("RGB", (1024, 640 * len(cases)), "white")
    for case_index, case in enumerate(cases):
        rows, traces = [], []
        for config in ("current", selected):
            folder = args.root / args.split / config / case["id"]
            rows.append(json.loads((folder / "result.json").read_text()))
            with np.load(folder / "trace.npz", allow_pickle=False) as arrays:
                traces.append(dict(arrays))
        tint = overlay(case["case"])
        movies = []
        for index in range(max(len(trace["frames"]) for trace in traces)):
            canvas = Image.new("RGB", (1024, 640), "white")
            draw = ImageDraw.Draw(canvas)
            for column, (label, row, trace) in enumerate(
                zip(("Current preset", selected), rows, traces)
            ):
                frame = min(index, len(trace["frames"]) - 1)
                x = column * 512
                draw.text((x + 10, 8), f"{case['id']} | {label}", font=title, fill="black")
                canvas.paste(
                    panel(trace["frames"][frame], tint, trace["states"], frame, row["goal_state"]),
                    (x, 40),
                )
                draw.text(
                    (x + 10, 558),
                    f"Safe completion: {row['safe_completion']} | {row['actions']} actions",
                    font=text,
                    fill="black",
                )
                draw.text(
                    (x + 10, 582),
                    f"Margin breach: {row['margin_breach'] or row['arena_margin_breach']}",
                    font=text,
                    fill="black",
                )
            draw.text(
                (10, 614),
                "Red: hazard. Amber: full-pusher exclusion. Purple: task goal. "
                "Green: fixed scene marker.",
                font=text,
                fill="black",
            )
            movies.append(np.asarray(canvas))
        gallery.paste(Image.fromarray(movies[-1]), (0, case_index * 640))
        movies.extend([movies[-1]] * 10)
        imageio.mimsave(
            destination / f"{case['id']}-comparison.mp4",
            movies,
            fps=10,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        )
    gallery.save(destination / f"{args.split}-comparison.png")
    print(f"Saved all {len(cases)} paired {args.split} replays and final-state gallery")


if __name__ == "__main__":
    main()
