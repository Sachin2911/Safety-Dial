"""Replay completed learned-checker traces; overlays never enter policy observations."""

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from scripts.media_pusht_smoke import FONT, overlay, panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    font, small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 14)
    gallery = Image.new("RGB", (1536, 660), "white")
    for col, (name, folder) in enumerate(
        (
            ("island", "island-r1"),
            ("bar", "bar"),
            ("gate", "gate"),
        )
    ):
        run = args.root / folder
        result = json.loads((run / "result.json").read_text())
        with np.load(run / "episode/trace.npz", allow_pickle=False) as data:
            trace = dict(data)
        tint = overlay(result["case"])
        frames = []
        for index, frame in enumerate(trace["frames"]):
            canvas = Image.new("RGB", (512, 660), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (10, 10),
                f"{name.capitalize()} | Learned safety + frozen LeWM",
                font=font,
                fill="black",
            )
            canvas.paste(panel(frame, tint, trace["states"], index, result["goal_state"]), (0, 42))
            draw.text(
                (10, 565),
                "Native position/velocity -> learned motion -> clearance",
                font=small,
                fill="black",
            )
            draw.text(
                (10, 590),
                "6 px margin around the full pusher. T-block excluded.",
                font=small,
                fill="black",
            )
            draw.text(
                (10, 615),
                "Purple: actual goal. Green: fixed scene marker.",
                font=small,
                fill="black",
            )
            draw.text(
                (10, 640),
                f"Safe completion: {result['safe_completion']} | "
                f"{result['actions']} actions | no reference fallback",
                font=small,
                fill="black",
            )
            frames.append(np.asarray(canvas))
        frames.extend([frames[-1]] * 10)
        imageio.mimsave(
            args.out / f"{name}.mp4", frames, fps=10, codec="libx264", quality=8, macro_block_size=2
        )
        gallery.paste(Image.fromarray(frames[-1]), (col * 512, 0))
    gallery.save(args.out / "learned-pusht-results.png")
    print("Saved three learned-controlled replays and the final-state gallery")


if __name__ == "__main__":
    main()
