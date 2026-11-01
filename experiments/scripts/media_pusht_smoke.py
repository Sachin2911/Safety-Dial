"""Render saved smoke-test traces only; overlays never enter the model observations."""

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymunk
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.pusht_smoke import signed_distance

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def load(folder):
    return json.loads((folder / "result.json").read_text()), dict(np.load(folder / "trace.npz"))


def overlay(case, size=512):
    y, x = np.mgrid[:size, :size]
    xy = np.stack([x, y], axis=-1) * 512 / size
    local = (xy - case["origin"]) @ np.asarray(case["basis"]).T
    distance = signed_distance(torch.tensor(local, dtype=torch.float64), case).numpy()
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[distance <= 21] = (244, 174, 55, 48)
    rgba[distance <= 0] = (215, 55, 62, 115)
    return Image.fromarray(rgba)


def panel(frame, case_overlay, states, index, goal, size=512):
    image = Image.fromarray(frame).convert("RGBA").resize((size, size))
    image = Image.alpha_composite(image, case_overlay)
    draw = ImageDraw.Draw(image)
    # The green silhouette is a fixed landmark, not the dataset-pair target.
    body = pymunk.Body(1, 1)
    body.center_of_gravity = (0, 45)
    body.angle, body.position = float(goal[4]), tuple(goal[2:4])
    purple = (135, 45, 160, 255)
    for polygon in [
        [(-60, 30), (60, 30), (60, 0), (-60, 0)],
        [(-15, 30), (-15, 120), (15, 120), (15, 30)],
    ]:
        target = [tuple(np.asarray(body.local_to_world(v)) * size / 512) for v in polygon]
        draw.line(target + [target[0]], fill=purple, width=2)
    gx, gy = np.asarray(goal[:2]) * size / 512
    radius = 15 * size / 512
    draw.ellipse((gx - radius, gy - radius, gx + radius, gy + radius), outline=purple, width=2)
    points = [tuple(p * size / 512) for p in states[: index + 1, :2]]
    if len(points) > 1:
        draw.line(points, fill=(25, 100, 205, 255), width=2)
    return image.convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads((args.root / "cases.json").read_text())
    ref, reference = load(args.root / "runs/reference")
    font = ImageFont.truetype(FONT, 19)
    small = ImageFont.truetype(FONT, 15)
    gallery = Image.new("RGB", (1536, 664), "white")
    for col, (kind, case) in enumerate(cases.items()):
        result, trace = load(args.root / "runs" / kind)
        tint = overlay(case)
        ref_score = json.loads((args.root / f"reference-rescored-{kind}.json").read_text())
        n = max(len(reference["frames"]), len(trace["frames"]))
        movie = []
        for i in range(n):
            left = min(i, len(reference["frames"]) - 1)
            right = min(i, len(trace["frames"]) - 1)
            canvas = Image.new("RGB", (1024, 674), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 12), "Unchecked reference", font=font, fill="black")
            draw.text((524, 12), f"Checked: {kind}", font=font, fill="black")
            canvas.paste(
                panel(
                    reference["frames"][left], tint, reference["states"], left, ref["goal_state"]
                ),
                (0, 45),
            )
            canvas.paste(
                panel(trace["frames"][right], tint, trace["states"], right, result["goal_state"]),
                (512, 45),
            )
            draw.text(
                (12, 567),
                f"Goal: {ref['success']} | margin breach: {ref_score['margin_breach']}",
                font=small,
                fill="black",
            )
            draw.text(
                (524, 567),
                f"Safe completion: {result['safe_completion']} | actions: {result['actions']}",
                font=small,
                fill="black",
            )
            draw.text(
                (12, 594),
                "Red: virtual hazard | amber: pusher-centre exclusion (body + 6 px)",
                font=small,
                fill="black",
            )
            draw.text(
                (12, 620),
                "Pusher safety only. T-block must still reach its goal. One seed.",
                font=small,
                fill="black",
            )
            draw.text(
                (12, 646),
                "Purple outline: actual goal | green silhouette: fixed scene marker",
                font=small,
                fill="black",
            )
            movie.append(np.asarray(canvas))
        movie.extend([movie[-1]] * 10)
        imageio.mimsave(
            args.root / f"{kind}-comparison.mp4",
            movie,
            fps=10,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        )
        if kind == "island":
            preview = [Image.fromarray(f).resize((768, 506)) for f in movie]
            preview[0].save(
                args.root / "island-comparison.gif",
                save_all=True,
                append_images=preview[1:],
                duration=100,
                loop=0,
            )
        image = panel(
            trace["frames"][-1],
            tint,
            trace["states"],
            len(trace["states"]) - 1,
            result["goal_state"],
        )
        gallery.paste(image, (col * 512, 45))
        draw = ImageDraw.Draw(gallery)
        draw.text((col * 512 + 12, 12), kind.capitalize(), font=font, fill="black")
        draw.text(
            (col * 512 + 12, 568),
            f"Safe completion: {result['safe_completion']}",
            font=font,
            fill="black",
        )
        draw.text(
            (col * 512 + 12, 600),
            f"Actions: {result['actions']} | clearance: {result['min_hazard_clearance_px']:.1f} px",
            font=small,
            fill="black",
        )
    ImageDraw.Draw(gallery).text(
        (12, 640),
        "Purple outlines: actual dataset-pair goal. Green silhouette: fixed scene marker.",
        font=font,
        fill="black",
    )
    gallery.save(args.root / "layouts-results.png")
    print("Saved three comparison videos, island preview, and final-state gallery.")


if __name__ == "__main__":
    main()
