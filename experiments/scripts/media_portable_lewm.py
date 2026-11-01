"""Render recorded body trajectories and task frames; no simulator or model calls."""

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.portable_motion import clearance, shape_constraints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root
    report = json.loads((out / "report.json").read_text())
    panel = json.loads((out / "panel.json").read_text())
    chosen = report["selection"]["selected"]["id"]
    dest = out / "media"
    dest.mkdir(exist_ok=False)
    fontpath = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    font = ImageFont.truetype(str(fontpath), 16)
    small = ImageFont.truetype(str(fontpath), 13)
    design = json.loads((ROOT / "configs/lewm_portable_20261030/design.json").read_text())
    radius = report["motion_report"]["spec"]["body_radius"]
    margin = design["margin"][report["environment"]]
    for condition in ("hazard", "control"):
        gallery = Image.new("RGB", (1024, 6 * 700), "white")
        for row, case in enumerate(panel["cases"][6:]):
            data = []
            results = []
            for name in ("authors", chosen):
                folder = out / "evaluation" / name / case["id"] / condition
                with np.load(folder / "trace.npz") as z:
                    data.append(dict(z))
                results.append(json.loads((folder / "result.json").read_text()))
            with np.load(out / "inputs" / (case["id"] + ".npz")) as z:
                reference = z["reference_route"]
            allpoints = np.concatenate([reference, *[d["dense"] for d in data]])
            lo = allpoints.min(0)
            hi = allpoints.max(0)
            span = max(float((hi - lo).max()), 0.1 if report["environment"] != "tworoom" else 40)
            centre = (lo + hi) / 2
            half_extent = np.array([480 / 310, 1.0]) * span * 0.7
            lo, hi = centre - half_extent, centre + half_extent

            def project(points):
                a = (np.asarray(points) - lo) / (hi - lo)
                if report["environment"] != "tworoom":
                    a[..., 1] = 1 - a[..., 1]
                return a * np.array([480, 310]) + np.array([16, 330])

            xx, yy = np.meshgrid(np.arange(480), np.arange(310))
            uv = np.stack([xx / 480, yy / 310], -1)
            if report["environment"] != "tworoom":
                uv[..., 1] = 1 - uv[..., 1]
            grid = torch.as_tensor(uv * (hi - lo) + lo)
            background = np.full((310, 480, 3), 255, dtype=np.uint8)
            if condition == "hazard":
                constraints = shape_constraints(case["shape"])
                actual = clearance(grid, constraints, 0).numpy()
                protected = actual - radius
                background[protected < margin] = [251, 227, 182]
                background[actual <= 0] = [184, 62, 74]
            backdrop = Image.fromarray(background)

            movies = []
            for step in range(max(len(d["frames"]) for d in data)):
                frame = Image.new("RGB", (1024, 700), "white")
                for col, (label, d, r) in enumerate(
                    zip(("Authors preset", "Selected: " + chosen), data, results)
                ):
                    tile = Image.new("RGB", (512, 700), "white")
                    td = ImageDraw.Draw(tile)
                    td.text((8, 5), case["id"] + " | " + condition, font=font, fill="black")
                    td.text((8, 26), label, font=small, fill="black")
                    idx = min(step, len(d["frames"]) - 1)
                    tile.paste(Image.fromarray(d["frames"][idx]).resize((256, 256)), (0, 48))
                    tile.paste(Image.fromarray(d["goal_image"]).resize((256, 256)), (256, 48))
                    td.text((8, 307), "Observed frame", font=small, fill="black")
                    td.text((264, 307), "Task goal frame", font=small, fill="purple")
                    td.rectangle((16, 330, 496, 640), outline="gray")
                    tile.paste(backdrop, (16, 330))
                    reference_points = project(reference)
                    td.line([tuple(p) for p in reference_points], fill="#999999", width=2)
                    # States give action-boundary body positions; dense trace is retained raw.
                    points = project(d["states"][: min(step + 1, len(d["states"])), -2:])
                    if len(points) > 1:
                        td.line([tuple(p) for p in points], fill="#1d826b", width=3)
                    for point, colour in ((points[0], "black"), (points[-1], "#1d826b")):
                        x, y = point
                        td.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colour)
                    td.text(
                        (8, 648),
                        f"Safe completion: {r['safe_completion']} | actions {r['actions']} | breach {r['margin_breach']}",
                        font=small,
                        fill="black",
                    )
                    td.text(
                        (8, 670),
                        "XY: red hazard, amber body+margin, grey reference, green actual.",
                        font=small,
                        fill="black",
                    )
                    frame.paste(tile, (col * 512, 0))
                movies.append(np.asarray(frame))
            gallery.paste(Image.fromarray(movies[-1]), (0, row * 700))
            movies.extend([movies[-1]] * 10)
            imageio.mimsave(
                dest / (case["id"] + "-" + condition + ".mp4"),
                movies,
                fps=10,
                codec="libx264",
                quality=8,
                macro_block_size=2,
            )
        gallery.save(dest / (condition + "-comparison.png"))
    print("Saved all paired reserved replays", flush=True)


if __name__ == "__main__":
    main()
