"""Render the saved three-arm reproduction; no new environment or model queries."""

import json
import os
from pathlib import Path
import sys


def main():
    root = Path("/workspace/Safety-Dial")
    design = json.loads((root / "design.json").read_text())
    if os.environ.get("RUNPOD_POD_ID") != design["pod_id"]:
        raise RuntimeError("Media generation belongs on the declared rented pod")
    import imageio.v2 as imageio
    import matplotlib
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(root / "current/experiments"))
    from scripts.media_pusht_smoke import overlay, panel

    report = json.loads((root / "results/report.json").read_text())
    if not report["verified"]:
        raise RuntimeError("Saved outcomes must pass audit before rendering")
    dest = root / "results/media"
    dest.mkdir(exist_ok=False)
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    font = ImageFont.truetype(str(font_path), 18)
    small = ImageFont.truetype(str(font_path), 15)
    gallery = Image.new("RGB", (1536, 3 * 650), "white")
    labels = ("Historical stack / CPU", "Current stack / CPU", "Current stack / GPU")
    for row, case in enumerate(design["case_order"]):
        inputs, results = [], []
        for arm in design["arm_order"]:
            folder = root / "results" / arm / case
            with np.load(folder / "episode/trace.npz", allow_pickle=False) as data:
                inputs.append(dict(data))
            results.append(json.loads((folder / "result.json").read_text()))
        tint = overlay(results[0]["case"])
        movie = []
        for step in range(max(len(d["frames"]) for d in inputs)):
            frame = Image.new("RGB", (1536, 650), "white")
            draw = ImageDraw.Draw(frame)
            for col, (label, data, result) in enumerate(zip(labels, inputs, results)):
                x = col * 512
                draw.text((x + 8, 6), case.capitalize() + " | " + label, font=font, fill="black")
                index = min(step, len(data["frames"]) - 1)
                frame.paste(
                    panel(data["frames"][index], tint, data["states"], index, result["goal_state"]),
                    (x, 34),
                )
                draw.text(
                    (x + 8, 558),
                    f"Safe completion {result['safe_completion']} | {result['actions']} actions",
                    font=small,
                    fill="black",
                )
                draw.text(
                    (x + 8, 582),
                    "Exact hand-set planning budget; frozen LeWM",
                    font=small,
                    fill="black",
                )
                draw.text(
                    (x + 8, 606),
                    "Pusher-only safety, 6 px margin; purple goal",
                    font=small,
                    fill="black",
                )
                draw.text(
                    (x + 8, 628),
                    "Known scene and seed; no tuning or training",
                    font=small,
                    fill="black",
                )
            movie.append(np.asarray(frame))
        gallery.paste(Image.fromarray(movie[-1]), (0, row * 650))
        movie.extend([movie[-1]] * 10)
        imageio.mimsave(
            dest / (case + ".mp4"), movie, fps=10, codec="libx264", quality=8, macro_block_size=2
        )
    gallery.save(dest / "comparison.png")
    print("Saved three paired reproduction replays and the nine-outcome gallery", flush=True)


if __name__ == "__main__":
    main()
