"""Render saved paired author-versus-selected PushT cases without new simulation."""

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
    args = parser.parse_args()
    out = args.root
    report = json.loads((out / "report.json").read_text())
    assert report["audited"]
    selected = report["selected"]["id"]
    cases = json.loads((out / "panel.json").read_text())["cases"]
    dest = out / "media"
    dest.mkdir(exist_ok=False)
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    font = ImageFont.truetype(str(font_path), 17)
    small = ImageFont.truetype(str(font_path), 14)
    for condition in ("hazard", "control"):
        gallery = Image.new("RGB", (1024, 640 * 6), "white")
        for index, case in enumerate(c for c in cases if c["split"] == "evaluation"):
            rows = []
            traces = []
            for config in ("authors", selected):
                folder = out / "evaluation" / config / case["id"] / condition
                rows.append(json.loads((folder / "result.json").read_text()))
                with np.load(folder / "trace.npz", allow_pickle=False) as arrays:
                    traces.append(dict(arrays))
            tint = overlay(rows[0]["case"])
            movie = []
            for t in range(max(len(trace["frames"]) for trace in traces)):
                frame = Image.new("RGB", (1024, 640), "white")
                draw = ImageDraw.Draw(frame)
                for col, (label, row, trace) in enumerate(
                    zip(("Authors preset", "Selected: " + selected), rows, traces)
                ):
                    x = col * 512
                    i = min(t, len(trace["frames"]) - 1)
                    draw.text((x + 8, 6), case["id"] + " | " + condition, font=font, fill="black")
                    draw.text((x + 8, 29), label, font=small, fill="black")
                    frame.paste(
                        panel(trace["frames"][i], tint, trace["states"], i, row["goal_state"]),
                        (x, 52),
                    )
                    draw.text(
                        (x + 8, 570),
                        f"Safe completion: {row['safe_completion']} | {row['actions']} actions",
                        font=small,
                        fill="black",
                    )
                    draw.text(
                        (x + 8, 592),
                        f"Margin breach: {row['margin_breach'] or row['arena_margin_breach']}",
                        font=small,
                        fill="black",
                    )
                draw.text(
                    (8, 618),
                    "Pusher safety only. Purple: actual goal. Green: fixed marker. "
                    "Amber: body + 6 px exclusion.",
                    font=small,
                    fill="black",
                )
                movie.append(np.asarray(frame))
            gallery.paste(Image.fromarray(movie[-1]), (0, index * 640))
            movie.extend([movie[-1]] * 10)
            imageio.mimsave(
                dest / (case["id"] + "-" + condition + ".mp4"),
                movie,
                fps=10,
                codec="libx264",
                quality=8,
                macro_block_size=2,
            )
        gallery.save(dest / (condition + "-comparison.png"))
    print("Rendered all twelve paired reserved comparisons", flush=True)


if __name__ == "__main__":
    main()
