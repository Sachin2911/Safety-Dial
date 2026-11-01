"""Annotate saved episode frames; no simulator stepping or model queries."""

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from scripts.media_pusht_smoke import overlay, panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.episode / "result.json").read_text())
    audit = json.loads((args.episode / "audit.json").read_text())
    with np.load(args.episode / "episode/trace.npz", allow_pickle=False) as data:
        trace = dict(data)
    media = args.episode / "media"
    media.mkdir(exist_ok=False)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = (
        ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    )
    tint = overlay(result["case"])
    summary = [
        "Known engineering case",
        "Adapter: seeded, untrained",
        "Optimizer updates: 0",
        "",
        f"Safe completion: {bool(audit['safe_completion'])}",
        f"Violation: {bool(audit['violation'])}",
        f"Unfinished: {bool(audit['unfinished'])}",
        f"Actions: {audit['actions']}",
        f"Replans: {audit['replans']}",
        f"Direct proposal choices: {audit['adapter_direct_selections']}",
        "",
        "Red: virtual hazard",
        "Orange: pusher margin",
        "Purple: task goal",
        "Blue: executed path",
    ]
    gallery = Image.new("RGB", (1536, 592), "white")
    selected = [0, (len(trace["frames"]) - 1) // 2, len(trace["frames"]) - 1]
    writer = imageio.get_writer(media / "episode.mp4", fps=10, macro_block_size=16)
    try:
        for i, frame in enumerate(trace["frames"]):
            canvas = Image.new("RGB", (832, 592), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 10),
                "Proposal adapter + SafeCEM | Engineering integration",
                fill="black",
                font=font,
            )
            view = panel(frame, tint, trace["states"], i, result["goal_state"])
            canvas.paste(view, (0, 48))
            for row, text in enumerate(summary):
                draw.text((524, 52 + row * 29), text, fill="black", font=font)
            draw.text(
                (12, 565),
                f"Executed actions: {i} | Hazards are replay overlays only",
                fill="black",
                font=font,
            )
            writer.append_data(np.asarray(canvas))
            for column, index in enumerate(selected):
                if i == index:
                    gallery.paste(view, (column * 512, 48))
                    ImageDraw.Draw(gallery).text(
                        (column * 512 + 12, 12), f"Action {i}", fill="black", font=font
                    )
    finally:
        writer.close()
    ImageDraw.Draw(gallery).text(
        (12, 565),
        "Seeded untrained proposal adapter; one episode; no baseline comparison",
        fill="black",
        font=font,
    )
    gallery.save(media / "gallery.png")
    (media / "provenance.json").write_text(
        json.dumps(
            {
                "saved_frames": len(trace["frames"]),
                "new_simulator_interactions": 0,
                "new_frozen_model_queries": 0,
                "overlays_enter_model_inputs": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"media": str(media), "frames": len(trace["frames"])}))


if __name__ == "__main__":
    main()
