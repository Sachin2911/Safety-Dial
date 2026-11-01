"""Paired validation replays from original saved frames; no model or simulator calls."""

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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.out / "optimizer-audit.json").read_text())
    if not audit["verified"]:
        raise RuntimeError("Saved results must pass audit first")
    media = args.out / "media"
    media.mkdir(exist_ok=False)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    gallery = Image.new("RGB", (1024, 6 * 288), "white")
    manifest = []
    for pair_index, pair in enumerate(audit["validation_pairs"]):
        case, seed = pair["case"], pair["seed"]
        rows, traces = [], []
        for arm in ("native", "selected"):
            folder = args.out / "episodes" / f"validation_{arm}_{case}_s{seed}"
            rows.append(json.loads((folder / "result.json").read_text()))
            with np.load(folder / "episode/trace.npz", allow_pickle=False) as data:
                traces.append(dict(data))
        name = f"{case}-s{seed}.mp4"
        writer = imageio.get_writer(media / name, fps=10, macro_block_size=16)
        count = max(len(t["frames"]) for t in traces)
        for frame_index in range(count):
            canvas = Image.new("RGB", (1024, 640), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 8), f"Known-scene validation | {case} | seed {seed}", fill="black", font=font
            )
            for col, (row, trace) in enumerate(zip(rows, traces, strict=True)):
                index = min(frame_index, len(trace["frames"]) - 1)
                label = "Native SafeCEM" if col == 0 else "Selected: " + audit["selection_winner"]
                draw.text((col * 512 + 12, 36), label, fill="black", font=font)
                view = panel(
                    trace["frames"][index],
                    overlay(row["case"]),
                    trace["states"],
                    index,
                    row["goal_state"],
                )
                canvas.paste(view, (col * 512, 66))
                text = f"Safe: {row['safe_completion']} | Unfinished: {row['unfinished']} | Actions: {row['actions']}"
                draw.text((col * 512 + 8, 584), text, fill="black", font=font)
                draw.text(
                    (col * 512 + 8, 610),
                    f"Frame/action {index}; hazard overlay outside policy input",
                    fill="black",
                    font=font,
                )
            writer.append_data(np.asarray(canvas))
        writer.close()
        thumbnail = canvas.crop((0, 16, 1024, 624)).resize((485, 288))
        gallery.paste(thumbnail, (0, pair_index * 288))
        ImageDraw.Draw(gallery).text(
            (500, pair_index * 288 + 12),
            f"{case}, seed {seed}\nNative safe: {rows[0]['safe_completion']} ({rows[0]['actions']} actions)\n"
            f"Selected {audit['selection_winner']}: {rows[1]['safe_completion']} ({rows[1]['actions']} actions)",
            fill="black",
            font=font,
        )
        manifest.append(
            {
                "video": name,
                "frames": count,
                "fps": 10,
                "saved_frames_per_arm": [len(t["frames"]) for t in traces],
            }
        )
    gallery.save(media / "gallery.png")
    (media / "provenance.json").write_text(
        json.dumps(
            {
                "replays": manifest,
                "new_simulator_interactions": 0,
                "new_frozen_model_queries": 0,
                "overlays_enter_model_inputs": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"media": str(media), "paired_replays": len(manifest)}))


if __name__ == "__main__":
    main()
