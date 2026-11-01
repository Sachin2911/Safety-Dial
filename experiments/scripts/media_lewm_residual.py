"""Render clearly labelled development comparisons from saved frames only."""

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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text())
    audit = json.loads((args.out / "audit.json").read_text())
    if not audit["verified"]:
        raise RuntimeError("Complete saved-evidence audit required")
    diagnostic = audit["phase"] == "influence_gate_failed"
    stage = "diagnostic" if diagnostic else "selection"
    arms = ("zero", "untrained") if diagnostic else ("zero", "untrained", "cma", "random")
    cases = design["diagnostic"] if diagnostic else design["selection"]
    media = args.out / "media"
    media.mkdir(exist_ok=False)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    gallery = Image.new("RGB", (1024, len(cases) * 320), "white")
    provenance = []
    for index, (case, seed) in enumerate(cases):
        for arm in arms:
            rows, traces = [], []
            for selected in ("native", arm):
                folder = args.out / "episodes" / f"{stage}_{selected}_{case}_s{seed}"
                rows.append(json.loads((folder / "result.json").read_text()))
                with np.load(folder / "episode/trace.npz", allow_pickle=False) as data:
                    traces.append(dict(data))
            name = f"{stage}-{arm}-{case}-s{seed}.mp4"
            writer = imageio.get_writer(media / name, fps=10, macro_block_size=16)
            count = max(len(t["frames"]) for t in traces)
            try:
                for frame_index in range(count):
                    canvas = Image.new("RGB", (1024, 640), "white")
                    draw = ImageDraw.Draw(canvas)
                    draw.text(
                        (12, 8),
                        f"Development {stage} | {case} | seed {seed} | known scene",
                        fill="black",
                        font=font,
                    )
                    for col, (row, trace) in enumerate(zip(rows, traces, strict=True)):
                        k = min(frame_index, len(trace["frames"]) - 1)
                        label = "Native SafeCEM" if col == 0 else "Residual: " + arm
                        draw.text((512 * col + 12, 36), label, fill="black", font=font)
                        view = panel(
                            trace["frames"][k],
                            overlay(row["case"]),
                            trace["states"],
                            k,
                            row["goal_state"],
                        )
                        canvas.paste(view, (512 * col, 66))
                        text = (
                            f"Complete safely: {row['safe_completion']} | Actions: {row['actions']}"
                        )
                        draw.text((512 * col + 8, 584), text, fill="black", font=font)
                        draw.text(
                            (512 * col + 8, 610),
                            f"Violation: {row['margin_breach'] or row['arena_margin_breach']} | Unfinished: {row['unfinished']} | Action: {k}",
                            fill="black",
                            font=font,
                        )
                    writer.append_data(np.asarray(canvas))
            finally:
                writer.close()
            if arm == ("untrained" if diagnostic else "cma"):
                gallery.paste(canvas.resize((512, 320)), (0, index * 320))
                ImageDraw.Draw(gallery).text(
                    (528, index * 320 + 15),
                    f"{case}, seed {seed}\nNative safe: {rows[0]['safe_completion']}\n{arm} safe: {rows[1]['safe_completion']}\n"
                    f"Actions: {rows[0]['actions']} / {rows[1]['actions']}\nDevelopment {stage}; not reserved validation",
                    fill="black",
                    font=font,
                )
            provenance.append(
                {
                    "video": name,
                    "frames": count,
                    "fps": 10,
                    "stage": stage,
                    "native_vs": arm,
                    "saved_frames_per_arm": [len(t["frames"]) for t in traces],
                }
            )
    gallery.save(media / "gallery.png")
    (media / "provenance.json").write_text(
        json.dumps(
            {
                "replays": provenance,
                "new_simulator_interactions": 0,
                "new_frozen_model_queries": 0,
                "overlays_enter_model_inputs": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"paired_replays": len(provenance), "media": str(media)}))


if __name__ == "__main__":
    main()
