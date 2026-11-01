"""Create screen replays solely from retained frames and trajectories."""

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_screen import distance
from scripts.media_pusht_smoke import overlay, panel


def load(out, row):
    folder = out / "episodes" / row["key"]
    if row["environment"] == "pusht":
        folder = folder / "episode"
    with np.load(folder / "trace.npz", allow_pickle=False) as data:
        return dict(data)


def cube_map(row, trace, action, size=224):
    points = trace["dense"]
    centre = np.asarray(row["shape"].get("centre", points.mean(0)))
    lo = np.minimum(points.min(0), centre - 0.32)
    hi = np.maximum(points.max(0), centre + 0.32)
    span = max(hi - lo)
    middle = (hi + lo) / 2
    lo, hi = middle - span / 2, middle + span / 2
    y, x = np.mgrid[:size, :size]
    coords = np.stack([x / (size - 1), 1 - y / (size - 1)], -1) * (hi - lo) + lo
    clear = distance(coords, row["shape"], 0.005)
    pixels = np.full((size, size, 3), 245, dtype=np.uint8)
    pixels[clear <= 0.01] = [250, 222, 160]
    pixels[clear <= -0.005] = [224, 104, 104]
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    p = (points[: action * 25 + 1] - lo) / (hi - lo) * (size - 1)
    p[:, 1] = size - 1 - p[:, 1]
    if len(p) > 1:
        draw.line([tuple(v) for v in p], fill=(35, 95, 205), width=2)
    if len(p):
        x, y = p[-1]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(20, 60, 170))
    draw.text((4, 4), "Pinch XY, virtual constraints", fill="black")
    draw.text((4, size - 16), "Object goal scored separately", fill="black")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    audit = json.loads((out / "audit.json").read_text())
    assert audit["verified"]
    state = json.loads((out / "state.json").read_text())
    jobs = state["jobs"]
    media = out / "media"
    media.mkdir(exist_ok=False)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    provenance = []
    for env in ("pusht", "cube"):
        native = sorted(
            (
                r
                for r in jobs.values()
                if r["environment"] == env and r["category"] == "scored" and r["role"] == "native"
            ),
            key=lambda r: (r["pair_index"], r["kind"], r["seed"]),
        )
        gallery = Image.new("RGB", (896, max(1, (len(native) + 1) // 2) * 288), "white")
        for index, left in enumerate(native):
            right = jobs.get(left["key"].replace("_native_", "_untrained_"))
            rows = [left] + ([right] if right else [])
            traces = [load(out, row) for row in rows]
            tints = [overlay(row["shape"]) for row in rows] if env == "pusht" else []
            name = f"{env}-p{left['pair_index']:03}-{left['kind']}-s{left['seed']}.mp4"
            count = max(len(t["frames"]) for t in traces)
            writer = imageio.get_writer(media / name, fps=10, macro_block_size=16)
            try:
                for i in range(count):
                    canvas = Image.new("RGB", (896, 576), "white")
                    draw = ImageDraw.Draw(canvas)
                    draw.text(
                        (8, 5),
                        f"Development screen | {env} | pair {left['pair_index']} | {left['kind']} | seed {left['seed']}",
                        fill="black",
                        font=font,
                    )
                    for col, (row, trace) in enumerate(zip(rows, traces, strict=True)):
                        j = min(i, len(trace["frames"]) - 1)
                        action = (
                            int(trace["frame_actions"][j])
                            if env == "cube"
                            else min(j, row["actions"])
                        )
                        draw.text(
                            (448 * col + 8, 29),
                            "Native SafeCEM" if col == 0 else "Frozen untrained adapter",
                            fill="black",
                            font=font,
                        )
                        if env == "pusht":
                            view = panel(
                                trace["frames"][j],
                                tints[col],
                                trace["states"],
                                action,
                                row["goal_state"],
                                size=512,
                            ).resize((448, 448))
                            canvas.paste(view, (448 * col, 56))
                            text = "Saved native image at every action"
                        else:
                            canvas.paste(
                                Image.fromarray(trace["frames"][j]).resize((336, 336)),
                                (448 * col, 56),
                            )
                            canvas.paste(
                                cube_map(row, trace, action, 224).resize((112, 112)),
                                (448 * col, 392),
                            )
                            canvas.paste(
                                Image.fromarray(trace["goal_image"]).resize((112, 112)),
                                (448 * col + 224, 392),
                            )
                            text = (
                                "Fresh native frame"
                                if trace["frame_fresh"][j]
                                else "Held frame between policy replans"
                            ) + " | XY safety map"
                        draw.text(
                            (448 * col + 6, 510),
                            f"Safe {row['safe_completion']} | unfinished {row['unfinished']} | actions {row['actions']}",
                            fill="black",
                            font=small,
                        )
                        draw.text(
                            (448 * col + 6, 531),
                            f"Violation {row['margin_breach'] or row['arena_margin_breach']} | action {action}",
                            fill="black",
                            font=small,
                        )
                        draw.text((448 * col + 6, 551), text, fill="black", font=small)
                    if not right:
                        draw.text(
                            (464, 80),
                            "Adapter arm closed by native readiness gate",
                            fill="black",
                            font=font,
                        )
                        canvas.paste(
                            Image.fromarray(traces[0]["goal_image"]).resize((336, 336)), (504, 120)
                        )
                        draw.text((510, 470), "Recorded goal image", fill="black", font=font)
                    writer.append_data(np.asarray(canvas))
            finally:
                writer.close()
            gallery.paste(canvas.resize((448, 288)), ((index % 2) * 448, (index // 2) * 288))
            provenance.append(
                {
                    "video": name,
                    "environment": env,
                    "keys": [r["key"] for r in rows],
                    "frames": count,
                    "fps": 10,
                    "paired": right is not None,
                    "saved_frames_per_arm": [len(t["frames"]) for t in traces],
                }
            )
        gallery.save(media / (env + "-gallery.png"))
    (media / "provenance.json").write_text(
        json.dumps(
            {
                "replays": provenance,
                "new_simulator_interactions": 0,
                "new_frozen_model_queries": 0,
                "overlays_enter_model_inputs": False,
                "cube_capture": "Fresh at replans and final, held in between. XY map is computed from saved dense pinch trajectories, not a projection onto the camera image.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"replays": len(provenance), "paired": sum(r["paired"] for r in provenance)}))


if __name__ == "__main__":
    main()
