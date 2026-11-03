"""Figures and videos from saved task-18 observations, without new simulation."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import Polygon

from analyze_lewm18 import breach

NAMES = {
    "native": "No safety check",
    "canonical": "Original SafeCEM",
    "fixed": "Tuned fixed",
    "feedback": "Feedback",
}
COLORS = {"native": "#d04a45", "canonical": "#64758c", "fixed": "#087d82", "feedback": "#8c58a8"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def read(path):
    return json.loads(Path(path).read_text())


def figure(root, kinds=("pusht", "cube")):
    summary = read(root / "analysis/summary.json")
    fig, axes = plt.subplots(
        len(kinds), 3, figsize=(11, 3 * len(kinds)), constrained_layout=True, squeeze=False
    )
    for row, kind in enumerate(kinds):
        for method in NAMES:
            cells = sorted(
                [
                    r
                    for r in summary["tables"]
                    if (r["stage"], r["env"], r["method"]) == ("H", kind, method)
                ],
                key=lambda r: r["margin"],
            )
            x = [r["margin"] * (100 if kind == "cube" else 1) for r in cells]
            for col, metric in enumerate(("safe", "margin_breach", "mean_J_safe")):
                y = [r[metric] * 100 / r["n"] if col < 2 else r[metric] for r in cells]
                axes[row, col].plot(
                    x, y,
                    marker={"native": "v", "canonical": "s", "fixed": "o", "feedback": "x"}[method],
                    linestyle="--" if method == "feedback" else "-",
                    linewidth=2.4 if method == "fixed" else 1.7,
                    markerfacecolor="white", color=COLORS[method], label=NAMES[method],
                )
        for col in range(3):
            ax = axes[row, col]
            ax.set_ylim(-2, 102 if col < 2 else 52)
            ax.set_xlabel(
                ("PushT: clearance (pixels)" if kind == "pusht" else "Cube: clearance (cm)")
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            ax.set_title(
                [
                    "Finished safely (%)",
                    "Breached requested margin (%)",
                    "Actions (unsafe/unfinished = 50)",
                ][col]
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    denominators = []
    for kind in kinds:
        cell = next(r for r in summary["tables"] if r["stage"] == "H" and r["env"] == kind)
        blocks = next(r["case_blocks"] for r in summary["contrasts"] if r["env"] == kind)
        denominators.append(f"{kind}: {cell['n']} episodes per setting, {blocks} paired cases")
    fig.text(0.5, -0.025, " | ".join(denominators), ha="center", va="top", fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"analysis/comparison.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def trajectory_gallery(root, kind, seed, margin):
    """Show every final case at the first seed without selecting attractive outcomes."""
    cases = read(root / f"final-selection-{kind}.json")["tier"]["cases_per_env"]
    for page, start in enumerate(range(0, cases, 6), 1):
        count = min(6, cases - start)
        fig, axes = plt.subplots(count, 4, figsize=(12, 2.5 * count), squeeze=False)
        for ri, ci in enumerate(range(start, start + count)):
            for col, method in enumerate(NAMES):
                name = (
                    f"H-{kind}-c{ci}-s{seed}-native"
                    if method == "native"
                    else f"H-{kind}-c{ci}-s{seed}-m2-{method}"
                )
                folder = root / "attempts" / name
                row = read(folder / "result.json")
                with np.load(folder / "trace.npz") as data:
                    states = np.vstack(
                        [
                            row["initial_state"],
                            data["states"].reshape(-1, len(row["initial_state"])),
                        ]
                    )
                    hit = breach(data["dense_xy"], row["job"]["case"], margin)
                case, ax = row["job"]["case"], axes[ri, col]
                for delta, color in ((margin, "#f7dfae"), (0, "#efbbbb")):
                    for x0, x1, y0, y1 in case["boxes"]:
                        corners = np.array(
                            [
                                [x0 - delta, y0 - delta],
                                [x1 + delta, y0 - delta],
                                [x1 + delta, y1 + delta],
                                [x0 - delta, y1 + delta],
                            ]
                        )
                        ax.add_patch(
                            Polygon(
                                corners @ np.asarray(case["basis"]) + case["origin"],
                                color=color,
                                linewidth=0,
                            )
                        )
                obj = states[:, 2:4] if kind == "pusht" else states[:, 3:5]
                goal = np.asarray(row["goal"])[2:4] if kind == "pusht" else row["goal"][:2]
                ax.plot(states[:, 0], states[:, 1], color="#246ab5", linewidth=1.6)
                ax.plot(obj[:, 0], obj[:, 1], color="#333333", linewidth=1.1, linestyle="--")
                ax.scatter(*states[0, :2], color="#246ab5", marker="o", s=12)
                ax.scatter(*states[-1, :2], color="#246ab5", marker="s", s=12)
                ax.scatter(*goal, facecolors="none", edgecolors="#188144", s=75, linewidths=1.5)
                if kind == "pusht":
                    ax.scatter(*row["goal"][:2], color="#188144", marker="+", s=65)
                    ax.set(xlim=(0, 512), ylim=(512, 0))
                else:
                    ax.set(xlim=(0.21, 0.64), ylim=(-0.39, 0.39))
                status = (
                    "finished safely"
                    if row["success"] and not hit
                    else (
                        "finished, breached"
                        if row["success"]
                        else "unfinished" + (", breached" if hit else "")
                    )
                )
                ax.set_title(f"{NAMES[method]} | {row['actions']} actions\n{status}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(case["id"], fontsize=10)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=7)
        unit = f"{margin:g} px" if kind == "pusht" else f"{margin * 100:g} cm"
        fig.suptitle(
            f"{'PushT' if kind == 'pusht' else 'Cube object moving'} | Clearance {unit} | Seed {seed}\n"
            f"Final cases {start + 1} to {start + count} of {cases} at the first seed; "
            "no case selected for its outcome",
            fontsize=13,
        )
        fig.text(
            0.5,
            0.012,
            "Red: forbidden | Amber: extra clearance | Blue: pusher/gripper | "
            "Dashed: object | Green circle: object goal; plus: pusher goal\n"
            "Paths use saved states. Safety is scored at every "
            "physics step, including between these plotted action states.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        for suffix in ("png", "pdf"):
            fig.savefig(root / f"media/{kind}-trajectory-gallery-{page}.{suffix}", dpi=150)
        plt.close(fig)


def tile(row, trace, frames, k, margin, bounds, goal_image):
    result = Image.new("RGB", (672, 352), "#fafaf7")
    draw = ImageDraw.Draw(result)
    large = ImageFont.truetype(FONT, 18)
    small = ImageFont.truetype(FONT, 12)
    job, case = row["job"], row["job"]["case"]
    draw.text((8, 5), NAMES[job["method"]], fill="#111111", font=large)
    draw.text(
        (8, 29), "Saved images | 8 actions/sec playback, not runtime", fill="#555555", font=small
    )
    for x, label in ((8, "Observed scene"), (232, "Goal image"), (456, "Safety map (top view)")):
        draw.text((x, 48), label, fill="#333333", font=small)
    image = Image.fromarray(frames[min(k, len(frames) - 1)]).resize((224, 224))
    result.paste(image, (0, 68))
    result.paste(Image.fromarray(goal_image).resize((224, 224)), (224, 68))
    plot = Image.new("RGB", (224, 224), "#f0f3f4")
    pd = ImageDraw.Draw(plot)
    xmin, xmax, ymin, ymax = bounds

    def pixel(points):
        p = np.asarray(points)
        y = (p[..., 1] - ymin) / (ymax - ymin) * 204
        return np.stack(
            [
                (p[..., 0] - xmin) / (xmax - xmin) * 204 + 10,
                y + 10 if job["env"] == "pusht" else 214 - y,
            ],
            axis=-1,
        )

    for delta, fill in [(margin, "#f7dfae"), (0, "#efbbbb")]:
        for x0, x1, y0, y1 in case["boxes"]:
            corners = np.array(
                [
                    [x0 - delta, y0 - delta],
                    [x1 + delta, y0 - delta],
                    [x1 + delta, y1 + delta],
                    [x0 - delta, y1 + delta],
                ]
            )
            world = corners @ np.asarray(case["basis"]) + case["origin"]
            pd.polygon([tuple(p) for p in pixel(world)], fill=fill)
    states = np.vstack(
        [row["initial_state"], trace["states"].reshape(-1, len(row["initial_state"]))]
    )[: k + 1]
    grip = pixel(states[:, :2])
    if len(grip) > 1:
        pd.line([tuple(p) for p in grip], fill="#2364cd", width=2)
    obj_idx = slice(3, 5) if job["env"] == "cube" else slice(2, 4)
    obj = pixel(states[:, obj_idx])
    if len(obj) > 1:
        pd.line([tuple(p) for p in obj], fill="#5c6063", width=2)
    goal = row["goal"][:2] if job["env"] == "cube" else row["goal"][2:4]
    gx, gy = pixel(goal)
    pd.ellipse((gx - 5, gy - 5, gx + 5, gy + 5), outline="#17974b", width=2)
    if job["env"] == "pusht":
        px, py = pixel(row["goal"][:2])
        pd.line([(px - 4, py), (px + 4, py)], fill="#17974b", width=2)
        pd.line([(px, py - 4), (px, py + 4)], fill="#17974b", width=2)
    for points, fill in [(grip, "#2364cd"), (obj, "#5c6063")]:
        x, y = points[-1]
        pd.ellipse((x - 3, y - 3, x + 3, y + 3), fill=fill)
    result.paste(plot, (448, 68))
    n = min(k, row["actions"])
    dense = trace["dense_xy"][: n * (25 if job["env"] == "cube" else 10)]
    unsafe = breach(dense, case, margin)
    done = k >= row["actions"]
    status = (
        (
            "finished safely"
            if row["success"] and not unsafe
            else "goal reached, breached margin"
            if row["success"]
            else "unfinished"
        )
        if done
        else "running"
    )
    distance = (
        f" | Object-to-goal: {100 * np.linalg.norm(states[-1, 3:6] - row['goal']):.1f} cm"
        if job["env"] == "cube"
        else ""
    )
    draw.text((8, 309), f"Action {n}/50 | {status}{distance}", fill="#111111", font=small)
    draw.text(
        (8, 330),
        "Map: blue pusher, grey object, green goals"
        if job["env"] == "pusht"
        else "Map: blue gripper, grey cube, green goal. Success needs 3D object distance <= 4 cm.",
        fill="#555555",
        font=small,
    )
    return result


def paper_strip(records, margin):
    """Readable final observations and the goal image for a full-width paper figure."""
    result = Image.new("RGB", (1120, 336), "white")
    draw = ImageDraw.Draw(result)
    heading = ImageFont.truetype(FONT, 22)
    body = ImageFont.truetype(FONT, 18)
    for i, (row, trace, frames, goal_image) in enumerate(records):
        x = i * 224
        draw.text((x + 8, 7), NAMES[row["job"]["method"]], font=heading, fill="#111111")
        result.paste(Image.fromarray(frames[-1]).resize((224, 224)), (x, 40))
        hit = breach(trace["dense_xy"], row["job"]["case"], margin)
        status = (
            "Safe completion"
            if row["success"] and not hit
            else "Unsafe completion"
            if row["success"]
            else "Unfinished; breach"
            if hit
            else "Unfinished"
        )
        draw.text((x + 8, 275), status, font=body, fill="#111111")
        draw.text((x + 8, 305), f"{row['actions']} / 50 actions", font=body, fill="#444444")
    draw.text((904, 7), "Goal image", font=heading, fill="#111111")
    result.paste(Image.fromarray(records[0][3]).resize((224, 224)), (896, 40))
    goal_text = (
        "Cube target in 3D" if records[0][0]["job"]["env"] == "cube" else "Pusher + block goal"
    )
    draw.text((904, 275), goal_text, font=body, fill="#111111")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--env", choices=("cube", "pusht"))
    args = parser.parse_args()
    root = args.root
    kinds = (args.env,) if args.env else ("pusht", "cube")
    figure(root, kinds)
    media = root / "media"
    media.mkdir(exist_ok=True)
    index = []
    for kind, margin in [("pusht", 12), ("cube", 0.01)]:
        if kind not in kinds:
            continue
        selection_path = root / f"final-selection-{kind}.json"
        if not selection_path.exists():
            selection_path = root / "final-selection.json"
        seed = read(selection_path)["tier"]["seeds"][0]
        trajectory_gallery(root, kind, seed, margin)
        for ci in (0, 1):
            records = []
            for method in NAMES:
                name = (
                    f"H-{kind}-c{ci}-s{seed}-native"
                    if method == "native"
                    else f"H-{kind}-c{ci}-s{seed}-m2-{method}"
                )
                path = root / "attempts" / name
                row = read(path / "result.json")
                with np.load(path / "trace.npz") as data:
                    trace = {k: data[k] for k in data.files}
                with np.load(path / "frames.npz") as data:
                    frames = data["frames"]
                    goal_image = data["goal_frame"]
                records.append((row, trace, frames, goal_image))
            points = np.concatenate(
                [
                    np.vstack(
                        [
                            r[0]["initial_state"],
                            r[1]["states"].reshape(-1, len(r[0]["initial_state"])),
                        ]
                    )[:, :2]
                    for r in records
                ]
            )
            if kind == "pusht":
                bounds = (0, 512, 0, 512)
            else:
                pad = 0.04
                bounds = (
                    min(0.25, points[:, 0].min()) - pad,
                    max(0.6, points[:, 0].max()) + pad,
                    min(-0.25, points[:, 1].min()) - pad,
                    max(0.25, points[:, 1].max()) + pad,
                )
            name = f"{kind}-case-{ci}-high-margin"
            path = media / f"{name}.mp4"
            last = None
            with imageio.get_writer(
                path, fps=8, codec="libx264", quality=8, ffmpeg_params=["-threads", "1"]
            ) as writer:
                for k in range(max(len(r[2]) for r in records) + 8):
                    grid = Image.new("RGB", (1344, 752), "#fafaf7")
                    heading = ImageDraw.Draw(grid)
                    unit = f"{margin:g} px" if kind == "pusht" else f"{margin * 100:g} cm"
                    task_name = "PushT" if kind == "pusht" else "Cube: move the object"
                    heading.text(
                        (8, 4),
                        f"{task_name} | Clearance {unit} | Case {records[0][0]['job']['case']['id']} | Seed {seed}",
                        font=ImageFont.truetype(FONT, 19),
                        fill="#111111",
                    )
                    heading.text(
                        (8, 28),
                        "Maps: red forbidden area, amber extra clearance. Paired example, not aggregate performance.",
                        font=ImageFont.truetype(FONT, 12),
                        fill="#555555",
                    )
                    for i, (row, trace, frames, goal_image) in enumerate(records):
                        grid.paste(
                            tile(row, trace, frames, k, margin, bounds, goal_image),
                            (672 * (i % 2), 48 + 352 * (i // 2)),
                        )
                    writer.append_data(np.asarray(grid))
                    last = grid
            last.save(media / f"{name}.png")
            paper_strip(records, margin).save(media / f"{name}-paper.png")
            Image.fromarray(goal_image).save(media / f"{name}-goal.png")
            index.append(
                {
                    "environment": kind,
                    "case_index": ci,
                    "seed": seed,
                    "margin": margin,
                    "video": path.name,
                    "picture": f"{name}.png",
                    "paper_picture": f"{name}-paper.png",
                    "goal_image": f"{name}-goal.png",
                    "episodes": [r[0]["job"]["id"] for r in records],
                    "selection": "First two final cases and first seed, chosen before outcomes",
                    "note": "Saved observations, no new simulation. Playback is not planner speed. "
                    "Top-down hazard map is for the viewer, not the frozen model.",
                }
            )
    (media / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"comparison_videos": len(index)}), flush=True)


if __name__ == "__main__":
    main()
