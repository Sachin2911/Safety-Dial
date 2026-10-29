"""Render videos of task-10 Push-T episodes from the saved simulator states.

No GPU and no world model: each frame puts the saved pusher and T-block pose into the Push-T
renderer and overlays what the planner was told to avoid. Red boxes are the true pillars
(invisible to the world model), the dashed amber outline is the pillar inflated by the
clearance dial, the blue line is the pusher's path so far, and the green T and ring are the
goal pose of the block and the pusher. Episodes are chosen by fixed rules recorded in
videos/manifest.json next to the videos, so they illustrate the statistics, not replace them.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.safecem10_geometry import design, gate_boxes, inside_any, layout
from helpers.safecem10_planner import make_state

REMOTE = ROOT / "outputs/safecem10/remote"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
RED, AMBER, TRAIL, GREEN = (214, 39, 40), (225, 120, 15), (31, 90, 200), (35, 150, 60)
INK, MUTED, SURFACE = (25, 25, 25), (105, 105, 100), (252, 252, 251)
HEADER, FOOTER, SIZE = 56, 40, 512
FPS, HOLD = 8, 14


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


class Replayer:
    """The Push-T renderer driven by saved states instead of physics."""

    def __init__(self, spec):
        from stable_worldmodel.envs.pusht.env import PushT

        env = spec["environment"]
        self.env = PushT(resolution=SIZE)
        self.start = make_state(env["start_pusher"], env["start_block"], env["block_theta"])
        goal = make_state(env["goal_pusher"], env["goal_block"], env["block_theta"])
        self.env.reset(seed=0, options={"state": self.start, "goal_state": goal})
        # Draw the task's goal pose of the block, not the env's default coverage target.
        self.env.goal_pose = np.array([*env["goal_block"], env["block_theta"]], dtype=float)
        self.goal_pusher = env["goal_pusher"]

    def render(self, state):
        e = self.env
        e.agent.velocity = (0.0, 0.0)
        e.agent.position = (float(state[0]), float(state[1]))
        e.block.angle = float(state[4])
        e.block.position = (float(state[2]), float(state[3]))
        e.space.reindex_shapes_for_body(e.agent)
        e.space.reindex_shapes_for_body(e.block)
        return np.array(e.render())


def dashed_rect(draw, box, color, width=2, dash=9, gap=6):
    x0, x1, y0, y1 = box
    for a, b, horizontal in (
        ((x0, y0), (x1, y0), True),
        ((x0, y1), (x1, y1), True),
        ((x0, y0), (x0, y1), False),
        ((x1, y0), (x1, y1), False),
    ):
        start, end = (a[0], b[0]) if horizontal else (a[1], b[1])
        pos = start
        while pos < end:
            stop = min(pos + dash, end)
            if horizontal:
                draw.line([(pos, a[1]), (stop, a[1])], fill=color, width=width)
            else:
                draw.line([(a[0], pos), (a[0], stop)], fill=color, width=width)
            pos = stop + gap


def scene(rep, state, boxes, dial, trail, inside_now):
    base = Image.fromarray(rep.render(state)).convert("RGBA")
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(over)
    for x0, x1, y0, y1 in boxes:
        draw.rectangle([x0, y0, x1, y1], fill=RED + (70,), outline=RED + (255,), width=2)
    if dial:
        for x0, x1, y0, y1 in boxes:
            dashed_rect(draw, (x0 - dial, x1 + dial, y0 - dial, y1 + dial), AMBER + (255,))
    gx, gy = rep.goal_pusher
    draw.ellipse([gx - 17, gy - 17, gx + 17, gy + 17], outline=GREEN + (255,), width=3)
    if len(trail) > 1:
        draw.line([tuple(p) for p in trail], fill=TRAIL + (190,), width=3)
    for p in trail[:-1]:
        draw.ellipse([p[0] - 2.5, p[1] - 2.5, p[0] + 2.5, p[1] + 2.5], fill=TRAIL + (220,))
    if inside_now:
        px, py = trail[-1]
        draw.ellipse([px - 22, py - 22, px + 22, py + 22], outline=RED + (255,), width=4)
    return Image.alpha_composite(base, over).convert("RGB")


def panel(image, title, subtitle, step, budget, inside_steps, outcome):
    canvas = Image.new("RGB", (SIZE, HEADER + SIZE + FOOTER), SURFACE)
    canvas.paste(image, (0, HEADER))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 6), title, font=font(20, True), fill=INK)
    draw.text((12, 32), subtitle, font=font(14), fill=MUTED)
    label = f"step {step}/{budget}"
    width = draw.textlength(label, font=font(14))
    draw.text((SIZE - 12 - width, 10), label, font=font(14), fill=MUTED)
    inside = f"steps inside a pillar: {inside_steps}"
    draw.text((12, HEADER + SIZE + 10), inside, font=font(15), fill=RED if inside_steps else MUTED)
    if outcome:
        text, color = outcome
        width = draw.textlength(text, font=font(15, True))
        draw.text((SIZE - 12 - width, HEADER + SIZE + 10), text, font=font(15, True), fill=color)
    return canvas


def load(job_id):
    folder = REMOTE / "attempts" / job_id
    result = json.loads((folder / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError(f"{job_id} is not a complete episode")
    return result, np.load(folder / "states.npz")["states"]


def describe(job):
    """Short title and subtitle for a job, in the paper's vocabulary."""
    stage, cond, mode = job["stage"], job.get("condition"), job["mode"]
    planner = job["planner"]
    if mode == "penalty":
        return (
            "unconstrained planner",
            f"layout {job['layout']}, seed {job['seed']}, no safety check (lambda {job['lam']:g})",
        )
    d0 = job["clearances"][0]
    if stage == "C2":
        params = job["params"]
        return (
            f"searched setting, seed {job['seed']}",
            f"d0 {params['d0']:.0f} px, slope {params['d1']:+.1f} px/block, "
            f"horizon {planner['horizon']}, tol {planner['feasible_tol']:.1f}",
        )
    name = {
        "exact": "exact check",
        "fixed": "grounded probe check",
        "bestonly": "frozen probe, best plan",
    }.get(cond, "frozen probe check (stage B)")
    quantile = job.get("quantile")
    dial = f"d = {d0:g} px" + (f" (p{quantile} of imagination error)" if quantile else "")
    return f"{name}, {dial}", f"layout {job['layout']}, seed {job['seed']}"


def episode_frames(rep, spec, job_id):
    result, states = load(job_id)
    job = result["job"]
    boxes = gate_boxes(layout(job["layout"], spec), spec)
    dial = job["clearances"][0] if job["mode"] == "safe" else 0.0
    budget = int(spec["environment"]["eval_budget_steps"])
    title, subtitle = describe(job)
    trail = [tuple(rep.start[:2])]
    frames = [
        panel(scene(rep, rep.start, boxes, dial, trail, False), title, subtitle, 0, budget, 0, None)
    ]
    inside_steps = 0
    for t, state in enumerate(states, 1):
        trail.append((float(state[0]), float(state[1])))
        inside = bool(inside_any(state[None, :2], boxes)[0])
        inside_steps += inside
        outcome = None
        if t == len(states):
            if result["success"]:
                ok = not result["any_violation"]
                outcome = (
                    f"SUCCESS in {t} steps" + ("" if ok else ", unsafe"),
                    GREEN if ok else RED,
                )
            else:
                outcome = (f"no success in {budget} steps", MUTED)
        frames.append(
            panel(
                scene(rep, state, boxes, dial, trail, inside),
                title,
                subtitle,
                t,
                budget,
                inside_steps,
                outcome,
            )
        )
    if inside_steps != result["hazard_steps"]:
        raise ValueError(
            f"{job_id}: overlay counts {inside_steps} pillar steps, record says {result['hazard_steps']}"
        )
    meta = {
        "job_id": job_id,
        "success": bool(result["success"]),
        "hazard_steps": int(result["hazard_steps"]),
        "J": result["J"],
        "steps": len(states),
    }
    return frames, meta


def write_mp4(frames, path, fps=FPS, hold=HOLD):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=16) as w:
        for f in frames:
            w.append_data(np.asarray(f))
        for _ in range(hold):
            w.append_data(np.asarray(frames[-1]))


def write_gif(frames, path, scale=0.5, fps=FPS, hold=HOLD):
    small = [f.resize((int(f.width * scale), int(f.height * scale)), Image.LANCZOS) for f in frames]
    small += [small[-1]] * hold
    path.parent.mkdir(parents=True, exist_ok=True)
    small[0].save(
        path,
        save_all=True,
        append_images=small[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )


def grid(panel_lists, ncols, heading, scale=0.75):
    """Synchronized grid: every panel advances one step per frame; finished episodes hold."""
    pw, ph = int(SIZE * scale), int((HEADER + SIZE + FOOTER) * scale)
    nrows = (len(panel_lists) + ncols - 1) // ncols
    top = 64
    length = max(len(p) for p in panel_lists)
    frames = []
    for i in range(length):
        height = -(-(top + ph * nrows) // 16) * 16  # codec-friendly multiple of 16
        canvas = Image.new("RGB", (pw * ncols, height), SURFACE)
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 10), heading[0], font=font(22, True), fill=INK)
        draw.text((14, 38), heading[1], font=font(15), fill=MUTED)
        for k, panels in enumerate(panel_lists):
            img = panels[min(i, len(panels) - 1)].resize((pw, ph), Image.LANCZOS)
            canvas.paste(img, ((k % ncols) * pw, top + (k // ncols) * ph))
        frames.append(canvas)
    return frames


def results_by_id():
    rows = {}
    for path in (REMOTE / "attempts").glob("*/result.json"):
        row = json.loads(path.read_text())
        if row["status"] == "complete":
            rows[row["job"]["id"]] = row
    return rows


def choose_seed(rows, lay):
    """First seed where the exact check at d = 0 succeeded and the unconstrained planner
    (best-plan execution) entered a pillar; else the first exact success; else seed 200."""
    seeds = range(200, 210)
    for s in seeds:
        exact = rows.get(f"B2-{lay}-{s}-exact-d0")
        free = rows.get(f"B2-{lay}-{s}-bestlam0-lam0")
        if exact and free and exact["success"] and free["any_violation"]:
            return s, "exact d=0 succeeded and the unconstrained planner entered a pillar"
    for s in seeds:
        exact = rows.get(f"B2-{lay}-{s}-exact-d0")
        if exact and exact["success"]:
            return s, "first seed where exact d=0 succeeded"
    return 200, "fallback"


def video_sets(rows):
    dials = [0, 10, 20, 30, 40, 50, 60]
    sets = []
    for lay in ("G80", "G0"):
        seed, rule = choose_seed(rows, lay)
        free = f"B2-{lay}-{seed}-bestlam0-lam0"
        sets.append(
            {
                "name": f"dial-sweep/{lay}-seed{seed}-exact-check",
                "heading": (
                    f"Exact check, layout {lay}, seed {seed}: one episode at seven dial settings",
                    "Top left: no safety check. Red: true pillars. Dashed amber: pillars inflated "
                    "by the dial. Green: goal pose of block and pusher.",
                ),
                "jobs": [free] + [f"B2-{lay}-{seed}-exact-d{d}" for d in dials],
                "rule": rule,
                "ncols": 4,
                "gif": lay == "G80",
            }
        )
        sets.append(
            {
                "name": f"dial-sweep/{lay}-seed{seed}-grounded-probe",
                "heading": (
                    f"Grounded probe check, layout {lay}, seed {seed}: one episode at seven dial settings",
                    "The check reads the pusher from the world model's latent state (probe v2, "
                    "anchored); it completes more but can cut through a pillar.",
                ),
                "jobs": [free] + [f"B2-{lay}-{seed}-fixed-d{d}" for d in dials],
                "rule": rule,
                "ncols": 4,
            }
        )
        sets.append(
            {
                "name": f"dial-sweep/{lay}-seed{seed}-frozen-probe-stage-B",
                "heading": (
                    f"Stage B, frozen probe check, layout {lay}, seed {seed}: the original pipeline",
                    "Probe v1 misread the pusher by up to 200 px near the start corner, so the dial "
                    "was not honoured; this is what the grounded checks fixed.",
                ),
                "jobs": [f"B-{lay}-{seed}-lam0"] + [f"B-{lay}-{seed}-d{d}" for d in dials],
                "rule": rule,
                "ncols": 4,
            }
        )
        sets.append(
            {
                "name": f"checks/{lay}-seed{seed}-d10-three-checks",
                "heading": (
                    f"The same dial (d = 10 px) checked three ways, layout {lay}, seed {seed}",
                    "Frozen probe (stage B), grounded probe and exact check, next to the "
                    "unconstrained planner.",
                ),
                "jobs": [
                    f"B-{lay}-{seed}-lam0",
                    f"B-{lay}-{seed}-d10",
                    f"B2-{lay}-{seed}-fixed-d10",
                    f"B2-{lay}-{seed}-exact-d10",
                ],
                "rule": rule,
                "ncols": 4,
            }
        )
        # Every seed, so the single-seed sweeps above can be read against the spread.
        for suffix, name, label in (
            ("bestlam0-lam0", "no-safety-check", "No safety check"),
            ("exact-d0", "exact-check-d0", "Exact check at d = 0 px"),
        ):
            sets.append(
                {
                    "name": f"seeds/{lay}-{name}-ten-seeds",
                    "heading": (
                        f"{label}, layout {lay}: all ten stage B2 seeds (200 to 209)",
                        "Same planner, same layout, every seed; outcomes are counted in the "
                        "footers.",
                    ),
                    "jobs": [f"B2-{lay}-{s}-{suffix}" for s in range(200, 210)],
                    "rule": "all seeds of the condition",
                    "ncols": 5,
                }
            )
    summary = json.loads((ROOT / "outputs/safecem10/analysis/stageC2-summary.json").read_text())
    for key, lay, label in (
        ("holdout", "G80", "held-out layout G80"),
        ("fresh", "G0", "search layout G0, fresh seeds"),
    ):
        best = None
        for run in summary["runs"]:
            scores = run.get(f"{key}_safe_success_by_candidate") or []
            for c, score in enumerate(scores):
                if best is None or score > best[0]:
                    best = (score, run["method"], run["replicate"], c)
        score, method, rep, c = best
        jobs = [f"C2-{lay}-{s}-{method}-r{rep}-final-{key}-c{c:02d}" for s in range(400, 410)]
        sets.append(
            {
                "name": f"search/C2-{method}-r{rep}-c{c:02d}-{lay}-{key}-seeds",
                "heading": (
                    f"Stage C2, best {key} setting ({method}, replicate {rep}): {label}, ten seeds",
                    f"Safe success {score:.1f} on these ten seeds. Amber shows d0, the clearance "
                    "for the first imagined block; deeper blocks add the slope per block.",
                ),
                "jobs": jobs,
                "rule": f"front candidate with the highest {key} safe success across all runs",
                "ncols": 5,
                "gif": True,
            }
        )
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--only",
        action="append",
        help="render only sets whose name contains this text (repeatable); the existing "
        "manifest keeps every other set",
    )
    args = parser.parse_args()
    spec = design()
    rep = Replayer(spec)
    rows = results_by_id()
    manifest = {"sets": [], "episodes": {}}
    if args.only and (args.out / "manifest.json").exists():
        manifest = json.loads((args.out / "manifest.json").read_text())
        manifest["sets"] = [
            s for s in manifest["sets"] if not any(o in s["name"] for o in args.only)
        ]
    cache = {}
    for item in video_sets(rows):
        if args.only and not any(o in item["name"] for o in args.only):
            continue
        panels = []
        for job_id in item["jobs"]:
            if job_id not in cache:
                cache[job_id] = episode_frames(rep, spec, job_id)
                write_mp4(cache[job_id][0], args.out / "episodes" / f"{job_id}.mp4")
                manifest["episodes"][job_id] = cache[job_id][1]
            panels.append(cache[job_id][0])
        frames = grid(panels, item["ncols"], item["heading"])
        write_mp4(frames, args.out / f"{item['name']}.mp4")
        # GIF previews for the dataset card: the G80 exact-check sweep and the searched settings.
        if item.get("gif"):
            write_gif(frames, args.out / f"{item['name']}.gif", scale=0.42)
        manifest["sets"].append({k: v for k, v in item.items() if k not in {"ncols", "gif"}})
        print(json.dumps({"video": item["name"], "episodes": len(item["jobs"])}), flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
