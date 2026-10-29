"""Render the dial-study videos of tasks 12 to 14 (TwoRoom, Reacher, Cube) from synced evidence.

Every episode is redrawn from its own record. TwoRoom and Reacher set the saved simulator
state for each frame; Cube replays the saved actions from the scene's dataset state and checks
that the replayed pinch point matches the saved one. On top of the simulator's frame: the
true hazard band (red), the band inflated by the episode's first-block clearance (dashed
amber), the target (green) and the agent's trail (blue). The world model never saw any of
these overlays. Each overlay recounts the steps inside the band and must agree with the record.

    render_dial_videos.py --task reacher13 --out outputs/safetydial-reacher/videos
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
RED, AMBER, TRAIL, GREEN = (214, 39, 40), (225, 120, 15), (31, 90, 200), (35, 150, 60)
INK, MUTED, SURFACE = (25, 25, 25), (105, 105, 100), (252, 252, 251)
HEADER, FOOTER, SIZE = 56, 40, 448
FPS, HOLD = 8, 14


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


class TwoRoom:
    ns, unit, per_unit, agent = "tworoom12", "px", 1.0, "agent"

    def __init__(self, spec):
        import torch
        from stable_worldmodel.envs.two_room.env import TwoRoomEnv

        from helpers import tworoom12_geometry as geo

        self.torch, self.geo, self.spec = torch, geo, spec
        self.env = TwoRoomEnv()
        self.env.reset(seed=0)
        self.scale = SIZE / self.env.IMG_SIZE
        env = spec["environment"]
        self.start = np.asarray(env["start_agent"], dtype=float)
        self.target = np.asarray(env["target"], dtype=float)
        self.target_radius = 16.0

    def boxes(self, job):
        return self.geo.hazard_boxes(self.geo.layout(job["layout"], self.spec), self.spec)

    def episode(self, result, record):
        points = [self.start] + [s[:2] for s in record["states"]]
        frames = []
        for p in points:
            pos = self.torch.as_tensor(np.asarray(p), dtype=self.torch.float32)
            img = self.env._render_frame(agent_pos=pos).cpu().numpy().transpose(1, 2, 0)
            frames.append(np.asarray(Image.fromarray(img).resize((SIZE, SIZE), Image.NEAREST)))
        return frames, np.asarray(points, dtype=float)

    def to_px(self, xy):
        return [(float(x) * self.scale, float(y) * self.scale) for x, y in np.atleast_2d(xy)]


class Reacher:
    ns, unit, per_unit, agent = "reacher13", "cm", 100.0, "fingertip"

    def __init__(self, spec):
        from helpers import reacher13_geometry as geo
        from helpers.reacher13_env import GoalReacherEnv

        self.geo, self.spec = geo, spec
        self.env = GoalReacherEnv()
        self.env.reset(seed=0)
        self.physics = self.env.env.physics
        env = spec["environment"]
        self.links = env["link_lengths"]
        self.start = np.asarray([*env["start_qpos"], 0.0, 0.0])
        self.target = geo.fingertip(env["target_qpos"], self.links)[0]
        self.target_radius = 0.012
        data, model = self.physics.data, self.physics.model
        self.cam_pos = np.asarray(data.cam_xpos[0]).copy()
        self.cam_mat = np.asarray(data.cam_xmat[0]).reshape(3, 3).copy()
        self.focal = (SIZE / 2) / math.tan(math.radians(float(model.cam_fovy[0])) / 2)

    def boxes(self, job):
        return self.geo.hazard_boxes(self.geo.layout(job["layout"], self.spec), self.spec)

    def frame(self, q):
        with self.physics.reset_context():
            self.physics.data.qpos[:] = q[:2]
            self.physics.data.qvel[:] = q[2:4]
        return np.asarray(self.physics.render(height=SIZE, width=SIZE, camera_id=0))

    def episode(self, result, record):
        states = record["states"]
        frames = [self.frame(self.start)] + [self.frame(s[:4]) for s in states]
        tips = [self.geo.fingertip(self.start[:2], self.links)[0]] + [s[4:6] for s in states]
        return frames, np.asarray(tips, dtype=float)

    def to_px(self, xy, z=0.01):
        out = []
        for x, y in np.atleast_2d(xy):
            c = self.cam_mat.T @ (np.array([x, y, z]) - self.cam_pos)
            out.append((SIZE / 2 + self.focal * c[0] / -c[2], SIZE / 2 - self.focal * c[1] / -c[2]))
        return out


REPLAY_TOLERANCE_M = 1e-3  # about one pixel at 448 px


class Cube:
    ns, unit, per_unit, agent = "cube14", "cm", 100.0, "gripper"
    departure = None

    def __init__(self, spec):
        from helpers import cube14_geometry as geo
        from helpers.cube14_env import GoalCubeEnv

        self.geo, self.spec = geo, spec
        scene = json.loads((ROOT / "outputs/cube14/scene/S15.json").read_text())
        radius = spec["environment"]["success"]["effector_radius_m"]
        self.options = {
            "state": np.asarray(scene["state"], dtype=float),
            "goal_cube_pos": np.asarray(scene["goal_cube_pos"], dtype=float),
            "marker_pos": np.asarray(scene["marker_pos"], dtype=float),
            "goal_effector_pos": np.asarray(scene["goal_effector_pos"], dtype=float),
            "effector_radius_m": radius,
        }
        self.env = GoalCubeEnv(height=SIZE, width=SIZE)
        self.env.reset(seed=0, options=self.options)
        self.target = np.asarray(scene["goal_effector_pos"], dtype=float)[:2]
        self.target_radius = radius
        u = self.env.unwrapped
        cam = u._model.camera("front_pixels").id
        self.cam_pos = np.asarray(u._data.cam_xpos[cam]).copy()
        self.cam_mat = np.asarray(u._data.cam_xmat[cam]).reshape(3, 3).copy()
        self.focal = (SIZE / 2) / math.tan(math.radians(float(u._model.cam_fovy[cam])) / 2)

    def boxes(self, job):
        return self.geo.hazard_boxes(self.geo.layout(job["layout"], self.spec), self.spec)

    def episode(self, result, record):
        env = self.env
        env.reset(seed=0, options=self.options)
        u = env.unwrapped
        frames = [np.asarray(env.render())]
        points = [u._data.site_xpos[u._pinch_site_id][:2].copy()]
        for a in record["actions"]:
            _, _, _, _, info = env.step(np.asarray(a))
            frames.append(np.asarray(env.render()))
            points.append(info["state"][:2].copy())
        points = np.asarray(points, dtype=float)
        # The local replay can drift from the rental's by floating point (up to about 0.05 mm
        # seen); the arm image is the replay's, the trail and band counts are the record's.
        self.departure = float(np.abs(points[1:] - record["states"][:, :2]).max())
        if self.departure > REPLAY_TOLERANCE_M:
            raise ValueError(
                f"{result['job']['id']}: replay departs from the record by {self.departure:.2e} m"
            )
        return frames, np.vstack([points[:1], record["states"][:, :2]])

    def to_px(self, xy, z=0.0):
        out = []
        for x, y in np.atleast_2d(xy):
            c = self.cam_mat.T @ (np.array([x, y, z]) - self.cam_pos)
            out.append((SIZE / 2 + self.focal * c[0] / -c[2], SIZE / 2 - self.focal * c[1] / -c[2]))
        return out


TASKS = {"tworoom12": TwoRoom, "reacher13": Reacher, "cube14": Cube}


def corners(box):
    x0, x1, y0, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def dashed_polygon(draw, points, color, width=2, dash=9, gap=6):
    for a, b in zip(points, points[1:] + points[:1], strict=True):
        a, b = np.asarray(a), np.asarray(b)
        length = float(np.linalg.norm(b - a))
        pos = 0.0
        while pos < length:
            stop = min(pos + dash, length)
            p, q = a + (b - a) * pos / length, a + (b - a) * stop / length
            draw.line([tuple(p), tuple(q)], fill=color, width=width)
            pos = stop + gap


def scene(adapter, image, boxes, dial, trail, inside_now):
    base = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGBA")
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(over)
    for box in boxes:
        draw.polygon(adapter.to_px(corners(box)), fill=RED + (70,), outline=RED + (255,))
    if dial:
        for x0, x1, y0, y1 in boxes:
            inflated = (x0 - dial, x1 + dial, y0 - dial, y1 + dial)
            dashed_polygon(draw, adapter.to_px(corners(inflated)), AMBER + (255,))
    ring = [
        adapter.target + adapter.target_radius * np.array([math.cos(a), math.sin(a)])
        for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)
    ]
    draw.polygon(adapter.to_px(np.asarray(ring)), outline=GREEN + (255,), width=3)
    px = adapter.to_px(np.asarray(trail))
    if len(px) > 1:
        draw.line(px, fill=TRAIL + (190,), width=3)
    for x, y in px[:-1]:
        draw.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=TRAIL + (220,))
    x, y = px[-1]
    draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=TRAIL + (255,))
    if inside_now:
        draw.ellipse([x - 18, y - 18, x + 18, y + 18], outline=RED + (255,), width=4)
    return Image.alpha_composite(base, over).convert("RGB")


def panel(image, title, subtitle, step, budget, inside_steps, outcome):
    canvas = Image.new("RGB", (SIZE, HEADER + SIZE + FOOTER), SURFACE)
    canvas.paste(image, (0, HEADER))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 6), title, font=font(18, True), fill=INK)
    draw.text((12, 32), subtitle, font=font(13), fill=MUTED)
    label = f"step {step}/{budget}"
    width = draw.textlength(label, font=font(13))
    draw.text((SIZE - 12 - width, 10), label, font=font(13), fill=MUTED)
    inside = f"steps inside the band: {inside_steps}"
    draw.text((12, HEADER + SIZE + 10), inside, font=font(14), fill=RED if inside_steps else MUTED)
    if outcome:
        text, color = outcome
        width = draw.textlength(text, font=font(14, True))
        draw.text((SIZE - 12 - width, HEADER + SIZE + 10), text, font=font(14, True), fill=color)
    return canvas


def describe(adapter, job):
    """Short title and subtitle for a job."""
    u, k = adapter.unit, adapter.per_unit
    if job["mode"] == "penalty":
        return "no safety check", f"layout {job['layout']}, seed {job['seed']}"
    d0 = job["clearances"][0] * k
    if job["stage"] == "C":
        p = job["params"]
        return (
            f"searched setting, seed {job['seed']}",
            f"d0 {p['d0'] * k:.1f} {u}, slope {p['d1'] * k:+.1f} {u}/block, "
            f"horizon {job['planner']['horizon']}, tol {job['planner']['feasible_tol'] * k:.1f}",
        )
    executes = "checked plan" if job["planner"].get("execute") == "mean_safe" else "elite mean"
    return (
        f"safety check, d = {d0:g} {u}",
        f"layout {job['layout']}, seed {job['seed']}, {executes}",
    )


def episode_frames(adapter, spec, remote, job_id):
    folder = remote / "attempts" / job_id
    result = json.loads((folder / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError(f"{job_id} is not a complete episode")
    record = dict(np.load(folder / "states.npz"))
    job = result["job"]
    boxes = adapter.boxes(job)
    dial = job["clearances"][0] if job["mode"] == "safe" else 0.0
    budget = int(spec["environment"]["eval_budget_steps"])
    title, subtitle = describe(adapter, job)
    images, points = adapter.episode(result, record)
    inside = adapter.geo.inside_any(points, boxes)
    frames, inside_steps = [], 0
    for t, (image, now) in enumerate(zip(images, inside, strict=True)):
        inside_steps += int(now) if t else 0
        outcome = None
        if t == len(images) - 1:
            if result["success"]:
                ok = not result["any_violation"]
                text = f"SUCCESS in {t} steps" + ("" if ok else ", unsafe")
                outcome = (text, GREEN if ok else RED)
            else:
                outcome = (f"no success in {budget} steps", MUTED)
        picture = scene(adapter, image, boxes, dial, points[: t + 1], bool(now) and t > 0)
        frames.append(panel(picture, title, subtitle, t, budget, inside_steps, outcome))
    if inside_steps != result["hazard_steps"]:
        raise ValueError(
            f"{job_id}: overlay counts {inside_steps} band steps, record says "
            f"{result['hazard_steps']}"
        )
    meta = {
        "job_id": job_id,
        "success": bool(result["success"]),
        "hazard_steps": int(result["hazard_steps"]),
        "J": result["J"],
        "steps": len(images) - 1,
    }
    if getattr(adapter, "departure", None) is not None:
        meta["replay_departure_m"] = adapter.departure
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
        draw.text((14, 10), heading[0], font=font(20, True), fill=INK)
        draw.text((14, 38), heading[1], font=font(14), fill=MUTED)
        for k, panels in enumerate(panel_lists):
            img = panels[min(i, len(panels) - 1)].resize((pw, ph), Image.LANCZOS)
            canvas.paste(img, ((k % ncols) * pw, top + (k // ncols) * ph))
        frames.append(canvas)
    return frames


def results_by_id(remote):
    rows = {}
    for path in (remote / "attempts").glob("*/result.json"):
        row = json.loads(path.read_text())
        if row["status"] == "complete":
            rows[row["job"]["id"]] = row
    return rows


def dial_tag(d):
    return f"d{d:g}"


def choose_seed(rows, lay, seeds):
    """First seed where the checked planner at d = 0 succeeded safely and the unconstrained
    planner entered the band; else the first safe checked success; else the first seed."""
    for s in seeds:
        safe, free = rows.get(f"B-{lay}-{s}-safe-d0"), rows.get(f"B-{lay}-{s}-free-lam0")
        if (
            safe
            and free
            and safe["success"]
            and not safe["any_violation"]
            and free["any_violation"]
        ):
            return s, "checked d=0 succeeded safely and the unconstrained planner entered the band"
    for s in seeds:
        safe = rows.get(f"B-{lay}-{s}-safe-d0")
        if safe and safe["success"] and not safe["any_violation"]:
            return s, "first seed where checked d=0 succeeded safely"
    return seeds[0], "fallback: first seed"


def video_sets(adapter, spec, rows, remote, with_c):
    stage = spec["stages"]["B"]
    seeds, dials = stage["seeds"], stage["dials"]
    u, k = adapter.unit, adapter.per_unit
    search = spec["stages"]["C"]["search_layout"]
    names = {lay["id"]: lay for lay in spec["hazard"]["layouts"]}
    sets = []
    for lay in names:
        seed, rule = choose_seed(rows, lay, seeds)
        offset = names[lay].get("offset_from_route_m", names[lay].get("offset_from_route_px", 0))
        sets.append(
            {
                "name": f"dial-sweep/{lay}-seed{seed}",
                "heading": (
                    f"Layout {lay} (gap {offset * k:g} {u} off the free route), seed {seed}: "
                    f"one episode at {len(dials)} dial settings",
                    "Top left: no safety check. Red: true band. Dashed amber: band inflated by the "
                    f"dial. Green: the {adapter.agent}'s target. Blue: its trail.",
                ),
                "jobs": [f"B-{lay}-{seed}-free-lam0"]
                + [f"B-{lay}-{seed}-safe-{dial_tag(d)}" for d in dials],
                "rule": rule,
                "ncols": 4,
                "gif": lay == search,
            }
        )
    summary = json.loads((ROOT / f"outputs/{adapter.ns}/analysis/stageB-summary.json").read_text())
    best = summary["best_fixed_dial"].get(search, {}).get("dial", 0.0)
    for suffix, name, label in (
        ("free-lam0", "no-safety-check", "No safety check"),
        ("safe-d0", "safety-check-d0", f"Safety check at d = 0 {u}"),
        (f"safe-{dial_tag(best)}", f"safety-check-best-{dial_tag(best)}", None),
    ):
        if label is None:
            if best == 0.0:
                continue
            label = f"Safety check at the best fixed dial, d = {best * k:g} {u}"
        sets.append(
            {
                "name": f"seeds/{search}-{name}-ten-seeds",
                "heading": (
                    f"{label}, layout {search}: all ten stage B seeds",
                    "Same planner, same layout, every seed; outcomes are counted in the footers.",
                ),
                "jobs": [f"B-{search}-{s}-{suffix}" for s in seeds],
                "rule": "all seeds of the condition",
                "ncols": 5,
            }
        )
    if with_c:
        c_summary = json.loads(
            (ROOT / f"outputs/{adapter.ns}/analysis/stageC-summary.json").read_text()
        )
        holdout = spec["stages"]["C"]["holdout_layout"]
        finals = spec["stages"]["C"]["final_seeds"]
        for key, lay, label in (
            ("holdout", holdout, f"held-out layout {holdout}"),
            ("fresh", search, f"search layout {search}, fresh seeds"),
        ):
            top = None
            for run in c_summary["runs"]:
                final = json.loads(
                    (remote / f"stageC/{run['method']}-r{run['replicate']}/final.json").read_text()
                )
                objectives = final.get(f"{key}_objectives") or []
                for c, score in enumerate(run.get(f"{key}_safe_success_by_candidate") or []):
                    steps = objectives[c][1] if c < len(objectives) else float("inf")
                    # Highest safe success; among ties, the fewest steps.
                    if top is None or (score, -steps) > (top[0], -top[4]):
                        top = (score, run["method"], run["replicate"], c, steps)
            if top is None:
                continue
            score, method, rep, c, steps = top
            jobs = [f"C-{lay}-{s}-{method}-r{rep}-final-{key}-c{c:02d}" for s in finals]
            if not all(j in rows for j in jobs):
                continue
            sets.append(
                {
                    "name": f"search/C-{method}-r{rep}-c{c:02d}-{lay}-{key}-seeds",
                    "heading": (
                        f"Stage C, best {key} setting ({method}, replicate {rep}): {label}",
                        f"Safe success {score:.1f} and mean J {steps:.1f} steps on these ten "
                        "seeds. Amber shows d0, the clearance for the first imagined block.",
                    ),
                    "jobs": jobs,
                    "rule": (
                        f"front candidate with the highest {key} safe success across all runs, "
                        "ties broken by the fewest mean steps"
                    ),
                    "ncols": 5,
                    "gif": key == "holdout",
                }
            )
    return [s for s in sets if all(j in rows for j in s["jobs"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--with-c", action="store_true", help="add the stage C final sets")
    parser.add_argument("--only", action="append", help="render only sets containing this text")
    parser.add_argument("--episode", help="render one episode clip and exit (a smoke test)")
    args = parser.parse_args()
    spec = json.loads((ROOT / f"configs/{args.task}/design.json").read_text())
    remote = ROOT / f"outputs/{args.task}/remote"
    adapter = TASKS[args.task](spec)
    if args.episode:
        frames, meta = episode_frames(adapter, spec, remote, args.episode)
        write_mp4(frames, args.out / "episodes" / f"{args.episode}.mp4")
        frames[-1].save(args.out / f"{args.episode}-last.png")
        print(json.dumps(meta))
        return
    rows = results_by_id(remote)
    manifest = {"sets": [], "episodes": {}}
    if args.only and (args.out / "manifest.json").exists():
        manifest = json.loads((args.out / "manifest.json").read_text())
        manifest["sets"] = [
            s for s in manifest["sets"] if not any(o in s["name"] for o in args.only)
        ]
    cache = {}
    for item in video_sets(adapter, spec, rows, remote, args.with_c):
        if args.only and not any(o in item["name"] for o in args.only):
            continue
        panels = []
        for job_id in item["jobs"]:
            if job_id not in cache:
                cache[job_id] = episode_frames(adapter, spec, remote, job_id)
                write_mp4(cache[job_id][0], args.out / "episodes" / f"{job_id}.mp4")
                manifest["episodes"][job_id] = cache[job_id][1]
            panels.append(cache[job_id][0])
        frames = grid(panels, item["ncols"], item["heading"])
        write_mp4(frames, args.out / f"{item['name']}.mp4")
        if item.get("gif"):
            write_gif(frames, args.out / f"{item['name']}.gif", scale=0.42)
        manifest["sets"].append({k: v for k, v in item.items() if k not in {"ncols", "gif"}})
        print(json.dumps({"video": item["name"], "episodes": len(item["jobs"])}), flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
