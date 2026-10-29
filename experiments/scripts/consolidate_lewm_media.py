"""Package tasks 16/17 and reconstruct illustrative videos without rerunning experiments."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm16_protocol import boxes, hits
from helpers.reacher13_geometry import fingertip

OUT = ROOT / "outputs/lewm-consolidation"
SNAP = ROOT / "outputs/lewm17/recovery/snapshot"
SOURCES = {16: ROOT / "outputs/lewm16/remote", 17: SNAP / "outputs/lewm17"}
S = 448
HEIGHT = 576
FPS = 8
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BG = (250, 250, 247)
RED, AMBER, BLUE, GREEN = (205, 55, 55), (220, 135, 20), (35, 100, 205), (30, 150, 80)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(task, env):
    result = []
    for folder in sorted((SOURCES[task] / "attempts").iterdir()):
        row = json.loads((folder / "result.json").read_text())
        ticket = json.loads((folder / "ticket.json").read_text())
        job = row.get("job") or ticket.get("job", ticket)
        if job["env"] == env:
            result.append((folder, job, row))
    return result


def choose_gallery(task, rows):
    """Fixed ordering plus labeled first failure per method/margin, never a performance sample."""
    selected, reasons = {}, collections.defaultdict(list)
    stages = {"A", "B"} if task == 16 else {"H"}
    eligible = [x for x in rows if x[1]["stage"] in stages and x[2]["status"] == "complete"]
    for stage in sorted(stages):
        stage_rows = [x for x in eligible if x[1]["stage"] == stage]
        case = min(x[1]["case"]["id"] for x in stage_rows)
        seed = min(x[1]["seed"] for x in stage_rows)
        for x in stage_rows:
            job = x[1]
            if job["case"]["id"] == case and job["seed"] == seed:
                selected[job["id"]] = x
                reasons[job["id"]].append("first case and first planner seed, all margins/methods")
    failures = {}
    for x in eligible:
        job, row = x[1:]
        key = (job["stage"], job["condition"], job["margin"])
        if not row["requested_safe_completion"] and key not in failures:
            failures[key] = x
    for x in failures.values():
        selected[x[1]["id"]] = x
        reasons[x[1]["id"]].append("first non-safe-completion in lexical episode order")
    return [(selected[k], reasons[k]) for k in sorted(selected)]


class Renderer:
    def __init__(self, env):
        self.kind = env
        if env == "tworoom":
            import torch
            from stable_worldmodel.envs.two_room.env import TwoRoomEnv

            torch.set_num_threads(1)
            self.torch = torch
            self.env = TwoRoomEnv()
            self.env.reset(seed=0)
        else:
            from helpers.reacher13_env import GoalReacherEnv

            self.env = GoalReacherEnv()
            self.env.reset(seed=0)
            self.physics = self.env.env.physics
            data, model = self.physics.data, self.physics.model
            self.cam_pos = data.cam_xpos[0].copy()
            self.cam_mat = data.cam_xmat[0].reshape(3, 3).copy()
            self.focal = (S / 2) / math.tan(math.radians(float(model.cam_fovy[0])) / 2)

    def pixels(self, xy):
        xy = np.asarray(xy).reshape(-1, 2)
        if self.kind == "tworoom":
            return [tuple(p * S / 224) for p in xy]
        out = []
        for x, y in xy:
            c = self.cam_mat.T @ (np.array([x, y, 0.01]) - self.cam_pos)
            out.append((S / 2 + self.focal * c[0] / -c[2], S / 2 - self.focal * c[1] / -c[2]))
        return out

    def frame(self, state):
        if self.kind == "tworoom":
            a = self.env._render_frame(agent_pos=self.torch.tensor(state, dtype=self.torch.float32))
            a = a.cpu().numpy().transpose(1, 2, 0)
            return Image.fromarray(a).resize((S, S), Image.Resampling.NEAREST)
        with self.physics.reset_context():
            self.physics.data.qpos[:] = state[:2]
            self.physics.data.qvel[:] = state[2:4]
        if len(state) >= 6:
            delta = np.abs(self.physics.named.data.geom_xpos["finger"][:2] - state[4:6]).max()
            if delta > 1e-9:
                raise ValueError("Rendered fingertip differs from recorded state")
        return Image.fromarray(self.physics.render(height=S, width=S, camera_id=0))

    def close(self):
        self.env.close()


def dashed(draw, poly, color):
    for a, b in zip(poly, poly[1:] + poly[:1], strict=True):
        a, b = np.array(a), np.array(b)
        length = np.linalg.norm(b - a)
        for t in np.arange(0, length, 12):
            stop = min(t + 7, length)
            draw.line(
                [tuple(a + (b - a) * t / length), tuple(a + (b - a) * stop / length)],
                fill=color,
                width=2,
            )


def render_episode(renderer, task, item, destination, reasons):
    folder, job, row = item
    with np.load(folder / "states.npz", allow_pickle=False) as data:
        states = data["states"].copy()
        fine = data["fine_states"].copy()
    assert len(states) == row["env_steps"]
    xy = states[:, :2] if renderer.kind == "tworoom" else states[:, 4:6]
    fine_xy = fine[:, :2] if renderer.kind == "tworoom" else fine[:, 4:6]
    hazard = boxes(renderer.kind, job["case"])
    hv, mv = hits(fine_xy, hazard), hits(fine_xy, hazard, job["margin"])
    assert bool(hv.any()) == row["any_violation"]
    assert bool(mv.any()) == row["margin_breach"]
    start = np.asarray(job["case"]["start"])
    start_xy = start if renderer.kind == "tworoom" else fingertip(start[:2])[0]
    goal = np.asarray(job["case"]["goal"])
    target = goal if renderer.kind == "tworoom" else fingertip(goal)[0]
    trajectory = np.vstack([start_xy, xy])
    all_states = [start] + list(states)
    stem = job["id"]
    video = destination / "videos" / f"{stem}.mp4"
    still = destination / "pictures/episodes" / f"{stem}.png"
    video.parent.mkdir(parents=True, exist_ok=True)
    still.parent.mkdir(parents=True, exist_ok=True)
    f18, f14, f12 = [ImageFont.truetype(FONT, n) for n in [18, 14, 12]]
    selected = None
    if task == 17 and "optimizer" in job:
        p = (
            SOURCES[17]
            / "optimizers"
            / (f"{renderer.kind}-{job['optimizer']}-r{job['replicate']}")
            / "selected.json"
        )
        if p.exists():
            info = json.loads(p.read_text())
            selected = info["selected_label"]
    unit = f"{job['margin']:g} px" if renderer.kind == "tworoom" else f"{job['margin'] * 100:g} cm"
    frames = []
    for step, state in enumerate(all_states):
        base = renderer.frame(state).convert("RGBA")
        overlay = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for x0, x1, y0, y1 in hazard:
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            draw.polygon(renderer.pixels(corners), fill=RED + (55,), outline=RED + (230,))
            m = job["margin"]
            inflated = [(x0 - m, y0 - m), (x1 + m, y0 - m), (x1 + m, y1 + m), (x0 - m, y1 + m)]
            if m:
                dashed(draw, renderer.pixels(inflated), AMBER + (255,))
        if renderer.kind == "tworoom":
            ring = [
                target + 16 * np.array([math.cos(t), math.sin(t)])
                for t in np.linspace(0, 2 * math.pi, 50, endpoint=False)
            ]
            draw.polygon(renderer.pixels(ring), outline=GREEN + (255,), width=2)
        else:
            elbow = 0.12 * np.array([math.cos(goal[0]), math.sin(goal[0])])
            dashed(draw, renderer.pixels([[0, 0], elbow, target]), GREEN + (230,))
        px, py = renderer.pixels([target])[0]
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=GREEN + (255,))
        trail = renderer.pixels(trajectory[: step + 1])
        if len(trail) > 1:
            draw.line(trail, fill=BLUE + (230,), width=3)
        px, py = trail[-1]
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=BLUE + (255,))
        frame = Image.new("RGB", (S, HEIGHT), BG)
        frame.paste(Image.alpha_composite(base, overlay).convert("RGB"), (0, 80))
        d = ImageDraw.Draw(frame)
        d.text(
            (10, 7),
            f"Task {task} | {renderer.kind} | {job['condition']}",
            font=f18,
            fill=(25, 25, 25),
        )
        d.text(
            (10, 32),
            f"{job['case']['id']} | seed {job['seed']} | {unit}",
            font=f12,
            fill=(65, 65, 65),
        )
        d.text(
            (10, 53), "Recorded-state replay | 8 actions/sec playback", font=f12, fill=(65, 65, 65)
        )
        end = step if renderer.kind == "tworoom" else min(2 * step, len(fine))
        d.text(
            (10, 533),
            f"Action {step}/{len(states)} | hazard {int(hv[:end].sum())} | margin {int(mv[:end].sum())}",
            font=f14,
            fill=(35, 35, 35),
        )
        outcome = (
            "safe completion"
            if row["requested_safe_completion"]
            else ("unsafe completion" if row["success"] else "unfinished")
        )
        label = outcome if step == len(states) else "in progress"
        if selected:
            label += f" | selected: {selected}"
        d.text(
            (10, 555),
            label,
            font=f12,
            fill=GREEN if label.startswith("safe completion") else (65, 65, 65),
        )
        frames.append(np.asarray(frame))
    with imageio.get_writer(
        video,
        fps=FPS,
        codec="libx264",
        quality=7,
        ffmpeg_params=["-threads", "1"],
        macro_block_size=16,
    ) as writer:
        for frame in frames + [frames[-1]] * 8:
            writer.append_data(frame)
    Image.fromarray(frames[-1]).save(still)
    reader = imageio.get_reader(video)
    decoded = sum(1 for _ in reader)
    reader.close()
    assert decoded == len(frames) + 8
    return {
        "episode_id": stem,
        "campaign": task,
        "environment": renderer.kind,
        "stage": job["stage"],
        "case_id": job["case"]["id"],
        "seed": job["seed"],
        "condition": job["condition"],
        "margin": job["margin"],
        "selection_reason": reasons,
        "selected_candidate": selected,
        "requested_safe_completion": row["requested_safe_completion"],
        "any_violation": row["any_violation"],
        "margin_breach": row["margin_breach"],
        "unfinished": row["unfinished"],
        "env_steps": row["env_steps"],
        "video": str(video.relative_to(destination)),
        "picture": str(still.relative_to(destination)),
        "video_frames": decoded,
        "fps": FPS,
        "provenance": "Re-rendered recorded states; no policy queries or simulated rollout steps",
        "states_sha256": sha(folder / "states.npz"),
    }


def copy_documents(task, env, destination):
    source_root = ROOT if task == 16 else SNAP
    for source in sorted((source_root / "progress/receipts").glob(f"{task}-*")):
        if source.is_file() and "workspace-manifest" not in source.name:
            target = destination / "protocol" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    configs = source_root / f"configs/lewm{task}"
    shutil.copytree(configs, destination / "code/configs" / f"lewm{task}", dirs_exist_ok=True)
    for group in ["helpers", "scripts", "tests"]:
        for pattern in [f"*lewm{task}*", "reacher13_*.py", "tworoom12_*.py", "safecem10_*.py"]:
            for source in (source_root / "experiments" / group).glob(pattern):
                if source.is_file():
                    target = destination / "code/experiments" / group / source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    # Task 17 imports reviewed task-16 helpers. Include their frozen bytes as dependencies.
    if task == 17:
        for source in (source_root / "experiments/helpers").glob("lewm16_*.py"):
            shutil.copy2(source, destination / "code/experiments/helpers" / source.name)
        shutil.copytree(
            source_root / "configs/lewm16", destination / "code/configs/lewm16", dirs_exist_ok=True
        )
    analysis = ROOT / f"outputs/lewm{task}/analysis"
    for source in analysis.glob("*"):
        if source.suffix in {".pdf", ".png", ".json"}:
            target = destination / "results" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if task == 17:
        names = [
            p for p in (SOURCES[task] / "optimizers").iterdir() if p.name.startswith(env + "-")
        ]
        for source in names:
            shutil.copytree(source, destination / "controllers" / source.name, dirs_exist_ok=True)
    campaign = "16-lewm-safecem-campaign-plan.md" if task == 16 else "17-lewm-evolution-campaign.md"
    p = source_root / "progress" / campaign
    if p.exists():
        shutil.copy2(p, destination / "protocol" / campaign)


def pack(task, env):
    dest = OUT / "stage" / env / "campaigns" / f"lewm{task}"
    dest.mkdir(parents=True, exist_ok=True)
    rows = load_rows(task, env)
    raw = dest / "data/raw-attempts.tar.gz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    manifest, episodes, controls, fine_rows = {}, [], [], []
    with tarfile.open(raw, "w:gz") as archive:
        for folder, job, row in rows:
            for source in sorted(folder.rglob("*")):
                if not source.is_file():
                    continue
                name = str(source.relative_to(SOURCES[task]))
                archive.add(source, arcname=name, recursive=False)
                manifest[name] = {"sha256": sha(source), "bytes": source.stat().st_size}
            episode = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
            episode.update(
                campaign=task,
                episode_id=job["id"],
                environment=env,
                stage=job["stage"],
                case_id=job["case"]["id"],
                seed=job["seed"],
                condition=job["condition"],
                margin=job["margin"],
                case_json=json.dumps(job["case"]),
                parameters_json=json.dumps(job["params"]),
                counters_json=json.dumps(row.get("counters", {})),
                job_json=json.dumps(job),
                result_json=json.dumps(row),
            )
            if "engaged" not in episode:
                episode["engaged"] = (row.get("path_length", 0) or 0) > (
                    1 if env == "tworoom" else 0.001
                )
                episode["engagement_definition"] = "Derived from recorded path length, task-16 rule"
            episodes.append(episode)
            with np.load(folder / "states.npz", allow_pickle=False) as data:
                states, actions, fine = data["states"], data["actions"], data["fine_states"]
                if row["status"] == "complete":
                    assert len(states) == row["env_steps"] == len(actions)
                for i, (s, a) in enumerate(zip(states, actions, strict=True)):
                    controls.append(
                        {
                            "episode_id": job["id"],
                            "action_index": i + 1,
                            "state": s.tolist(),
                            "recorded_environment_action": a.tolist(),
                            "reconstructed_bounded_simulator_control": np.clip(a, -1, 1).tolist(),
                        }
                    )
                for i, s in enumerate(fine):
                    fine_rows.append(
                        {"episode_id": job["id"], "integration_index": i + 1, "state": s.tolist()}
                    )
    write_json(dest / "data/raw-files.json", manifest)
    with (dest / "data/episodes.jsonl").open("w") as f:
        for row in episodes:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    for name, data in [
        ("episodes", episodes),
        ("trajectories", controls),
        ("dense-states", fine_rows),
    ]:
        pq.write_table(
            pa.Table.from_pylist(data), dest / "data" / f"{name}.parquet", compression="zstd"
        )
    copy_documents(task, env, dest)
    gallery = [
        {"episode_id": x[1]["id"], "reasons": reasons} for x, reasons in choose_gallery(task, rows)
    ]
    write_json(dest / "gallery-selection.json", gallery)
    summary = {
        "task": task,
        "environment": env,
        "attempts": len(rows),
        "complete": sum(x[2]["status"] == "complete" for x in rows),
        "failed": sum(x[2]["status"] != "complete" for x in rows),
        "stage_counts": dict(collections.Counter(x[1]["stage"] for x in rows)),
        "control_rows": len(controls),
        "dense_state_rows": len(fine_rows),
        "raw_files": len(manifest),
        "gallery_episodes": len(gallery),
    }
    write_json(dest / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def render(task, env, limit=None):
    dest = OUT / "stage" / env / "campaigns" / f"lewm{task}"
    selected = choose_gallery(task, load_rows(task, env))
    if limit:
        selected = selected[:limit]
    renderer = Renderer(env)
    gallery = []
    try:
        for i, (item, reasons) in enumerate(selected):
            gallery.append(render_episode(renderer, task, item, dest, reasons))
            if (i + 1) % 10 == 0:
                print(
                    json.dumps(
                        {"task": task, "env": env, "rendered": i + 1, "total": len(selected)}
                    ),
                    flush=True,
                )
    finally:
        renderer.close()
    write_json(dest / ("media-index-preview.json" if limit else "media-index.json"), gallery)
    cols = 4
    thumbw, thumbh = 224, 288
    sheet = Image.new("RGB", (cols * thumbw, math.ceil(len(gallery) / cols) * thumbh), BG)
    for i, row in enumerate(gallery):
        im = Image.open(dest / row["picture"]).resize((thumbw, thumbh), Image.Resampling.LANCZOS)
        sheet.paste(im, ((i % cols) * thumbw, (i // cols) * thumbh))
    sheet.save(dest / "pictures/gallery-contact-sheet.jpg", quality=90)
    print(json.dumps({"task": task, "env": env, "videos": len(gallery), "done": True}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["pack", "render"])
    parser.add_argument("--task", type=int, choices=[16, 17], required=True)
    parser.add_argument("--env", choices=["tworoom", "reacher"], required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "pack":
        pack(args.task, args.env)
    else:
        render(args.task, args.env, args.limit)


if __name__ == "__main__":
    main()
