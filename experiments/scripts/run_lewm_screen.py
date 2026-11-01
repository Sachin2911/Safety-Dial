"""Two-environment readiness and fixed-adapter replication, on one declared RunPod."""

import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import importlib

importlib.import_module("triton")
import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_proposal_pilot import write_json
from helpers.lewm_screen import KINDS, distance, geometry, image_stats, no_hazard_gate, summarize
from helpers.lewm_screen_runtime import load_runtime, trial


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(design, out):
    freeze = json.loads((out / "source-freeze.json").read_text())
    if digest(ROOT / design["protocol_path"]) != freeze["protocol_sha256"]:
        raise RuntimeError("Frozen protocol changed")
    for names in (freeze["sources"], design["asset_sha256"]):
        for name, sha in names.items():
            if digest(ROOT / name) != sha:
                raise RuntimeError("Frozen source/asset changed: " + name)


def reserve(design):
    if dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10) >= dt.datetime.fromisoformat(
        design["deadline_utc"]
    ):
        raise RuntimeError("Ten-minute preservation reserve reached")


def run_job(design, out, state, rt, case, role, kind, seed, *, engineering=False):
    label = "engineering" if engineering else "scored"
    key = f"{rt.environment}_{role}_p{case['index']:03}_{kind}_s{seed}"
    if engineering:
        key = "engineering_" + key
    if key in state["jobs"]:
        return state["jobs"][key]
    reserve(design)
    started = list((out / "episodes").glob("*/started.json"))
    engineering_count = sum(p.parent.name.startswith("engineering_") for p in started)
    if len(started) >= 100 or (engineering and engineering_count >= 4):
        raise RuntimeError("Frozen rollout/engineering-start limit reached")
    folder = out / "episodes" / key
    if folder.exists():
        if (folder / "result.json").exists():
            result = json.loads((folder / "result.json").read_text())
        else:
            raise RuntimeError("Incomplete job requires diagnosis, never an outcome replay: " + key)
    else:
        state["active"] = key
        write_json(out / "state.json", state)
        try:
            result = trial(
                rt,
                case,
                role,
                kind,
                seed,
                design["environments"][rt.environment],
                folder,
                cap=6 if engineering else 150,
            )
        except BaseException:
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / "result.json").exists():
                write_json(
                    folder / "technical-error.json",
                    {
                        "error": traceback.format_exc(),
                        "resources": dict(getattr(rt, "counters", {})),
                        "key": key,
                    },
                )
            raise
    result.update(key=key, category=label)
    write_json(folder / "result.json", result)
    state["jobs"][key] = result
    state["active"] = None
    write_json(out / "state.json", state)
    return result


def reset_push(pair, seed, counters):
    from stable_worldmodel.envs.pusht.env import PushT

    class Counted(PushT):
        def _set_state(self, state):
            original = self.space.step

            def step(dt):
                counters["reset_physics_steps"] += 1
                return original(dt)

            self.space.step = step
            try:
                return super()._set_state(state)
            finally:
                self.space.step = original

        def render(self):
            counters["render_frames"] += 1
            return super().render()

    env = Counted()
    counters["environment_resets"] += 1
    try:
        observed, _ = env.reset(
            seed=seed, options={"state": pair["initial_state"], "goal_state": pair["goal_state"]}
        )
        if env.eval_state(pair["goal_state"], observed["state"])[0]:
            raise RuntimeError("Already successful preparation start")
        frame = env.render()
        return np.asarray(observed["state"]).copy(), frame
    finally:
        env.close()


def prepare_environment(design, out, rt):
    name = rt.environment
    counters = collections.Counter() if name == "pusht" else rt.counters
    startup = dict(counters)
    counters.clear()
    cases = []
    images = []
    for source in design["sources"][name]:
        reserve(design)
        index = source["index"]
        path = ROOT / source["path"]
        with np.load(path, allow_pickle=False) as data:
            pair = dict(data)
        seed = design["planner_seed_base"][name] + 10 * index
        if name == "pusht":
            initial, frame = reset_push(pair, seed, counters)
            endpoint = np.asarray(pair["goal_state"])[:2]
            reference = None
            shape = source["source_metadata"]["shape"]
        else:
            initial = rt.env.reset(seed, pair["initial"], pair["goal"], pair["goal_image"])
            if rt.env.success():
                raise RuntimeError("Already successful Cube preparation start")
            frame = rt.env.render()
            reference = pair["reference_route"]
            endpoint = reference[-1]
            shape = source["source_metadata"]["shape"]
        shapes = geometry(name, index, initial, endpoint, reference, shape)
        actual = initial[:2] if name == "pusht" else initial[-2:]
        admission = {}
        threshold = design["environments"][name]["margin"]
        threshold += (
            float(rt.head.error_allowance) if name == "pusht" else rt.head.calibration["allowance"]
        )
        for kind in KINDS[1:]:
            clear = distance(
                np.stack([actual, endpoint]),
                shapes[kind],
                design["environments"][name]["body_radius"],
            )
            admission[kind] = clear.tolist()
            if bool((clear < threshold).any()):
                write_json(
                    out / "preparation" / (name + "-rejected.json"),
                    {
                        "pair": index,
                        "kind": kind,
                        "clearance": clear.tolist(),
                        "threshold": threshold,
                        "resources": dict(counters),
                    },
                )
                raise RuntimeError(
                    "Frozen geometry has an inadmissible endpoint; never replace after outcomes"
                )
        image_stats(frame)
        image_stats(pair["goal_image"])
        case = {
            "index": index,
            "pair_path": str(path),
            "pair_sha256": digest(path),
            "shapes": shapes,
            "settled_initial": initial.tolist(),
            "endpoint_clearance": admission,
            "source_episode": source["source_episode"],
            "source_row": source["source_row"],
        }
        cases.append(case)
        images.extend([frame, pair["goal_image"]])
        write_json(
            out / "preparation" / (name + "-resources.json"),
            {"constructor": startup, "preparation": dict(counters)},
        )
    gallery(
        images,
        [f"pair {i // 2}: {'start' if i % 2 == 0 else 'goal'}" for i in range(6)],
        out / "quality" / (name + "-prepared.png"),
        columns=2,
    )
    write_json(
        out / "preparation" / (name + "-panel.json"),
        {
            "cases": cases,
            "constructor": startup,
            "resources": dict(counters),
            "prepared_without_controller_outcomes": True,
        },
    )
    counters.clear()
    return cases


def gallery(images, labels, path, columns=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    height = 248
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 224, rows * height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    for i, (frame, label) in enumerate(zip(images, labels, strict=True)):
        x, y = i % columns * 224, i // columns * height
        canvas.paste(Image.fromarray(np.asarray(frame)), (x, y + 24))
        draw.text((x + 4, y + 3), label, fill="black", font=font)
    canvas.save(path)


def control_gallery(out, state, name):
    images, labels = [], []
    for key, row in state["jobs"].items():
        if (
            row["category"] != "scored"
            or row["environment"] != name
            or row["kind"] != "none"
            or row["role"] != "native"
        ):
            continue
        base = out / "episodes" / key
        if name == "pusht":
            base = base / "episode"
        with np.load(base / "trace.npz", allow_pickle=False) as trace:
            frames = trace["frames"]
            for index in np.linspace(0, len(frames) - 1, 5).round().astype(int):
                images.append(frames[index])
                labels.append(f"p{row['pair_index']} seed{row['seed']} a{index}")
    gallery(images, labels, out / "quality" / (name + "-controls.png"))


def engineering_equivalence(out, state, name, seed):
    keys = [f"engineering_{name}_{role}_p000_none_s{seed}" for role in ("native", "zero")]
    traces = []
    for key in keys:
        base = out / "episodes" / key
        if name == "pusht":
            base = base / "episode"
        with np.load(base / "trace.npz", allow_pickle=False) as data:
            traces.append({k: data[k] for k in ("actions", "states", "dense", "frames")})
    if any(not np.array_equal(traces[0][k], traces[1][k]) for k in traces[0]):
        raise RuntimeError("Zero/native engineering probe differs")
    for field in ("predictor_steps", "model_batch_calls", "model_candidate_rollouts"):
        if state["jobs"][keys[0]]["resources"].get(field, 0) != state["jobs"][keys[1]][
            "resources"
        ].get(field, 0):
            raise RuntimeError("Zero correction changed native model work")
    write_json(
        out / "engineering" / (name + "-equivalence.json"),
        {
            "verified": True,
            "seed": seed,
            "episodes": 2,
            "max_actions_per_episode": 6,
            "new_simulator_interactions_in_audit": 0,
        },
    )


def controls(design, out):
    if out.exists():
        raise FileExistsError("Preserve attempts; use a new output namespace")
    out.mkdir(parents=True)
    write_json(
        out / "source-freeze.json",
        {
            "protocol_sha256": digest(ROOT / design["protocol_path"]),
            "sources": {
                str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "experiments").rglob("*.py")
            },
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "pod_id": design["execution_pod_id"],
        },
    )
    verify(design, out)
    state = {"phase": "controls", "jobs": {}, "active": None, "environment_states": {}}
    write_json(out / "state.json", state)
    for name in ("pusht", "cube"):
        rt = None
        try:
            reserve(design)
            rt = load_runtime(
                ROOT, design, name, out / "preparation" / (name + "-controls-constructor.json")
            )
            cases = prepare_environment(design, out, rt)
            write_json(out / "panels" / (name + ".json"), cases)
            seed = design["engineering_seed_base"][name]
            for role in ("native", "zero"):
                run_job(design, out, state, rt, cases[0], role, "none", seed, engineering=True)
            engineering_equivalence(out, state, name, seed)
            rows = []
            for case in cases:
                for repeat in range(2):
                    seed = design["planner_seed_base"][name] + 10 * case["index"] + repeat
                    rows.append(run_job(design, out, state, rt, case, "native", "none", seed))
            state["environment_states"][name] = {
                "phase": "awaiting_image_review",
                "native_control": summarize(rows),
                "performance_gate": no_hazard_gate(rows),
            }
            control_gallery(out, state, name)
        except Exception:
            state["environment_states"][name] = {
                "phase": "technical_failure",
                "error": traceback.format_exc(),
            }
            write_json(out / "technical" / (name + ".json"), state["environment_states"][name])
        finally:
            if rt is not None and rt.env is not None:
                rt.env.close()
            write_json(out / "state.json", state)
    state["phase"] = "awaiting_control_review"
    write_json(out / "state.json", state)


def main_grid(design, out):
    verify(design, out)
    state = json.loads((out / "state.json").read_text())
    if state["phase"] == "complete_with_declared_gates":
        return
    if state["phase"] != "awaiting_control_review":
        raise RuntimeError("Inspect the current worker before any continuation")
    review = json.loads((out / "control-review.json").read_text())
    for name in ("pusht", "cube"):
        status = state["environment_states"][name]
        if status["phase"] == "technical_failure":
            continue
        if name not in review or not review[name]["images_clean"] or not status["performance_gate"]:
            status["phase"] = "closed_at_readiness"
            continue
        rt = None
        try:
            reserve(design)
            rt = load_runtime(
                ROOT, design, name, out / "preparation" / (name + "-main-constructor.json")
            )
            if rt.env is not None:
                write_json(
                    out / "preparation" / (name + "-main-constructor.json"), dict(rt.counters)
                )
                rt.counters.clear()
            cases = json.loads((out / "panels" / (name + ".json")).read_text())
            # Finish native baseline, then paired fixed guidance. No parameter updates.
            for role in ("native", "untrained"):
                for case in cases:
                    for kind in KINDS:
                        for repeat in range(2):
                            seed = design["planner_seed_base"][name] + 10 * case["index"] + repeat
                            row = run_job(design, out, state, rt, case, role, kind, seed)
                            if row["frame_flags"]:
                                status["phase"] = "closed_for_flagged_input_review"
                                raise RuntimeError(
                                    "Potential policy-image corruption requires visual review; flags are not a validated classifier"
                                )
            status["phase"] = "complete"
        except Exception:
            if status["phase"] != "closed_for_flagged_input_review":
                status["phase"] = "technical_failure"
            status["error"] = traceback.format_exc()
        finally:
            if rt is not None and rt.env is not None:
                rt.env.close()
            write_json(out / "state.json", state)
    state["phase"] = "complete_with_declared_gates"
    state["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(out / "state.json", state)
    print(
        json.dumps(
            {
                "phase": state["phase"],
                "environments": state["environment_states"],
                "jobs": len(state["jobs"]),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("controls", "main"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text())
    if os.environ.get("RUNPOD_POD_ID") != design["execution_pod_id"]:
        raise RuntimeError("Only the declared rented GPU may run this screen")
    torch.set_num_threads(4)
    (controls if args.mode == "controls" else main_grid)(design, args.out)


if __name__ == "__main__":
    main()
