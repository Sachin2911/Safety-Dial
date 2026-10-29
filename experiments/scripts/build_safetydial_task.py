"""Assemble a private Hugging Face dataset for one dial study: task 12 (TwoRoom), 13 (Reacher)
or 14 (Cube), in the layout of SafetyDial-PushT.

From the synced evidence: every scored episode as a table row and a trajectory, the raw
attempt records by stage, the stage C search records, the diagnosis, the LeWM checkpoint
(checked against its pin), the analysis, the frozen protocol and the source at the C freeze
(checked file by file against that freeze), and the videos from render_dial_videos.py. Then
a scan for credentials, rental endpoints and local home paths (inside archives too), a
manifest, and with --upload a private upload verified file by file at the pinned revision.

    build_safetydial_task.py --task reacher13 [--upload] [--skip-build]
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc

FINAL = re.compile(r"-final-(fresh|holdout)-c(\d+)$")
SEARCH = re.compile(r"-g(\d+)-c(\d+)$")
METHOD = re.compile(r"^C-[^-]+-\d+-([a-z0-9]+)-r(\d+)-")
TEXT = {".json", ".jsonl", ".md", ".txt", ".py", ".sh", ".tex", ".csv", ".log", ".toml", ".cfg"}
# Not frozen with the run: the analysis (and the task-10 helpers it imports), the renderer
# and this builder, placed in the same tree so it runs as is.
EXTRAS = (
    "experiments/scripts/render_dial_videos.py",
    "experiments/scripts/build_safetydial_task.py",
    "experiments/scripts/analyze_safecem10.py",
    "experiments/helpers/safecem10_evo.py",
    "experiments/helpers/safecem10_geometry.py",
)

TASKS = {
    "tworoom12": {
        "number": 12,
        "repo": "Sachioster/SafetyDial-TwoRoom",
        "title": "SafetyDial-TwoRoom",
        "env": "TwoRoom",
        "prefix": "12-tworoom-",
        "plan": "progress/12-tworoom-dial-plan.md",
        "unit": "px",
        "per_unit": 1.0,
        "agent": "agent",
        "state_columns": ("agent_x", "agent_y"),
        "model_dir": "data/tworoom12/hf_tworoom",
        "converted": "data/tworoom12/stablewm/checkpoints/tworoom",
        "model_name": "lewm-tworoom",
        "metric_key": "min_target_distance_px",
        "environment": (
            "TwoRoom from stable-worldmodel 0.1.1: a 224 x 224 px top-down world of two rooms "
            "joined by a door, in which a dot moves under 2-D velocity commands (up to 5 px "
            "per step) and stops at walls."
        ),
        "task": (
            "Fixed start (164, 120) px and target (164, 170) px in the right room, 50 steps. "
            "Success is the simulator's own flag: the agent within 16 px of the target."
        ),
        "check": (
            "Exact. TwoRoom's dynamics (velocity integration with wall collisions) are known and "
            "deterministic, so each candidate plan's agent path is computed exactly from the "
            "true position at every replan; stage A replayed every executed step to 5e-5 px."
        ),
        "caveats": [
            "LeWM's TwoRoom goal cost is informative only within about 30 px of the goal, so "
            "the task is a short 50 px move; a door-crossing design was dropped as hopeless "
            "for the planner (see the plan in `protocol/`).",
        ],
    },
    "reacher13": {
        "number": 13,
        "repo": "Sachioster/SafetyDial-Reacher",
        "title": "SafetyDial-Reacher",
        "env": "Reacher",
        "prefix": "13-reacher-",
        "plan": None,
        "unit": "cm",
        "per_unit": 100.0,
        "agent": "fingertip",
        "state_columns": ("q1", "q2", "v1", "v2", "finger_x", "finger_y"),
        "model_dir": "data/reacher13/hf_model",
        "converted": "data/reacher13/stablewm/checkpoints/reacher",
        "model_name": "lewm-reacher",
        "metric_key": "min_target_distance_m",
        "environment": (
            "DeepMind Control Reacher in stable-worldmodel 0.1.1 (the qpos_match task): a "
            "two-link planar arm (links of 0.12 m) driven by joint torques, seen from above."
        ),
        "task": (
            "Fixed start pose (0.7, 0.9) rad and target pose (-0.7, 0.9) rad, 50 steps; the "
            "fingertip arcs from (0.088, 0.197) m to (0.209, -0.053) m. Success is the "
            "simulator's own flag: both joints within 0.05 rad of the target pose."
        ),
        "check": (
            "Exact. The arm's own MuJoCo model is rolled forward from the true joint state for "
            "every candidate (mujoco.rollout, each action held for the wrapper's two physics "
            "steps); stage A replayed every executed step to 1e-16 m."
        ),
        "caveats": [
            "The fixed sweep was chosen from a diagnosis of the free planner (6 of 8 seeds at "
            "this sweep; 13 of 24 random dataset pairs), so the study sits where the planner "
            "already works without a hazard.",
            "Task 12's 1000 samples with the top 10 left the free planner 0 of 6, so Reacher "
            "uses LeWM's own 300 samples with the top 30 in every condition.",
        ],
    },
    "cube14": {
        "number": 14,
        "repo": "Sachioster/SafetyDial-Cube",
        "title": "SafetyDial-Cube",
        "env": "OGBench Cube",
        "prefix": "14-cube-",
        "plan": None,
        "unit": "cm",
        "per_unit": 100.0,
        "agent": "gripper",
        "state_columns": (
            "effector_x",
            "effector_y",
            "effector_z",
            "cube_x",
            "cube_y",
            "cube_z",
            "gripper_opening",
        ),
        "model_dir": "data/cube14/hf_model",
        "converted": "data/cube14/stablewm/checkpoints/cube",
        "model_name": "lewm-cube",
        "metric_key": "min_effector_goal_m",
        "environment": (
            "OGBench Cube (one cube) in stable-worldmodel 0.1.1: a UR5e arm with a Robotiq "
            "gripper under relative end-effector control. Each action moves the gripper's "
            "target by up to 5 cm; a Python inverse-kinematics controller and joint PD loops "
            "chase it for 25 physics steps."
        ),
        "task": (
            "A reach on one LeWM dataset pair (rows 1899137 -> 1899162): start from the "
            "recorded simulator state and plan toward the frame recorded 25 steps later, "
            "50 steps. Success: the gripper's pinch point within 0.04 m (OGBench's own cube "
            "tolerance) of its position in that frame."
        ),
        "check": (
            "Approximate. Simulating the controller per candidate is too slow, so the check "
            "predicts the pinch point from the commands with a fitted tracking gain (0.40, "
            "0.41, 0.335 of each commanded step), interpolating four points per step. In "
            "stage A its one-step error was median 4.3 mm, max 1.8 cm; the dial is therefore "
            "also a margin for the check's own error."
        ),
        "caveats": [
            "The free planner completed no pick-and-place carry in the diagnosis (0 of 24 "
            "pairs moving the cube 8 cm or more), so the study is a reach task, not LeWM's "
            "cube-placement task.",
            "The arm is drawn opaque, as in the dataset frames (OGBench's pixel default is "
            "translucent); the target marker is drawn at the dataset's data-collection target.",
            "Stage C searched on layout C1, not C2 as first designed: every C2 cell of stage B "
            "scored 0 safe success, which would make every hypervolume 0.",
            "The videos draw the arm by replaying each episode's saved commands in a local "
            "simulator, which stays within 0.7 mm of the rental's record (floating point, under "
            "a pixel); the trail and the band counts come from the record itself.",
        ],
    },
}


QUESTIONS = (
    "Three questions: does the dial keep the {agent} out of the band, what does it cost in "
    "task completion, and can an evolutionary search over the dial's settings (NSGA-II) beat "
    "random search, a grid and any single fixed dial?"
)

# The card's plain-language part. Stage C findings are written after its analysis; the builder
# refuses to publish stage C results without them.
STORY = {
    "tworoom12": {
        "summary": "A safety dial for a world-model planner, tested on TwoRoom.",
        "idea": (
            "A dot moves through a two-room world toward a green target, steered by a planner "
            "that uses LeWorldModel (LeWM), a learned world model that imagines what the camera "
            "will see after a few moves. It plans with CEM: imagine 1000 candidate move "
            "sequences, refit to the best 10, repeat 10 times, carry out the first block of five "
            "moves, replan.",
            "We put a red band across the room between the start and the target, with one 24 px "
            "gap. The world model was never trained with the band, so it cannot see it. Safety "
            "comes from a check added to the planner (SafeCEM): every imagined plan whose dot "
            "enters the band *inflated by d pixels* ranks below every plan that stays clear. The "
            "dot's motion is known exactly, so the check is exact. **d is the safety dial.** At "
            "d = 12 px the gap is closed as far as the planner knows.",
        ),
        "findings": [
            "**With the exact check, the dial is a guarantee.** None of the 280 checked episodes "
            "in stage B entered the band, against 20 of 30 for the planner with no check.",
            "**A small dial makes the planner safer and faster.** Without the check the planner "
            "succeeds 6 times in 10 and its safe success is 0.4, 0.3 and 0.1 on the three "
            "layouts. With the check and d at most 2 px, safe success is 0.9 to 1.0 on every "
            "layout and episodes are 12 to 14 steps shorter. Part of that gain comes from the "
            "layout: the gap lines the dot up above the target, where the world model's goal "
            "cost is flat, so a layout without that alignment need not show the same gain.",
            "**The dial is a cliff, not a slope.** Safe success holds up to 4 px and is 0 from "
            "6 px on every layout, well before the gap closes at 12 px; the planner waits safely "
            "above the band.",
            "**The best fixed dial already reaches safe success 1.0,** so the stage C search can "
            "only improve how fast the planner gets there.",
        ],
        "findings_c": [
            "**Searching the planner's settings made it faster, but evolution did not win the "
            "search.** Random search beat NSGA-II and the grid on every hypervolume measure, in "
            "all 9 replicate pairs. Its best settings reach the target safely in 12 to 13 steps "
            "on fresh seeds and on the held-out layout, against about 21 for the best fixed dial "
            "in stage B (other seeds).",
            "**The lever was the planning horizon, not the dial.** Every top random-search setting "
            "plans 3 blocks ahead (stage B planned 5); the dial settings that reach safe success "
            "1.0 vary widely. NSGA-II kept a horizon-3 setting in one replicate of three, and the "
            "frozen grid (horizons 4 to 6) could not reach it.",
            "**The tolerance gene trades a little safety for flexibility.** 14 of 2892 stage C "
            "episodes entered the band, for 1 or 2 steps each, all with settings that tolerated "
            "at least 3.2 px of imagined penetration.",
        ],
    },
    "reacher13": {
        "summary": "A safety dial for a world-model planner, tested on DeepMind Control Reacher.",
        "idea": (
            "A two-link arm swings its fingertip from a start pose to a target pose, steered by "
            "a planner that uses LeWorldModel (LeWM), a learned world model that imagines what "
            "the camera will see after a few torque commands. It plans with CEM: imagine 300 "
            "candidate action sequences, refit to the best 30, repeat 10 times, carry out the "
            "first block of five actions, replan.",
            "We put a red band across the fingertip's route with one 5 cm gap. The world model "
            "was never trained with the band, so it cannot see it. Safety comes from a check "
            "added to the planner (SafeCEM): every imagined plan whose fingertip enters the band "
            "*inflated by d* ranks below every plan that stays clear. The check rolls the arm's "
            "own physics model forward, so it is exact. **d is the safety dial.** At d = 2.5 cm "
            "the gap is closed as far as the planner knows.",
        ),
        "findings": [
            "**With the exact check, the dial is a guarantee.** None of the 240 checked episodes "
            "in stage B entered the band, against 13 of 30 for the planner with no check.",
            "**The check helps where the free planner cuts through the band.** Without the check "
            "the planner succeeds 7 times in 10 on every layout, but its safe success falls to "
            "0.7, 0.4 and 0.1 as the gap moves 0, 1.25 and 2.5 cm off its route. With the check "
            "the best fixed dial reaches 0.8, 0.8 and 0.5. Where the free planner already "
            "threads the gap (R0), the check at d = 0 costs success: 0.3 against 0.7.",
            "**The dial is a cliff.** Every layout keeps most of its success up to 1 cm and "
            "falls to 0 to 0.1 from 1.5 cm, well before the gap closes at 2.5 cm; beyond the "
            "cliff the arm stops 7 to 9 cm short of the band.",
        ],
        "findings_c": [
            "**Searching the settings raised safe success where it searched, not on the "
            "held-out layout.** Every run's final front reached safe success 0.6 to 0.9 on fresh "
            "seeds of the search layout R2 (mean 0.74), against 0.5 for the best fixed dial in "
            "stage B (other seeds). On the held-out layout R0 the fronts averaged 0.62, against "
            "0.8 for the best fixed dial: the same pattern as on Push-T.",
            "**NSGA-II and random search tie; both beat a fixed grid.** The two are within noise "
            "on every measure (NSGA-II wins 5 of 9 replicate pairs on each hypervolume measure, "
            "and both reach best safe success 0.77 on fresh seeds).",
            "**The lever is again the planning horizon.** Every top setting plans 3 blocks ahead. "
            "Across the whole search, safe success was 0.59 at horizon 3 against 0.26 at stage "
            "B's horizon of 5; the frozen grid (horizons 4 to 6) could not reach it.",
            "**Safety held.** 2 of 2792 stage C episodes entered the band, for one step each, "
            "both with settings that tolerated more than 1 cm of imagined penetration.",
        ],
    },
    "cube14": {
        "summary": "A safety dial for a world-model planner, tested on a robot arm (OGBench Cube).",
        "idea": (
            "A UR5e robot arm moves its gripper to where it is in a goal image, steered by a "
            "planner that uses LeWorldModel (LeWM), a learned world model that imagines what the "
            "camera will see after a few commands. Each command moves the gripper's target by "
            "up to 5 cm and a controller chases it. The planner uses CEM: imagine 300 candidate "
            "command sequences, refit to the best 30, repeat 10 times, carry out the first block "
            "of five commands, replan.",
            "We put a keep-out band across the reach with one 8 cm gap. The world model was "
            "never trained with the band, so it cannot see it. Safety comes from a check added "
            "to the planner (SafeCEM): every imagined plan whose gripper enters the band "
            "*inflated by d* ranks below every plan that stays clear. Unlike the TwoRoom and "
            "Reacher studies, this check is approximate: it predicts the gripper's path from the "
            "commands with a fitted tracking gain (the arm covers about 41% of each commanded "
            "step), so the dial is also a margin for the check's own error. **d is the safety "
            "dial.** At d = 4 cm the gap is closed as far as the planner knows.",
        ),
        "findings": [
            "**Without the check the arm cuts through the band.** It reaches the goal 8 times in "
            "10 on every layout but enters the band in 23 of 30 episodes, so its safe success "
            "is only 0.2, 0.2 and 0.0.",
            "**The approximate check still keeps the gripper out almost always.** 1 of 200 "
            "checked episodes in stage B touched the band, for one saved step at d = 0.",
            "**At d = 0 the check gains a little; any margin stops the arm.** Safe success at "
            "d = 0 is 0.3 on C0 and C1 against 0.2 without the check, and 0 on C2. From d = 1 cm "
            "no checked episode reached the goal: the gripper descends without progress and "
            "stalls 13 to 21 cm away. LeWM's goal cost is flat over the last 15 to 20 cm, so once "
            "the direct plans are ruled out nothing pulls the planner on.",
        ],
        "findings_c": [
            "**On the search layout, searching the settings did not beat the best fixed dial.** "
            "The final fronts reached safe success 0.0 to 0.5 on fresh seeds of C1 (mean 0.24), "
            "against 0.3 for the check at d = 0 in stage B (other seeds).",
            "**On the held-out layout, random search's settings did.** On C0, where the gap sits "
            "on the free route, they reached 0.5, 0.7 and 0.9 (mean 0.70) against 0.3 for the "
            "best fixed dial. The best of them scored only 0.1 on fresh C1 seeds, so it suits the "
            "easy layout rather than being robust.",
            "**NSGA-II came last, behind even a fixed grid.** Its fronts settled on large "
            "near-term clearances (1.7 to 2.8 cm) that scored well on their 3 search seeds but "
            "reached safe success 0.0 to 0.3 on the final seeds, as stage B predicts for margins "
            "of 1 cm or more. With successes this sparse (8.8% across the search), a 3-seed "
            "score is very noisy; a plausible reading is that elitist selection kept the lucky "
            "candidates (one front is four copies of one setting).",
            "**Safety mostly held with the approximate check.** 9 of 2872 stage C episodes "
            "entered the band, for one saved step each: three at zero margin and zero tolerance "
            "(the check's own error), six with settings that tolerated imagined penetration.",
        ],
    },
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


class Task:
    def __init__(self, ns):
        self.ns = ns
        self.cfg = TASKS[ns]
        self.spec = json.loads((ROOT / f"configs/{ns}/design.json").read_text())
        self.remote = ROOT / f"outputs/{ns}/remote"
        self.analysis = ROOT / f"outputs/{ns}/analysis"
        self.cache = ROOT / f"outputs/{ns}/hf-verify-cache"
        geo = __import__(f"helpers.{ns}_geometry", fromlist=["hazard_boxes"])
        self.geo = geo
        self.home_allowed = {f"code/{ns}/experiments/helpers/dsrl06_infra.py"}

    def start_row(self, record):
        env = self.spec["environment"]
        if self.ns == "tworoom12":
            return np.asarray(env["start_agent"], dtype=float)
        if self.ns == "reacher13":
            tip = self.geo.fingertip(env["start_qpos"], env["link_lengths"])[0]
            return np.asarray([*env["start_qpos"], 0.0, 0.0, *tip], dtype=float)
        return np.asarray(record["start"], dtype=float)

    def point_columns(self):
        return {"tworoom12": (0, 1), "reacher13": (4, 5), "cube14": (0, 1)}[self.ns]


def flatten(prefix, node, out):
    for key, value in node.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flatten(f"{name}_", value, out)
        elif isinstance(value, list):
            if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                out[f"{name}_min"], out[f"{name}_max"] = float(value[0]), float(value[1])
            else:
                out[name] = json.dumps(value)
        else:
            out[name] = value


def planner_label(job):
    execute = job["planner"].get("execute", "mean")
    if job["stage"] == "C":
        return f"searched setting, safety check, {execute} execution"
    if job["mode"] == "penalty":
        return f"no safety check, {execute} execution"
    return f"safety check, {execute} execution"


def episode_row(task, result, videos):
    job, planner = result["job"], result["job"]["planner"]
    final, search, method = (
        FINAL.search(job["id"]),
        SEARCH.search(job["id"]),
        METHOD.search(job["id"]),
    )
    params = job.get("params") or {}
    k = task.cfg["per_unit"]
    # A stage C row carries a schedule (genes d0, d1), not one dial.
    dial = job["clearances"][0] if job["mode"] == "safe" and job["stage"] != "C" else None
    row = {
        "episode_id": job["id"],
        "stage": job["stage"],
        "layout": job["layout"],
        "seed": job["seed"],
        "planner_label": planner_label(job),
        "condition": job.get("condition"),
        "mode": job["mode"],
        "lam": job["lam"],
        "execute": planner.get("execute", "mean"),
        "dial": dial,
        f"dial_{task.cfg['unit']}": None if dial is None else dial * k,
        "clearances": json.dumps(job["clearances"]),
        "method": method.group(1) if method else None,
        "replicate": int(method.group(2)) if method else None,
        "phase": "final" if final else ("search" if search else None),
        "generation": int(search.group(1)) if search else None,
        "final_set": final.group(1) if final else None,
        "candidate": int((final or search).group(2)) if (final or search) else None,
        "gene_d0": params.get("d0"),
        "gene_d1": params.get("d1"),
        "gene_horizon": params.get("horizon"),
        "gene_topk_frac": params.get("topk_frac"),
        "gene_tol": params.get("tol"),
        "num_samples": planner["num_samples"],
        "cem_iters_per_solve": planner["n_steps"],
        "topk": planner["topk"],
        "horizon": planner["horizon"],
        "receding_horizon": planner["receding_horizon"],
        "action_block": planner["action_block"],
        "feasible_tol": planner["feasible_tol"],
        "safe_success": bool(result["success"]) and not result["any_violation"],
    }
    rest = {}
    skip = {"job", "status", "worker"}
    flatten("", {key: v for key, v in result.items() if key not in skip}, rest)
    for key, value in rest.items():
        row.setdefault(key, value)
    row["video"] = videos.get(job["id"])
    return row


def build_tables(task, out):
    """data/episodes.{parquet,csv} and data/trajectories.parquet from the synced attempts."""
    import pandas as pd

    manifest = json.loads((out / "videos/manifest.json").read_text())
    videos = {job_id: f"videos/episodes/{job_id}.mp4" for job_id in manifest["episodes"]}
    rows, trajectories = [], []
    for path in sorted((task.remote / "attempts").glob("*/result.json")):
        result = json.loads(path.read_text())
        if result["status"] != "complete":
            continue
        record = dict(np.load(path.parent / "states.npz"))
        states = record["states"]
        if not np.isfinite(states).all() or len(states) != result["n_states"]:
            raise ValueError(f"{path.parent.name}: saved states do not match the record")
        if digest(path.parent / "states.npz") != result["states_sha256"]:
            raise ValueError(f"{path.parent.name}: states.npz differs from its recorded hash")
        boxes = task.geo.hazard_boxes(
            task.geo.layout(result["job"]["layout"], task.spec), task.spec
        )
        rows.append(episode_row(task, result, videos))
        full = np.vstack([task.start_row(record)[None, : states.shape[1]], states])
        columns = task.cfg["state_columns"][: states.shape[1]]
        frame = pd.DataFrame(full, columns=columns)
        frame.insert(0, "t", np.arange(len(full), dtype=np.int16))
        frame.insert(0, "episode_id", result["job"]["id"])
        actions = record.get("actions")
        if actions is not None and len(actions) == len(states):
            padded = np.vstack([actions, np.full((1, actions.shape[1]), np.nan)])
            for i in range(actions.shape[1]):
                frame[f"action_{i}"] = padded[:, i]  # action taken from this row's state
        x, y = task.point_columns()
        frame["inside_band"] = task.geo.inside_any(full[:, [x, y]], boxes)
        trajectories.append(frame)
    table = pd.DataFrame(rows)
    for column in ("replicate", "generation", "candidate", "gene_horizon", "steps_to_success"):
        if column in table:
            table[column] = table[column].astype("Int64")
    order = {"A": 0, "B": 1, "C": 2}
    table = table.sort_values(
        ["stage", "episode_id"], key=lambda c: c.map(order) if c.name == "stage" else c
    )
    (out / "data").mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "data/episodes.parquet", index=False)
    table.to_csv(out / "data/episodes.csv", index=False)
    traj = pd.concat(trajectories, ignore_index=True)
    traj.to_parquet(out / "data/trajectories.parquet", index=False)
    missing = set(videos) - set(table["episode_id"])
    if missing:
        raise ValueError(f"videos without a scored episode: {sorted(missing)[:3]}")
    return table, len(traj)


def checkpoint_files(task, out):
    pins = task.spec["pins"]
    model = out / "checkpoints" / task.cfg["model_name"]
    source = ROOT / task.cfg["model_dir"]
    for name in ("config.json", "weights.pt"):
        copy(source / name, model / name)
    weights = model / "weights.pt"
    if (
        weights.stat().st_size != pins["model_weights_bytes"]
        or digest(weights) != pins["model_weights_sha256"]
    ):
        raise ValueError("weights.pt differs from the pinned upstream checkpoint")
    converted = ROOT / task.cfg["converted"]
    for name in ("lewm_object.ckpt", "lewm_object.json"):
        copy(converted / name, model / "converted" / name)
    meta = json.loads((converted / "lewm_object.json").read_text())
    if meta.get("source_sha256") != pins["model_weights_sha256"]:
        raise ValueError("the converted checkpoint was not built from the pinned weights")
    from huggingface_hub import hf_hub_download

    card = hf_hub_download(
        pins["model_repo"], "README.md", revision=pins["model_revision"], cache_dir=task.cache
    )
    copy(Path(card), model / "UPSTREAM_README.md")
    return Path(card).read_text()


def tar_stage(task, stage, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in (task.remote / "attempts").glob(f"{stage}-*") if p.is_dir())
    with tarfile.open(target, "w:gz") as archive:
        for folder in folders:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    info = archive.gettarinfo(
                        path, arcname=f"attempts/{path.relative_to(task.remote / 'attempts')}"
                    )
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
    return len(folders)


def data_files(task, out):
    counts = {}
    for stage in ("A", "B", "C"):
        counts[stage] = tar_stage(task, stage, out / "data/raw" / f"attempts-{stage}.tar.gz")
    if (task.remote / "stageC").exists():
        shutil.copytree(task.remote / "stageC", out / "data/search/stageC")
    diagnosis = ROOT / f"outputs/{task.ns}/diagnosis"
    if diagnosis.exists():
        for path in sorted(diagnosis.iterdir()):
            if path.is_file():
                copy(path, out / "data/diagnosis" / path.name)
    for extra in ("scene", "side-probes"):
        folder = ROOT / f"outputs/{task.ns}/{extra}"
        if folder.exists():
            for path in sorted(folder.iterdir()):
                copy(path, out / f"data/{extra}" / path.name)
    return counts


def result_files(task, out):
    for path in sorted(task.analysis.iterdir()):
        if path.is_file():
            copy(path, out / "results/analysis" / path.name)


def protocol_files(task, out):
    protocol = out / "protocol"
    for path in sorted((ROOT / f"configs/{task.ns}").iterdir()):
        copy(path, protocol / "configs" / path.name)
    if task.cfg["plan"] and (ROOT / task.cfg["plan"]).exists():
        copy(ROOT / task.cfg["plan"], protocol / Path(task.cfg["plan"]).name)
    for path in sorted((ROOT / "progress/receipts").glob(f"{task.cfg['prefix']}*")):
        if "preservation" in path.name and path.suffix == ".json":
            continue  # markers hold local paths only; the facts are in the receipts
        copy(path, protocol / "receipts" / path.name)
    copy(ROOT / f"outputs/{task.ns}/engineering.json", protocol / "engineering.json")
    markers = ["prepare-complete.json", *sorted(p.name for p in task.remote.glob("stage-*.json"))]
    markers += sorted(p.name for p in task.remote.glob("STOP-*.txt"))
    for name in markers:
        copy(task.remote / name, protocol / "run-markers" / name)
    for name in ("versions.json", "installed-freeze.txt", "render-backend.json"):
        if (task.remote / "environment" / name).exists():
            copy(task.remote / "environment" / name, protocol / "environment" / name)


def code_files(task, out):
    """The source at the stage C freeze, each file checked against the freeze's hash."""
    freezes = sorted((ROOT / "progress/receipts").glob(f"{task.cfg['prefix']}C-v*.json"))
    if not freezes:
        raise ValueError("No stage C freeze")
    freeze = json.loads(freezes[-1].read_text())
    for rel, sha in sorted(freeze["source_sha256"].items()):
        src = ROOT / rel
        if digest(src) != sha:
            raise ValueError(f"{rel} changed after the stage C freeze")
        copy(src, out / "code" / task.ns / rel)
    for rel in (f"experiments/scripts/analyze_{task.ns}.py", *EXTRAS):
        if rel in freeze["source_sha256"]:
            raise ValueError(f"{rel} is frozen; it would be overwritten")
        copy(ROOT / rel, out / "code" / task.ns / rel)
    return freezes[-1].name


def rentals(task):
    return sorted((ROOT / "progress").glob(f"rental-{task.ns}*.json"))


def sensitive_values(task):
    """Byte strings that must never appear in the dataset: credentials and rental endpoints."""
    from dotenv import dotenv_values

    values = [
        v for k, v in dotenv_values(ROOT / ".env").items() if v and ("TOKEN" in k or "KEY" in k)
    ]

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if ("ipaddr" in key or key == "ssh_host") and value:
                    values.append(str(value))
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for path in rentals(task):
        walk(json.loads(path.read_text()))
    return [v.encode() for v in values if len(v) >= 6]


def scan(task, out):
    """Fail on any credential, rental endpoint or local home path, inside archives too."""
    needles = sensitive_values(task)
    home = str(Path.home()).encode()

    def check(label, data, text):
        if any(n in data for n in needles):
            raise ValueError(f"sensitive value in {label}")
        if text and home in data:
            raise ValueError(f"local home path in {label}")

    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        label = str(path.relative_to(out))
        check(label, path.read_bytes(), path.suffix in TEXT and label not in task.home_allowed)
        if path.name.endswith(".tar.gz"):
            with tarfile.open(path, mode="r|gz") as archive:
                for member in archive:
                    if member.isfile():
                        inner = archive.extractfile(member).read()
                        check(f"{path.name}:{member.name}", inner, Path(member.name).suffix in TEXT)


def fmt(x, digits=2):
    return "" if x is None else f"{x:.{digits}f}"


def layout_role(task, lid):
    c = task.spec["stages"]["C"]
    if lid == c["search_layout"]:
        return "stage C search"
    if lid == c["holdout_layout"]:
        return "stage C held out"
    return "stage B only"


def stage_b_table(task):
    summary = json.loads((task.analysis / "stageB-summary.json").read_text())
    u, k = task.cfg["unit"], task.cfg["per_unit"]
    groups = summary["groups"]
    lines = [
        "| layout | gap offset | free: success / safe success / entries | best checked dial | "
        "its safe success | checked entries, all dials | gap closes at |",
        "|---|---|---|---|---|---|---|",
    ]
    for lay in task.spec["hazard"]["layouts"]:
        lid = lay["id"]
        free = next(g for g in groups if g["layout"] == lid and g["condition"] == "free")
        safe = [g for g in groups if g["layout"] == lid and g["condition"] == "safe"]
        best = summary["best_fixed_dial"].get(lid, {})
        offset = lay.get("offset_from_route_m", lay.get("offset_from_route_px", 0.0)) * k
        entries = sum(round(g["violation_episode_rate"] * g["n"]) for g in safe)
        lines.append(
            f"| {lid} ({layout_role(task, lid)}) | {offset:g} {u} | "
            f"{fmt(free['success_rate'])} / {fmt(free['safe_success_rate'])} / "
            f"{round(free['violation_episode_rate'] * free['n'])} of {free['n']} | "
            f"{best.get('dial', 0) * k:g} {u} | {fmt(best.get('safe_success_rate'))} | "
            f"{entries} of {sum(g['n'] for g in safe)} | {lay['gap_width'] / 2 * k:g} {u} |"
        )
    return "\n".join(lines), summary


def stage_c_table(task):
    path = task.analysis / "stageC-summary.json"
    if not path.exists():
        return None, None
    summary = json.loads(path.read_text())
    labels = {"nsga2": "NSGA-II", "random": "random search", "grid": "grid search"}
    lines = [
        "| method | replicates | search HV | fresh-seed HV | held-out HV | "
        "best safe success, fresh seeds | best safe success, held-out layout |",
        "|---|---|---|---|---|---|---|",
    ]
    for method, entry in summary["methods"].items():
        cells = []
        for key in (
            "search_hv",
            "fresh_hv",
            "holdout_hv",
            "fresh_best_safe_success",
            "holdout_best_safe_success",
        ):
            value = entry.get(key)
            cells.append(f"{value['mean']:.2f} ± {value['sd']:.2f}" if value else "")
        lines.append(
            f"| {labels.get(method, method)} | {entry['replicates']} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines), summary


def gene_range(task, name, gene):
    lo, hi = gene["low"], gene["high"]
    if gene["unit"].startswith("m"):
        k, u = task.cfg["per_unit"], task.cfg["unit"]
        return f"{lo * k:g} to {hi * k:g} {u}"
    if gene["unit"].startswith("px"):
        return f"{lo:g} to {hi:g} px"
    return f"{lo:g} to {hi:g}"


def stage_c_design(task):
    c = task.spec["stages"]["C"]
    genes = c["genes"]
    rung = c["ladder"][0]
    labels = {
        "d0": "the clearance for the first imagined block, d0",
        "d1": "its change per block, d1",
        "horizon": "the horizon in blocks",
        "topk_frac": "the elite fraction",
        "tol": "the tolerated summed penetration",
    }
    parts = ", ".join(f"{labels[n]} ({gene_range(task, n, genes[n])})" for n in genes)
    return (
        f"NSGA-II, random search and a fixed grid each tune five settings: {parts}. The dial "
        "becomes a schedule over the imagined blocks, d(t) = d0 + d1 t, clipped at 0. "
        f"Population {c['population']}, {rung['generations']} generations, "
        f"{rung['replicates']} replicates per method; each candidate is scored on "
        f"{c['eval_seeds_per_candidate']} seeds of layout {c['search_layout']} by two "
        "minimized objectives, mean exposure and mean J (steps to success, the "
        f"{task.spec['environment']['eval_budget_steps']}-step budget when the episode does "
        f"not succeed). Each run's final front (up to {c['front_cap']} settings) is re-run "
        f"on {len(c['final_seeds'])} fresh seeds of layout {c['search_layout']} and on the "
        f"held-out layout {c['holdout_layout']}."
    )


def video_rows(out):
    manifest = json.loads((out / "videos/manifest.json").read_text())
    lines, gifs = [], []
    for item in manifest["sets"]:
        lines.append(
            f"- `videos/{item['name']}.mp4`: {item['heading'][0]}. Chosen by: {item['rule']}."
        )
        if (out / "videos" / f"{item['name']}.gif").exists():
            gifs.append((item["name"], item["heading"][0]))
    return lines, gifs, len(manifest["episodes"])


def upstream_license(card):
    match = re.search(r"^license:\s*(\S+)", card, re.MULTILINE)
    return match.group(1) if match else "unknown"


def readme(task, out, table, n_traj, card, freeze_name, counts):
    cfg, spec = task.cfg, task.spec
    pins = spec["pins"]
    u, k = cfg["unit"], cfg["per_unit"]
    b_table, b_summary = stage_b_table(task)
    c_table, c_summary = stage_c_table(task)
    vlines, gifs, n_clips = video_rows(out)
    stages = table["stage"].value_counts().to_dict()
    hz = spec["hazard"]
    dials = ", ".join(f"{d * k:g}" for d in spec["stages"]["B"]["dials"])
    lic = upstream_license(card)
    planner = spec["planner"]
    c = spec["stages"]["C"]
    gpu = json.loads(rentals(task)[-1].read_text())["offer"]["gpu_name"]
    story = STORY[task.ns]
    if c_table and not story["findings_c"]:
        raise ValueError("write the stage C findings (STORY) before publishing stage C results")
    findings = "\n".join(
        f"{i}. {text}" for i, text in enumerate(story["findings"] + story["findings_c"], 1)
    )
    top_gif = c_gif = ""
    for name, heading in gifs:
        if name.startswith("dial-sweep/") and not top_gif:
            lay, seed = name.split("/")[1].split("-seed")
            top_gif = (
                f"![{heading}](videos/{name}.gif)\n\n*One episode (layout {lay}, seed {seed}) "
                "at every stage B dial, next to the planner with no safety check (top left). "
                "Red: the true band. Dashed amber: the band inflated by the dial. Blue: the "
                f"{cfg['agent']}'s path. Green: the target.*"
            )
        elif name.startswith("search/") and not c_gif:
            c_gif = f"![{heading}](videos/{name}.gif)\n\n*{heading}.*\n"
    caveats = "\n".join(f"- {line}" for line in cfg["caveats"])
    preserved = sorted((ROOT / "progress/receipts").glob(f"{cfg['prefix']}preservation*.json"))
    evidence = ""
    if preserved:
        marker = json.loads(preserved[-1].read_text())
        revs = marker.get("pinned_revisions") or {}
        evidence = "; ".join(
            f"`{repo}` (under `recordings/{task.ns}` and `receipts/{task.ns}`) at revision `{rev}`"
            for repo, rev in revs.items()
        )
    provenance = [
        f"World model `{pins['model_repo']}` at revision `{pins['model_revision']}`; `weights.pt` "
        f"sha256 `{pins['model_weights_sha256']}`, checked against the bytes here when this "
        "dataset was built.",
        f"Training data (not redistributed): dataset repo `{pins['dataset_repo']}` at revision "
        f"`{pins['dataset_revision']}`, file `{pins['dataset_file']}` (sha256 "
        f"`{pins['dataset_sha256']}`).",
        f"Every stage was frozen before its first episode: `protocol/receipts/{cfg['prefix']}*.json`.",
        *(
            [f"Raw evidence with run logs is preserved privately in {evidence}."]
            if evidence
            else []
        ),
        f"Software: torch {pins['torch']}, stable-worldmodel {pins['stable_worldmodel']}, "
        f"stable-pretraining {pins['stable_pretraining']} (`protocol/environment`). Hardware: "
        f"one rented {gpu}.",
    ]
    c_section = (
        f"""### Stage C: searching the planner's settings

{c_table}

Mean ± sd over replicates. Hypervolume (HV) is measured against the reference point
(exposure 1, J {c["reference_point"][1]}); higher is better. Search HV covers every candidate
the run scored during the search; fresh-seed and held-out HV re-run the run's final front on
{len(c["final_seeds"])} new seeds of layout {c["search_layout"]} and on the held-out layout
{c["holdout_layout"]}. Best safe success is the best of up to {c["front_cap"]} front settings
on those seeds, so it is optimistic. Pairwise tests: `results/analysis/stageC-summary.json`.

![Stage C: fronts per method](results/analysis/stageC-fronts.png)

{c_gif}"""
        if c_table
        else "Stage C results are not in this version.\n"
    )
    return f"""---
license: mit
pretty_name: {cfg["title"]}
tags:
  - world-models
  - model-predictive-control
  - safe-planning
  - robotics
  - lewm
configs:
  - config_name: episodes
    data_files: data/episodes.parquet
  - config_name: trajectories
    data_files: data/trajectories.parquet
---

# {cfg["title"]}

**{story["summary"]}** Every episode, trajectory, checkpoint, figure and video from task
{cfg["number"]} of the Safety-Dial project (Sachin Mohan, BSc Honours in Computer Science,
University of the Witwatersrand, 2026): {len(table)} scored episodes (stage A
{stages.get("A", 0)}, stage B {stages.get("B", 0)}, stage C {stages.get("C", 0)}),
{n_traj} trajectory rows and {n_clips} episode videos.

{top_gif}

## The idea in one minute

{chr(10).join(p + chr(10) for p in story["idea"])}
{QUESTIONS.format(agent=cfg["agent"])}

## What we found

{findings}

## The task

- **Environment.** {cfg["environment"]}
- **Task.** {cfg["task"]}
- **Hazard.** {hz["description"]}
- **Violation.** {spec["environment"]["violation_rule"].capitalize()}. Exposure is the fraction of an
  episode's saved states inside the band; safe success is success with no state inside it.
- **Layouts.** {", ".join(f"{lay['id']} ({layout_role(task, lay['id'])}, gap {lay.get('offset_from_route_m', lay.get('offset_from_route_px', 0)) * k:g} {u} off the free route)" for lay in hz["layouts"])}.

## The planner

- World model: LeWM ([`{pins["model_repo"]}`](https://huggingface.co/{pins["model_repo"]}),
  pixels only), frozen; no weights are updated anywhere in this study.
- CEM: {planner["num_samples"]} samples, {planner["n_steps"]} iterations per replan, top
  {planner["topk"]}, horizon {planner["horizon"]} blocks of {planner["action_block"]} actions,
  replanning every block, 50-step budget.
- Safety check: {cfg["check"]}
- Ranking: feasible candidates (summed penetration of the inflated band within the
  tolerance) rank before infeasible ones; execution `mean_safe` runs the elite mean if it
  passes the same check, else the best checked candidate.
- Dial values in stage B: {dials} {u}.
- Stage C: {stage_c_design(task)}

## Results

### Stage B: the dial sweep (10 paired seeds per cell)

{b_table}

![Stage B: safe success against the dial](results/analysis/stageB-response.png)

Full table: `results/analysis/stageB-table.md`.

{c_section}
## Videos

Every video is redrawn from saved records; the red band, the dashed amber band inflated by
the dial, the green target and the blue trail are overlays the world model never saw.

{chr(10).join(vlines)}

Single-episode clips: `videos/episodes/<episode_id>.mp4`, linked from the `video` column.

## Files

- `data/episodes.parquet` (and `.csv`): one row per scored episode with the job (stage,
  layout, seed, condition, dial, genes), the planner settings and every recorded outcome.
- `data/trajectories.parquet`: one row per saved state, t = 0 is the start; state columns
  {", ".join(f"`{c}`" for c in cfg["state_columns"])}, actions where recorded, and
  `inside_band`.
- `data/raw/attempts-{{A,B,C}}.tar.gz`: the raw attempt folders ({counts.get("A", 0)},
  {counts.get("B", 0)} and {counts.get("C", 0)} episodes): `result.json`, `states.npz`,
  `history.json` (per-iteration CEM records).
- `data/search/stageC`: the search records per method and replicate.
- `data/diagnosis` and similar: the planner diagnosis that fixed the design.
- `checkpoints/{cfg["model_name"]}`: the LeWM weights as published upstream, the converted
  object checkpoint the planner loads, and the upstream card.
- `results/analysis`: summaries, tables and figures.
- `protocol`: the design, every freeze (job list, design and source hashes, written before
  the stage's first episode) and receipt, the engineering admission and run markers.
- `code/{task.ns}`: the source as frozen for stage C (`{freeze_name}`, which lists the
  frozen files and their hashes), each file checked against that freeze when this dataset was
  built, plus the analysis, the renderer and this builder in the same tree.

## Loading

```python
import pandas as pd
from huggingface_hub import hf_hub_download

def get(name):
    return hf_hub_download("{cfg["repo"]}", name, repo_type="dataset")

episodes = pd.read_parquet(get("data/episodes.parquet"))
sweep = episodes[episodes.stage == "B"]
print(sweep.groupby(["layout", "condition", "dial_{u}"]).safe_success.mean())
```

## Provenance

{chr(10).join(f"- {line}" for line in provenance)}

## Caveats

- Ten seeds per stage B cell and three replicates per method in stage C; differences
  between search methods are not conclusive.
{caveats}

## License

The world-model files in `checkpoints/{cfg["model_name"]}` are redistributed unchanged (and
in converted form) from [`{pins["model_repo"]}`](https://huggingface.co/{pins["model_repo"]})
under its license ({lic}); its card is included as `UPSTREAM_README.md`. Everything else here
is released under the MIT license.

## Citation

```bibtex
@misc{{mohan2026{cfg["title"].lower().replace("-", "")},
  author = {{Mohan, Sachin}},
  title = {{{cfg["title"]}: a safety dial for a world-model planner on {cfg["env"]}}},
  year = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{cfg["repo"]}}}}}
}}
```
"""


def write_manifest(out):
    path = out / "sha256-manifest.json"
    manifest = {
        str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(out.rglob("*"))
        if p.is_file() and p != path
    }
    atomic_json(path, manifest)
    return manifest


def build(task, out):
    if not (out / "videos/manifest.json").exists():
        raise ValueError(
            f"render the videos first: render_dial_videos.py --task {task.ns} --out {out}/videos"
        )
    for child in out.iterdir():
        if child.name != "videos":
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    table, n_traj = build_tables(task, out)
    card = checkpoint_files(task, out)
    counts = data_files(task, out)
    result_files(task, out)
    protocol_files(task, out)
    freeze_name = code_files(task, out)
    (out / "README.md").write_text(readme(task, out, table, n_traj, card, freeze_name, counts))
    scan(task, out)
    return write_manifest(out)


def upload(task, out):
    from dotenv import dotenv_values
    from huggingface_hub import HfApi, hf_hub_download

    repo = task.cfg["repo"]
    manifest = json.loads((out / "sha256-manifest.json").read_text())
    scan(task, out)
    token = dotenv_values(ROOT / ".env").get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo, repo_type="dataset")
    if not info.private:
        raise ValueError("Destination is not private")
    commit = api.upload_folder(
        folder_path=out,
        repo_id=repo,
        repo_type="dataset",
        parent_commit=info.sha,
        commit_message=f"{task.cfg['title']}: task-{task.cfg['number']} episodes, checkpoint, results and videos",
    )
    revision = commit.oid
    remote = set(api.list_repo_files(repo, repo_type="dataset", revision=revision))
    if remote - {".gitattributes"} != set(manifest) | {"sha256-manifest.json"}:
        raise ValueError("Uploaded inventory differs from the local folder")

    def verify(rel):
        copied = Path(
            hf_hub_download(
                repo, rel, repo_type="dataset", revision=revision, token=token, cache_dir=task.cache
            )
        )
        item = manifest.get(rel) or {
            "bytes": (out / rel).stat().st_size,
            "sha256": digest(out / rel),
        }
        if copied.stat().st_size != item["bytes"] or digest(copied) != item["sha256"]:
            raise ValueError(f"Uploaded file differs: {rel}")
        return rel

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(verify, sorted(set(manifest) | {"sha256-manifest.json"})))
    record = {
        "utc": utc(),
        "repo": repo,
        "private": True,
        "revision": revision,
        "verified_files": len(checked),
        "manifest_sha256": digest(out / "sha256-manifest.json"),
    }
    atomic_json(out.parent / f"{task.cfg['title'].lower()}-upload.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--upload", action="store_true", help="upload the built folder and verify it"
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="upload the folder as already built"
    )
    args = parser.parse_args()
    task = Task(args.task)
    out = args.out or ROOT / "outputs" / task.cfg["title"].lower()
    if not args.skip_build:
        manifest = build(task, out)
        print(
            json.dumps(
                {"files": len(manifest), "bytes": sum(x["bytes"] for x in manifest.values())}
            ),
            flush=True,
        )
    if args.upload:
        print(json.dumps(upload(task, out)), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Messages from the hub client are withheld; this script's own messages name files only.
        own = not type(error).__module__.startswith(
            ("huggingface_hub", "requests", "urllib3", "httpx")
        )
        report = {"error_type": type(error).__name__, "error": str(error)[:300] if own else ""}
        print(json.dumps(report), file=sys.stderr)
        raise SystemExit(1) from None
