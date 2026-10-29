"""Assemble SafetyDial-PushT, the browsable dataset of task 10, and optionally upload it.

One folder holds everything a reader needs: the LeWM Push-T checkpoint the planner used
(upstream bytes, pinned) and its converted form, both pusher probes, one table row per
episode, every saved trajectory, the raw per-episode records, the search records, the
analysis, the paper, the protocol, the source that ran, and the videos rendered beforehand by
render_safecem10_videos.py into <out>/videos. Rental records and preservation markers stay
out because they carry host addresses and local paths and say nothing about the experiment.
With --upload the folder goes to a private dataset repo and every file is verified by pinned
revision.
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
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from analyze_safecem10 import realized_clearance
from helpers.dsrl06_infra import atomic_json, utc
from helpers.safecem10_geometry import design, gate_boxes, inside_any, layout
from helpers.safecem10_planner import make_state
from helpers.safecem10_protocol import digest

REPO = "Sachioster/SafetyDial-PushT"
EVIDENCE = (
    "Sachioster/maniguard-jar-replication-20261022",
    "c799121672be8557425f06cc8d538d0f6369f11b",
)
REMOTE = ROOT / "outputs/safecem10/remote"
HF_STAGE = ROOT / "outputs/safecem10/hf-stage"
CACHE = ROOT / "outputs/safecem10/hf-verify-cache"
RENTALS = ("progress/rental-safecem10-20261025.json", "progress/rental-safecem10b-20261025.json")
PAPER = "results/safecem10-gate-dial-study"
FIGURES = ("gate-response", "calibration", "b2-response", "c2-fronts")
SCRIPTS = ("analyze_safecem10.py", "render_safecem10_videos.py", "build_safetydial_pusht.py")
FINAL = re.compile(r"-final-(fresh|holdout)-c(\d+)$")
SEARCH = re.compile(r"-g(\d+)-c(\d+)$")
INTEGER_COLUMNS = (
    "quantile",
    "replicate",
    "generation",
    "candidate",
    "gene_horizon",
    "steps_to_success",
    "first_violation_step",
)
STATE_COLUMNS = (
    "pusher_x",
    "pusher_y",
    "block_x",
    "block_y",
    "block_theta",
    "pusher_vx",
    "pusher_vy",
)
# The frozen source names the local vastai binary. It stays byte-exact (its hash is in
# code/source-bundles.json), and a user name in a path is not a credential.
HOME_PATH_ALLOWED = {"code/safecem10/experiments/helpers/dsrl06_infra.py"}
TEXT = {".json", ".jsonl", ".md", ".txt", ".py", ".sh", ".tex", ".csv", ".log", ".toml", ".cfg"}


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def check_name(planner):
    check = planner.get("check", "probe")
    if check == "exact":
        return "exact check"
    if check == "anchored":
        return f"grounded probe check ({planner.get('probe_version', 'v1')})"
    return "frozen probe check" if planner.get("probe_version", "v1") == "v1" else "probe v2 check"


def planner_label(job):
    """The planner in the paper's words, for browsing the table."""
    planner = job["planner"]
    execute = planner.get("execute", "mean")
    if job["stage"] in ("C", "C2"):
        return f"searched setting, {check_name(planner)}, {execute} execution"
    if job["mode"] == "penalty" and job["lam"] == 0:
        return f"no safety check, {execute} execution"
    if job["mode"] == "penalty":
        return f"penalty lambda {job['lam']:g}, {check_name(planner)}, {execute} execution"
    return f"{check_name(planner)}, {execute} execution"


def episode_row(result, states, boxes, videos):
    job, planner = result["job"], result["job"]["planner"]
    params = job.get("params") or {}
    final, search = FINAL.search(job["id"]), SEARCH.search(job["id"])
    summary = result.get("planner") or {}
    controller = result.get("controller_step_error_px") or {}
    return {
        "episode_id": job["id"],
        "stage": job["stage"],
        "layout": job["layout"],
        "seed": job["seed"],
        "planner_label": planner_label(job),
        "mode": job["mode"],
        "lam": job["lam"],
        "condition": job.get("condition"),
        "check": planner.get("check", "probe"),
        "probe_version": planner.get("probe_version", "v1"),
        "execute": planner.get("execute", "mean"),
        "dial_px": job["clearances"][0] if job["mode"] == "safe" else None,
        "quantile": job.get("quantile"),
        "clearances": json.dumps(job["clearances"]),
        "method": job.get("method"),
        "replicate": job.get("replicate"),
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
        "success": bool(result["success"]),
        "safe_success": bool(result["success"]) and not result["any_violation"],
        "steps_to_success": result["steps_to_success"],
        "J": result["J"],
        "hazard_steps": result["hazard_steps"],
        "exposure": result["exposure"],
        "any_violation": bool(result["any_violation"]),
        "first_violation_step": result["first_violation_step"],
        "realized_clearance_px": realized_clearance(states, boxes),
        "crossing": result["crossing"],
        "oob_steps": result["oob_steps"],
        "final_block_err_px": result["final_block_err_px"],
        "min_block_err_px": result["min_block_err_px"],
        "block_moved_px": result["block_moved_px"],
        "px_per_step": result["px_per_step"],
        "n_states": result["n_states"],
        "env_steps": result["env_steps"],
        "replans": summary.get("replans"),
        "cem_iterations": summary.get("cem_iterations"),
        "frac_feasible_mean": summary.get("frac_feasible_mean"),
        "final_frac_feasible": summary.get("final_frac_feasible"),
        "final_elite_feasible": summary.get("final_elite_feasible"),
        "final_elite_viol_mean": summary.get("final_elite_viol_mean"),
        "final_elite_goal_mean": summary.get("final_elite_goal_mean"),
        "controller_step_error_px_max": controller.get("max"),
        "model_candidate_rollouts": result["model_candidate_rollouts"],
        "planner_seconds": result["planner_seconds"],
        "episode_seconds": result["episode_seconds"],
        "utc": result["utc"],
        "states_sha256": result["states_sha256"],
        "video": videos.get(job["id"]),
    }


def build_tables(out, spec):
    """data/episodes.{parquet,csv} and data/trajectories.parquet from the synced attempts."""
    import pandas as pd

    env = spec["environment"]
    start = make_state(env["start_pusher"], env["start_block"], env["block_theta"])
    manifest = json.loads((out / "videos/manifest.json").read_text())
    videos = {job_id: f"videos/episodes/{job_id}.mp4" for job_id in manifest["episodes"]}
    rows, trajectories = [], []
    for path in sorted((REMOTE / "attempts").glob("*/result.json")):
        result = json.loads(path.read_text())
        if result["status"] != "complete":
            continue
        states = np.load(path.parent / "states.npz")["states"]
        if not np.isfinite(states).all() or len(states) != result["n_states"]:
            raise ValueError(f"{path.parent.name}: saved states do not match the record")
        boxes = gate_boxes(layout(result["job"]["layout"], spec), spec)
        rows.append(episode_row(result, states, boxes, videos))
        full = np.vstack([start[None, :], states])
        frame = pd.DataFrame(full, columns=STATE_COLUMNS)
        frame.insert(0, "t", np.arange(len(full), dtype=np.int16))
        frame.insert(0, "episode_id", result["job"]["id"])
        frame["inside_pillar"] = inside_any(full[:, :2], boxes)
        trajectories.append(frame)
    table = pd.DataFrame(rows)
    # Integer columns that are empty for some stages stay integers (nullable), not floats.
    for column in INTEGER_COLUMNS:
        table[column] = table[column].astype("Int64")
    order = {"A": 0, "B": 1, "B2": 2, "C": 3, "C2": 4}
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


def checkpoint_files(out, spec):
    pins = spec["pins"]
    model = out / "checkpoints/lewm-pusht"
    source = ROOT / "data/safecem10/hf_pusht"
    for name in ("config.json", "weights.pt"):
        copy(source / name, model / name)
    weights = model / "weights.pt"
    if (
        weights.stat().st_size != pins["model_weights_bytes"]
        or digest(weights) != pins["model_weights_sha256"]
    ):
        raise ValueError("weights.pt differs from the pinned upstream checkpoint")
    converted = ROOT / "data/safecem10/stablewm/checkpoints/pusht"
    for name in ("lewm_object.ckpt", "lewm_object.json"):
        copy(converted / name, model / "converted" / name)
    if (
        json.loads((converted / "lewm_object.json").read_text())["source_sha256"]
        != pins["model_weights_sha256"]
    ):
        raise ValueError("the converted checkpoint was not built from the pinned weights")
    from huggingface_hub import hf_hub_download

    card = hf_hub_download(
        pins["model_repo"], "README.md", revision=pins["model_revision"], cache_dir=CACHE
    )
    copy(Path(card), model / "UPSTREAM_README.md")
    probes = out / "checkpoints/probes"
    copy(REMOTE / "probes/probe.pt", probes / "probe-v1.pt")
    copy(REMOTE / "probes/probe-v2.pt", probes / "probe-v2.pt")
    expected = {
        "probe-v1.pt": json.loads((REMOTE / "prepare-complete.json").read_text())["probe_sha256"],
        "probe-v2.pt": json.loads((REMOTE / "probe-v2.json").read_text())["probe_sha256"],
    }
    for name, sha in expected.items():
        if digest(probes / name) != sha:
            raise ValueError(f"{name} differs from its recorded hash")
    for name in ("probe-v2.json", "probe-v2-eval.npz"):
        copy(REMOTE / name, probes / "reports" / name)
    copy(REMOTE / "diag/probe-map.json", probes / "reports/probe-map.json")


def data_files(out):
    preserved = json.loads((HF_STAGE / "receipts/safecem10/sha256-manifest.json").read_text())
    for stage in ("A", "B", "B2", "C", "C2"):
        rel = f"recordings/safecem10/attempts-{stage}.tar.gz"
        target = out / "data/raw" / f"attempts-{stage}.tar.gz"
        copy(HF_STAGE / rel, target)
        if digest(target) != preserved[rel]["sha256"]:
            raise ValueError(f"{rel} differs from the preserved archive")
    for name, label in (("stageC2", "stageC2"), ("stageC", "stageC-paused")):
        shutil.copytree(REMOTE / name, out / "data/search" / label)
    for name in ("imagination.json", "imagination-v2.json", "imagination-errors.npz"):
        copy(REMOTE / name, out / "data/imagination" / name)


def result_files(out):
    for path in (ROOT / "outputs/safecem10/analysis").iterdir():
        copy(path, out / "results/analysis" / path.name)
    for suffix in (".pdf", ".tex"):
        copy(ROOT / f"{PAPER}{suffix}", out / "results/paper" / f"{Path(PAPER).name}{suffix}")
    for name in FIGURES:
        for suffix in (".pdf", ".png"):
            copy(
                ROOT / f"results/safecem10-{name}{suffix}",
                out / f"results/paper/safecem10-{name}{suffix}",
            )


def protocol_files(out):
    protocol = out / "protocol"
    for path in (ROOT / "configs/safecem10").iterdir():
        copy(path, protocol / "configs" / path.name)
    copy(ROOT / "progress/10-safecem-dial-plan.md", protocol / "10-safecem-dial-plan.md")
    for path in sorted((ROOT / "progress/receipts").glob("10-safecem-*")):
        if path.name.startswith("10-safecem-preservation") and path.suffix == ".json":
            continue  # markers: local paths only, the facts are in 10-safecem-preservation.md
        copy(path, protocol / "receipts" / path.name)
    markers = ["prepare-complete.json", *sorted(p.name for p in REMOTE.glob("stage-*.json"))]
    markers += sorted(p.name for p in REMOTE.glob("STOP-*.txt"))
    for name in markers:
        copy(REMOTE / name, protocol / "run-markers" / name)
    for name in ("versions.json", "installed-freeze.txt"):
        copy(REMOTE / "environment" / name, protocol / "environment" / name)


def safe_member(name):
    return not name.startswith("/") and ".." not in Path(name).parts


def code_files(out):
    """The C2 source tree as files, every bundle's file hashes, and the local scripts."""
    bundles = {}
    for bundle in sorted((ROOT / "outputs/safecem10/source").glob("bundle-*.tar.gz")):
        entry = {"sha256": digest(bundle), "files": {}, "withheld": []}
        with tarfile.open(bundle, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not safe_member(member.name):
                    continue
                data = archive.extractfile(member).read()
                entry["files"][member.name] = hashlib.sha256(data).hexdigest()
                if member.name.startswith("progress/rental-"):
                    entry["withheld"].append(member.name)
                elif bundle.name == "bundle-10-safecem-C2-freeze.tar.gz" and member.name.startswith(
                    ("configs/", "experiments/")
                ):
                    target = out / "code/safecem10" / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
        bundles[bundle.name] = entry
    atomic_json(out / "code/source-bundles.json", bundles)
    for name in SCRIPTS:
        copy(ROOT / "experiments/scripts" / name, out / "code/scripts" / name)


def sensitive_values():
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

    for name in RENTALS:
        walk(json.loads((ROOT / name).read_text()))
    return [v.encode() for v in values if len(v) >= 6]


def scan(out):
    """Fail on any credential, rental endpoint or local home path, inside archives too."""
    needles = sensitive_values()
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
        check(label, path.read_bytes(), path.suffix in TEXT and label not in HOME_PATH_ALLOWED)
        if path.name.endswith(".tar.gz"):
            # Stream mode: members are read in order, never by seeking back in the gzip stream.
            with tarfile.open(path, mode="r|gz") as archive:
                for member in archive:
                    if member.isfile():
                        inner = archive.extractfile(member).read()
                        check(f"{path.name}:{member.name}", inner, Path(member.name).suffix in TEXT)


def fmt(x, digits=2):
    return f"{x:.{digits}f}"


def stage_b_table(table):
    """Paper table 1 from the episode rows: completion and exposure per layout and dial."""
    b = table[table.stage == "B"]
    gaps = {
        "G0": "60 px at x = 80",
        "G40": "60 px at x = 120",
        "G80": "60 px at x = 160",
        "N40": "40 px at x = 80",
    }
    lines = [
        "| layout | gap | no safety check | d = 0 | d = 10 | d = 20 | exposure peak | threading the gap, d <= 20 / d >= 25 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for lay, gap in gaps.items():
        cell = b[b.layout == lay]
        free = cell[(cell["mode"] == "penalty") & (cell.lam == 0)]
        safe = cell[cell["mode"] == "safe"]
        parts = [f"{fmt(free.success.mean())} / {fmt(free.exposure.mean(), 3)}"]
        for d in (0, 10, 20):
            group = safe[safe.dial_px == d]
            parts.append(f"{fmt(group.success.mean())} / {fmt(group.exposure.mean(), 3)}")
        exposure = safe.groupby("dial_px").exposure.mean()
        open_, closed = safe[safe.dial_px <= 20], safe[safe.dial_px >= 25]
        threads = (
            f"{(open_.crossing == 'gap').sum()} of {len(open_)} / "
            f"{(closed.crossing == 'gap').sum()} of {len(closed)}"
        )
        peak = f"{fmt(exposure.max(), 3)} at d = {exposure.idxmax():g}"
        lines.append(f"| {lay} | {gap} | {' | '.join(parts)} | {peak} | {threads} |")
    return "\n".join(lines)


def best_single_dial(table, lay):
    """Best safe success over the single settings of the grounded pipeline in stage B2."""
    b2 = table[(table.stage == "B2") & (table.layout == lay) & (table.condition == "fixed")]
    rates = b2.groupby("dial_px").safe_success.mean()
    return rates.max(), rates.idxmax()


def stage_b2_table():
    labels = {
        "frozen": "frozen probe (stage B)",
        "bestonly": "probe v1, best candidate (G0 only)",
        "fixed": "grounded probe (v2, anchored)",
        "exact": "exact controller check",
    }
    rows = (ROOT / "outputs/safecem10/analysis/stageB2-compact.md").read_text().strip().splitlines()
    out = []
    for row in rows:
        cells = row.split("|")
        key = cells[1].strip()
        if key in labels:
            cells[1] = f" {labels[key]} "
        out.append("|".join(cells))
    return "\n".join(out)


def video_rows(out):
    manifest = json.loads((out / "videos/manifest.json").read_text())
    what = [
        (
            "dial-sweep/",
            "exact-check",
            "One episode at seven dials with the exact check (the pusher's true controller path), next to the planner with no check (top left).",
        ),
        (
            "dial-sweep/",
            "grounded-probe",
            "The same seed with the grounded probe check (probe v2, anchored), the pipeline the search used. It completes more often but can clip a pillar.",
        ),
        (
            "dial-sweep/",
            "frozen-probe",
            "The original stage B pipeline (probe v1, elite mean). The probe misread the pusher near the start, so the dial was not honoured.",
        ),
        (
            "checks/",
            "",
            "The same dial, d = 10 px, checked three ways, next to the planner with no check.",
        ),
        ("seeds/", "no-safety-check", "The planner with no safety check on all ten seeds."),
        ("seeds/", "exact-check-d0", "The exact check at d = 0 px on the same ten seeds."),
        (
            "search/",
            "holdout",
            "The best setting found by the search, on the held-out layout G80, ten fresh seeds.",
        ),
        (
            "search/",
            "fresh",
            "The best setting found by the search, on ten fresh seeds of the search layout G0.",
        ),
    ]
    order = [p + k for p, k, _ in what]
    lines = ["| video | what it shows | in these episodes |", "|---|---|---|"]

    def rank(item):
        name = item["name"]
        lay = 0 if "G80" in name else 1
        for i, (prefix, key, _) in enumerate(what):
            if name.startswith(prefix) and key in name:
                return (i if prefix != "search/" else len(what) + i, lay)
        return (len(order) * 2, lay)

    for item in sorted(manifest["sets"], key=rank):
        name = item["name"]
        text = next(t for p, k, t in what if name.startswith(p) and k in name)
        eps = [manifest["episodes"][j] for j in item["jobs"]]
        entered = sum(e["hazard_steps"] > 0 for e in eps)
        safe = sum(e["success"] and e["hazard_steps"] == 0 for e in eps)
        unsafe = sum(e["success"] and e["hazard_steps"] > 0 for e in eps)
        tally = f"safe successes: {safe}, unsafe successes: {unsafe}, entered a pillar: {entered} of {len(eps)}"
        links = f"[{name}.mp4](videos/{name}.mp4)"
        if (out / f"videos/{name}.gif").exists():
            links += f" ([gif](videos/{name}.gif))"
        lines.append(f"| {links} | {text} | {tally} |")
    return "\n".join(lines), manifest


def readme(out, table, n_traj, spec):
    counts = table.groupby("stage").size().to_dict()
    pins = spec["pins"]
    env = spec["environment"]
    videos, manifest = video_rows(out)
    g0, d_g0 = best_single_dial(table, "G0")
    g80, d_g80 = best_single_dial(table, "G80")
    c2 = (ROOT / "outputs/safecem10/analysis/stageC2-table.md").read_text().strip()
    c2 += f"\n| best single dial (B2 grounded pipeline, B2 seeds) | | | | | {fmt(g0)} (d = {d_g0:g}) | {fmt(g80)} (d = {d_g80:g}) |"
    exact = table[(table.stage == "B2") & (table.condition == "exact")]
    sets = len(manifest["sets"])
    clips = len(manifest["episodes"])
    probe1 = json.loads((REMOTE / "prepare-complete.json").read_text())["probe_sha256"]
    probe2 = json.loads((REMOTE / "probe-v2.json").read_text())["probe_sha256"]
    scalers = json.loads((REMOTE / "prepare-complete.json").read_text())["scalers_sha256"]
    sweep = next(
        s["name"]
        for s in manifest["sets"]
        if s["name"].startswith("dial-sweep/G80") and "exact" in s["name"]
    )
    holdout = next(
        s["name"]
        for s in manifest["sets"]
        if s["name"].startswith("search/") and "holdout" in s["name"]
    )
    fresh = next(
        s["name"]
        for s in manifest["sets"]
        if s["name"].startswith("search/") and "fresh" in s["name"]
    )
    seeds = {
        lay: re.search(r"seed(\d+)", s["name"]).group(1)
        for s in manifest["sets"]
        if s["name"].startswith("dial-sweep/") and "exact" in s["name"]
        for lay in ("G80", "G0")
        if s["name"].startswith(f"dial-sweep/{lay}-")
    }
    methods = json.loads((ROOT / "outputs/safecem10/analysis/stageC2-summary.json").read_text())[
        "methods"
    ]
    best = {m: fmt(v["fresh_best_safe_success"]["mean"]) for m, v in methods.items()}
    held = {m: fmt(v["holdout_best_safe_success"]["mean"]) for m, v in methods.items()}
    loop = json.loads((REMOTE / "probe-v2.json").read_text())["evaluation"]["closed_loop_stage_AB"]
    return f"""---
license: mit
pretty_name: SafetyDial-PushT
language:
- en
tags:
- robotics
- world-models
- model-predictive-control
- safe-planning
- evolutionary-algorithms
- push-t
size_categories:
- 1K<n<10K
configs:
- config_name: episodes
  data_files: data/episodes.parquet
  default: true
- config_name: trajectories
  data_files: data/trajectories.parquet
---

# SafetyDial-PushT

**A safety dial for a world-model planner, tested on Push-T.** Every episode, trajectory,
checkpoint, figure and video from task 10 of the Safety-Dial project (Sachin Mohan, BSc
Honours in Computer Science, University of the Witwatersrand, 2026).

![One episode at seven dial settings]({f"videos/{sweep}.gif"})

*One episode (layout G80, seed {seeds["G80"]}) at seven dial settings, next to the planner with no
safety check (top left). Red: the true pillars. Dashed amber: the pillars inflated by the
dial. Blue: the pusher's path. Green: the goal pose. A red ring marks a step inside a pillar.*

## The idea in one minute

A planner pushes the grey T to the green target using LeWorldModel (LeWM), a learned world
model that imagines what the camera will see after a few pusher moves. It plans with CEM:
imagine 300 candidate move sequences, refit to the best 30, repeat 30 times, carry out the
first block of five moves, replan.

We put a wall of two red pillars across the route with one gap. The world model was never
trained with pillars, so it cannot see them. Safety comes from a check added to the planner
(SafeCEM): every imagined plan whose pusher enters a pillar *inflated by d pixels* ranks below
every plan that stays clear. **d is the safety dial.** A small d lets the planner thread the
gap. Once d reaches half the gap width, the gap is closed as far as the planner knows, and it
has to find another way.

Three questions: does the dial keep the pusher out of the pillars, what does it cost in task
completion, and can an evolutionary search over the dial's settings (NSGA-II) beat any single
fixed dial?

## What we found

1. **With the exact check, the dial is a guarantee.** When the check follows the pusher's true
   controller path, {int(exact.any_violation.sum())} of {len(exact)} episodes ever had the
   pusher inside a pillar, and every episode kept at least the requested clearance. The cost
   is that the planner often stalls above the wall.
2. **The dial is a sharp route switch, not a smooth trade-off.** Below closure the planner
   threads the gap. Past it, the planner hovers above the wall for all 50 steps and never finds
   the way around: success 0 on every layout.
3. **The first, probe-based check could not see the pusher.** It read the pusher's position
   from the world model's latent state and was off by a median of {loop["v1"]["all"]["median"]:.1f}
   px ({loop["v1"]["corner"]["median"]:.0f} px in the start corner), so the requested clearance
   was not the realized clearance. A retrained probe (v2) cut that to
   {loop["v2"]["all"]["median"]:.1f} px, yet the planner still cut corners.
4. **Searching the dial's settings beats any single dial where it searched.** On fresh seeds of
   the search layout, the best searched setting reaches safe success {best["nsga2"]} (NSGA-II),
   {best["random"]} (random search) and {best["grid"]} (grid) on average, against {fmt(g0)} for
   the best single dial. On the held-out layout it does not on average ({held["nsga2"]},
   {held["random"]} and {held["grid"]} against {fmt(g80)}), and with three replicates the
   NSGA-II against random search comparison is not settled.

The full write-up is [`results/paper/safecem10-gate-dial-study.pdf`](results/paper/safecem10-gate-dial-study.pdf).

## Videos

{sets} grid videos and {clips} single-episode clips. Every frame is drawn from the saved
simulator state of a scored episode, with nothing re-simulated, so what you see is what was
scored. A grid plays its episodes in sync, one simulator step per frame at 8 frames per
second, then holds the last frame. Each footer counts the steps with the pusher's centre
inside a true pillar and gives the outcome; the count was checked against the scored record
for every episode.

{videos}

Every episode in a grid also has its own clip in `videos/episodes/<episode_id>.mp4`, linked
from the `video` column of the episode table.

**How the episodes were chosen.** By fixed rules, recorded per video in
`videos/manifest.json`. The single-seed sweeps use, for each layout, the first seed (200 to
209) on which the exact check at d = 0 succeeded and the planner with no check entered a
pillar (seed {seeds["G80"]} on G80, {seeds["G0"]} on G0). The ten-seed grids show every seed of their condition.
The search videos show, over all nine search runs, the front setting with the highest safe
success on that evaluation. So the sweeps illustrate the statistics, and the ten-seed grids
show the spread.

| best searched settings | |
|---|---|
| ![held-out]({f"videos/{holdout}.gif"}) | ![fresh seeds]({f"videos/{fresh}.gif"}) |

## The task

- Push-T (`{env["env_name"]}`, stable-worldmodel {pins["stable_worldmodel"]}), a 512 x 512 px
  arena; the world model sees {env["image_size"]} x {env["image_size"]} images.
- Start: pusher at (40, 40), T-block at (300, 310), angle 0.785 rad. Goal: pusher at
  (250, 310), block at (360, 310), same angle.
- Success uses the environment's own rule: the joint distance of the pusher and block
  positions from the goal is below 20 px and the block angle is within pi/9 (20 degrees).
  {env["eval_budget_steps"]} steps per episode. J = steps to success, 50 when the episode fails.
- Wall band y in [190, 230] from x = 0 to 330, with one gap. Layouts G0, G40, G80: a 60 px gap
  centred at x = 80, 120, 160. N40: a 40 px gap centred at x = 80.
- Exposure = the fraction of saved states with the pusher's centre inside a true pillar. Safe
  success = success with exposure 0.
- Planner: CEM with 300 samples, top 30, 30 iterations per solve, a 5-block horizon (5
  actions per block), replanning after every block, feasibility-first ranking.

## Stages

| stage | what ran | episodes |
|---|---|---|
| A | engineering check | {counts.get("A", 0)} |
| B | the dial sweep with the frozen probe check: 4 layouts x 10 seeds x 11 settings | {counts.get("B", 0)} |
| B2 | the sweep again with grounded checks (probe v2 anchored, and exact) plus references, on G0 and G80, same seeds | {counts.get("B2", 0)} |
| C | first search, paused when the probe fault was found (kept for the record) | {counts.get("C", 0)} |
| C2 | NSGA-II against random search against a grid over five genes, 3 replicates each, finals on fresh seeds and on held-out G80 | {counts.get("C2", 0):,} |

The C2 genes: initial clearance d0 in [0, 60] px, a clearance slope over the imagined blocks
d1 in [-8, 8] px per block, horizon 3 to 7 blocks, elite fraction 0.05 to 0.3, and a
feasibility tolerance of 0 to 15 px. Objectives: mean exposure and mean J.

## Results

**Stage B**, frozen probe check, ten seeds per cell (success / exposure):

{stage_b_table(table)}

**Stage B2**, pooled over G0 and G80. Open dials leave the gap passable (d < 30):

{stage_b2_table()}

**Stage C2**, mean ± sd over three replicates. Hypervolume (HV) of (exposure, J) against the
reference point (1, 50); larger is better. The search column is optimistic because those
objectives selected the front; only the fresh-seed and held-out columns support claims.

{c2}

## Files

```
README.md                    this card
videos/                      grid videos (mp4, three gif previews) and manifest.json
  episodes/                  one mp4 per episode shown in a grid
data/
  episodes.parquet, .csv     one row per scored episode ({len(table):,} rows)
  trajectories.parquet       the saved simulator state at every step ({n_traj:,} rows)
  raw/attempts-*.tar.gz      per-episode folders: result.json, states.npz, the planner's
                             per-iteration history.json, ticket.json
  search/stageC2/            per-run generations (gen-00 to gen-07) and final fronts
  search/stageC-paused/      the paused first search
  imagination/               imagined-pusher error measurements behind the uncertainty dials
checkpoints/
  lewm-pusht/                the upstream LeWM Push-T world model, pinned (config.json,
                             weights.pt), converted/ for stable-worldmodel, upstream card
  probes/                    latent-to-pusher probes: probe-v1.pt (stages A to C) and
                             probe-v2.pt (B2 and C2), with their evaluation reports
results/
  paper/                     the write-up: pdf, tex and figures
  analysis/                  summaries, tables and figures from analyze_safecem10.py
protocol/                    design, plan, the freeze and receipt of every stage, run markers,
                             software versions
code/
  safecem10/                 the source that ran stage C2 (helpers, scripts, tests, configs)
  scripts/                   analysis, video rendering and this dataset's build script
  source-bundles.json        sha256 of every file in every frozen source bundle
sha256-manifest.json         size and sha256 of every file here
```

Main columns of `data/episodes.parquet`: `episode_id`, `stage`, `layout`, `seed`,
`planner_label` (the planner in words), `mode` (`safe` = feasibility-first check, `penalty` =
violation added to the cost with weight `lam`; `lam` 0 means no safety check), `check`,
`probe_version`, `execute` (`mean` of the elite set or `best` checked candidate), `dial_px`,
`quantile` (90 or 95 for the dials set in uncertainty units), `clearances` (per imagined
block), the C2 fields `method`, `replicate`, `phase` (`search` or `final`), `generation`,
`final_set` (`fresh` or `holdout`), `candidate`, `gene_*`, then the outcomes `success`,
`safe_success`, `J`, `hazard_steps`, `exposure`, `realized_clearance_px` (minimum signed
distance from the pusher's centre to a true pillar; negative means inside), `crossing` (how the
pusher crossed the wall band), planner statistics, and `video`.

`data/trajectories.parquet`: `episode_id`, `t` (0 is the start state; t is the state after step
t), `pusher_x`, `pusher_y`, `block_x`, `block_y`, `block_theta`, `pusher_vx`, `pusher_vy`,
`inside_pillar`.

## How to load

```python
import pandas as pd
from huggingface_hub import hf_hub_download

def get(name):
    return hf_hub_download("{REPO}", name, repo_type="dataset")

episodes = pd.read_parquet(get("data/episodes.parquet"))
exact = episodes[(episodes.stage == "B2") & (episodes.condition == "exact")]
print(exact.groupby(["layout", "dial_px"]).safe_success.mean())

steps = pd.read_parquet(get("data/trajectories.parquet"))
one = steps[steps.episode_id == "B2-G80-201-exact-d0"]
```

or `datasets.load_dataset("{REPO}", "episodes")`. Each probe is a small MLP (192 to 256 to
128 to 2 units, ReLU) that maps a LeWM latent to the pusher's (x, y) in arena pixels:

```python
import torch
blob = torch.load(get("checkpoints/probes/probe-v2.pt"), map_location="cpu", weights_only=False)
# blob["state_dict"] loads into MLPProbe in code/safecem10/experiments/helpers/safecem10_planner.py
```

## Provenance

- World model: [`{pins["model_repo"]}`](https://huggingface.co/{pins["model_repo"]}) at revision
  `{pins["model_revision"]}`; `weights.pt` sha256 `{pins["model_weights_sha256"]}`, checked
  against the bytes here when this dataset was built.
- Probe hashes: v1 `{probe1}`, v2 `{probe2}`, both checked against the run's records.
- Training data (not redistributed): `{pins["dataset_repo"]}` dataset at revision
  `{pins["dataset_revision"]}`, file `{pins["dataset_file"]}` (sha256 `{pins["dataset_sha256"]}`).
  The planner's data scalers (sha256 `{scalers}`) are rebuilt from that file by
  `build_scalers` in `code/safecem10/experiments/scripts/prepare_safecem10.py`; the file itself
  was not kept.
- Every stage was frozen (job list, design and source hashes) before its first episode:
  `protocol/receipts/10-safecem-*-freeze.json`.
- The complete raw evidence, with run logs and rental records, is preserved privately in
  `{EVIDENCE[0]}` at revision `{EVIDENCE[1]}`.
- Software: torch {pins["torch"]}, stable-worldmodel {pins["stable_worldmodel"]},
  stable-pretraining {pins["stable_pretraining"]}, transformers {pins["transformers"]}
  (`protocol/environment`). Hardware: a rented RTX 3090, plus a second one for part of C2.

## Caveats

- Ten seeds per cell in B and B2, three replicates per method in C2. The NSGA-II against random
  search differences are not conclusive.
- Best-of-front safe success is a maximum over up to six settings on the same ten seeds, so it
  is optimistic. The best single dial is a maximum over the single settings and is optimistic
  by a similar margin. Hypervolume on fresh seeds and on the held-out layout is the primary
  comparison.
- The videos were chosen by the fixed rules above, not at random, except the ten-seed grids.
- Stage C used the frozen probe and was paused when its fault was found; C2 supersedes it.
- Past gap closure, no planner in this study found the route around the wall. We attribute
  this to the 5-block imagination horizon and a goal cost that pulls straight toward the goal.
- Some receipts written during the run describe success as block coverage of 0.95 or more. The
  rule actually used throughout is the environment's own, stated under The task.

## License

The world-model files in `checkpoints/lewm-pusht` are redistributed unchanged (and in converted
form) from [`{pins["model_repo"]}`](https://huggingface.co/{pins["model_repo"]}) under its MIT
license; its model card is included as `UPSTREAM_README.md`. Everything else here (probes,
records, analysis, videos and code) is released under the MIT license.

## Citation

```bibtex
@misc{{mohan2026safetydialpusht,
  author = {{Mohan, Sachin}},
  title = {{SafetyDial-PushT: a safety dial for a world-model planner on Push-T}},
  year = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{REPO}}}}}
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


def build(out):
    if not (out / "videos/manifest.json").exists():
        raise ValueError("render the videos first: render_safecem10_videos.py --out <out>/videos")
    for child in out.iterdir():
        if child.name != "videos":
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    spec = design()
    table, n_traj = build_tables(out, spec)
    checkpoint_files(out, spec)
    data_files(out)
    result_files(out)
    protocol_files(out)
    code_files(out)
    (out / "README.md").write_text(readme(out, table, n_traj, spec))
    scan(out)
    return write_manifest(out)


def upload(out):
    from dotenv import dotenv_values
    from huggingface_hub import HfApi, hf_hub_download

    manifest = json.loads((out / "sha256-manifest.json").read_text())
    scan(out)
    token = dotenv_values(ROOT / ".env").get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(REPO, repo_type="dataset")
    if not info.private:
        raise ValueError("Destination is not private")
    commit = api.upload_folder(
        folder_path=out,
        repo_id=REPO,
        repo_type="dataset",
        parent_commit=info.sha,
        commit_message="SafetyDial-PushT: task-10 episodes, checkpoints, results and videos",
    )
    revision = commit.oid
    remote = set(api.list_repo_files(REPO, repo_type="dataset", revision=revision))
    if remote - {".gitattributes"} != set(manifest) | {"sha256-manifest.json"}:
        raise ValueError("Uploaded inventory differs from the local folder")

    def verify(rel):
        copied = Path(
            hf_hub_download(
                REPO, rel, repo_type="dataset", revision=revision, token=token, cache_dir=CACHE
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
        "repo": REPO,
        "private": True,
        "revision": revision,
        "verified_files": len(checked),
        "manifest_sha256": digest(out / "sha256-manifest.json"),
    }
    atomic_json(out.parent / "safetydial-pusht-upload.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/safetydial-pusht")
    parser.add_argument(
        "--upload", action="store_true", help="upload the built folder and verify it"
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="upload the folder as already built"
    )
    args = parser.parse_args()
    if not args.skip_build:
        manifest = build(args.out)
        print(
            json.dumps(
                {"files": len(manifest), "bytes": sum(x["bytes"] for x in manifest.values())}
            ),
            flush=True,
        )
    if args.upload:
        print(json.dumps(upload(args.out)), flush=True)


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
