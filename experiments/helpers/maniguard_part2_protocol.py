"""Prospective followup schedules and immutable source receipts."""

import json
from dataclasses import asdict
from pathlib import Path

from helpers.maniguard_followup_budget import ALLOCATION, atomic
from helpers.maniguard_part2_controller import conditions
from helpers.maniguard_phase2_families import ROOT, validate_scene_access
from helpers.maniguard_phase3_protocol import check_sources, sha
from helpers.maniguard_phase3_protocol import implementation_files as part1_files

RUN = ROOT / "runs/maniguard/phase3-parts2-3/20261018-56582094"
TRAIN = ("task_0039/base", "task_0010/base")
ENGINEERING = "task_0040/base"


def implementation_files():
    files = part1_files()
    for pattern in ("maniguard_part2_*.py", "maniguard_followup_*.py"):
        for directory in ("experiments/helpers", "experiments/scripts"):
            files += list((ROOT / directory).glob(pattern))
    files += [ALLOCATION]
    return sorted(set(files))


def plan():
    jobs = []
    # Cyclic Latin order balances each condition's ordinal position once. Seeds and
    # cases are chosen prospectively, without inspecting these tasks' outcomes.
    for block, (scene, seed) in enumerate(
        (scene, 4000 + 10 * i + r) for i, scene in enumerate(TRAIN) for r in range(2)
    ):
        order = "ABCD"[block:] + "ABCD"[:block]
        for condition in order:
            jobs.append({
                "episode_id": f"part2-b{block}-{condition}", "part": 2,
                "purpose": "train", "condition": condition, "block": block,
                "scene": scene, "seed": seed, "action_horizon": 1100,
                "controller": asdict(conditions()[condition]),
            })
    return {"schema": "maniguard-part2-schedule-v1", "jobs": jobs}


def draft():
    return {
        "schema": "maniguard-part2-v1", "status": "draft", "planning_commit": "d4b8182",
        "conditions": {name: asdict(cfg) for name, cfg in conditions().items()},
        "correction": {
            "units": "L2 arm joint target difference in radians / (sqrt(7) * 0.04 rad)",
            "accumulation": "sum after actual environment input, after downstream clipping",
            "numerical_tolerance_rad": 1e-6, "threshold": 0.5,
            "reset": "on each query, preserve executed and finger history across queries",
            "precedence": "execute, account, observe, goal; if unfinished: max8 then threshold after4",
        },
        "common_restraint": "Part1 finalist componentwise joint-target caps .04/.02; smoothing1",
        "grasp_heuristic": "both abs fingers .003..035, prior delta<.002, close command<0",
        "passive_capture": "every existing evaluator observation, lossless RGB/state NPZ; no new render",
        "evaluation_targets": "native local+Z float32 tilt margin per active protected object; target-goal distance",
        "controller_information": "proposal, current joints/fingers, causal actual command history only",
        "engineering": {"attempts": 4, "max_successful_actions": 96, "max_actions_each": 32, "conditions": ["A", "C", "D"],
                        "scene": ENGINEERING, "seed": 3900,
                        "failure_allowance": "one pre-app import failure, no simulation launched; then A/C/D checks; no full retries"},
        "full_episode_allowance": 16, "native_action_horizon": 1100,
        "timeouts_seconds": 600, "retries": "none; an invalid full slot stops the comparison",
        "schedule": plan(), "allocation": json.loads(ALLOCATION.read_text()),
        "fixed_reference_for_part3": "best B/C by mean U, then lower queries, then B; training only",
        "interpretation": "four blocks, descriptive mechanism screen, no generalization claim",
    }


def freeze(path, audits):
    if Path(path).exists():
        raise FileExistsError("Frozen protocol is immutable")
    if (len(audits) != 3 or [a.get("condition") for a in audits] != ["A", "C", "D"]
            or any(not a.get("engineering_valid") for a in audits)):
        raise ValueError("Require the three valid bounded engineering checks")
    value = draft()
    value.update(status="frozen", sources=check_sources(), engineering_audits=audits,
                 code_hashes={str(p.relative_to(ROOT)): sha(p) for p in implementation_files()})
    value["scenes"] = []
    for purpose, scenes in (("engineering", [ENGINEERING]), ("train", TRAIN)):
        for scene in scenes:
            validate_scene_access("clutter", scene, purpose)
            source = ROOT / "data/maniguard/benchmark/clutter_pickup" / scene / "scene_ep1.json"
            value["scenes"].append({"scene": scene, "purpose": purpose, "sha256": sha(source)})
    value["checkpoint_receipt_sha256"] = sha(
        ROOT / "data/maniguard/policies/clutter/download-provenance.json"
    )
    atomic(path, value)
    return value


def validate_frozen(path):
    value = json.loads(Path(path).read_text())
    if value.get("status") != "frozen":
        raise ValueError("Full episodes require a frozen protocol")
    for name, digest in value["code_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen followup implementation changed: " + name)
    for scene in value.get("scenes", []):
        source = ROOT / "data/maniguard/benchmark/clutter_pickup" / scene["scene"] / "scene_ep1.json"
        if sha(source) != scene["sha256"]:
            raise ValueError("Scene changed")
    check_sources()
    return value
