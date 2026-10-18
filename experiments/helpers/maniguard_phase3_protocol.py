"""Prospective Part 1 protocol and source/scene checks, independent of outcomes."""

import hashlib
import json
import subprocess
from pathlib import Path

from helpers.maniguard_phase2_families import (
    ROOT,
    family_spec,
    load_sources,
    validate_scene_access,
)
from helpers.maniguard_phase3_adapter import DOMAIN, NEUTRAL
from helpers.maniguard_phase3_budget import atomic
from helpers.maniguard_phase3_search import COMPARISON, ENGINEERING, TRAIN

CONFIG = ROOT / "configs/maniguard/phase3/part1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_files():
    files = list((ROOT / "experiments/helpers").glob("maniguard_phase3_*.py"))
    files += list((ROOT / "experiments/scripts").glob("maniguard_phase3_*.py"))
    files += [
        ROOT / f"experiments/helpers/{name}.py"
        for name in (
            "maniguard_runtime",
            "maniguard_source",
            "maniguard_adaptation",
            "maniguard_client",
            "maniguard_phase2_outcomes",
            "maniguard_phase2_families",
            "maniguard_results",
        )
    ]
    files += [
        ROOT / "configs/maniguard/phase2" / name
        for name in ("families.json", "sources.json", "splits.json", "completion.json")
    ]
    files += [ROOT / "configs/maniguard/phase3/allocation.json"]
    files += [
        ROOT / name
        for name in (
            "experiments/scripts/maniguard_serve.py",
            "experiments/helpers/maniguard_policy.py",
            "experiments/helpers/maniguard_parallel_policy.py",
            "experiments/helpers/maniguard_policy_provenance.py",
        )
    ]
    return sorted(set(files))


def check_sources():
    sources = load_sources()
    for key, directory in (
        ("maniguard", "ManiGuard"),
        ("behavior", "ManiGuard/behavior-1k"),
        ("openpi", "openpi"),
    ):
        folder = ROOT / "third_party" / directory
        revision = subprocess.check_output(
            ["git", "-C", str(folder), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != sources[key]["revision"]:
            raise ValueError("Source revision mismatch: " + key)
        subprocess.run(
            ["git", "-C", str(folder), "diff", "--exit-code", "HEAD", "--"],
            check=True,
            capture_output=True,
        )
    spec = family_spec("clutter")
    if sha(ROOT / "third_party/ManiGuard" / spec["eval_config"]) != spec["eval_config_sha256"]:
        raise ValueError("Clutter evaluator configuration hash mismatch")
    contract = json.loads((ROOT / "configs/maniguard/phase2/completion.json").read_text())
    for name, expected in contract["sources"].items():
        if sha(ROOT / "third_party/ManiGuard" / name) != expected:
            raise ValueError("Physical completion source changed: " + name)
    return sources


def scene_receipts():
    receipts = []
    for purpose, scenes in (
        ("engineering", [ENGINEERING]),
        ("train", TRAIN),
        ("validation", COMPARISON),
    ):
        for scene in scenes:
            validate_scene_access("clutter", scene, purpose)
            path = ROOT / "data/maniguard/benchmark/clutter_pickup" / scene / "scene_ep1.json"
            receipts.append(
                {
                    "scene": scene,
                    "purpose": purpose,
                    "sha256": sha(path),
                    "partition_verified": True,
                }
            )
    return receipts


def draft():
    return {
        "schema": "maniguard-phase3-part1-v1",
        "status": "draft_requires_engineering",
        "planning_commit": "8df7ac9",
        "family": "clutter",
        "source_profile": "legacy",
        "domain": DOMAIN,
        "grid_cardinality": 144,
        "neutral_genome": list(NEUTRAL),
        "search_seed": 20261018,
        "schedule_seed": 20261019,
        "rng": "numpy.random.Generator(PCG64)",
        "population_size": 4,
        "allocations": {
            "48": {"populations": 3, "search": 24, "comparison": 24},
            "28": {"populations": 2, "search": 16, "comparison": 12},
        },
        "utility": "int(C_req and not V) - 2 * int(V)",
        "V": "raw ltl_violated",
        "fitness": "mean of two valid full train episodes at dial 1",
        "tie_rule": "stable slot birth order, copied elite first",
        "finalist_rule": "best of final evaluated population only; freeze before comparison",
        "neutral_finalist_rule": "stop; no replacement or redundant comparison",
        "mutation": {
            "kernel": "AdapterSearch",
            "reject_noop": True,
            "reject_population_duplicate": True,
            "failed_draws_before_uniform": 100,
        },
        "native_action_horizon": 1100,
        "process_timeout_seconds": 1200,
        "cleanup_margin_seconds": 30,
        "bookkeeping_reserve_seconds": 300,
        "engineering": {
            "scene": ENGINEERING,
            "genome": [3, 3, 2, 0],
            "dials": [0, 0.5, 1],
            "action_horizon": 32,
            "max_total_actions": 96,
        },
        "training": {"scenes": list(TRAIN), "seed_formula": "1000+10*g+i"},
        "comparison": {
            "scenes": list(COMPARISON),
            "dials": [0, 0.5, 1],
            "seed_formula": "2000+10*j+r",
            "repetitions": {"48": 4, "28": 2},
            "order": "adjacent matched blocks; shuffled six-permutation cycles",
            "videos": "save all native comparison videos, no physics replay",
        },
        "exposure": "Previously used development family; conservatively treat scenes as exposed. "
        "Fresh seed repetitions only; no unseen-scene or family-transfer claim.",
        "allocation": json.loads((ROOT / "configs/maniguard/phase3/allocation.json").read_text()),
        "retry_policy": "no automatic retries; invalid evidence stops population",
        "information": "proposals, measured joints/fingers and previous final environment inputs",
        "disabled_controls": ["settling", "retry", "release_delay", "monitor_requery"],
    }


def freeze(path, engineering_audits):
    if len(engineering_audits) != 3 or [x.get("dial") for x in engineering_audits] != [0, 0.5, 1]:
        raise ValueError("Freeze requires all three declared engineering checks")
    if any(
        not a.get("engineering_valid") or a.get("min_chunk_length", 0) < 16
        for a in engineering_audits
    ):
        raise ValueError("Engineering validity and >=16 usable chunk actions are required")
    protocol = draft()
    protocol.update(
        status="frozen",
        sources=check_sources(),
        scenes=scene_receipts(),
        engineering_audits=engineering_audits,
        code_hashes={str(p.relative_to(ROOT)): sha(p) for p in implementation_files()},
    )
    inputs = [
        ROOT / "data/maniguard/policies/clutter/download-provenance.json",
        ROOT / "data/maniguard/benchmark/clutter_pickup/download-provenance.json",
        ROOT / "data/maniguard/datasets/omnigibson-robot-assets/models/franka/"
        "franka_panda_longfinger/download-provenance.json",
    ]
    protocol["input_receipts"] = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    if Path(path).exists():
        raise FileExistsError("Never replace an existing frozen protocol")
    atomic(path, protocol)
    return protocol


def validate_frozen(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("status") != "frozen" or protocol.get("domain") != DOMAIN:
        raise ValueError("Physical search requires the frozen checked Part 1 protocol")
    for relative, expected in protocol["code_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen implementation changed: " + relative)
    for relative, expected in protocol["input_receipts"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen input receipt changed: " + relative)
    if protocol["scenes"] != scene_receipts() or protocol["sources"] != check_sources():
        raise ValueError("Frozen scene/source identity changed")
    return protocol
