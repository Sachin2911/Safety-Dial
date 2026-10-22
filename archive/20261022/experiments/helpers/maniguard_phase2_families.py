"""Pinned family routing and scene-exposure checks, independent of the simulator.

This registry is coordinator metadata. It must never be part of the primary
controller observation: the policy may use task text, but not these identifiers.
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/maniguard/phase2"
PARTITIONS = ("engineering", "train", "validation", "test")


def load_registry(path=None):
    """Return family specifications keyed by their short coordinator names."""
    return json.loads(Path(path or CONFIG / "families.json").read_text())["families"]


def family_spec(family, path=None):
    try:
        return load_registry(path)[family]
    except KeyError as error:
        raise ValueError(f"Unknown ManiGuard family: {family}") from error


def load_sources(path=None):
    return json.loads(Path(path or CONFIG / "sources.json").read_text())


def load_splits(path=None):
    return json.loads(Path(path or CONFIG / "splits.json").read_text())


def scene_base_id(family, scene, registry_path=None):
    """Normalize a task or task/condition path without accepting arbitrary paths."""
    spec = family_spec(family, registry_path)
    parts = str(scene).split("/")
    if parts[0] == spec["benchmark_dir"]:
        parts = parts[1:]
    if len(parts) not in (1, 2) or not re.fullmatch(r"task_\d{4}", parts[0]):
        raise ValueError(f"Invalid benchmark scene: {scene}")
    if len(parts) == 2 and parts[1] not in spec["conditions"]:
        raise ValueError(f"Unknown benchmark condition: {parts[1]}")
    if parts[0] not in spec["task_ids"]:
        raise ValueError(f"Unknown base task for {family}: {parts[0]}")
    return parts[0]


def task_split(family, scene, splits_path=None, registry_path=None):
    task = scene_base_id(family, scene, registry_path)
    partitions = load_splits(splits_path)["families"][family]
    matched = [name for name in PARTITIONS if task in partitions[name]]
    if len(matched) != 1:
        raise ValueError(f"Base task must belong to exactly one partition: {family}/{task}")
    return matched[0]


def validate_scene_access(family, scene, purpose, *, frozen_evaluation=False,
                          splits_path=None, registry_path=None):
    """Reject accidental outcome exposure, including transfer-family training.

    ``metadata`` is a non-action-bearing schema/availability check, not permission
    to inspect outcome records. ``frozen_evaluation`` records a coordinator's
    already frozen protocol; this helper does not itself establish that freeze.
    Whole-family transfer can use any partition because the entire family was
    excluded. Lid is explicitly design-exposed, so it cannot support that stronger
    transfer claim even when its actor was excluded from optimization.
    """
    if purpose not in (*PARTITIONS, "transfer", "metadata"):
        raise ValueError(f"Unknown scene-use purpose: {purpose}")
    split = task_split(family, scene, splits_path, registry_path)
    role = family_spec(family, registry_path)["role"]
    if purpose == "metadata":
        return split
    if role in ("reserved_transfer", "design_exposed_transfer"):
        if purpose != "transfer" or not frozen_evaluation:
            raise ValueError(f"{family} is excluded from development; freeze transfer evaluation")
        return split
    if purpose == "transfer":
        raise ValueError(f"{family} is a development family, not whole-family transfer")
    if split != purpose:
        raise ValueError(f"{family}/{scene} is {split}, not permitted for {purpose}")
    if purpose == "test" and not frozen_evaluation:
        raise ValueError("Reserved test outcomes require a frozen evaluation protocol")
    return split


def deterministic_partitions(family, task_ids, seed):
    """New-family procedure frozen before outcomes: hash order, 4 / rest / 4 / 12."""
    if len(task_ids) < 21 or len(set(task_ids)) != len(task_ids):
        raise ValueError("Need at least 21 distinct base tasks")
    ordered = sorted(task_ids, key=lambda task: (
        hashlib.sha256(f"{seed}:{family}:{task}".encode()).hexdigest(), task,
    ))
    return {"engineering": ordered[:4], "train": ordered[4:-16],
            "validation": ordered[-16:-12], "test": ordered[-12:]}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_registry(config_dir=CONFIG, root=ROOT, *, verify_upstream=True):
    """Audit assignments and pinned configuration without loading any rollout."""
    config_dir, root = Path(config_dir), Path(root)
    manifest = json.loads((config_dir / "families.json").read_text())
    families = manifest["families"]
    splits = load_splits(config_dir / "splits.json")
    sources = load_sources(config_dir / "sources.json")
    legacy_path = root / "configs/maniguard/splits.json"
    legacy_sources_path = root / "configs/maniguard/sources.json"
    if _sha(legacy_path) != manifest["legacy_splits_sha256"]:
        raise ValueError("Phase-one split bytes changed")
    if _sha(legacy_sources_path) != manifest["legacy_sources_sha256"]:
        raise ValueError("Phase-one source bytes changed")
    legacy = json.loads(legacy_path.read_text())
    previous_sources = json.loads(legacy_sources_path.read_text())
    expected_names = {"jar", "lid", "stack", "clutter", "cabinet", "dusty"}
    if set(families) != expected_names or set(splits["families"]) != expected_names:
        raise ValueError("Phase-two registry must cover all six families")
    if sources["maniguard"]["revision"] != manifest["upstream_revision"]:
        raise ValueError("Upstream source pin mismatch")
    for name, spec in families.items():
        parts = splits["families"][name]
        if set(parts) != set(PARTITIONS):
            raise ValueError(f"Unexpected partitions for {name}")
        assigned = [task for split in PARTITIONS for task in parts[split]]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(spec["task_ids"]):
            raise ValueError(f"Incomplete or overlapping partitions for {name}")
        if len(assigned) != spec["base_tasks"]:
            raise ValueError(f"Wrong task count for {name}")
        if name in legacy["families"]:
            if parts != legacy["families"][name]:
                raise ValueError(f"Phase-one assignments changed for {name}")
            if sources["policies"][name] != previous_sources["policies"][name]:
                raise ValueError(f"Phase-one policy pin changed for {name}")
        elif parts != deterministic_partitions(name, spec["task_ids"], splits["seed"]):
            raise ValueError(f"New assignments do not follow declared hashing: {name}")
        policy = sources["policies"][spec["policy_key"]]
        if not re.fullmatch(r"[0-9a-f]{40}", policy["revision"]):
            raise ValueError(f"Unpinned policy: {name}")
        if spec["action_dim"] != 8 or spec["execute_horizon"] != 8:
            raise ValueError(f"Unexpected baseline action contract: {name}")
        if name in ("cabinet", "dusty") and (
            spec["role"] != "reserved_transfer" or spec["phase1_outcomes_exposed"]
        ):
            raise ValueError(f"Transfer-family exposure classification changed: {name}")
        if verify_upstream:
            path = root / "third_party/ManiGuard" / spec["eval_config"]
            if _sha(path) != spec["eval_config_sha256"]:
                raise ValueError(f"Pinned evaluator configuration changed: {name}")
            text = path.read_text()
            for key in ("max_steps", "grasping_mode", "execute_horizon", "action_dim",
                        "controller_preset"):
                match = re.search(r"^" + key + r":\s*([^#\n]+)", text, re.MULTILINE)
                if match is None or match.group(1).strip() != str(spec[key]):
                    raise ValueError(f"Upstream {key} mismatch: {name}")
            match = re.search(r"^serve_config_name:\s*([^#\n]+)", text, re.MULTILINE)
            expected_config = spec.get("upstream_serve_config", policy["config"])
            override = spec.get("policy_config_override")
            if override and (name != "lid" or override["config"] != policy["config"]):
                raise ValueError(f"Invalid preserved policy exception: {name}")
            if expected_config != policy["config"] and not override:
                raise ValueError(f"Undocumented policy exception: {name}")
            if match is None or match.group(1).strip() != expected_config:
                raise ValueError(f"Checkpoint configuration mismatch: {name}")
    return {"families": len(families), "base_tasks": sum(s["base_tasks"] for s in families.values()),
            "pilot_families": [name for name, spec in families.items() if spec["pilot"]],
            "behavioral_outcomes_loaded": False}
