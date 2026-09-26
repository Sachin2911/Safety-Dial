"""Audit source-trajectory roles before training or evaluating a new study run.

Test and stress are two views of the same final-test role and may share sources.
All other role overlaps are errors. Root identifiers alone do not establish independence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def partition_source_roles(families: dict, seed: int) -> dict:
    """Partition familiar trajectories into dev/test BEFORE sampling roots or pairs."""
    familiar = np.asarray(families["familiar"], dtype=np.int64)
    heldout = np.asarray(families["heldout"], dtype=np.int64)
    if len(familiar) < 2 or not len(heldout):
        raise ValueError("Need at least two familiar and one heldout source trajectories")
    if len(set(familiar)) != len(familiar) or len(set(heldout)) != len(heldout):
        raise ValueError("Source family contains duplicate trajectory IDs")
    if set(familiar) & set(heldout):
        raise ValueError("Familiar and heldout source families overlap")
    shuffled = np.random.default_rng(seed).permutation(familiar)
    split = max(1, len(shuffled) // 2)
    return {
        "dev": {"familiar": sorted(int(x) for x in shuffled[:split])},
        "test": {
            "familiar": sorted(int(x) for x in shuffled[split:]),
            "heldout": sorted(int(x) for x in heldout),
        },
    }


def inspect_bank_splits(bank_dirs: dict[str, Path], *,
                        allowed_overlaps=(("test", "stress"),)) -> dict:
    """Return an auditable role-overlap report from existing roots.json files.

    Push-T roots identify their source episode in meta.episode. Walker roots use
    episode at the top level. This assumes all supplied banks use one source dataset.
    """
    allowed = {frozenset(pair) for pair in allowed_overlaps}
    sources, roots, errors, summaries = {}, {}, [], {}
    for role, path in bank_dirs.items():
        blob = json.loads((Path(path) / "roots.json").read_text())
        source_ids, root_ids = [], []
        for root in blob["roots"]:
            episode = root.get("meta", {}).get("episode", root.get("episode"))
            if episode is None:
                errors.append(f"{role}: root {root.get('root_id')} has no source episode")
            else:
                source_ids.append(int(episode))
            if root.get("root_id") is None:
                errors.append(f"{role}: root has no root_id")
            else:
                root_ids.append(str(root["root_id"]))
        if not blob["roots"]:
            errors.append(f"{role}: bank has no roots")
        if len(root_ids) != len(set(root_ids)):
            errors.append(f"{role}: duplicate root IDs within the bank")
        sources[role], roots[role] = set(source_ids), set(root_ids)
        fingerprint = hashlib.sha256(json.dumps(sorted(sources[role])).encode()).hexdigest()
        summaries[role] = {
            "path": str(path), "n_roots": len(blob["roots"]),
            "n_source_episodes": len(sources[role]), "source_ids_sha256": fingerprint,
        }
    overlaps = []
    names = list(sources)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_source, shared_root = sorted(sources[a] & sources[b]), sorted(roots[a] & roots[b])
            permitted = frozenset((a, b)) in allowed
            overlaps.append({"roles": [a, b], "source_episodes": shared_source,
                             "root_ids": shared_root, "allowed": permitted})
            if (shared_source or shared_root) and not permitted:
                errors.append(f"{a}/{b}: {len(shared_source)} shared source episodes, "
                              f"{len(shared_root)} shared root IDs")
    return {"passes": not errors, "banks": summaries, "overlaps": overlaps, "errors": errors}


def validate_bank_splits(bank_dirs: dict[str, Path], *,
                         allowed_overlaps=(("test", "stress"),)) -> dict:
    report = inspect_bank_splits(bank_dirs, allowed_overlaps=allowed_overlaps)
    if not report["passes"]:
        raise ValueError("Source trajectory split integrity failed: " + "; ".join(report["errors"]))
    return report


def bank_identity(bank_dir: Path) -> dict:
    """Exact bank bytes, including tapes/truth, layouts and source roots."""
    from helpers.runManifest import file_sha256

    bank_dir = Path(bank_dir)
    return {name: file_sha256(bank_dir / name) for name in
            ("roots.json", "branches.h5", "layouts.json")}


def decomposition_gate(table: dict, *, min_false_safe=5, min_imagination=3,
                       min_imagination_share=0.25) -> dict:
    """A declared development-only diagnosis gate; defaults remain provisional."""
    if min_false_safe < 1 or min_imagination < 1 or not 0 <= min_imagination_share <= 1:
        raise ValueError("Invalid decomposition gate thresholds")
    attribution = table.get("attribution", {})
    n = sum(int(attribution.get(k, 0)) for k in ("temporal", "readout", "imagination"))
    imagined = int(attribution.get("imagination", 0))
    share = imagined / n if n else None
    passes = n >= min_false_safe and imagined >= min_imagination and share >= min_imagination_share
    return {"passes": bool(passes), "role": "development", "provisional_thresholds": True,
            "n_attributable_false_safe": n, "n_imagination": imagined,
            "imagination_share": share, "min_false_safe": min_false_safe,
            "min_imagination": min_imagination, "min_imagination_share": min_imagination_share,
            "excluded_from_attribution": "domain exits and censored futures"}


def validate_decomposition_gate(report: dict, banks_dir: Path, probes_run: Path) -> str:
    """Require passing diagnosis on the exact bank/probe bytes consumed by E2."""
    from helpers.runManifest import file_sha256

    if report.get("gate", {}).get("passes") is not True:
        raise ValueError("E1 decomposition gate has not passed; E2 predictor repair is not justified")
    if report["gate"].get("role") != "development":
        raise ValueError("E1 gate must use development outcomes only")
    identities = report.get("bank_identities", {})
    for name in ("dev", "test", "stress"):
        if identities.get(name) != bank_identity(Path(banks_dir) / name):
            raise ValueError(f"E1/E2 {name} bank identities differ")
    probe_name = report.get("probe")
    if probe_name not in {"block_pose_mlp", "block_pose_linear"}:
        raise ValueError("E1 report does not identify a supported frozen block-pose probe")
    if report.get("probe_sha256") != file_sha256(Path(probes_run) / f"{probe_name}.pt"):
        raise ValueError("E1/E2 frozen probe identities differ")
    return probe_name
