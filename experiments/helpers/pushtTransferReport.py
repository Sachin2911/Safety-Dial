"""CPU-only E4 statistics on immutable E3 rows, with source-episode clustering."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from helpers.decomposition import ordinary_motion
from helpers.dialMetrics import cluster_bootstrap, fsa
from helpers.pushtSourceFamilies import SOURCE_FAMILY_PROTOCOL, geometric_source_family
from helpers.pushtGeometry import Disc, hazard_from_dict
from helpers.pushtLayouts import FAMILIES as LAYOUT_FAMILIES, cell_of
from helpers.runManifest import file_sha256

FAMILIES = ("familiar", "heldout")
CELLS = tuple(f"layout={hazard}/source={source}" for hazard in FAMILIES for source in FAMILIES)


def source_metadata(roots: list[dict]) -> dict[int, dict]:
    metadata = {}
    for index, root in enumerate(roots):
        meta = root["meta"]
        family = geometric_source_family(meta["state_at_root"], root["goal_state"])
        if (family not in FAMILIES or family != meta.get("source_family")
                or meta.get("source_family_protocol") != SOURCE_FAMILY_PROTOCOL["version"]):
            raise ValueError("E4 requires actual geometric root/goal families")
        metadata[index] = {"family": family, "episode": int(meta["episode"]),
                           "root_id": root["root_id"]}
    return metadata


def validate_layouts(layouts, sources):
    expected = {(source["root_id"], family) for source in sources.values() for family in FAMILIES}
    keys = [(layout["root_id"], layout["family"]) for layout in layouts]
    if set(keys) != expected or len(keys) != len(set(keys)):
        raise ValueError("Each final root must have exactly both frozen hazard families")
    for layout in layouts:
        hazard, family = hazard_from_dict(layout["hazard"]), LAYOUT_FAMILIES[layout["family"]]
        size = hazard.r if isinstance(hazard, Disc) else np.sqrt((hazard.x1 - hazard.x0) * (hazard.y1 - hazard.y0)) / 2
        if (cell_of(hazard.centre) not in set(map(tuple, family["cells"]))
                or not family["size"][0] <= size <= family["size"][1]):
            raise ValueError("Hazard geometry disagrees with its claimed held-out/familiar family")


def row_key(row):
    return int(row["root"]), int(row["branch"]), row["layout"]


def cell_key(row, sources):
    return f"layout={row['layout']}/source={sources[int(row['root'])]['family']}"


def load_rows(directory: Path, reference: dict) -> dict:
    path = (Path(directory) / reference["file"]).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError("E3 row artifact must remain inside its result directory")
    if file_sha256(path) != reference["sha256"]:
        raise ValueError("E3 evaluation row artifact changed")
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["rows"]))


def validate_rows(rows, sources, expected_keys):
    keys = [row_key(row) for row in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(expected_keys):
        raise ValueError("E4 rows must cover every frozen branch/layout exactly once")
    for row in rows:
        if int(row["root"]) not in sources or row["layout"] not in FAMILIES:
            raise ValueError("Invalid E4 root/layout identity")
        if not np.isfinite(row["cmin_imagined"]) or not np.isfinite(row["cmin_dense"]):
            raise ValueError("E4 clearances must be finite; censoring is a separate mask")
        for key in ("unsafe_composite", "censored", "arena_exit"):
            if type(row.get(key)) is not bool:
                raise ValueError(f"Explicit observed outcome metadata required: {key}")


def _arrays(rows, sources):
    return {"c": np.array([r["cmin_imagined"] for r in rows], float),
            "u": np.array([r["unsafe_composite"] for r in rows], bool),
            "censored": np.array([r["censored"] for r in rows], bool),
            "episode": np.array([sources[int(r["root"])]["episode"] for r in rows]),
            "cell": np.array([cell_key(r, sources) for r in rows])}


def _bootstrap_interval(stat_fn, episode, *, n_boot, seed, **arrays):
    """Expose discarded replicates without treating their conditional CI as a point."""
    result = cluster_bootstrap(stat_fn, episode, n_boot=n_boot, seed=seed, **arrays)
    usable = int(result["n_boot"])
    return {**result, "n_boot_requested": int(n_boot), "n_boot_usable": usable,
            "n_boot_dropped": int(n_boot) - usable,
            "point_defined": bool(np.isfinite(result["point"])),
            "interval_defined": bool(np.isfinite(result["lo"]) and np.isfinite(result["hi"])),
            "interval_scope": "Usable bootstrap replicates only; a finite interval cannot resolve an undefined full-data point"}


def cell_statistics(rows, sources, margin, *, n_boot=1000, seed=0):
    out = {}
    for cell in CELLS:
        subset = [row for row in rows if cell_key(row, sources) == cell]
        if not subset:
            raise ValueError(f"Missing E4 transfer cell: {cell}")
        arrays = _arrays(subset, sources)
        episode = arrays.pop("episode")
        arrays.pop("cell")
        point = fsa(arrays["c"], arrays["u"], margin, censored=arrays["censored"])
        intervals = {}
        for metric in ("fsa", "acceptance_rate"):
            intervals[metric] = _bootstrap_interval(
                lambda c, u, censored, key=metric: fsa(c, u, margin, censored=censored)[key],
                episode, n_boot=n_boot, seed=seed, **arrays)
        out[cell] = {**point, "n_roots": len({r["root"] for r in subset}),
            "n_source_episodes": int(len(np.unique(episode))), "n_branches": len(subset),
            "n_unsafe_observed": int(arrays["u"].sum()),
            "n_censored_futures": int(arrays["censored"].sum()),
            "n_arena_exits": sum(r["arena_exit"] for r in subset), "intervals": intervals,
            "bootstrap_cluster": "source episode; all root siblings and tapes sampled together"}
    return out


def paired_statistics(random_rows, boundary_rows, sources, random_margin, boundary_margin,
                      *, n_boot=1000, seed=0):
    a, b = sorted(random_rows, key=row_key), sorted(boundary_rows, key=row_key)
    if [row_key(r) for r in a] != [row_key(r) for r in b]:
        raise ValueError("E4 comparisons require the same tapes/layouts")
    for left, right in zip(a, b):
        if any(left[key] != right[key] for key in ("unsafe_composite", "censored", "cmin_dense", "arena_exit")):
            raise ValueError("Paired E4 rows disagree on observed truth")
    arrays = _arrays(a, sources)
    arrays["cb"] = np.array([r["cmin_imagined"] for r in b], float)
    episode = arrays.pop("episode")

    def delta(c, cb, u, censored, cell, target):
        take = cell == target
        return (fsa(cb[take], u[take], boundary_margin, censored=censored[take])["fsa"]
                - fsa(c[take], u[take], random_margin, censored=censored[take])["fsa"])

    differences, shifts = {}, {}
    for target in CELLS:
        differences[target] = _bootstrap_interval(
            lambda **data: delta(**data, target=target), episode,
            n_boot=n_boot, seed=seed, **arrays)
        if target == CELLS[0]:
            continue
        interval = _bootstrap_interval(
            lambda **data: delta(**data, target=target) - delta(**data, target=CELLS[0]),
            episode, n_boot=n_boot, seed=seed, **arrays)
        reference = differences[CELLS[0]]
        if not interval["point_defined"]:
            status = "undefined_outcome"
        elif not interval["interval_defined"]:
            status = "insufficient_bootstrap_evidence"
        else:
            status = ("boundary_advantage_shrinks" if interval["lo"] > 0 else
                      "boundary_advantage_expands" if interval["hi"] < 0 else "inconclusive")
        premise = bool(reference["point_defined"] and reference["interval_defined"]
                       and reference["hi"] < 0)
        if not reference["point_defined"] or not interval["point_defined"]:
            interpretation = "undefined_outcome"
        elif not reference["interval_defined"] or not interval["interval_defined"]:
            interpretation = "insufficient_bootstrap_evidence"
        else:
            interpretation = status if premise else "no_demonstrated_familiar_cell_advantage"
        shifts[target] = {**interval, "contrast_direction": status,
            "reference_boundary_advantage_demonstrated": premise,
            "reference_point_defined": reference["point_defined"],
            "reference_interval_defined": reference["interval_defined"],
            "interpretation": interpretation,
            "contrast": "(boundary-random FSA) in target minus familiar-layout/familiar-source"}
    return {"boundary_minus_random_fsa": differences, "h4_advantage_change": shifts,
            "reference_cell": CELLS[0], "bootstrap_cluster": "source episode with paired method rows"}


def check_acquisition(report):
    if (report.get("status") != "complete" or report.get("gate", {}).get("diagnostic")
            or not report.get("gate", {}).get("e2_passed")
            or not report.get("split_audit", {}).get("passes")
            or report.get("source_family_protocol") != SOURCE_FAMILY_PROTOCOL):
        raise ValueError("E4 requires a complete qualified geometric acquisition study")
    arms = report.get("arms", {})
    if not {"random", "boundary"} <= set(arms):
        raise ValueError("E4 requires both prospective acquisition arms")
    seeds = set(arms["random"])
    if len(seeds) < 3 or any(set(values) != seeds for values in arms.values()):
        raise ValueError("E4 requires the same three or more acquisition seeds per arm")
    planned = np.cumsum(report["rounds"]).tolist()
    for values in arms.values():
        for curve in values.values():
            if [p["budget_added"] for p in curve["curve"]] != planned:
                raise ValueError("E4 requires every planned acquisition budget")
    return sorted(seeds), planned


def summarize_variant(rows, sources, margin, *, n_boot=1000, seed=0):
    return {"margin_calibrated_on_development": float(margin),
            "cells": cell_statistics(rows, sources, margin, n_boot=n_boot, seed=seed),
            "ordinary_motion": ordinary_motion(rows)}


def h4_conclusion(paired, seeds, budget):
    out = {}
    for bank in ("test", "stress"):
        out[bank] = {}
        for cell in CELLS[1:]:
            statuses = {seed: paired[str(budget)][seed][bank]["h4_advantage_change"][cell]["interpretation"]
                        for seed in seeds}
            unique = set(statuses.values())
            conclusion = (next(iter(unique)) if len(unique) == 1 else
                          "undefined_outcome_in_some_acquisition_seeds" if "undefined_outcome" in unique else
                          "mixed_across_acquisition_seeds")
            out[bank][cell] = {"conclusion": conclusion, "per_seed": statuses}
    return {"budget_added": budget, "by_bank": out,
            "criterion": "Direction requires finite full-data contrast and reference points and finite intervals in every acquisition seed; exploratory, no multiple-comparison correction",
            "acceptance_caveat": "Margins are frozen from development calibration; achieved acceptance is reported per cell and is not retuned on final test data"}
