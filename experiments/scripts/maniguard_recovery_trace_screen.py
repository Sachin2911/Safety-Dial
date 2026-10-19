"""Run the prospective cap/refresh diagnostic on verified historical records.

No alternate command is executed in a simulator. Threshold crossings describe
opportunities within factual query prefixes, not a counterfactual episode rollout.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_recovery_controller import (
    Config,
    cap_targets,
    direction_angle,
    final_input,
    normalized_correction,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def keyed(rows, kind):
    selected = [r for r in rows if r["type"] == kind]
    result = {r["step"]: r for r in selected}
    if len(result) != len(selected):
        raise ValueError("Duplicate recorded step for " + kind)
    return result


def load_episodes(run, protocol):
    receipt = read(run / "donor-restore.json")
    spec = protocol["donors"]
    if (not receipt["archive_download_hash_verified"] or not receipt["files_hash_verified"]
            or not receipt["private"] or receipt["archive_sha256"] != spec["sha256"]
            or receipt["revision"] != spec["revision"]):
        raise ValueError("Require the exact private, hash-verified donor archive")
    expected = {r["path"]: r["sha256"] for r in receipt["files"]}
    donor = run / "donors"
    episodes, inventory = [], []
    for job_path in sorted(donor.glob("**/data/job.json")):
        audit_path = job_path.parent.parent / "audit.json"
        audit, job = read(audit_path), read(job_path)
        if job["purpose"] == "engineering":
            continue
        if not audit.get("valid") or not audit.get("full_episode") or job["purpose"] != "train":
            raise ValueError("All selected donors must be valid full training records")
        paths = [job_path, audit_path, job_path.parent / "trace.jsonl"]
        for path in paths:
            if digest(path) != expected[str(path.relative_to(donor))]:
                raise ValueError("Restored evidence changed: " + str(path))
        trace = [json.loads(line) for line in paths[-1].read_text().splitlines()]
        p, u, bounds = [keyed(trace, kind) for kind in ("proposal", "executed", "action_space")]
        n = audit["steps"]
        if any(set(v) != set(range(n)) for v in (p, u, bounds)):
            raise ValueError("Incomplete proposal/executed/bounds alignment")
        queries = sorted(keyed(trace, "query"))
        if not queries or queries[0] != 0:
            raise ValueError("Missing initial query")
        commands = []
        for t in range(n):
            proposal = np.asarray(p[t]["original"], dtype=p[t]["event"]["proposal_dtype"])
            joints = np.asarray(p[t]["frame"]["joints"], dtype=np.float64)
            low, high = [np.asarray(bounds[t][key], dtype=np.float64) for key in ("low", "high")]
            if not np.array_equal(np.asarray(u[t]["action"]), np.asarray(bounds[t]["clipped"])):
                raise ValueError("Final recorded environment input differs from clipping receipt")
            if not np.array_equal(final_input(np.asarray(p[t]["adapted"], dtype=proposal.dtype),
                                             low, high), np.asarray(u[t]["action"])):
                raise ValueError("Recorded action-space replay failed")
            commands.append((proposal, joints, low, high))
        chunk_lengths = [end - start for start, end in zip(queries, queries[1:] + [n])]
        episodes.append({"job": job, "commands": commands, "queries": queries,
                         "chunk_lengths": chunk_lengths})
        inventory.append({"episode_id": job["episode_id"], "task": job["scene"].split("/")[0],
                          "condition": job["condition"], "controls": n,
                          "actual_queries": len(queries), "actual_horizon": job["effective_horizon"],
                          "source_hashes": {str(p.relative_to(donor)): digest(p) for p in paths}})
    if len(episodes) != spec["expected_full_episodes"]:
        raise ValueError("Unexpected full-episode denominator")
    return episodes, inventory


def geometry_summary(episode, cap, geometry):
    config = Config(cap=cap, geometry=geometry)
    angles, post_angles, errors = [], [], []
    changes = clip_changes = undefined = 0
    maximum = 0.0
    for proposal, joints, low, high in episode["commands"]:
        adapted = cap_targets(proposal, joints, config)
        executed = final_input(adapted, low, high)
        delta = float(np.max(np.abs(proposal[:7].astype(np.float64) - executed[:7])))
        changes += delta > 1e-6
        clip_changes += not np.array_equal(adapted[:7], executed[:7])
        maximum = max(maximum, float(np.max(np.abs(executed[:7] - joints))))
        if delta > 1e-6:
            before = direction_angle(proposal, adapted, joints)
            after = direction_angle(proposal, executed, joints)
            if before is None or after is None:
                undefined += 1
            else:
                angles.append(before)
                post_angles.append(after)
            errors.append(delta)
    job = episode["job"]
    return {"episode_id": job["episode_id"], "task": job["scene"].split("/")[0],
            "geometry": geometry, "cap_rad": cap, "controls": len(episode["commands"]),
            "changed_commands": changes, "angle_denominator": len(angles),
            "angles_above_5_degrees": int(np.sum(np.asarray(angles) > 5)),
            "post_clip_angles_above_5_degrees": int(np.sum(np.asarray(post_angles) > 5)),
            "maximum_angle_degrees": max(angles, default=0.0),
            "maximum_post_clip_angle_degrees": max(post_angles, default=0.0),
            "maximum_target_error_rad": maximum, "downstream_arm_clipping_changes": clip_changes,
            "undefined_zero_vector_angles": undefined}


def prefixes(episode, cap):
    result = []
    for start, length in zip(episode["queries"], episode["chunk_lengths"], strict=True):
        if length != 8:
            continue
        correction, cumulative = 0.0, []
        for p, q, low, high in episode["commands"][start:start + 8]:
            executed = final_input(cap_targets(p, q, Config(cap=cap)), low, high)
            correction += normalized_correction(p, executed)
            cumulative.append(correction)
        result.append({"query_step": start, "cumulative": cumulative})
    return result


def timing_rows(episodes, thresholds, minimum, cap, role):
    rows = []
    for episode in episodes:
        chunks = prefixes(episode, cap)
        for name, value in thresholds:
            ages = [next((age for age in range(minimum, 8)
                          if chunk["cumulative"][age - 1] >= value), None) for chunk in chunks]
            counts = Counter(age for age in ages if age is not None)
            job = episode["job"]
            rows.append({"role": role, "episode_id": job["episode_id"],
                         "task": job["scene"].split("/")[0], "threshold_name": name,
                         "threshold": value, "minimum_interval": minimum,
                         "complete_eight_action_prefixes": len(chunks),
                         "prefixes_with_early_opportunity": sum(counts.values()),
                         "age4": counts[4], "age5": counts[5],
                         "age6": counts[6], "age7": counts[7],
                         "discarded_incomplete_prefixes": sum(
                             length != 8 for length in episode["chunk_lengths"])})
    return rows


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(directory, geometry, timing):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), constrained_layout=True)
    colors = ["#bc6045", "#237e8a"]
    for i, kind in enumerate(("componentwise", "direction")):
        chosen = [r for r in geometry if r["cap_rad"] == .04 and r["geometry"] == kind]
        numerator = sum(r["angles_above_5_degrees"] for r in chosen)
        denominator = sum(r["angle_denominator"] for r in chosen)
        value = 100 * numerator / denominator if denominator else 0
        axes[0].bar(i, value, color=colors[i], width=.58)
        axes[0].text(i, value + 1.5, f"{numerator}/{denominator}", ha="center", fontsize=9)
    axes[0].set(xticks=[0, 1], xticklabels=["Componentwise", "Vector scaling"],
                ylabel="Changed commands with >5 deg distortion (%)",
                title="Uniform 0.04 rad cap; all 24 recordings", ylim=(0, 110))
    names = ["q50", "q75", "q90", "original_0.5"]
    labels = ["Median", "75th", "90th", "Original 0.5"]
    for i, name in enumerate(names):
        selected = [r for r in timing if r["role"] == "diagnostic" and
                    r["threshold_name"] == name and r["minimum_interval"] == 4]
        numerator = sum(r["prefixes_with_early_opportunity"] for r in selected)
        denominator = sum(r["complete_eight_action_prefixes"] for r in selected)
        value = 100 * numerator / denominator if denominator else 0
        axes[1].bar(i, value, color="#237e8a" if i < 3 else "#8c939c", width=.62)
        axes[1].text(i, value + .35, f"{numerator}/{denominator}", ha="center", fontsize=9)
    axes[1].set(xticks=range(4), xticklabels=labels, ylabel="Prefixes with early-query opportunity (%)",
                title="Frozen thresholds; 4 diagnostic recordings")
    axes[1].set_ylim(0, max(5, axes[1].get_ylim()[1] * 1.25))
    fig.suptitle("Saved-trace recovery diagnostic: command geometry and trigger opportunity",
                 fontsize=12)
    fig.savefig(directory / "diagnostic.pdf")
    fig.savefig(directory / "diagnostic.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    protocol = read(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "experiments/helpers/maniguard_recovery_controller.py",
               ROOT / "experiments/tests/test_maniguard_recovery_controller.py", args.protocol]
    freeze = {"schema": "maniguard-recovery-diagnostic-source-freeze-v1",
              "frozen_utc": datetime.now(UTC).isoformat(),
              "before_diagnostic_scoring": True,
              "files": {str(p.resolve().relative_to(ROOT)): digest(p) for p in sources}}
    save(args.output / "source-freeze.json", freeze)
    for source in sources:
        destination = args.output / "study_source" / source.resolve().relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    episodes, inventory = load_episodes(args.run, protocol)
    geometry = []
    for episode in episodes:
        if time.time() - started > protocol["budget"]["maximum_analysis_seconds"]:
            raise TimeoutError("Prospective offline diagnostic budget exhausted")
        for cap in protocol["geometry"]["caps_rad"]:
            for name in protocol["geometry"]["maps"]:
                geometry.append(geometry_summary(episode, cap, name))
    spec = protocol["timing"]
    calibration, diagnostic = [[e for e in episodes
                               if e["job"]["scene"].split("/")[0] in spec[role + "_tasks"]
                               and e["job"]["condition"] in spec[role + "_conditions"]]
                              for role in ("calibration", "diagnostic")]
    for role, selected in (("calibration", calibration), ("diagnostic", diagnostic)):
        if len(selected) != spec[role + "_expected_episodes"]:
            raise ValueError("Unexpected " + role + " episode denominator")
    positives = [chunk["cumulative"][age - 1] for e in calibration
                 for chunk in prefixes(e, spec["common_cap_rad"])
                 for age in spec["eligible_ages"] if chunk["cumulative"][age - 1] > 0]
    thresholds = [("q" + str(int(100 * q)), float(np.quantile(positives, q, method="linear")))
                  for q in spec["threshold_quantiles"]] if positives else []
    selected_threshold = thresholds[0][1] if thresholds else None
    save(args.output / "frozen-thresholds.json", {
        "frozen_before_diagnostic_prefix_scoring": True,
        "calibration_episodes": [e["job"]["episode_id"] for e in calibration],
        "positive_eligible_correction_values": len(positives),
        "thresholds": dict(thresholds), "selected_threshold": selected_threshold,
        "derivation": spec["quantile_method"], "protocol_sha256": digest(args.protocol)})
    thresholds += [("original_0.5", .5)]
    timing = [r for role, selected in (("calibration", calibration), ("diagnostic", diagnostic))
              for minimum in spec["minimum_intervals"]
              for r in timing_rows(selected, thresholds, minimum, spec["common_cap_rad"], role)]
    result = {"schema": "maniguard-recovery-saved-trace-diagnostic-v1",
              "study_id": protocol["study_id"], "started_utc": datetime.fromtimestamp(
                  started, UTC).isoformat(), "elapsed_analysis_seconds": time.time() - started,
              "protocol_sha256": digest(args.protocol), "source_freeze_sha256": digest(
                  args.output / "source-freeze.json"),
              "thresholds": dict(thresholds), "selected_threshold": selected_threshold,
              "positive_eligible_calibration_values": len(positives),
              "full_episodes": len(episodes), "tasks": sorted({i["task"] for i in inventory}),
              "historical_controls_reused": sum(i["controls"] for i in inventory),
              "historical_queries_reused": sum(i["actual_queries"] for i in inventory),
              "new_simulator_controls": 0, "new_physics_steps": 0,
              "new_policy_queries": 0, "new_encoder_calls": 0, "trained_model_fits": 0,
              "physical_safety_improvement_evaluated": False,
              "fresh_recovery_audit": False,
              "episodes": inventory, "geometry": geometry, "timing": timing,
              "limitations": protocol["limitations"]}
    write_csv(args.output / "geometry.csv", geometry)
    write_csv(args.output / "timing.csv", timing)
    plot(args.output, geometry, timing)
    result["elapsed_analysis_seconds"] = time.time() - started
    result["derived_files"] = {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}
    save(args.output / "diagnostic.json", result)
    print(json.dumps({k: result[k] for k in (
        "study_id", "elapsed_analysis_seconds", "full_episodes", "historical_controls_reused",
        "selected_threshold", "positive_eligible_calibration_values", "new_simulator_controls"
    )}, indent=2))


if __name__ == "__main__":
    main()
