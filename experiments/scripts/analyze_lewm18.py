"""Independently score saved trajectories; paired case-level uncertainty and work totals."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def breach(xy, case, margin):
    if len(xy) == 0:
        return False
    local = (np.asarray(xy)[..., :2] - case["origin"]) @ np.asarray(case["basis"]).T
    return any(
        bool(
            (
                (local[:, 0] >= x0 - margin)
                & (local[:, 0] <= x1 + margin)
                & (local[:, 1] >= y0 - margin)
                & (local[:, 1] <= y1 + margin)
            ).any()
        )
        for x0, x1, y0, y1 in case["boxes"]
    )


def score(row, trace, margin):
    job, states = row["job"], trace["states"]
    goal = np.asarray(row["goal"])
    n = len(trace["actions"])
    assert n == len(states) == row["actions"]
    assert np.isfinite(states).all() and np.isfinite(trace["dense_xy"]).all()
    substeps, dimensions, timestep = (25, 3, 0.05) if job["env"] == "cube" else (10, 2, 0.1)
    assert len(trace["dense_xy"]) == substeps * n
    initial = np.asarray(row["initial_state"])
    path = np.vstack([initial, states.reshape(-1, len(initial))])[:, :dimensions]
    path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum())
    engaged = path_length > (1 if job["env"] == "pusht" else 0.001)
    assert bool(engaged) == row["engaged"]
    assert np.isclose(path_length, row["path_length"], atol=1e-10, rtol=0)
    assert np.isclose(row["simulated_seconds"], n * timestep, atol=1e-10, rtol=0)
    np.testing.assert_allclose(row["final_state"], states[-1] if n else initial, atol=1e-10, rtol=0)
    if len(states) == 0:
        success = False
    elif job["env"] == "cube":
        success = bool(np.any(np.linalg.norm(states[:, 3:6] - goal, axis=1) <= 0.04))
    else:
        angle = np.abs((states[:, 4] - goal[4] + np.pi) % (2 * np.pi) - np.pi)
        success = bool(
            np.any((np.linalg.norm(states[:, :4] - goal[:4], axis=1) < 20) & (angle < np.pi / 9))
        )
    hit = breach(trace["dense_xy"], job["case"], margin)
    raw_hit = breach(trace["dense_xy"], job["case"], 0)
    safe = success and not hit and row["error"] is None and row["controller_failure"] is None
    recorded_replans = len(row["replans"])
    # Every solve encodes exactly one current and one goal frame. Some Cube
    # prefix failures return before appending a replan record. Include that solve
    # in latency's denominator; remove the separate engineering parity call.
    encoder_frames = row["resources"].get("encoder_frames", 0)
    assert encoder_frames % 2 == 0
    replans = encoder_frames // 2 - int(row.get("native_parity_error") is not None)
    assert replans - recorded_replans in (0, 1)
    if replans != recorded_replans:
        assert row["controller_failure"] is not None or row["error"] is not None
    return {
        "id": job["id"],
        "case": job["case"]["id"],
        "seed": job["seed"],
        "stage": job["stage"],
        "env": job["env"],
        "method": job["method"],
        "margin": margin,
        "success": success,
        "safe": safe,
        "violation": raw_hit,
        "margin_breach": hit,
        "unfinished": not success,
        "engaged": bool(engaged),
        "controller_failure": row["controller_failure"] is not None,
        "error": row["error"] is not None,
        "initially_solved": row["initially_solved"],
        "actions": row["actions"],
        "J_safe": row["actions"] if safe else 50,
        "simulated_seconds": row["simulated_seconds"],
        "planning_seconds": row["planning_seconds"],
        "episode_seconds": row["episode_seconds"],
        "resources": row["resources"],
        "replans": replans,
        "recorded_replans": recorded_replans,
        "final_object_goal_distance": row["final_object_goal_distance"],
    }


def aggregate(rows):
    obj = {"n": len(rows)}
    for key in (
        "success",
        "safe",
        "violation",
        "margin_breach",
        "unfinished",
        "engaged",
        "controller_failure",
        "error",
        "initially_solved",
    ):
        obj[key] = sum(r[key] for r in rows)
    for key in ("J_safe", "planning_seconds", "episode_seconds", "final_object_goal_distance"):
        obj["mean_" + key] = float(np.mean([r[key] for r in rows]))
    safe_rows = [r for r in rows if r["safe"]]
    obj["safe_speed_n"] = len(safe_rows)
    for key in ("actions", "simulated_seconds"):
        obj["mean_safe_" + key] = float(np.mean([r[key] for r in safe_rows])) if safe_rows else None
    obj["resources"] = dict(
        sum((collections.Counter(r["resources"]) for r in rows), collections.Counter())
    )
    plans = sum(r["replans"] for r in rows)
    obj["planning_seconds_per_replan"] = (
        sum(r["planning_seconds"] for r in rows) / plans if plans else None
    )
    return obj


def contrast(rows, first, second, metric):
    groups = collections.defaultdict(dict)
    for row in rows:
        if row["method"] in (first, second):
            groups[(row["case"], row["seed"])][row["method"]] = row[metric]
    blocks = collections.defaultdict(list)
    for (case, seed), pair in groups.items():
        if set(pair) != {first, second}:
            raise RuntimeError(f"Incomplete paired comparison: {case}, {seed}")
        blocks[case].append(float(pair[first]) - float(pair[second]))
    values = np.asarray([np.mean(v) for _, v in sorted(blocks.items())])
    rng = np.random.default_rng(180990)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "first": first,
        "second": second,
        "metric": metric,
        "estimate": float(values.mean()),
        "interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        "confidence": 0.95,
        "constant_observed_case_effect": bool(np.ptp(values) == 0),
        "interval_limit": "When every observed case has the same effect, the bootstrap "
        "range collapses. This does not establish population equivalence or zero uncertainty.",
        "case_blocks": len(values),
        "pairs": len(groups),
        "interpretation": "Descriptive paired case-block bootstrap, no multiplicity claim gate",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--env", choices=("cube", "pusht"))
    args = parser.parse_args()
    root = args.root
    kinds = (args.env,) if args.env else ("cube", "pusht")
    if not (root / "campaign-complete.json").exists():
        raise RuntimeError("Campaign must finish before final analysis")
    protocol = read(Path(__file__).resolve().parents[2] / "configs/lewm18/protocol.json")
    rows, work, invalid = [], collections.Counter(), []
    paired_initial_states = {}
    feedback_activity = collections.defaultdict(collections.Counter)
    attempts = list((root / "attempts").iterdir())
    for path in sorted(attempts):
        if not (path / "result.json").exists():
            invalid.append(path.name)
            continue
        row = read(path / "result.json")
        assert row["job"]["env"] in kinds, path.name
        if row["job"]["stage"] == "H" and row["job"]["method"] == "feedback":
            params = row["job"]["params"]
            changed = [
                any(s[k] != params[k] for k in ("horizon", "variance", "retain"))
                for s in row["replans"]
            ]
            feedback_activity[(row["job"]["env"], row["job"]["margin"])].update(
                episodes=1,
                episodes_changed=int(any(changed)),
                recorded_solves=len(changed),
                changed_solves=sum(changed),
            )
        pair_key = (row["job"]["env"], row["job"]["case"]["id"], row["job"]["seed"])
        if pair_key in paired_initial_states:
            np.testing.assert_allclose(
                row["initial_state"],
                paired_initial_states[pair_key],
                atol=1e-10,
                rtol=0,
                err_msg=f"Paired initial state changed: {pair_key}",
            )
        else:
            paired_initial_states[pair_key] = row["initial_state"]
        work.update(row["resources"])
        with np.load(path / "trace.npz") as trace:
            audited = score(row, trace, row["job"]["margin"])
            assert audited["success"] == row["success"], path.name
            assert audited["safe"] == row["requested_safe_completion"], path.name
            assert audited["margin_breach"] == row["margin_breach"], path.name
            assert audited["violation"] == row["any_violation"], path.name
            assert audited["unfinished"] == row["unfinished"], path.name
            assert audited["J_safe"] == row["J_safe"], path.name
            if row["job"]["stage"] == "H" and row["job"]["method"] == "native":
                rows.extend(score(row, trace, m) for m in protocol["dials"][row["job"]["env"]])
            else:
                rows.append(audited)
    validation_work = collections.Counter()
    host_roots = sorted((root / "hosts").iterdir()) if (root / "hosts").exists() else [root]
    excluded_host_work = collections.Counter()
    excluded_host_attempts = []
    derived_initialization_work = collections.Counter()
    derived_validation_initialization = collections.Counter()
    cube_reset_actions = 0
    complete = read(root / "campaign-complete.json")
    selected_hosts = complete.get("hosts", {})
    for host_root in host_roots:
        for path in host_root.glob("validation-*-result.json"):
            validation = read(path)
            validation_work.update(validation["resources"])
            # Both retained pre-data checks reset two Cube environments, whose
            # upstream task-mode reset takes two random stabilization actions.
            # The successful check also resets PushT once.
            assert validation["resources"]["resets"] in (2, 3)
            derived_validation_initialization["reset_actions"] += 4
            derived_validation_initialization["reset_physics_steps"] += 100
            if validation["resources"]["resets"] == 3:
                derived_validation_initialization["reset_physics_steps"] += 2
                derived_validation_initialization["reset_render_frames"] += 1
        if (host_root / "synthetic-model-check.json").exists():
            validation_work.update(read(host_root / "synthetic-model-check.json")["resources"])
        for path in host_root.glob("serial-warmup-*.json"):
            validation_work.update(read(path)["resources"])
        if host_root != root:
            for path in (host_root / "attempts").glob("*/result.json"):
                row = read(path)
                if row["job"]["env"] not in kinds:
                    continue
                if row["job"]["env"] == "cube":
                    # Already included in DenseCube's executed physics counter;
                    # do not add these fifty steps to the total a second time.
                    assert row["resources"]["executed_physics_steps"] == 25 * (row["actions"] + 2)
                    cube_reset_actions += 2
                if row["job"]["env"] == "pusht" and "reset_physics_steps" not in row["resources"]:
                    # Pinned PushT.reset calls _set_state twice, each integrating
                    # once, and renders one goal image. These were not instrumented
                    # in the original cohort, so keep the derivation explicit.
                    derived_initialization_work["reset_physics_steps"] += 2
                    derived_initialization_work["reset_render_frames"] += 1
                if selected_hosts.get(row["job"]["env"]) != host_root.name:
                    excluded_host_work.update(row["resources"])
                    excluded_host_attempts.append(f"{host_root.name}/{path.parent.name}")
    all_work = work + excluded_host_work + validation_work
    accounting = {
        "environment_actions_excluding_reset": all_work["environment_actions"],
        "reset_stabilization_actions": cube_reset_actions
        + derived_validation_initialization["reset_actions"],
        "environment_resets": all_work["environment_resets"] + all_work["resets"],
        "physics_integration_steps_including_reset_and_previews": sum(
            all_work[k]
            for k in (
                "executed_physics_steps",
                "checker_physics_steps",
                "counterfactual_physics_steps",
                "reset_physics_steps",
            )
        )
        + derived_initialization_work["reset_physics_steps"]
        + derived_validation_initialization["reset_physics_steps"],
        "counterfactual_physics_steps": all_work["checker_physics_steps"]
        + all_work["counterfactual_physics_steps"],
        "predictor_steps": all_work["predictor_steps"],
        "model_batch_calls": all_work["model_batch_calls"],
        "model_candidate_rollouts": all_work["model_candidate_rollouts"],
        "render_frames_including_reset": all_work["render_frames"]
        + all_work["renders"]
        + all_work["reset_render_frames"]
        + derived_initialization_work["reset_render_frames"]
        + derived_validation_initialization["reset_render_frames"],
    }
    grouped = collections.defaultdict(list)
    for row in rows:
        if row["stage"] in ("H", "C", "T"):
            grouped[(row["stage"], row["env"], row["margin"], row["method"])].append(row)
    tables = [
        {"stage": k[0], "env": k[1], "margin": k[2], "method": k[3], **aggregate(v)}
        for k, v in sorted(grouped.items())
    ]
    contrasts = []
    for kind in kinds:
        for margin in protocol["dials"][kind]:
            cell = [r for r in rows if (r["stage"], r["env"], r["margin"]) == ("H", kind, margin)]
            for first, second in [("fixed", "canonical"), ("feedback", "fixed")]:
                for metric in ("safe", "J_safe"):
                    contrasts.append(
                        {"env": kind, "margin": margin, **contrast(cell, first, second, metric)}
                    )
    context_strata = []
    for kind in kinds:
        for initially_solved in (False, True):
            group = [
                r
                for r in rows
                if r["stage"] == "C"
                and r["env"] == kind
                and r["initially_solved"] == initially_solved
            ]
            if group:
                context_strata.append(
                    {"env": kind, "initially_solved_group": initially_solved, **aggregate(group)}
                )
    report = {
        "schema": "lewm18-analysis-v1",
        "environments": list(kinds),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unique_attempts": len(attempts),
        "missing_result_attempts": invalid,
        "tables": tables,
        "contrasts": contrasts,
        "benchmark_context_strata": context_strata,
        "feedback_activity": [
            {"env": k[0], "margin": k[1], **v} for k, v in sorted(feedback_activity.items())
        ],
        "feedback_activity_scope": "Changes from the base settings in recorded replans only; "
        "failed final prefix solves may have no replan entry. Descriptive, post-hoc audit.",
        "episode_work": dict(work),
        "predata_validation_work": dict(validation_work),
        "excluded_host_attempts": excluded_host_attempts,
        "excluded_host_work": dict(excluded_host_work),
        "excluded_cohort_note": "The original PushT cohort is excluded after a reset-anchor "
        "repair; its attempts and work remain charged, even on the same physical host.",
        "derived_original_pusht_reset_work": dict(derived_initialization_work),
        "derived_predata_reset_work": dict(derived_validation_initialization),
        "cube_reset_accounting": "Two reset stabilization actions and fifty physics steps "
        "per episode. The fifty steps are already included in executed_physics_steps; "
        "only the reset action count is additional. Pre-data reset steps are derived separately.",
        "all_charged_work": accounting,
        "derived_reset_work_basis": "Two _set_state integrations and one goal render per "
        "completed PushT reset, from pinned stable_worldmodel 0.1.1; not measured counters.",
        "hosts_by_environment": selected_hosts,
        "physical_instances_by_environment": complete.get("physical_instances", selected_hosts),
        "independent_scoring": "All recorded results checked against saved state and dense path",
        "paired_initial_state_check": "Recorded initial state equal across methods and margins "
        "for every shared environment/case/seed, tolerance 1e-10.",
        "native_rescoring": "One final rollout scored at three margins; not three independent attempts",
        "timing": "Final episodes have two concurrent workers. Serial T-stage latency is separate.",
        "replan_accounting": "Count solve calls from two encoded frames per solve, excluding "
        "the engineering native-parity check. This includes a final failed Cube prefix solve "
        "that returns before appending the usual replan record.",
        "scope": "Nontrivial dataset-pair goals with synthetic keep-outs; original object objective "
        "for Cube, not full five-goal OGBench leaderboard.",
        "no_evolutionary_claim": True,
    }
    target = root / "analysis"
    target.mkdir(exist_ok=True)
    write(target / "summary.json", report)
    write(target / "scored-rows.json", rows)
    print(
        json.dumps({"unique_attempts": len(attempts), "tables": len(tables), "invalid": invalid}),
        flush=True,
    )


if __name__ == "__main__":
    main()
