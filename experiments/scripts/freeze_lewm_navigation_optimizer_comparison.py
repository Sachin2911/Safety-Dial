"""Freeze NAV optimizer comparisons using JSON metadata and source hashes only.

This stdlib-only builder never imports experiment helpers, reads array/checkpoint
contents, accesses final pixels, starts workers or launches controller episodes.
"""

import argparse
import ast
import copy
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("none", "circle3", "circle5", "circle7")
STRONG = [0.25, 0.25, 0.1, 0.25, 0.1, 2.0, 0.25, 2.0, 4.0, 1.3862943611198906]
HOSTS = [
    {"repeat_id": 0, "host": "navigation", "pod_id": "vukovc3t4o1dui"},
    {"repeat_id": 1, "host": "pusht", "pod_id": "wq4ar2qx2ff6r4"},
    {"repeat_id": 2, "host": "navigation", "pod_id": "vukovc3t4o1dui"},
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_source_closure(initial):
    """Pin statically imported project helpers, including entrypoint overrides."""
    pending, seen = list(initial), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = ROOT / name
        require(path.is_file(), "Missing declared project source: " + name)
        seen.add(name)
        tree = ast.parse(path.read_text(), filename=name)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
                if node.module == "helpers":
                    modules.extend("helpers." + alias.name for alias in node.names)
        for module in modules:
            candidates = [
                ROOT / "experiments" / (module.replace(".", "/") + ".py"),
                ROOT / "experiments/scripts" / (module.replace(".", "/") + ".py"),
            ]
            for candidate in candidates:
                if candidate.is_file():
                    pending.append(str(candidate.relative_to(ROOT)))
    return sorted(seen)


def check_lever(base_path, evidence_path, training_ids):
    """A fixed prespecified panel may gate launch, never be outcome-filtered."""
    state = json.loads(evidence_path.read_text())
    require(state.get("phase") == "complete", "A completed prospective lever pilot is required")
    source = json.loads((evidence_path.parent / "source-freeze.json").read_text())
    require(
        source["protocol_sha256"] == digest(base_path), "Lever evidence belongs to another pilot"
    )
    rows = list(state["jobs"].values())
    selected = {"native": [], "fixed_geometry_progress": []}
    for row in rows:
        require(
            not row.get("error") and not row.get("extra", {}).get("frame_flags"),
            "Technical or image failure requires review before optimizer freeze",
        )
        if row["case_id"] in training_ids and row["method"] in selected:
            selected[row["method"]].append(row)
    require(
        all(
            len(rows) == len(training_ids) and {r["case_id"] for r in rows} == training_ids
            for rows in selected.values()
        ),
        "Exactly one native/fixed result is required per prespecified TRAIN case",
    )
    for row in selected["fixed_geometry_progress"]:
        failure = row.get("extra", {}).get("controller_failure") or ""
        require(
            not (row["environment_actions"] == 0 and "No admitted local" in failure),
            "Initial empty-support TRAIN case needs separately frozen zero-decision PG handling; do not omit the charged case",
        )
    counts = {
        method: sum(bool(row["safe_completion"]) for row in values)
        for method, values in selected.items()
    }
    require(
        counts["native"] < counts["fixed_geometry_progress"] < len(training_ids),
        "Prespecified TRAIN panel needs a useful non-ceiling lever; do not substitute cases based on outcomes",
    )
    return {
        "state_sha256": digest(evidence_path),
        "source_freeze_sha256": digest(evidence_path.parent / "source-freeze.json"),
        "training_case_ids": sorted(training_ids),
        "safe_completions": counts,
        "denominator_each": len(training_ids),
        "role": "Launch gate only, no outcome-based panel substitution",
    }


def write(path, value):
    if path.exists():
        raise FileExistsError("Never overwrite a prospective freeze: " + str(path))
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return {"path": str(path), "sha256": digest(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def balanced(cases, role, quota):
    selected = []
    for condition in CONDITIONS:
        members = sorted(
            (c for c in cases if c["split"] == role and c["condition"] == condition),
            key=lambda c: (c["source_episode"], c["source_step"], c["case_id"]),
        )
        require(len(members) >= quota, "Insufficient frozen metadata quota: " + role + condition)
        selected.extend(copy.deepcopy(members[:quota]))
    return sorted(selected, key=lambda c: (c["source_episode"], c["case_id"]))


def fixed_grid():
    """48 distinct nonzero candidates, plus the separately evaluated common anchor."""
    result = []
    for geometry, persistence, progress in itertools.product(
        (0.5, 1.0, 2.0, 4.0), (0.5, 1.0, 2.0, 4.0), (1.0, 2.0, 4.0)
    ):
        genes = STRONG.copy()
        genes[5], genes[7], genes[8] = geometry, persistence, progress
        if genes != STRONG:
            result.append(genes)
    # The factorial contains the separately measured anchor. Replace that duplicate
    # with a prospective stronger-latent-match setting, retaining48 novel candidates.
    result.append([0.5, 0.5, 0.2, 0.5, 0.2, 2.0, 0.5, 2.0, 4.0, STRONG[9]])
    require(len(result) == len({tuple(p) for p in result}) == 48, "Fixed grid must be distinct")
    require(all(all(value != 0 for value in p) for p in result), "Fixed features must be nonzero")
    return result


def seeded(cases, repeat, stage):
    offset = (repeat + 1) * 10000 + {"training": 0, "selection": 100000}[stage]
    result = copy.deepcopy(cases)
    for case in result:
        case["seed"] += offset
    return result


def evaluation(base, pod_id, cases, cap):
    result = copy.deepcopy(base["evaluation"])
    result["execution_pod_id"] = pod_id
    result["maximum_episode_actions"] = cap
    result["allowed_source_episodes"] = sorted({c["source_episode"] for c in cases})
    for case in cases:
        result["asset_sha256"][case["input_path"]] = case["input_sha256"]
    return result


def comparison(base, name, namespace, design, jobs, sources, maximum_actions):
    return {
        "schema": "lewm-navigation-comparison-v1",
        "name": name,
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output_namespace": namespace,
        "evaluation": design,
        "case_preparation": copy.deepcopy(base["case_preparation"]),
        "source_files": list(sources),
        "source_sha256": sources,
        "jobs": jobs,
        "maximum_episode_starts": len(jobs),
        "maximum_native_resets": len(jobs),
        "maximum_environment_actions": maximum_actions,
        "controller_static_parameters": {
            "common_strong_anchor": STRONG,
            "provenance": "All job parameter vectors are explicit; common anchor remains separate from trained candidate provenance",
        },
        "information": base.get("information"),
        "optimizer_updates": 0,
    }


def make_jobs(cases, candidates, cap, repeat):
    jobs = []
    for label, method, parameters in candidates:
        for case in cases:
            jobs.append(
                {
                    "job_id": label + "_" + case["case_id"] + "_s" + str(case["seed"]),
                    "method": method,
                    "parameters": parameters.copy(),
                    "case": copy.deepcopy(case),
                    "environment_seed": case["seed"],
                    "policy_seed": case["seed"],
                    "repeat_id": repeat,
                    "max_actions": cap,
                }
            )
    return jobs


def freeze(args):
    base_path = args.base_pilot.resolve()
    base = json.loads(base_path.read_text())
    require(
        base["evaluation"]["controller_backend"] == "observed-target-v2",
        "Ten-gene v2 family required",
    )
    require(base.get("parameter_count") == 10, "Shared ten-gene representation must be explicit")
    require(
        base["evaluation"]["progress_backend"] == "navigation-append-fsync-v1",
        "Durable accounting required",
    )
    require(base["evaluation"]["planner"]["checked"], "Learned action checks must remain enabled")
    metadata_panel = base.get("metadata_panel")
    panel_ref = metadata_panel or {
        "path": base["case_preparation"]["panel_path"],
        "sha256": base["case_preparation"]["panel_sha256"],
    }
    panel_path = args.panel if args.panel else ROOT / panel_ref["path"]
    require(digest(panel_path) == panel_ref["sha256"], "Frozen metadata panel identity differs")
    episode_actions = args.episode_actions
    if episode_actions is None:
        episode_actions = base["evaluation"]["maximum_episode_actions"]
    require(
        episode_actions == base["evaluation"]["maximum_episode_actions"],
        "Episode horizon must match the qualified lever pilot; a different horizon requires new matching prospective pilot evidence",
    )
    panel = json.loads(panel_path.read_text())
    require(panel.get("final_pixels_read", 0) == 0, "Final pixels were already accessed")
    # The four-case training panel is selected from the original eight pilot cases,
    # without consulting any pilot/training outcome, score or optimizer artifact.
    allowed = set(base["pilot_case_ids"])
    native_cases = [copy.deepcopy(j["case"]) for j in base["jobs"] if j["method"] == "native"]
    require(
        {c["case_id"] for c in native_cases} == allowed,
        "Base pilot must retain every prespecified case",
    )
    train = balanced(native_cases, "development_training", 1)
    lever = check_lever(base_path, args.lever_evidence, {c["case_id"] for c in train})
    selection_metadata = panel
    selection_metadata_sha256 = digest(panel_path)
    if args.selection_panel:
        require(
            args.selection_panel_sha256 is not None, "Explicit selection metadata SHA is required"
        )
        require(
            digest(args.selection_panel) == args.selection_panel_sha256,
            "Selection metadata changed",
        )
        selection_metadata = json.loads(args.selection_panel.read_text())
        selection_metadata_sha256 = args.selection_panel_sha256
    elif metadata_panel is not None:
        raise ValueError(
            "Changed geometry requires separately frozen matching fresh selection metadata; old radii are not a fallback"
        )
    require(
        selection_metadata.get("final_pixels_read", 0) == 0,
        "Selection metadata must preserve the final seal",
    )
    if metadata_panel is not None:
        require(
            "geometry_transform" in panel
            and selection_metadata.get("geometry_transform") == panel["geometry_transform"],
            "Training and selection geometry transformations differ",
        )
    selection = balanced(selection_metadata["cases"], "development_selection", 2)
    train_episodes = {c["source_episode"] for c in train}
    selection_episodes = {c["source_episode"] for c in selection}
    final_episodes = set(panel["sealed_final_episode_ids"])
    require(
        not (
            train_episodes & selection_episodes
            or (train_episodes | selection_episodes) & final_episodes
        ),
        "Frozen roles overlap",
    )
    hosts = json.loads(args.repeat_hosts.read_text()) if args.repeat_hosts else copy.deepcopy(HOSTS)
    require(
        sorted(h["repeat_id"] for h in hosts) == [0, 1, 2],
        "Exactly three independent learner repeats",
    )
    require(
        len({h["pod_id"] for h in hosts}) == 2,
        "Use two physical hosts, with all methods on each assigned repeat host",
    )
    require(
        args.population >= 2 and args.generations >= 1 and episode_actions >= 1,
        "Positive optimizer limits required",
    )
    require(
        all(
            math.isfinite(value) and value > 0
            for value in (args.sigma, args.learning_rate, args.temperature)
        ),
        "Finite positive optimizer/learner scales required",
    )
    candidate_panels = args.population * args.generations
    require(candidate_panels % 2 == 0, "PG needs an even total sample/evaluation panel allowance")
    require(
        candidate_panels == 48 or args.allow_budget_override,
        "Root must explicitly freeze any changed48-panel budget",
    )
    episode_starts = (candidate_panels + 1) * len(train)
    action_cap = episode_starts * episode_actions
    fixed = fixed_grid()
    require(
        candidate_panels <= len(fixed),
        "The prospective fixed-grid pool has at most48 novel settings",
    )
    if candidate_panels != len(fixed):
        # A root-approved smaller budget retains an outcome-independent subset.
        fixed.sort(key=lambda genes: hashlib.sha256(json.dumps(genes).encode()).hexdigest())
        fixed = fixed[:candidate_panels]
    entrypoint = args.entrypoint
    files = sorted(
        set(base["source_files"])
        | set(
            project_source_closure(
                [entrypoint, *args.extra_source, str(Path(__file__).relative_to(ROOT))]
            )
        )
    )
    for name, expected in base["source_sha256"].items():
        require(digest(ROOT / name) == expected, "Parent pilot source changed: " + name)
    sources = {name: digest(ROOT / name) for name in files}
    out = args.out_dir.resolve()
    require(not out.exists(), "Use a new comparison freeze namespace")
    out.mkdir(parents=True)
    written, schedule = [], []
    common = {
        "schema": "lewm-campaign-training-v1",
        "deadline_utc": base["evaluation"]["deadline_utc"],
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "initial_parameters": STRONG,
        "max_environment_actions": action_cap,
        "max_episode_actions": episode_actions,
        "max_episode_starts": episode_starts,
        "population": args.population,
        "sigma": args.sigma,
        "parameter_bounds": [-6.0, 6.0],
        "preservation_reserve_seconds": args.reserve_seconds,
        "reinforce": {
            "learning_rate": args.learning_rate,
            "baseline_decay": 0.9,
            "temperature": args.temperature,
            "max_grad_norm": 1.0,
        },
        "case_preparation": copy.deepcopy(base["case_preparation"]),
        "source_files": files,
        "source_sha256": sources,
        "source_ids": {
            "sources": sources,
            "parent_pilot": {"path": str(base_path), "sha256": digest(base_path)},
            "panel_sha256": digest(panel_path),
            "selection_metadata_sha256": selection_metadata_sha256,
            "controller_representation": "observed-target-v2 ten genes,11 features",
            "entrypoint": entrypoint,
        },
        "budget_semantics": {
            "total_panel_maximum_including_anchor": candidate_panels + 1,
            "cma_random_candidate_panels": candidate_panels,
            "reinforce_updates": candidate_panels // 2,
            "cma_generation_updates": args.generations,
            "reinforce_panels_per_update": 2,
            "same_hard_actual_action_allowance": True,
            "actual_consumption_and_residual_unspent_reported_separately": True,
            "unfinished_panels_supply_no_nominee_fitness_or_CMA_tell": True,
        },
        "lever_gate": lever,
        "fitness": "Lexicographic safe completions, then fewer violations, then original-task progress sum. CMA consumes within-generation minimization ranks, without arbitrary safety penalties.",
        "ordinary_learning_objective": "For N=4, r=5*safe_completion-violation+progress/5. Mean over episodes of -previous-update-case-EMA-advantage times SUM categorical decision log probabilities; exact shared masked ten-gene scores, on-policy traces, Adam, norm clip1, shared bounds. Both stochastic training and deterministic post-update panels charged.",
        "zero_decision_precondition": "Current unchanged harness rejects zero-decision categorical episodes. Any such support refusal requires separately frozen NAV-only handling before launching PG; do not filter cases or omit charged failures.",
        "learning_curve_reporting": {
            "x_axis": "Cumulative actual environment-action attempts, including anchor, sampled/failed policies and stochastic PG training; report model/readout queries and elapsed time separately",
            "action_landmarks": sorted(
                {0, action_cap // 4, action_cap // 2, 3 * action_cap // 4, action_cap}
            ),
            "nominee_rule": "At each landmark use only a policy whose complete deterministic candidate panel finished by that actual-action count; anchor exists only after its whole panel. No interpolation, partial-panel fitness or inference from unspent allowance",
            "checkpoint_provenance": "Keep per-job ledgers/checkpoints and generation/update/candidate parameter files; final report states actual used and residual unspent allowances, never calls unequal consumption equal",
        },
        "anchor_provenance": "Common strong fixed nominee is separate from trained-candidate provenance. Strictly tied selection retains anchor; report selected_is_trained=false and no evolutionary credit for selecting it.",
    }
    for host in sorted(hosts, key=lambda h: h["repeat_id"]):
        repeat = host["repeat_id"]
        train_r, select_r = (
            seeded(train, repeat, "training"),
            seeded(selection, repeat, "selection"),
        )
        design = evaluation(base, host["pod_id"], train_r, episode_actions)
        methods = ("cma", "random", "reinforce")
        order = methods[repeat:] + methods[:repeat]
        run_refs = {}
        for method in methods:
            protocol = copy.deepcopy(common)
            namespace = args.namespace + f"/r{repeat}/{method}"
            protocol.update(
                method=method,
                repeat_id=repeat,
                optimizer_seed=args.optimizer_seed + repeat * 101,
                allowed_pod_ids=[host["pod_id"]],
                training_panel=train_r,
                evaluation=design,
                output_namespace=namespace,
                max_candidate_panels=candidate_panels // 2
                if method == "reinforce"
                else candidate_panels,
            )
            path = out / f"training-{method}-r{repeat}.json"
            written.append(write(path, protocol))
            run_refs[method] = {
                "protocol": str(path),
                "output_namespace": namespace,
                "nominees": namespace + "/nominees.json",
                "entrypoint": entrypoint,
            }
        candidates = [("anchor", "fixed_geometry_progress", STRONG)] + [
            (f"grid{i:03}", "fixed_geometry_progress", parameters)
            for i, parameters in enumerate(fixed)
        ]
        tune_ns = args.namespace + f"/r{repeat}/fixed-tuning"
        tune = comparison(
            base,
            f"navigation-fixed-grid-r{repeat}",
            tune_ns,
            design,
            make_jobs(train_r, candidates, episode_actions, repeat),
            sources,
            action_cap,
        )
        tune.update(
            fixed_grid={
                "novel_candidates": candidate_panels,
                "common_anchor_panels": 1,
                "factorial": "geometry(.5,1,2,4) x repetition(.5,1,2,4) x visualprogress(1,2,4); remove separately measured anchor and replace with prospective doubled-latent-match setting",
                "all_features_nonzero": True,
                "explicit_smaller_budget_subset": "If overridden below48, first N candidates by SHA256(JSON genes), before any outcomes",
            },
            selection_rule="Choose highest complete four-case training tuple; exact ties prefer anchor, then ascending grid index. No partial or technical panel can nominate.",
        )
        written.append(write(out / f"fixed-tuning-r{repeat}.json", tune))
        controls = comparison(
            base,
            f"navigation-native-strong-controls-r{repeat}",
            args.namespace + f"/r{repeat}/training-controls",
            design,
            make_jobs(
                train_r,
                [("native", "native", STRONG), ("strong", "fixed_geometry_progress", STRONG)],
                episode_actions,
                repeat,
            ),
            sources,
            2 * len(train) * episode_actions,
        )
        written.append(write(out / f"training-controls-r{repeat}.json", controls))
        select_design = evaluation(base, host["pod_id"], select_r, episode_actions)
        selection_spec = {
            "schema": "lewm-navigation-selection-reservation-v1",
            "repeat_id": repeat,
            "host": host,
            "entrypoint": entrypoint,
            "evaluation": select_design,
            "case_preparation": base["case_preparation"],
            "selection_panel": select_r,
            "source_files": files,
            "source_sha256": sources,
            "candidate_sources": {
                "native": "Original visual-goal native policy",
                "strong_anchor": STRONG,
                "tuned_fixed": {"output": tune_ns, "rule": tune["selection_rule"]},
                **run_refs,
            },
            "maximum_episode_starts": 6 * len(select_r),
            "maximum_environment_actions": 6 * len(select_r) * episode_actions,
            "materialization_before_any_selection_episode": "Require all relevant training runs complete or explicitly budget-stopped, pin their protocol/nominees/state hashes and selected weight vectors. Include each optimizer best-trained nominee if present, else labeled anchor fallback. Freeze six-arm8-case runtime protocol before reading any selection outcome.",
            "selection": "Native, strong anchor, tuned-fixed training winner, CMA best-trained, random best-trained and REINFORCE best-trained. Within each search arm compare its candidate to the common strong-anchor selection tuple; retain candidate only if strictly better, otherwise labeled untrained anchor. Exact ties retain anchor. All six arms use the same case/environment seeds and deterministic argmax policies, including duplicate-parameter nominees, and every execution is charged.",
            "repeat_policy": "Keep all three learner-repeat selections. Never select the best repeat. Aggregate uncertainty by repeat and source case, with physical host retained as a block.",
            "final_lock": "After selection, persist each selected parameter vector, is_trained flag, producer run/update/candidate provenance and selection score/input/output hashes. No further tuning once final reservation opens.",
        }
        written.append(write(out / f"selection-reservation-r{repeat}.json", selection_spec))
        schedule.append(
            {
                **host,
                "methods_sequential": list(order),
                "runs": run_refs,
                "fixed_tuning": tune_ns,
                "training_controls": controls["output_namespace"],
                "selection_protocol_to_materialize": str(
                    out / f"selection-reservation-r{repeat}.json"
                ),
                "rule": "One active experiment worker per physical host. Run all methods and native/fixed controls for the repeat on its assigned host; do not compare a method on another host without local controls.",
            }
        )
    final_reservation = {
        "schema": "lewm-navigation-final-reservation-v1",
        "sealed_source_episode_ids": sorted(final_episodes),
        "final_pixels_accessed_by_builder": 0,
        "episode_actions": episode_actions,
        "geometry_transform": panel.get("geometry_transform"),
        "endpoint": "Safe original-task completion within the prospectively chosen base-pilot action horizon; no retroactive change to earlier150-action results",
        "case_quota_per_condition": 4,
        "cases": 16,
        "per_repeat_six_arm_start_cap": 96,
        "per_repeat_six_arm_action_cap": 96 * episode_actions,
        "opening_precondition": "All three repeat-specific selection locks, source/protocol/weights/provenance hashes and final preparation protocol are persisted before ANY final pixel/image encoding/controller episode.",
        "admission": "Use unchanged v3 public metadata/temporal/within-room/endpoint-clearance rules, numeric episode modulo4 condition assignment, ascending episodes, first eligible cases percondition. Freeze exactly four percondition (16cases), no outside-pool replacement or changed margins/readout/locality. If a quota is unavailable, report insufficiency; no outcome-based repair.",
        "seeds": {
            "rule": "Final environment/planner seed=713000+source_episode; identical within each physical-host comparison; categorical sampling disabled",
            "repeat_policy": "All selected learner repeats retained; each evaluated on its own training host",
        },
        "host_controls": "On EACH physical host evaluate native, strong fixed and each repeat's tuned-fixed selection beside that host's CMA/random/REINFORCE selections. Retain separate denominators; do not pool away host effects. Shared native/strong control rows may be reused only if exact parameters/case/seed/runtime hashes match and the reuse rule is frozen before final execution.",
        "frozen_final_budget_rule": "After immutable metadata-only final-case admission, freeze exact case IDs/input hashes, all six-arm jobs per repeat, host/seed ordering and action/start caps before loading final model inputs. At16cases, full repeat six-arm maximum=96starts and14400actualactions for150-action cap; charge preparation/reset/encoding work separately, with all failures/partial starts retained.",
        "claims": "Report safe completion, violations, unfinished and engagement with denominators, original progress, actual actions, model/readout queries and elapsed time. Architecture/readout gains are shared and not EA gains. No optimizer advantage claim absent all native, tuned-fixed, random-search and genuine ordinary-learning comparisons with the same information and common hard allowance; distinguish unspent allowance from equal actual consumption.",
    }
    written.append(write(out / "final-reservation.json", final_reservation))
    manifest = {
        "schema": "lewm-navigation-optimizer-comparison-freeze-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_pilot": {"path": str(base_path), "sha256": digest(base_path)},
        "builder_sha256": digest(Path(__file__)),
        "lever_gate": lever,
        "entrypoint": entrypoint,
        "training_case_ids": [c["case_id"] for c in train],
        "selection_case_ids": [c["case_id"] for c in selection],
        "source_sha256": sources,
        "schedule": schedule,
        "files": written,
        "no_workers_launched": True,
        "per_optimizer_repeat_hard_action_cap": action_cap,
        "three_method_three_repeat_action_cap": 9 * action_cap,
        "fixed_tuning_three_repeat_action_cap": 3 * action_cap,
        "all_preparation_and_engineering_costs_remain_separately_charged": True,
        "deadline": common["deadline_utc"],
        "final_pixels_read": 0,
    }
    manifest_ref = write(out / "manifest.json", manifest)
    return {
        "manifest": manifest_ref,
        "protocol_files": len(written),
        "training_runs": 9,
        "per_optimizer_action_allowance": action_cap,
        "episode_starts": episode_starts,
        "ordinary_learning_updates": candidate_panels // 2,
        "new_model_simulator_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pilot", type=Path, required=True)
    parser.add_argument("--lever-evidence", type=Path, required=True)
    parser.add_argument("--extra-source", action="append", default=[])
    parser.add_argument(
        "--entrypoint", default="experiments/scripts/run_lewm_navigation_campaign_v2.py"
    )
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--selection-panel", type=Path)
    parser.add_argument("--selection-panel-sha256")
    parser.add_argument("--repeat-hosts", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--episode-actions", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--optimizer-seed", type=int, default=919100)
    parser.add_argument("--reserve-seconds", type=int, default=1800)
    parser.add_argument("--allow-budget-override", action="store_true")
    args = parser.parse_args()
    print(json.dumps(freeze(args), indent=2))


if __name__ == "__main__":
    main()
