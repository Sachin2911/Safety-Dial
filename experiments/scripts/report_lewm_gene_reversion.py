"""Frozen exploratory JSON-only gene-reversion report, rented Nav after completion.

Never loads arrays/models/images or changes controllers, selection or final data.
All twelve arms and ten paired contrasts remain in their prospectively fixed order.
"""

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NAV = "vukovc3t4o1dui"
PROTOCOL_SHA = "4112648831854f0007bfac6fe22df79a352136230f6d4f11045ee6c51492fce9"
ARMS = ("cma_full", "strong_anchor", *(f"revert_gene_{gene:02}" for gene in range(10)))
METRICS = ("safe_completion", "violation", "unfinished", "engaged")
QUERIES = (
    "actual_encoder_call_attempts",
    "actual_cost_call_attempts",
    "actual_safety_call_attempts",
    "goal_readout_call_attempts",
    "actual_candidate_block_steps_attempted",
)
SCOPE = (
    "Exploratory development TRAIN mechanism study of one fixed R0 nominee and eight fixed "
    "cases. All 12 arms and 10 gene contrasts are retained. No CIs, p-values, gene/case selection "
    "or optimizer-superiority inference. This study cannot change selected controllers, final "
    "cases or the frozen primary/secondary analyses. Its extra work is outside optimizer budgets "
    "and belongs in the original owning-host final collector."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def resolve(root, value):
    path = Path(value)
    if path.is_absolute():
        require("Safety-Dial" in path.parts, "Unexpected absolute evidence path")
        path = Path(*path.parts[path.parts.index("Safety-Dial") + 1 :])
    require(".." not in path.parts, "Evidence traversal")
    result = (root / path).resolve()
    require(
        result.is_relative_to(root.resolve()) and result.suffix == ".json",
        "Read only JSON evidence inside the owning project",
    )
    return result


def reference(root, item, hashes):
    path = resolve(root, item["path"])
    require(sha(path) == item["sha256"], "Frozen lineage reference differs: " + str(path))
    hashes[str(path.relative_to(root))] = item["sha256"]
    return load(path)


def write(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def validate_protocol(root, protocol, hashes):
    meta = protocol["exploratory_gene_reversion"]
    producer = reference(root, meta["producer_protocol"], hashes)
    state = reference(root, meta["producer_state"], hashes)
    source = reference(root, meta["producer_source_freeze"], hashes)
    nominees = reference(root, meta["producer_nominees"], hashes)
    require(
        state["phase"] == "complete"
        and state["active"] is None
        and state["optimizer_updates"] == 6
        and state["candidate_panels_completed"] == 48
        and not state["unfinalized_job_ids"]
        and state["action_accounting_exact"],
        "Original full CMA producer is incomplete",
    )
    require(
        source["protocol"] == producer
        and source["protocol_sha256"]
        == canonical(producer)
        == meta["producer_protocol_canonical_json_sha256"]
        and source["identity"] == state["identity"] == meta["producer_identity"],
        "Producer source/protocol identity differs",
    )
    full, anchor = meta["producer_nominee"], meta["common_anchor"]
    require(
        nominees["trained"] == state["best_trained"] == full
        and nominees["anchor"] == state["anchor"] == anchor
        and full["provenance"]["completed_updates_before_proposal"] == 5,
        "Full/anchor producer lineage differs",
    )
    arms = meta["arms"]
    require(
        tuple(arm["arm"] for arm in arms) == ARMS
        and arms[0]["parameters"] == full["parameters"]
        and arms[1]["parameters"] == anchor["parameters"],
        "Frozen arm order differs",
    )
    for gene, arm in enumerate(arms[2:]):
        expected = list(full["parameters"])
        expected[gene] = anchor["parameters"][gene]
        require(
            arm["reverted_gene"] == gene and arm["parameters"] == expected,
            "A reversion changes more than its declared coordinate",
        )
    cases = producer["training_panel"]
    require(
        len(cases) == len(meta["producer_panel"]) == 8
        and len({case["case_id"] for case in cases}) == 8
        and collections.Counter(case["condition"] for case in cases)
        == {"none": 2, "circle3": 2, "circle5": 2, "circle7": 2},
        "Keep every original balanced eight-case training panel",
    )
    for index, panel in enumerate(meta["producer_panel"]):
        started = reference(root, panel["started"], hashes)
        result = reference(root, panel["result"], hashes)
        inner = reference(root, panel["episode_result"], hashes)
        spec = started["job"]
        expected_id = f"cma_g005_c001_p{index:03}"
        policy_seed = int.from_bytes(
            hashlib.sha256(f"{producer['optimizer_seed']}/{expected_id}".encode()).digest()[:4],
            "big",
        ) % (2**31 - 1)
        require(
            panel["job_id"] == expected_id == full["job_ids"][index]
            and result["job_id"] == expected_id in state["completed_job_ids"]
            and spec["case"] == cases[index]
            and spec["parameters"] == full["parameters"]
            and spec["environment_seed"] == inner["environment_seed"] == cases[index]["seed"]
            and spec["policy_seed"] == inner["policy_seed"] == policy_seed
            and result["identity"] == started["identity"] == state["identity"]
            and result["input_sha256"] == started["input_sha256"] == canonical(spec)
            and spec["job_id"] == expected_id
            and spec["case_id"] == cases[index]["case_id"]
            and spec["method"] == "cma"
            and spec["mode"] == "deterministic"
            and spec["generation"] == 5
            and spec["candidate_index"] == 1
            and spec["repeat_id"] == 0
            and spec["max_actions"] == 30
            and spec["stage"] == "cma_g005_c001"
            and result["extra"]["episode_result_sha256"] == panel["episode_result"]["sha256"]
            and result["accounting_finalized"]
            and not result["error"],
            "The declared full-CMA producer panel lineage differs",
        )
    expected_jobs = []
    for index, old_case in enumerate(cases):
        case = {**old_case, "seed": 623000 + old_case["source_episode"]}
        for arm in arms[index:] + arms[:index]:
            expected_jobs.append(
                {
                    "job_id": f"{arm['arm']}_{case['case_id']}_s{case['seed']}",
                    "arm": arm["arm"],
                    "method": "fixed_geometry_progress",
                    "parameters": arm["parameters"],
                    "case": case,
                    "environment_seed": case["seed"],
                    "policy_seed": case["seed"],
                    "repeat_id": 0,
                    "max_actions": 30,
                    "ablation_reverted_gene": arm["reverted_gene"],
                }
            )
    require(
        protocol["jobs"] == expected_jobs
        and len(expected_jobs) == 96
        and protocol["maximum_episode_starts"] == protocol["maximum_native_resets"] == 96
        and protocol["maximum_environment_actions"] == 2880
        and protocol["evaluation"] == producer["evaluation"]
        and protocol["source_sha256"] == producer["source_sha256"]
        and meta["new_work_is_separate_from_optimizer_comparison"] is True,
        "Study cases/seeds/parameters/order/source/budgets differ from its exact freeze",
    )
    return arms, [case["case_id"] for case in cases]


def retained_rows(root, protocol, hashes):
    folder = root / protocol["output_namespace"]
    state_path, source_path = folder / "state.json", folder / "source-freeze.json"
    state, source = load(state_path), load(source_path)
    require(
        state["phase"] == "complete"
        and state["active"] is None
        and set(state["jobs"]) == {job["job_id"] for job in protocol["jobs"]},
        "Require every one of96 completed study jobs",
    )
    require(
        source["pod_id"] == NAV
        and source["protocol_sha256"] == PROTOCOL_SHA
        and source["sources"] == protocol["source_sha256"],
        "Study source freeze differs",
    )
    hashes[str(state_path.relative_to(root))] = sha(state_path)
    hashes[str(source_path.relative_to(root))] = sha(source_path)
    rows, totals = [], collections.Counter()
    for planned in protocol["jobs"]:
        prefix = folder / "jobs" / planned["job_id"]
        started_path, outer_path = prefix / "started.json", prefix / "result.json"
        row = load(outer_path)
        require(
            load(started_path) == planned and row == state["jobs"][planned["job_id"]],
            "Retained planned/state/result row differs",
        )
        for key in ("job_id", "method", "parameters", "environment_seed", "policy_seed"):
            require(row[key] == planned[key], "Study execution differs: " + key)
        require(
            row["case_id"] == planned["case"]["case_id"]
            and not row["error"]
            and not row["extra"]["frame_flags"],
            "A technical/input failure blocks this report; no case replacement",
        )
        inner_path = resolve(root, row["extra"]["episode_result"])
        require(
            inner_path == prefix / "episode/result.json"
            and sha(inner_path) == row["extra"]["episode_result_sha256"],
            "Inner outcome lineage differs",
        )
        inner = load(inner_path)
        for key in ("job_id", "method", "case_id", "environment_seed", "policy_seed"):
            require(inner[key] == row[key], "Inner execution metadata differs: " + key)
        require(
            inner["mode"] == "deterministic"
            and inner["maximum_actions"] == 30
            and inner["case_sha256"] == planned["case"]["input_sha256"]
            and inner["resources"] == row["resources"]
            and not inner["error"],
            "Inner source/counter contract differs",
        )
        for metric in METRICS:
            require(
                isinstance(row[metric], bool) and row[metric] == inner[metric],
                "Binary study metric differs",
            )
        require(
            not (row["safe_completion"] and (row["violation"] or row["unfinished"]))
            and isinstance(row["environment_actions"], int)
            and 0 <= row["environment_actions"] <= 30
            and row["environment_actions"]
            == inner["recorded_successful_actions"]
            == row["resources"]["environment_actions"]
            and 0 <= row["progress"] <= 1
            and row["progress"] == inner["progress"],
            "Inconsistent study metric/action/progress ledger",
        )
        require(
            all(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                for value in row["resources"].values()
            ),
            "Invalid resource counter",
        )
        totals.update(row["resources"])
        for path in (started_path, outer_path, inner_path):
            hashes[str(path.relative_to(root))] = sha(path)
        rows.append(
            {
                "arm": planned["arm"],
                "case_id": planned["case"]["case_id"],
                "condition": planned["case"]["condition"],
                "source_episode": planned["case"]["source_episode"],
                **{key: int(row[key]) for key in METRICS},
                "progress": row["progress"],
                "actual_actions": row["environment_actions"],
                **{key: row["resources"].get(key, 0) for key in QUERIES},
                "episode_seconds": inner["episode_seconds"],
                "planning_seconds": inner["planning_seconds"],
                "resources": row["resources"],
            }
        )
    require(
        totals == collections.Counter(state["resource_totals"])
        and totals["environment_actions"] <= 2880,
        "Study aggregate ledger does not reconstruct",
    )
    return rows, dict(totals)


def csv_file(path, rows, fields):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def report(args):
    require(
        os.environ.get("RUNPOD_POD_ID") == NAV,
        "All report arithmetic/plots belong only on rented navigation",
    )
    require(args.study_complete_confirmed_by_root, "Root must confirm study completion first")
    require(not args.out.exists(), "Use a fresh exploratory report namespace")
    plan = load(args.plan)
    require(
        plan["schema"] == "lewm-gene-reversion-report-plan-v1"
        and plan["allowed_pod_ids"] == [NAV]
        and plan["script_sha256"] == sha(Path(__file__))
        and plan["protocol"]["sha256"] == PROTOCOL_SHA
        and plan["arm_order"] == list(ARMS),
        "Exploratory report source/plan differs",
    )
    root = args.artifact_root.resolve()
    hashes = {}
    protocol = reference(root, plan["protocol"], hashes)
    require(protocol["evaluation"]["execution_pod_id"] == NAV, "Study belongs to another pod")
    for name, expected in protocol["source_sha256"].items():
        require(sha(ROOT / name) == expected, "Frozen scientific source changed")
    arms, cases = validate_protocol(root, protocol, hashes)
    require(
        cases == plan["case_order"]
        and plan["denominator_each_arm"] == 8
        and plan["paired_reversion_arms"] == list(ARMS[2:]),
        "Prospective all-arm/all-case exploratory report contract differs",
    )
    rows, resource_totals = retained_rows(root, protocol, hashes)
    ordered = {(row["arm"], row["case_id"]): row for row in rows}
    require(
        len(ordered) == 96 and set(ordered) == {(a, c) for a in ARMS for c in cases},
        "Complete unfiltered12x8 outcomes required",
    )
    summaries = []
    for arm in ARMS:
        selected = [ordered[(arm, case)] for case in cases]
        totals = collections.Counter()
        for row in selected:
            totals.update(row["resources"])
        summaries.append(
            {
                "arm": arm,
                "denominator": 8,
                **{metric: sum(row[metric] for row in selected) for metric in METRICS},
                "mean_progress": sum(row["progress"] for row in selected) / 8,
                "actual_actions": sum(row["actual_actions"] for row in selected),
                **{key: totals[key] for key in QUERIES},
                "episode_seconds": sum(row["episode_seconds"] for row in selected),
                "planning_seconds": sum(row["planning_seconds"] for row in selected),
                "resource_counter_totals_by_field": dict(totals),
            }
        )
    paired, per_case = [], []
    diff_fields = (*METRICS, "progress", "actual_actions", *QUERIES)
    for gene, arm in enumerate(ARMS[2:]):
        differences = []
        for case in cases:
            first, full = ordered[(arm, case)], ordered[("cma_full", case)]
            row = {
                "gene": gene,
                "arm": arm,
                "case_id": case,
                **{key: first[key] - full[key] for key in diff_fields},
            }
            differences.append(row)
            per_case.append(row)
        paired.append(
            {
                "gene": gene,
                "arm": arm,
                "gene_feature": arms[gene + 2]["gene_feature"],
                "denominator": 8,
                **{
                    key + "_mean_difference": sum(row[key] for row in differences) / 8
                    for key in diff_fields
                },
                **{
                    key + "_count_difference": sum(row[key] for row in differences)
                    for key in METRICS
                },
            }
        )
    args.out.mkdir(parents=True)
    csv_file(
        args.out / "arm-summary.csv",
        summaries,
        [key for key in summaries[0] if key != "resource_counter_totals_by_field"],
    )
    csv_file(
        args.out / "case-outcomes.csv",
        [ordered[(a, c)] for a in ARMS for c in cases],
        [key for key in rows[0] if key != "resources"],
    )
    csv_file(args.out / "paired-gene-differences.csv", paired, list(paired[0]))
    csv_file(args.out / "paired-case-differences.csv", per_case, list(per_case[0]))
    csv_file(
        args.out / "safe-completion-heatmap.csv",
        [{"arm": a, **{c: ordered[(a, c)]["safe_completion"] for c in cases}} for a in ARMS],
        ["arm", *cases],
    )
    tex = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Arm & SC & V & Unfinished & Engaged & Progress & Actions \\",
        r"\midrule",
    ]
    for row in summaries:
        label = row["arm"].replace("_", r"\_")
        tex.append(
            f"{label} & "
            + " & ".join(str(row[m]) + "/8" for m in METRICS)
            + f" & {row['mean_progress']:.3f} & {row['actual_actions']} "
            + r"\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", "% " + SCOPE]
    (args.out / "arm-summary.tex").write_text("\n".join(tex) + "\n")
    tex = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Reversion & $\Delta$SC/8 & $\Delta$Progress & $\Delta$Actions/episode \\",
        r"\midrule",
    ]
    for row in paired:
        tex.append(
            f"Gene {row['gene']:02} & {row['safe_completion_count_difference']:+d}/8"
            + f" & {row['progress_mean_difference']:+.3f}"
            + f" & {row['actual_actions_mean_difference']:+.2f} "
            + r"\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", "% " + SCOPE]
    (args.out / "paired-gene-differences.tex").write_text("\n".join(tex) + "\n")
    # No numerical/plot library imports precede actual Nav and completion/lineage guards.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    heat = [[ordered[(a, c)]["safe_completion"] for c in cases] for a in ARMS]
    labels = ["Full CMA", "Strong anchor", *(f"Revert gene {i:02}" for i in range(10))]
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.imshow(heat, vmin=0, vmax=1, cmap=ListedColormap(["#e5e7eb", "#2563eb"]), aspect="auto")
    axis.set_xticks(range(8), [c.replace("tworoom-", "").replace("-", "\n") for c in cases])
    axis.set_yticks(range(12), labels)
    for i in range(12):
        for j in range(8):
            axis.text(
                j,
                i,
                str(heat[i][j]),
                ha="center",
                va="center",
                color="white" if heat[i][j] else "black",
            )
    axis.set_title("Exploratory TRAIN: safe completion for all 12 arms and 8 fixed cases")
    axis.set_xlabel("Original frozen case order; 1 = safe completion, 0 = otherwise")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(args.out / ("safe-completion-heatmap." + suffix), dpi=240)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.bar(
        range(10), [100 * row["safe_completion_mean_difference"] for row in paired], color="#475569"
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(10), [f"{gene:02}" for gene in range(10)])
    axis.set_xlabel("Reverted gene, fixed order")
    axis.set_ylabel("Reversion minus full CMA: SC percentage points")
    axis.set_title("Exploratory paired differences on the same 8 TRAIN cases; no uncertainty claim")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(args.out / ("paired-gene-safe-completion." + suffix), dpi=240)
    plt.close(figure)
    report = {
        "schema": "lewm-gene-reversion-exploratory-report-v1",
        "scope": SCOPE,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pod_id": NAV,
        "plan_sha256": sha(args.plan),
        "report_script_sha256": sha(Path(__file__)),
        "study_protocol_sha256": PROTOCOL_SHA,
        "arm_order": list(ARMS),
        "frozen_arms_and_parameters": arms,
        "case_order": cases,
        "job_count": 96,
        "denominator_each_arm": 8,
        "all_cases_and_arms_retained": True,
        "arm_summaries": summaries,
        "all_ten_paired_reversion_minus_full_differences": paired,
        "resource_counter_totals_by_field": resource_totals,
        "hard_study_action_allowance": 2880,
        "actual_study_actions": resource_totals["environment_actions"],
        "unspent_study_allowance": 2880 - resource_totals["environment_actions"],
        "query_fields_are_separate_not_added_across_aliases": True,
        "source_evidence_sha256": hashes,
        "extra_work_outside_optimizer_budgets": True,
        "all_extra_physical_work_required_in_own_host_final_collector": True,
        "no_CIs_pvalues_or_superiority_inference": True,
        "new_model_simulator_calls_by_report": 0,
    }
    write(args.out / "report.json", report)
    (args.out / "captions.txt").write_text(SCOPE + "\n")
    write(
        args.out / "artifact-manifest.json",
        {
            "files": {
                path.name: sha(path) for path in sorted(args.out.iterdir()) if path.is_file()
            },
            "new_model_simulator_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "report": str(args.out / "report.json"),
                "sha256": sha(args.out / "report.json"),
                "arms": 12,
                "jobs": 96,
                "exploratory_only": True,
                "new_model_simulator_calls": 0,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--study-complete-confirmed-by-root", action="store_true")
    report(parser.parse_args())


if __name__ == "__main__":
    main()
