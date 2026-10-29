"""Paired case-block analysis and diagnostic summaries for the task-16 campaign."""

import collections
import json

import numpy as np


def read_rows(root, stages=None):
    rows = []
    for path in sorted(root.glob("*/result.json")):
        row = json.loads(path.read_text())
        if stages is None or row["job"]["stage"] in stages:
            rows.append(row)
    return rows


def paired_contrasts(rows, seed=160026, repetitions=10000):
    """Resample whole cases, preserving all paired seeds and margins in each case."""
    pairs = collections.defaultdict(dict)
    for row in rows:
        j = row["job"]
        if row["status"] != "complete" or j["condition"] not in ("fixed", "adaptive"):
            continue
        key = j["stage"], j["env"], j["case"]["id"], j["margin"], j["seed"]
        if j["condition"] in pairs[key]:
            raise ValueError(f"Duplicate paired arm: {key}")
        pairs[key][j["condition"]] = row
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    missing = []
    for (stage, env, case, margin, planner_seed), pair in pairs.items():
        if set(pair) != {"fixed", "adaptive"}:
            missing.append([stage, env, case, margin, planner_seed])
            continue
        for metric in ("requested_safe_completion", "any_violation", "margin_breach", "J"):
            difference = float(pair["adaptive"][metric]) - float(pair["fixed"][metric])
            for level in (str(margin), "all"):
                groups[(stage, env, level, metric)][case].append(difference)
    results = []
    rng = np.random.default_rng(seed)
    for (stage, env, margin, metric), cases in sorted(groups.items()):
        values = np.array([np.mean(v) for _, v in sorted(cases.items())])
        sampled = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(1)
        results.append(
            dict(
                stage=stage,
                env=env,
                margin=margin,
                metric=metric,
                paired_cases=len(values),
                paired_episodes=sum(map(len, cases.values())),
                adaptive_minus_fixed=float(values.mean()),
                interval_95=np.quantile(sampled, [0.025, 0.975]).tolist(),
                method="Percentile bootstrap over whole case blocks, equal case weights",
                repetitions=repetitions,
                seed=seed,
                caution="Three familiar layouts give limited uncertainty resolution"
                if stage == "A"
                else "Conditional on selected controllers; not optimizer replication",
            )
        )
    return {"contrasts": results, "unpaired": missing}


def cells(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        j = row["job"]
        if row["status"] == "complete":
            grouped[(j["stage"], j["env"], j["case"]["id"], j["margin"], j["condition"])].append(
                row
            )
    result = []
    for (stage, env, case, margin, condition), group in sorted(grouped.items()):
        item = dict(
            stage=stage, env=env, case=case, margin=margin, condition=condition, n=len(group)
        )
        for metric in (
            "success",
            "requested_safe_completion",
            "benchmark_safe_completion",
            "any_violation",
            "margin_breach",
            "unfinished",
            "censored",
        ):
            item[metric] = sum(int(r[metric]) for r in group)
        for metric in ("J", "path_length", "action_norm_mean", "seconds", "final_goal_distance"):
            item[f"mean_{metric}"] = float(np.mean([r[metric] for r in group]))
        item["engaged"] = sum(r["path_length"] > (1 if env == "tworoom" else 0.001) for r in group)
        item["resources"] = dict(
            sum((collections.Counter(r["counters"]) for r in group), collections.Counter())
        )
        result.append(item)
    return result


def diagnostics(root, stages=("Q",)):
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in read_rows(root, stages):
        if row["status"] != "complete":
            continue
        j = row["job"]
        group = groups[(j["env"], j["condition"], j["margin"])]
        attempt = root / j["id"]
        for plan in json.loads((attempt / "replans.json").read_text()):
            for key, value in plan["features"].items():
                if key != "replan" and plan["features"]["replan"] > 0:
                    group[key].append(value)
        path = attempt / "candidate-snapshots.json"
        if not path.exists():
            continue
        saved = np.load(attempt / "states.npz")
        states = saved["states"]
        for snapshot in json.loads(path.read_text()):
            feasible = np.asarray(snapshot["feasible"], dtype=bool)
            if not feasible.any():
                continue
            learned, geometric = (
                np.asarray(snapshot[k])[feasible] for k in ("learned", "geometric")
            )
            replan = snapshot["iteration"] // 10
            current = j["case"]["start"] if replan == 0 else states[5 * replan - 1]
            distance = (
                np.linalg.norm(np.asarray(current[:2]) - j["case"]["goal"])
                if j["env"] == "tworoom"
                else np.abs(np.asarray(current[:2]) - j["case"]["goal"]).max()
            )
            group["sampled_safe_candidates_progress_fraction"].append(
                float((geometric < distance).mean())
            )
            group["sampled_best_learned_true_distance"].append(float(geometric[learned.argmin()]))
            group["sampled_best_true_distance"].append(float(geometric.min()))
            if len(learned) > 2 and learned.std() > 1e-9 and geometric.std() > 1e-9:
                group["sampled_cost_distance_correlation"].append(
                    float(np.corrcoef(learned, geometric)[0, 1])
                )
    result = []
    for (env, condition, margin), group in sorted(groups.items()):
        result.append(
            dict(
                env=env,
                condition=condition,
                margin=margin,
                summaries={
                    key: {
                        "n_records": len(values),
                        "quantiles_0_25_50_75_100": np.quantile(
                            values, [0, 0.25, 0.5, 0.75, 1]
                        ).tolist(),
                    }
                    for key, values in group.items()
                },
            )
        )
    return {
        "sampling": "Deterministic uniform sample plus objective-selected top candidates; descriptive and correlated, not population estimates",
        "groups": result,
    }
