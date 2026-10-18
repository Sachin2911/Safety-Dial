"""Descriptive tables from an immutable Part 1 audit, including per-scene costs."""

import argparse
import json
from collections import Counter
from pathlib import Path


def tables(audit_dir):
    audit_dir = Path(audit_dir)
    ledger = json.loads((audit_dir / "attempt-ledger.json").read_text())
    rows = ledger["attempts"]
    categories = ("safe_complete", "unsafe_complete", "safe_unfinished", "unsafe_unfinished")
    validation = [r for r in rows if r.get("purpose") == "validation" and r["valid"]]
    scenes = sorted({r["scene"] for r in validation})
    output = []
    for scene in [*scenes, "all"]:
        for dial in (0, 0.5, 1):
            selected = [
                r
                for r in validation
                if r["dial"] == dial and (scene == "all" or r["scene"] == scene)
            ]
            counts = Counter(categories[(0 if r["C_req"] else 2) + int(r["V"])] for r in selected)
            actions = sum(r["costs"]["policy_control_steps"] for r in selected)
            changed = sum(r["changed_commands"] for r in selected)
            output.append(
                {
                    "scene": scene,
                    "dial": dial,
                    "n": len(selected),
                    **{k: counts[k] for k in categories},
                    "safe_requested_completion": sum(r["S_req"] for r in selected),
                    "requested_completion": sum(r["C_req"] for r in selected),
                    "benchmark_success": sum(r["benchmark_success"] for r in selected),
                    "violations": sum(r["V"] for r in selected),
                    "ever_contacted": sum(r["ever_contacted"] for r in selected),
                    "ever_grasped": sum(r["ever_grasped"] for r in selected),
                    "policy_actions": actions,
                    "changed_commands": changed,
                    "changed_fraction": changed / actions if actions else None,
                    "grasp_heuristic_actions": sum(r["grasp_heuristic_actions"] for r in selected),
                    "anchor_cap_active_actions": sum(r["cap_active_actions"] for r in selected),
                    "queries": sum(r["costs"]["policy_queries"] for r in selected),
                    "query_seconds": sum(r["costs"]["policy_query_seconds"] for r in selected),
                    "process_seconds": sum(r["process"]["elapsed_seconds"] for r in selected),
                }
            )
    by_purpose = []
    for purpose in ("engineering", "train", "validation"):
        selected = [r for r in rows if r.get("purpose") == purpose]
        by_purpose.append(
            {
                "purpose": purpose,
                "attempts": len(selected),
                "valid_completed": sum(r["valid"] for r in selected),
                "costs": {
                    k: sum(r["costs"].get(k) or 0 for r in selected)
                    for k in ledger["costs"]["recorded_sums"]
                },
                "unknown_costs": {
                    k: sum(r["costs"].get(k) is None for r in selected)
                    for k in ledger["costs"]["recorded_sums"]
                },
            }
        )
    totals = [r for r in output if r["scene"] == "all"]
    primary = None
    if all(r["n"] for r in totals):
        zero, _, full = totals
        primary = {
            "full_minus_zero_safe_completion_count": full["safe_requested_completion"]
            - zero["safe_requested_completion"],
            "full_minus_zero_safe_completion_rate": full["safe_requested_completion"] / full["n"]
            - zero["safe_requested_completion"] / zero["n"],
            "full_minus_zero_violating_episode_count": full["violations"] - zero["violations"],
        }
    result = {
        "descriptive_only": True,
        "per_scene_and_setting": output,
        "attempts_and_costs_by_purpose": by_purpose,
        "primary_contrast": primary,
    }
    (audit_dir / "tables.json").write_text(json.dumps(result, indent=2) + "\n")
    if validation:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, len(scenes) + 1, figsize=(12, 4), layout="constrained")
        for axis, scene in zip(axes, [*scenes, "all"], strict=True):
            selected = [r for r in output if r["scene"] == scene]
            base = [0, 0, 0]
            for key, color in zip(
                categories, ("#228833", "#cc6677", "#88ccee", "#aa4499"), strict=True
            ):
                heights = [r[key] for r in selected]
                axis.bar(range(3), heights, bottom=base, color=color, label=key.replace("_", " "))
                base = [a + b for a, b in zip(base, heights, strict=True)]
            labels = [
                f"{r['dial']}\nQ={r['queries']}\nt={r['process_seconds']:.0f}s" for r in selected
            ]
            axis.set(
                xticks=range(3),
                xticklabels=labels,
                title=scene,
                ylabel="Audited episode counts",
                xlabel="Dial; queries; process time",
            )
            for i, row in enumerate(selected):
                axis.text(i, row["n"] + 0.05, f"n={row['n']}", ha="center")
            axis.set_ylim(0, max(1, max(r["n"] for r in selected)) + 1)
        handles, labels = axes[-1].get_legend_handles_labels()
        figure.legend(handles, labels, fontsize=8, loc="outside lower center", ncol=4)
        figure.savefig(audit_dir / "per-scene-outcomes-and-costs.png", dpi=160)
        plt.close(figure)
    print(
        json.dumps(
            {"primary_contrast": primary, "totals": totals, "by_purpose": by_purpose}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    args = parser.parse_args()
    tables(args.audit_dir)
