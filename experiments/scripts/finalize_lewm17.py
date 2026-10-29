"""Freeze all validation-selected controllers before accessing the sealed final cases."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.lewm17_protocol import CONFIG, ENVIRONMENTS, OUT, digest, freeze, job, write_json
from helpers.lewm17_selector import MARGINS, mode_params


def rank(rows):
    return (
        not any(r["margin_breach"] or r["any_violation"] for r in rows),
        sum(r["requested_safe_completion"] for r in rows),
        -sum(r["J"] for r in rows),
        -sum(r["counters"]["predictor_steps"] for r in rows),
    )


def main():
    config = json.loads((CONFIG / "comparison.json").read_text())
    selected_paths = sorted((OUT / "optimizers").glob("*/selected.json"))
    if len(selected_paths) != 30:
        raise RuntimeError("All 30 independent optimizer runs must finish before final access")
    selected = [json.loads(p.read_text()) for p in selected_paths]
    assert all(
        s["status"] == "complete" and s["episodes"] == config["episodes_per_run"] for s in selected
    )
    keys = {(s["env"], s["method"], s["replicate"]) for s in selected}
    assert keys == {(e, m, r) for e in ENVIRONMENTS for m in config["methods"] for r in range(5)}
    rows = [json.loads(p.read_text()) for p in (OUT / "attempts").glob("*/result.json")]
    assert all(r["status"] == "complete" for r in rows)
    fixed = [r for r in rows if r["job"]["stage"] == "F"]
    assert len(fixed) == 288
    tuned = {}
    for env in ENVIRONMENTS:
        tuned[env] = {}
        for margin in MARGINS[env]:
            by_mode = [
                [
                    r
                    for r in fixed
                    if r["job"]["env"] == env
                    and r["job"]["margin"] == margin
                    and r["job"]["condition"] == f"mode{i}"
                ]
                for i in range(6)
            ]
            assert all(len(x) == 12 for x in by_mode)
            eligible = [i for i in range(6) if rank(by_mode[i])[0]]
            if not eligible:
                raise RuntimeError("No safe tuned fixed mode")
            index = max(eligible, key=lambda i: rank(by_mode[i]))
            tuned[env][str(margin)] = {
                "mode": index,
                "params": mode_params(env, index),
                "ranks": [list(rank(x)) for x in by_mode],
            }
    historical_path = ROOT / "configs/lewm16/selected-controllers.json"
    historical = json.loads(historical_path.read_text())["methods"]
    selection_path = OUT / "final-selection.json"
    if selection_path.exists():
        raise FileExistsError("Final selection already frozen")
    write_json(
        selection_path,
        {
            "frozen_epoch": time.time(),
            "adaptive": selected,
            "tuned": tuned,
            "historical": historical,
            "historical_source_sha256": digest(historical_path),
            "optimizer_selection_sha256": {
                str(p.relative_to(ROOT)): digest(p) for p in selected_paths
            },
            "code_sha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in sorted(ROOT.glob("experiments/helpers/lewm17*.py"))
            },
            "finalizer_sha256": digest(Path(__file__)),
            "rule": "All selected parameters frozen before first final panel access or outcome",
        },
    )
    sealed = CONFIG / "sealed/final.json"
    assert digest(sealed) == config["final_panel_sha256"]
    write_json(
        OUT / "final-access.json",
        {
            "utc_epoch": time.time(),
            "path": str(sealed),
            "sha256": digest(sealed),
            "selection_sha256": digest(selection_path),
            "purpose": "One-time final job construction after complete parameter freeze",
        },
    )
    panels = json.loads(sealed.read_text())
    jobs = []
    for env in ENVIRONMENTS:
        for case in panels[env][: config["final_cases"]]:
            for margin in MARGINS[env]:
                for seed in (179710, 179711):
                    for s in [x for x in selected if x["env"] == env]:
                        condition = f"{s['method']}-r{s['replicate']}"
                        params = {"weights": s["weights"], "margin": margin, "stochastic": False}
                        trial = job(env, "H", case, margin, seed, condition, params)
                        trial.update(
                            optimizer=s["method"], replicate=s["replicate"], diagnostic=False
                        )
                        jobs.append(trial)
                    for condition, params in [
                        ("historical", historical[env]["fixed"][f"{margin:g}"]["params"]),
                        ("tuned", tuned[env][str(margin)]["params"]),
                    ]:
                        trial = job(env, "H", case, margin, seed, condition, params)
                        trial["diagnostic"] = False
                        jobs.append(trial)
    assert len(jobs) == 2 * config["final_cases"] * 2 * 2 * 17
    print(
        freeze(
            "final-evaluation",
            jobs,
            {
                "selection_sha256": digest(selection_path),
                "comparison_sha256": digest(CONFIG / "comparison.json"),
                "total_final_episodes": len(jobs),
                "all_parameters_frozen": True,
            },
        )
    )


if __name__ == "__main__":
    main()
