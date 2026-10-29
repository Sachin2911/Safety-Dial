"""Prospective task-17 panel and engineering manifest freeze, no rollout launch."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import json

from helpers.lewm17_protocol import CONFIG, ENVIRONMENTS, freeze, initialize_panels, job
from helpers.lewm17_selector import MARGINS, anchors, mode_params


def main():
    initialize_panels()
    panels = json.loads((CONFIG / "development.json").read_text())
    jobs = []
    for env in ENVIRONMENTS:
        for index in range(2):
            case, margin, seed = panels[env][index], MARGINS[env][index], 170010 + index
            for condition in ("native", "canonical", "fixed", "selector"):
                params = mode_params(env, index) if condition == "fixed" else {}
                if condition == "selector":
                    params = {
                        "weights": anchors(env)[index].tolist(),
                        "margin": margin,
                        "stochastic": False,
                        "mode_seed": 17010010 + index,
                    }
                j = job(env, "E", case, margin, seed, condition, params)
                j.update(native_parity=condition == "native", save_frames=True)
                jobs.append(j)
    print(
        freeze(
            "engineering-v1",
            jobs,
            {
                "purposes": [
                    "native identity",
                    "constant mode parity",
                    "dense checker replay",
                    "rendering",
                    "resource counters",
                    "throughput",
                ],
                "total_episodes": 16,
                "max_development_episodes": 48,
                "gate": "native max abs <=1e-5; fixed/selector states exact; replay <=1e-7 Reacher, <=1e-3 TwoRoom",
                "outcome_selection": "No tuning or outcome-based panel acceptance in engineering",
            },
        )
    )


if __name__ == "__main__":
    main()
