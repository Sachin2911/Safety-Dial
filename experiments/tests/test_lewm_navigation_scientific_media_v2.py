"""Rented-only scientific media smoke check with synthetic arrays and no final data."""

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/render_lewm_navigation_scientific_media_v2.py"
spec = importlib.util.spec_from_file_location("scientific_media_synthetic", SCRIPT)
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUNPOD_POD_ID") not in ("vukovc3t4o1dui", "wq4ar2qx2ff6r4", "nj2fqkp2prgk8x"),
    reason="Synthetic renderer checks belong on the declared rented hosts only",
)


def test_showcase_choice_and_two_endpoint_png_svg_renderer_preserve_dummy_inputs(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    protocols = []
    for family in media.FAMILIES:
        cases = []
        for episode in reversed(range(5000, 5016)):
            condition = media.CONDITIONS[episode % 4]
            cases.append(
                {
                    "case_id": f"DUMMY-{episode}",
                    "source_episode": episode,
                    "source_step": 0,
                    "condition": condition,
                    "arbitrary_outcome_not_used": episode > 5003,
                }
            )
        protocols.append(
            {
                "family": family,
                "repeat_id": 0,
                "protocol": {"jobs": [{"case": case} for case in cases]},
            }
        )
    chosen = media.choose_cases(protocols)
    assert all(
        list(chosen[family].values()) == [f"DUMMY-{episode}" for episode in range(5000, 5004)]
        for family in media.FAMILIES
    )
    panels, before = {}, {}
    for family in media.FAMILIES:
        for condition in media.CONDITIONS:
            for arm in media.ARMS:
                states = np.array([[35.0, 35.0], [45.0, 40.0], [55.0, 45.0]])
                proposed = np.array([[0.4, 0.2], [0.3, 0.1]])
                goal = np.array([150.0, 160.0])
                for array in (states, proposed, goal):
                    array.setflags(write=False)
                key = (family, condition, arm)
                panels[key] = {
                    "states": states,
                    "proposed": proposed,
                    "original_goal": goal,
                    "shape": {"kind": "none"}
                    if condition == "none"
                    else {"kind": "island", "centre": [95.0, 100.0], "radius": int(condition[6:])},
                    "selector": []
                    if arm == "native"
                    else [
                        {
                            "target_kind": "observed_successor",
                            "selected_target_xy_estimate": [60.0, 55.0],
                        },
                        {
                            "target_kind": "original_goal",
                            "selected_target_xy_estimate": [148.0, 158.0],
                        },
                    ],
                    "source_episode": 5000 + media.CONDITIONS.index(condition),
                    "metrics": {
                        "safe_completion": False,
                        "violation": False,
                        "unfinished": True,
                        "environment_actions": 2,
                    },
                    "model_work": {"actual_cost_call_attempts": 8},
                }
                before[key] = tuple(array.tobytes() for array in (states, proposed, goal))
    plotting = {
        "body_radius": 7.0,
        "margin": 3.0,
        "empirical_planning_allowance": 12.405431747436523,
        "original_goal_radius": 16.0,
        "proposed_command_glyph_scale": 8.0,
    }
    for family in media.FAMILIES:
        evidence = media.render_profile(plt, np, panels, family, plotting, tmp_path)
        assert len(evidence) == 24
        assert all(
            row["body_disk_radius"] == 7
            and row["margin"] == 3
            and row["goal_radius"] == 16
            and row["empirical_allowance"] == 12.405431747436523
            and row["overlay_is_analysis_only"]
            for row in evidence
        )
    for key, panel in panels.items():
        assert (
            tuple(panel[name].tobytes() for name in ("states", "proposed", "original_goal"))
            == before[key]
        )
    expected = {
        family + "-showcase-trajectories." + suffix
        for family in media.FAMILIES
        for suffix in ("png", "svg")
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    for family in media.FAMILIES:
        assert (tmp_path / f"{family}-showcase-trajectories.png").read_bytes()[
            :8
        ] == b"\x89PNG\r\n\x1a\n"
        svg = (tmp_path / f"{family}-showcase-trajectories.svg").read_text()
        assert "Analysis-only overlays" in svg and "not a state prediction" in svg
        assert "Actual virtual hazard" in svg and "Predicted-centre buffer" in svg
