import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/pusht_e2_figures.py"
spec = importlib.util.spec_from_file_location("e2_figures_test", path)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def metric(**changes):
    return {"n": 10, "n_accepted": 5, "n_false_safe": 0,
            "n_accepted_censored": 1, "acceptance_rate": 0.5,
            "fsa": None, "fsa_lower": 0.0, "fsa_upper": 0.2, **changes}


def repair():
    grid = [{"cfg": {"steps": 1000 if i % 2 == 0 else 3000, "index": i},
             "dev": {"at_matched": {"fsa": float("nan")}},
             "retention": {"latent_mse_tf": 0.8}} for i in range(16)]
    return {"recipe_grid": grid, "chosen_recipe": grid[0]["cfg"],
            "no_update_retention": {"latent_mse_tf": 1.0},
            "arms": {"fixed_margin": {"margin": 80},
                     "adapted": {"dev": {"at_matched": {"fsa": None}}}}}


def test_all_undefined_grid_is_zero_eligible_first_fallback_not_winner():
    value = figures.selection_context(repair())
    assert value["eligible_count"] == 0 and value["recipe_count"] == 16
    assert value["first_recipe_fallback"]
    assert value["fixed_margin_undefined_target_fallback"]


def test_saved_fallback_identity_must_be_preserved():
    value = repair()
    value["chosen_recipe"] = value["recipe_grid"][1]["cfg"]
    with pytest.raises(ValueError, match="first grid"):
        figures.selection_context(value)


def test_undefined_target_does_not_invent_a_fixed_margin_match():
    value = repair()
    value["arms"]["fixed_margin"]["margin"] = 10
    with pytest.raises(ValueError, match="margin80"):
        figures.selection_context(value)


def test_censored_zero_lower_bound_is_not_formatted_as_zero_point():
    cells = figures.metric_cells(metric())
    assert cells[2] == "undefined"
    assert cells[3] == "[0, 0.2]"


def test_zero_acceptance_retains_undefined_point_and_bounds():
    cells = figures.metric_cells(metric(n_accepted=0, n_accepted_censored=0,
                                        acceptance_rate=0.0, fsa_lower=None, fsa_upper=None))
    assert cells[2:] == ["undefined", "[undefined, undefined]", "0"]


@pytest.mark.parametrize("changes", [{"fsa": 0.0},
    {"n_accepted": 0, "n_accepted_censored": 0, "fsa": 0.0},
    {"n_accepted": 0, "n_accepted_censored": 0, "fsa_lower": 0.0}])
def test_invalid_defined_values_fail_before_rendering(changes):
    with pytest.raises(ValueError):
        figures.metric_cells(metric(**changes))


def test_box_hazard_beyond_old_plot_window_is_fully_included():
    states = np.array([[300.0, 300.0, 300.0, 380.0, 0.0]])
    hazard = {"kind": "box", "x0": 245.6, "x1": 340.8, "y0": 496.3, "y1": 561.2}
    x, y = figures.plot_limits(states, hazard)
    assert y[0] > 561.2 and y[1] < 0
    assert x[0] < 0 and x[1] > 512


def test_rotated_footprints_and_optional_goal_cannot_be_clipped():
    states = np.array([[540.0, -30.0, 540.0, 490.0, 0.7],
                       [550.0, -50.0, 510.0, 530.0, 1.1]])
    goal = [800.0, 750.0, 0.5]
    hazard = {"kind": "disc", "cx": -30, "cy": 250, "r": 40}
    x, y = figures.plot_limits(states, hazard, goal_pose=goal)
    for pose in [*states[:, 2:5], goal]:
        for polygon in figures.t_polygons(pose):
            assert np.all((polygon[:, 0] > x[0]) & (polygon[:, 0] < x[1]))
            assert np.all((polygon[:, 1] < y[0]) & (polygon[:, 1] > y[1]))
    assert x[0] < -70 and y[1] < -50


@pytest.mark.parametrize("states,hazard", [
    ([], {"kind": "disc", "cx": 1, "cy": 1, "r": 2}),
    ([[0, 0, 0, 0, float("nan")]], {"kind": "disc", "cx": 1, "cy": 1, "r": 2}),
    ([[0, 0, 0, 0, 0]], {"kind": "disc", "cx": 1, "cy": 1, "r": -2}),
    ([[0, 0, 0, 0, 0]], {"kind": "box", "x0": 5, "x1": 2, "y0": 1, "y1": 3}),
])
def test_bad_geometry_fails_closed(states, hazard):
    with pytest.raises(ValueError):
        figures.plot_limits(states, hazard)


def test_each_bank_gets_a_complete_four_arm_markdown_table():
    bank = {"arms": {a: {"margin": 1, "at_matched": metric()} for a in figures.ARMS},
            "paired_adapted_minus_no_update": {"point": None, "lo": -0.02,
                                              "hi": -0.01, "n_boot": 3},
            "examples": {}}
    summary = {"mean_final_coverage": 0.5, "mean_final_pose_error_px": 20,
               "arena_exits": 0, "n_valid": 20}
    paired = {"run_id": "fixture", "gate": {"passes": False, "retention_complete": False},
              "banks": {name: copy.deepcopy(bank) for name in ["dev", "test", "stress"]},
              "retention": {"goal_comparison": {"passes": False, "complete": False},
                            "no_update_clips": {"latent_mse_tf": 1, "latent_mse_rollout": 2},
                            "adapted_clips": {"latent_mse_tf": 1, "latent_mse_rollout": 2},
                            "goal_no_update_summary": summary, "goal_adapted_summary": summary},
              "costs": {"all_run_simulator_steps": {}, "training": {"wall_clock_s": 1},
                        "wall_clock_s": 2}}
    text = figures.markdown_report(paired, figures.selection_context(repair()))
    assert text.count("| Arm | Margin (px)") == 3
    for name in ["Dev", "Test", "Stress"]:
        block = text.split(f"### {name} bank\n", 1)[1].split("Paired adapted", 1)[0]
        assert sum(line.startswith("|---") for line in block.splitlines()) == 1
        assert all(block.count(f"| {arm} |") == 1 for arm in figures.ARMS)
    assert "0 eligible recipes out of 16" in text
    assert "not a demonstrated development winner" in text
    assert "not confidence intervals" in text
    assert "not evidence of a directional improvement" in text


def test_censored_horizon_annotation_keeps_unobserved_suffix_explicit():
    example = {"trace": {"observed_state_steps": list(range(14)),
                         "planned_horizon_steps": 25},
               "unresolved_future": True, "known_unsafe": False, "censored": True}
    text = figures.horizon_text(example)
    assert "UNRESOLVED FUTURE" in text and "step 13 of 25" in text
    assert "does not establish safety" in text
