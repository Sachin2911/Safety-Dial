import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/pusht_e1_figures.py"
spec = importlib.util.spec_from_file_location("e1_figures_test", path)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def row(**changes):
    return {
        "m": 0.0,
        "acceptance_rate": 0.5,
        "n_accepted": 10,
        "n_false_safe": 2,
        "fsa": 0.2,
        **changes,
    }


def test_defined_fsa_retains_saved_value():
    value = figures.dial_value(row())
    assert value["status"] == "defined"
    assert value["point"] == value["lower"] == value["upper"] == 0.2


def test_censored_fsa_keeps_point_undefined_and_validates_saved_bounds():
    value = figures.dial_value(
        row(fsa=float("nan"), n_accepted_censored=1, fsa_lower=0.2, fsa_upper=0.3)
    )
    assert value["status"] == "undefined_censored" and value["point"] is None
    assert value["lower"] == 0.2 and value["upper"] == 0.3


def test_legacy_dial_without_censor_count_has_no_invented_upper_bound():
    value = figures.dial_value(row(fsa=float("nan")))
    assert value["status"] == "undefined_lower_bound_only" and value["point"] is None
    assert value["lower"] == 0.2 and value["upper"] is None


def test_zero_accepted_plans_remain_undefined_even_without_unsafe_events():
    value = figures.dial_value(
        row(
            acceptance_rate=0.0,
            n_accepted=0,
            n_false_safe=0,
            fsa=float("nan"),
            n_accepted_censored=0,
            fsa_lower=float("nan"),
            fsa_upper=float("nan"),
        )
    )
    assert value["status"] == "zero_acceptance"
    assert value["point"] is value["lower"] is value["upper"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"n_accepted": 0, "n_false_safe": 0, "acceptance_rate": 0.0, "fsa": 0.0},
        {"fsa": 0.2, "n_accepted_censored": 1},
        {"fsa": float("nan"), "n_accepted_censored": 1, "fsa_lower": 0.2, "fsa_upper": 0.4},
        {"n_false_safe": 11},
        {"n_accepted_censored": 9},
        {"acceptance_rate": -0.1},
    ],
)
def test_inconsistent_saved_counts_or_bounds_fail_closed(changes):
    with pytest.raises(ValueError):
        figures.dial_value(row(**changes))


def test_zero_observed_unsafe_with_unresolved_acceptance_is_only_a_zero_lower_bound():
    value = figures.dial_value(
        row(n_false_safe=0, fsa=float("nan"), n_accepted_censored=2, fsa_lower=0.0, fsa_upper=0.2)
    )
    assert value["point"] is None and value["lower"] == 0.0 and value["upper"] == 0.2


def test_m0_anchor_must_match_immutable_curve_counts():
    sources = {s: [row()] for s in figures.SOURCES}
    anchors = {s: row() for s in figures.SOURCES}
    anchors["imagined"] = row(n_false_safe=3, fsa=0.3)
    with pytest.raises(ValueError, match="differs from its dial row"):
        figures.prepare_dial({"banks": {"dev": {"dial_curves": sources, "at_m0": anchors}}})
