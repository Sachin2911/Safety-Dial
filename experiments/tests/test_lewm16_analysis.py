"""Check pairing and case weighting, which determine final uncertainty claims."""

from helpers.lewm16_analysis import paired_contrasts


def row(case, seed, condition, success):
    return {
        "status": "complete",
        "job": dict(
            stage="B", env="tworoom", case={"id": case}, margin=6, seed=seed, condition=condition
        ),
        "requested_safe_completion": success,
        "any_violation": False,
        "margin_breach": False,
        "J": 10 if success else 50,
    }


def test_case_weighting_does_not_turn_more_seeds_into_more_tasks():
    rows = []
    for case, count, fixed, adaptive in [
        ("a", 1, False, True),
        ("b", 5, True, True),
        ("c", 25, True, False),
    ]:
        for seed in range(count):
            rows += [row(case, seed, "fixed", fixed), row(case, seed, "adaptive", adaptive)]
    result = paired_contrasts(rows, repetitions=1000)
    effect = next(
        r
        for r in result["contrasts"]
        if r["margin"] == "all" and r["metric"] == "requested_safe_completion"
    )
    assert effect["adaptive_minus_fixed"] == 0
    assert effect["paired_cases"] == 3 and effect["paired_episodes"] == 31
    assert result["unpaired"] == []


def test_missing_arm_is_reported_and_paired_effect_direction_is_correct():
    rows = [row("a", 0, "fixed", False), row("a", 0, "adaptive", True), row("b", 0, "fixed", False)]
    result = paired_contrasts(rows, repetitions=100)
    assert len(result["unpaired"]) == 1
    effect = next(
        r
        for r in result["contrasts"]
        if r["margin"] == "all" and r["metric"] == "requested_safe_completion"
    )
    assert effect["adaptive_minus_fixed"] == 1
    assert effect["interval_95"] == [1, 1]
