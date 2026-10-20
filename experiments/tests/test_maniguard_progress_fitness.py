"""Progress supervision must retain terminal ordering and match all optimizer arms."""

from itertools import product

import pytest

from helpers.maniguard_progress_fitness import FactualProgress, ProgressObjective


def test_safe_closure_is_retained_and_later_violation_removes_its_progress():
    objective = ProgressObjective(("case0", "case1"))
    idle, closed = FactualProgress(False, False, 0), FactualProgress(False, False, 1)
    inactive = objective.score({"case0": idle, "case1": idle})
    closure = objective.score({"case0": closed, "case1": idle})
    violating = objective.score({"case0": FactualProgress(False, True, 4), "case1": idle})
    assert inactive.rank_key == (0, 0) and inactive.scalar == 0
    assert closure.rank_key == (0, 1) and closure.scalar == 1
    assert violating.rank_key == (-2, 0) and violating.scalar == -18
    assert FactualProgress(True, True, 4).primary == -2


def test_exhaustive_two_case_panel_scalar_preserves_lexicographic_terminal_order():
    objective = ProgressObjective(("a", "b"))
    possible = [FactualProgress(False, True, 4)] + [
        FactualProgress(complete, False, stage)
        for complete, stage in product((False, True), range(5))
    ]
    scores = [
        objective.score(dict(zip(objective.case_ids, pair))) for pair in product(possible, repeat=2)
    ]
    for left, right in product(scores, repeat=2):
        assert (left.scalar > right.scalar) == (left.rank_key > right.rank_key)
        assert (left.scalar == right.scalar) == (left.rank_key == right.rank_key)


@pytest.mark.parametrize("panel_size", [1, 2, 7])
def test_maximum_secondary_credit_cannot_outweigh_one_observed_primary_point(panel_size):
    objective = ProgressObjective(tuple(str(i) for i in range(panel_size)))
    progress = {case: FactualProgress(False, False, 4) for case in objective.case_ids}
    completed = {case: FactualProgress(False, False, 0) for case in objective.case_ids}
    completed[objective.case_ids[0]] = FactualProgress(True, False, 0)
    assert objective.score(progress).scalar == 4 * panel_size
    assert objective.score(completed).scalar == 4 * panel_size + 1


def test_ordinary_learning_rewards_sum_to_identical_candidate_fitness():
    objective = ProgressObjective(("a", "b"))
    outcomes = {"b": FactualProgress(False, True, 3), "a": FactualProgress(False, False, 2)}
    assert objective.episode_rewards(outcomes) == {"a": 2, "b": -18}
    assert sum(objective.episode_rewards(outcomes).values()) == objective.score(outcomes).scalar
    assert objective.specification()["dispatch_authorized"] is False


@pytest.mark.parametrize("ids", [(), ("a", "a"), ("",), "case0", (1,)])
def test_case_panels_cannot_be_empty_duplicated_or_ambiguous(ids):
    with pytest.raises(ValueError, match="distinct case identifiers"):
        ProgressObjective(ids)


@pytest.mark.parametrize(
    "labels",
    [(0, False, 0), (False, 0, 0), (False, False, True), (False, False, -1), (False, False, 5)],
)
def test_outcomes_require_factual_boolean_labels_and_bounded_integer_stages(labels):
    with pytest.raises(ValueError):
        FactualProgress(*labels)


def test_partial_extra_or_unvalidated_outcomes_cannot_receive_a_panel_score():
    objective, idle = ProgressObjective(("a", "b")), FactualProgress(False, False, 0)
    for outcomes in ({"a": idle}, {"a": idle, "b": idle, "c": idle}, {"a": idle, "b": {}}):
        with pytest.raises(ValueError, match="exactly one validated factual outcome"):
            objective.score(outcomes)
