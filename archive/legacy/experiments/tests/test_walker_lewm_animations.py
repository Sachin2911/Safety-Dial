"""Selection rules of the Walker2d LeWM animations act on saved rows only and are deterministic."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import walker_lewm_animations as wa  # noqa: E402


def row(branch, dense, imagined, *, fell=False, dense_block=None, imag_block=None, n=10):
    def cumulative(final, block):
        return [1.0 if block is None or k + 1 < block else final for k in range(n)]

    return {"branch": branch, "cmin_dense_health": dense, "cmin_imagined_health": imagined, "fell": fell,
            "cmin_by_block_dense_health": cumulative(dense, dense_block), "cmin_by_block_imagined_health": cumulative(imagined, imag_block)}


def test_categories_follow_truth_and_the_margin():
    assert wa.categorise(row(0, 0.5, 0.4), 0.0) == "safe_accepted"
    assert wa.categorise(row(0, -0.5, 0.4), 0.0) == "fall_missed"
    assert wa.categorise(row(0, -0.5, -0.1), 0.0) == "fall_caught"
    assert wa.categorise(row(0, 0.5, -0.1), 0.0) == "false_alarm"
    # the margin moves the decision, never the truth
    assert wa.categorise(row(0, 0.5, 0.4), 0.45) == "false_alarm"
    assert wa.categorise(row(0, -0.5, 0.4), 0.45) == "fall_caught"


def test_first_block_is_one_based_and_none_when_never_below():
    assert wa.first_block_below([1.0, 0.5, -0.1, -0.2], 0.0) == 3
    assert wa.first_block_below([-0.1, -0.1], 0.0) == 1
    assert wa.first_block_below([1.0, 0.5], 0.0) is None


def test_median_true_clearance_with_branch_tiebreak():
    rows = [row(3, 0.9, 0.5), row(1, 0.2, 0.5), row(2, 0.6, 0.5), row(7, 0.6, 0.5), row(5, -1.0, 0.5)]
    picked, info = wa.select_example(rows, "safe_accepted", 0.0)
    assert info["n_in_category"] == 4 and info["n_rows"] == 5
    assert picked["branch"] == 7  # sorted by (clearance, branch): 1, 2, 7, 3 -> index 2
    assert wa.select_example(list(reversed(rows)), "safe_accepted", 0.0)[0]["branch"] == 7  # order independent


def test_unsafe_categories_prefer_rows_that_fell():
    rows = [row(0, -0.2, 0.5), row(1, -0.4, 0.5, fell=True), row(2, -0.9, 0.5), row(3, -1.5, 0.5, fell=True), row(4, -2.0, 0.5, fell=True)]
    picked, info = wa.select_example(rows, "fall_missed", 0.0)
    assert info["n_in_category"] == 5 and info["n_fell_in_pool"] == 3
    assert picked["branch"] == 3  # median of the three that fell
    none_fell = [row(0, -0.2, 0.5), row(1, -0.4, 0.5), row(2, -0.9, 0.5)]
    assert wa.select_example(none_fell, "fall_missed", 0.0)[0]["branch"] == 1  # falls back to the whole category


def test_caught_falls_prefer_timing_consistent_rejections():
    early = row(0, -0.7, -0.01, fell=True, dense_block=7, imag_block=1)  # rejected six blocks early: not an imagined fall
    on_time = [row(1, -0.4, -0.1, dense_block=8, imag_block=8), row(2, -1.0, -0.02, dense_block=8, imag_block=9), row(3, -0.6, -0.05, dense_block=8, imag_block=6)]
    picked, info = wa.select_example([early, *on_time], "fall_caught", 0.0)
    assert info["n_in_category"] == 4 and info["n_timing_consistent"] == 3 and info["n_fell_in_pool"] == 0
    assert picked["branch"] == 3  # median true clearance among the timing-consistent rows
    # with nothing timing-consistent the early rejection is still shown, and counted as such
    picked, info = wa.select_example([early], "fall_caught", 0.0)
    assert picked["branch"] == 0 and info["n_timing_consistent"] == 0


def test_empty_category_is_reported_not_invented():
    picked, info = wa.select_example([row(0, 0.5, 0.4)], "fall_caught", 0.0)
    assert picked is None and info["n_in_category"] == 0
