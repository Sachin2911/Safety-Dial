import numpy as np
import pytest

from helpers.unisafe09_statistics import paired_binary, paired_mean
from helpers.unisafe09p_statistics import blocked_binary, blocked_mean, matched_groups


def test_one_host_binary_reduces_to_the_original_interval():
    a = [{"C": x} for x in [1, 0, 1, 1, 0]]
    b = [{"C": x} for x in [0, 0, 0, 1, 1]]
    old = paired_binary([r["C"] for r in a], [r["C"] for r in b], .03)
    new = blocked_binary({"one": (a, b)}, "C", .03)
    assert new["difference"] == old["difference"]
    assert new["interval"] == old["interval"]


def test_weighted_hosts_use_seed_counts_and_all_marginal_errors():
    groups = {"a": ([{"C": True}]*2, [{"C": False}]*2),
              "b": ([{"C": False}]*6, [{"C": True}]*6)}
    result = blocked_binary(groups, "C", .04)
    assert result["difference"] == -.5
    assert result["marginal_error"] == .01
    assert result["interval"][0] <= -.5 <= result["interval"][1]
    assert all(x["alpha"] == .02 for x in result["by_host"].values())


def test_one_host_bootstrap_reduces_to_original():
    a = [{"J": x} for x in [3, 5, 15]]
    b = [{"J": x} for x in [15, 4, 15]]
    draws = np.random.default_rng(2026102609).integers(0, 3, size=(10000, 3))
    old = paired_mean([r["J"] for r in a], [r["J"] for r in b], draws, .05)
    new = blocked_mean({"a": (a, b)}, {"a": draws})
    assert old["difference"] == new["difference"]
    assert old["interval"] == new["interval"]


def test_host_resampling_cannot_import_another_hosts_indices():
    groups = {"a": ([{"J": 5}], [{"J": 15}])}
    with pytest.raises(ValueError, match="within each physical host"):
        blocked_mean(groups, {"a": np.ones((10000, 1), dtype=int)})


def test_pairs_cannot_cross_hosts():
    rows = [{"physical_host": "one", "condition": "a", "seed": 7},
            {"physical_host": "two", "condition": "b", "seed": 7}]
    with pytest.raises(ValueError, match="Every host"):
        matched_groups(rows, "a", "b")
