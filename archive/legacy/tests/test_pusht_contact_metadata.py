"""Aggregate collisions cannot silently become contact or free-motion evidence."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from helpers.decomposition import by_regime, contact_metadata


def test_unknown_contact_neither_contact_nor_free_motion():
    metadata = contact_metadata({"n_contacts": np.array([0, 5, 0]), "contact_kind": "any_collision"})
    assert metadata["contact"] is None
    assert metadata["any_collision"]
    row = {**metadata, "root": 0, "rotation_deg": 0., "cmin_dense": 1., "cmin_imagined": 1.,
           "unsafe_composite": False, "censored": False}
    groups = by_regime([row], 0., sources=["imagined"])
    assert "contact" not in groups
    assert "free" not in groups
    assert groups["contact_unknown"]["n"] == 1


def test_wall_only_contact_and_unobserved_suffix_are_not_pusher_contact():
    branch = {"n_contacts": np.array([0, 5, 0]), "contact_kind": "pusher_block",
              "pusher_block_contacts": np.array([0, 0, 7]),
              "block_wall_contacts": np.array([0, 5, 0]), "observed": np.array([True, True, False])}
    metadata = contact_metadata(branch)
    assert metadata["contact"] is False
    assert metadata["contact_mechanism_identifiable"]
    assert metadata["block_wall_contact"]
    branch["pusher_block_contacts"][1] = 2
    assert contact_metadata(branch)["contact"] is True
