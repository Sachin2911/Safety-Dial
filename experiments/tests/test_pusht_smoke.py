"""Check body-radius inflation, rounded corners, and gaps independently of the model."""

import torch

from helpers.pusht_smoke import arena_clearance, physical_clearance, signed_distance


def test_split_scratch_preserves_hdf5_random_access(tmp_path):
    import h5py
    import numpy as np
    from scripts.prepare_pusht_smoke import SplitReader

    original = tmp_path / "original.h5"
    values = np.arange(200).reshape(20, 10)
    with h5py.File(original, "w") as output:
        output["values"] = values
    content = original.read_bytes()
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(content[: len(content) // 2])
    second.write_bytes(content[len(content) // 2 :])
    with SplitReader([first, second]) as reader, h5py.File(reader, "r") as dataset:
        assert np.array_equal(dataset["values"][:], values)
        assert np.array_equal(dataset["values"][3], values[3])


def test_full_pusher_disk_crosses_region_while_centre_is_outside():
    shape = {"kind": "island", "radius_px": 12, "origin": [0, 0], "basis": [[1, 0], [0, 1]]}
    xy = torch.tensor([[20.0, 0], [33, 0], [34, 0]])
    assert signed_distance(xy, shape)[0] == 8
    assert torch.allclose(physical_clearance(xy, shape), torch.tensor([-7.0, 6.0, 7.0]))


def test_gate_corners_use_euclidean_disk_clearance():
    shape = {
        "kind": "gate",
        "gap_offset_px": 0,
        "gap_width_px": 60,
        "band_half_width_px": 8,
        "extent_px": 400,
        "origin": [0, 0],
        "basis": [[1, 0], [0, 1]],
    }
    xy = torch.tensor([[0.0, 0], [27, 12], [30, 8]])
    assert torch.allclose(signed_distance(xy, shape), torch.tensor([30.0, 5.0, 0.0]))
    assert physical_clearance(xy, shape)[0] == 15


def test_rounded_bar_end_and_arena_body_clearance():
    shape = {"kind": "bar", "segment_half_length_px": 30, "radius_px": 7}
    xy = torch.tensor([[0.0, 10], [33, 4]])
    assert torch.allclose(signed_distance(xy, shape), torch.tensor([3.0, -2.0]))
    assert torch.allclose(
        arena_clearance(torch.tensor([[15.0, 200], [20, 200]])), torch.tensor([0.0, 5.0])
    )
