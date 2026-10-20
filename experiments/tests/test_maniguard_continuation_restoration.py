"""Mutable wave metadata must never authorize changes to physical evidence."""

import pytest

from helpers.maniguard_continuation_restoration import merge_recovered_waves


def test_layers_only_the_named_metadata_and_keeps_source_trees(tmp_path):
    waves = []
    for index in range(2):
        base = tmp_path / str(index)
        (base / "screen").mkdir(parents=True)
        (base / "screen/screen.json").write_text(str(index))
        (base / "screen/episode.txt").write_text("unchanged physical bytes")
        waves.append({"merged": str(base)})
    changes = merge_recovered_waves(waves, tmp_path / "merged")
    assert [row["path"] for row in changes] == ["screen/screen.json"]
    assert (tmp_path / "merged/screen/screen.json").read_text() == "1"
    assert (tmp_path / "0/screen/screen.json").read_text() == "0"


@pytest.mark.parametrize("name", ["episodes/example/trace.jsonl", "source-records/protocol.json"])
def test_refuses_differing_physical_or_frozen_source_bytes(tmp_path, name):
    waves = []
    for index in range(2):
        base = tmp_path / str(index)
        target = base / "screen" / name
        target.parent.mkdir(parents=True)
        target.write_text(str(index))
        waves.append({"merged": str(base)})
    with pytest.raises(ValueError, match="Immutable screen evidence differs"):
        merge_recovered_waves(waves, tmp_path / "merged")
