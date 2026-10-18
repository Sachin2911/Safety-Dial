"""Reserved evaluation cannot use the legacy boolean/actor-list episode gate."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import maniguard_phase2_episode as episode


@pytest.mark.parametrize('purpose,old_protocol', [
    ('test', False), ('transfer', False), ('engineering', True), ('train', True),
])
def test_reserved_gate_rejects_before_assets_models_or_output(tmp_path, monkeypatch,
                                                            purpose, old_protocol):
    output = tmp_path / 'must-not-exist'
    argv = ['episode', '--family', 'jar', '--scene', 'task_0000/base', '--purpose', purpose,
            '--seed', '0', '--model', str(tmp_path / 'missing-model'), '--output', str(output)]
    if old_protocol:
        argv += ['--frozen-protocol', str(tmp_path / 'old-protocol.json')]
    monkeypatch.setattr(sys, 'argv', argv)

    def forbidden():
        pytest.fail('Asset/source validation should not run for a forbidden entry point')

    monkeypatch.setattr(episode, 'validate_registry', forbidden)
    with pytest.raises(ValueError, match='exact-job maniguard_main_episode.py'):
        episode.main()
    assert not output.exists()
