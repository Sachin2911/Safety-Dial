"""Post-intervention cost attribution must distinguish holds from renewed guard blocks."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import maniguard_diagnose


def test_interrupted_hold_excludes_guard_and_includes_final_zero_countdown(tmp_path, monkeypatch):
    monkeypatch.setattr(maniguard_diagnose, 'read_episode',
                        lambda _: {'adapter': {'kind': 'handback'}})
    # One four-step hold is interrupted after two actions; another finishes.
    states = [(True, 0), (False, 3), (False, 2), (True, 2), (True, 2),
              (False, 3), (False, 2), (False, 1), (False, 0), (False, 0)]
    rows = []
    for step, (blocked, remaining) in enumerate(states):
        event = dict.fromkeys(('changed', 'release_delayed', 'retry_started',
                              'retry_active', 'grasp_evidence'), False)
        event.update(guard_blocked=blocked, handback_remaining=remaining)
        rows.append({'type': 'proposal', 'step': step, 'event': event,
                     'original': [0] * 7 + [-1], 'adapted': [0] * 7 + [-1]})
    (tmp_path / 'trace.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = maniguard_diagnose.diagnose(tmp_path)
    assert result['events']['guard_entry'] == 2
    assert result['events']['guard_exit'] == 2
    assert result['events']['guard_blocked'] == 3
    assert result['events']['handback_hold_inferred'] == 6


def test_guard_with_no_exit_has_no_handback_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(maniguard_diagnose, 'read_episode',
                        lambda _: {'adapter': {'kind': 'handback'}})
    event = dict.fromkeys(('changed', 'release_delayed', 'retry_started',
                          'retry_active', 'grasp_evidence'), False)
    event.update(guard_blocked=True, handback_remaining=0)
    row = {'type': 'proposal', 'step': 0, 'event': event,
           'original': [0] * 7 + [-1], 'adapted': [0] * 7 + [-1]}
    (tmp_path / 'trace.jsonl').write_text(json.dumps(row) + '\n')
    result = maniguard_diagnose.diagnose(tmp_path)
    assert result['events']['guard_entry'] == 1
    assert result['events']['guard_exit'] == 0
    assert result['events']['handback_hold_inferred'] == 0
