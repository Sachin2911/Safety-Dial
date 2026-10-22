"""Ensure broken evaluations never enter safe-completion denominators silently."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.maniguard_results import read_episode


def write_fixture(path, success=False, violation=False):
    (path / 'episode/task_0000').mkdir(parents=True)
    result = {'status': 'completed', 'scene_name': 'task_0000/base', 'steps': 2,
              'ltl_monitored': True, 'ltl_violated': violation, 'success': success,
              'ever_contacted': True, 'ever_grasped': success}
    files = {
        'episode/results.jsonl': result,
        'episode/task_0000/base_ltl.json': {
            'total_steps_monitored': 3, 'violated': violation, 'error': None,
            'log': [{'step': i} for i in range(3)],
        },
        'ledger.json': {'physics_accounting_attached': True, 'policy_control_steps': 2},
        'job.json': {'family': 'jar', 'seed': 0, 'adapter': {'kind': 'identity'},
                     'max_steps_override': None},
    }
    for name, value in files.items():
        (path / name).write_text(json.dumps(value) + '\n')
    return files


@pytest.mark.parametrize(('success', 'violation', 'expected'), [
    (True, False, 'safe_completion'), (True, True, 'unsafe_completion'),
    (False, False, 'safe_unfinished'), (False, True, 'unsafe_unfinished'),
])
def test_four_physical_outcomes_remain_distinct(tmp_path, success, violation, expected):
    write_fixture(tmp_path, success, violation)
    row = read_episode(tmp_path)
    assert row[expected]
    assert sum(row[k] for k in ('safe_completion', 'unsafe_completion',
                               'safe_unfinished', 'unsafe_unfinished')) == 1


@pytest.mark.parametrize(('filename', 'key', 'value'), [
    ('episode/results.jsonl', 'status', 'monitor_failed'),
    ('episode/results.jsonl', 'ltl_monitored', False),
    ('episode/results.jsonl', 'ltl_violated', None),
    ('episode/task_0000/base_ltl.json', 'error', {'step': 1}),
    ('episode/task_0000/base_ltl.json', 'log', [{'step': 0}, {'step': 2}]),
    ('episode/task_0000/base_ltl.json', 'violated', True),
    ('ledger.json', 'physics_accounting_attached', False),
    ('ledger.json', 'policy_control_steps', 1),
])
def test_missing_or_inconsistent_evidence_is_rejected(tmp_path, filename, key, value):
    files = write_fixture(tmp_path)
    files[filename][key] = value
    (tmp_path / filename).write_text(json.dumps(files[filename]) + '\n')
    with pytest.raises(ValueError):
        read_episode(tmp_path)
