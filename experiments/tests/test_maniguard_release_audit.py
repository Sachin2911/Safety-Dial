"""Release coverage must use pre-action holds and preserve the per-request delay bound."""

import json
import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1] / 'scripts')]

import maniguard_diagnose

from helpers.maniguard_release_audit import audit_release_trace

CONFIG = {'kind': 'release', 'release_delay': 2, 'horizon': 8}


def make_trace(openings=(True, True, True), delays=(True, True, False), holds=None):
    if holds is None:
        holds = {1: True, 2: True, 3: False}
    records = []
    for step, (opening, delay) in enumerate(zip(openings, delays)):
        original = [0.] * 7 + [1. if opening else -1.]
        adapted = original.copy()
        if delay:
            adapted[-1] = -1.
        event = dict.fromkeys(('guard_blocked', 'release_delayed', 'retry_started',
                              'retry_active', 'grasp_evidence', 'changed'), False)
        event.update(release_delayed=delay, changed=delay)
        records.extend([
            {'type': 'proposal', 'step': step, 'original': original,
             'adapted': adapted, 'event': event},
            {'type': 'executed', 'step': step, 'action': adapted.copy()},
        ])
        if step + 1 in holds:
            records.append({'type': 'goal', 'step': step + 1,
                            'detail': {'held': holds[step + 1]}})
    return records


def test_holding_after_first_action_does_not_count_as_holding_before_it():
    result = audit_release_trace(make_trace(), CONFIG, 3, first_violation_step=2)
    assert result['delayed_with_goal_target_held_before_action'] == 1
    assert result['delayed_with_pre_action_hold_unknown'] == 1
    assert result['delayed_with_goal_target_unheld_before_action'] == 0
    assert result['delayed_actions_strictly_before_first_violation'] == 1
    assert result['first_violation_action_delayed']
    assert result['delayed_runs'][0]['first_executed_opening_proposal'] == 2


def test_missing_goal_observations_remain_unknown():
    result = audit_release_trace(make_trace(holds={}), CONFIG, 3)
    assert result['delayed_with_pre_action_hold_unknown'] == 2
    assert result['delayed_with_goal_target_unheld_before_action'] == 0


def test_delay_bound_applies_to_each_consecutive_opening_request():
    records = make_trace((True, True, False, True, True, True),
                         (True, True, False, True, True, False), holds={})
    result = audit_release_trace(records, CONFIG, 6)
    assert result['maximum_delayed_prefix'] == 2
    assert result['delayed_actions'] == 4
    assert result['delayed_opening_runs'] == 2
    assert result['delayed_runs'][0]['first_executed_opening_proposal'] is None
    assert result['delayed_runs'][1]['first_executed_opening_proposal'] == 5


@pytest.mark.parametrize('flags', [(True, True, True), (False, True, False)])
def test_rejects_overlong_or_late_started_delay(flags):
    with pytest.raises(ValueError, match='bounded prefix'):
        audit_release_trace(make_trace(delays=flags), CONFIG, 3)


def test_rejects_undocumented_arm_changes():
    rows = make_trace()
    rows[0]['adapted'][0] = rows[1]['action'][0] = .1
    with pytest.raises(ValueError, match='may only replace'):
        audit_release_trace(rows, CONFIG, 3)


def test_rejects_different_execution():
    rows = make_trace()
    rows[1]['action'][-1] = 1.
    with pytest.raises(ValueError, match='Adapted and executed'):
        audit_release_trace(rows, CONFIG, 3)


def test_no_intervention_is_not_evidence_of_grasp_preservation():
    result = audit_release_trace(make_trace(delays=(False, False, False)), CONFIG, 3)
    assert result['delayed_actions'] == result['maximum_delayed_prefix'] == 0
    assert result['first_delayed_proposal'] is None
    assert result['delayed_with_goal_target_held_before_action'] == 0


def test_diagnostic_includes_audit_and_rejects_inconsistent_ledger(tmp_path, monkeypatch):
    verdict = {'adapter': CONFIG, 'steps': 3, 'ledger': {'changed_steps': 2}}
    monkeypatch.setattr(maniguard_diagnose, 'read_episode', lambda _: verdict)
    (tmp_path / 'trace.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in make_trace()))
    result = maniguard_diagnose.diagnose(tmp_path)
    assert result['release_audit']['delayed_actions'] == 2
    verdict['ledger']['changed_steps'] = 3
    with pytest.raises(ValueError, match='physical ledger'):
        maniguard_diagnose.diagnose(tmp_path)
