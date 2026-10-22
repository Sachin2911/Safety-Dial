"""Tracking audits must reject broken evidence without treating caps as velocity limits."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_motion_tracking import audit_trace


def trace_case():
    verdict = {
        'scene': 'task_0010/base', 'seed': 0, 'steps': 2,
        'adapter': {'kind': 'motion', 'free_step_limit': .01, 'grasp_step_limit': .005},
        'safe_completion': True, 'unsafe_completion': False,
        'safe_unfinished': False, 'unsafe_unfinished': False,
    }
    rows = []
    for i, (q, target, after, grasp, cap) in enumerate([
        (0., .01, .02, False, .01), (.02, .025, .03, True, .005),
    ]):
        rows.extend([
            {'type': 'proposal', 'step': i, 'frame': {'joints': [q] * 7},
             'original': [.1] * 7 + [-1], 'adapted': [target] * 7 + [-1],
             'event': {'changed': True, 'grasp_evidence': grasp, 'step_limit': cap}},
            {'type': 'executed', 'step': i, 'action': [target] * 7 + [-1]},
            {'type': 'observed', 'step': i + 1, 'state': [after] * 7,
             'monitor': {'valid': True, 'observation': {'step': i + 1, 'doomed': False}}},
        ])
    return verdict, rows


def test_measured_motion_can_exceed_a_correctly_executed_command_cap():
    verdict, rows = trace_case()
    result = audit_trace(verdict, rows)
    assert result['all_commands_within_active_cap_plus_1e_6']
    assert result['command_step_max_rad'] == pytest.approx(.01)
    assert result['measured_step_max_rad'] == pytest.approx(.02)
    assert result['measured_steps_above_active_cap_plus_1e_6'] == 2


@pytest.mark.parametrize(('damage', 'message'), [
    ('execution', 'Adapted and executed'),
    ('ordering', 'complete and ordered'),
    ('bound', 'exceeds the active command cap'),
    ('active_cap', 'Logged active caps'),
    ('state_alignment', 'next proposal frame'),
    ('nonfinite', 'finite arm joints'),
])
def test_rejects_unreliable_command_or_state_evidence(damage, message):
    verdict, rows = trace_case()
    if damage == 'execution':
        rows[1]['action'][0] += .001
    elif damage == 'ordering':
        rows[1]['step'] = 1
    elif damage == 'bound':
        rows[0]['adapted'][0] = rows[1]['action'][0] = .02
    elif damage == 'active_cap':
        rows[3]['event']['step_limit'] = .01
    elif damage == 'state_alignment':
        rows[3]['frame']['joints'][0] += .001
    elif damage == 'nonfinite':
        rows[5]['state'][0] = float('nan')
    with pytest.raises(ValueError, match=message):
        audit_trace(verdict, rows)


def test_first_action_violation_has_no_preceding_motion():
    verdict, rows = trace_case()
    verdict.update(safe_completion=False, unsafe_completion=True)
    rows[2]['monitor']['observation']['doomed'] = True
    rows[5]['monitor']['observation']['doomed'] = True
    result = audit_trace(verdict, rows)
    timing = result['violation_timing']
    assert timing['strictly_before_violation_max_measured_step_rad'] is None
    assert timing['strictly_before_violation_peak_observation'] is None
    assert timing['strictly_before_violation_steps_above_active_cap_plus_1e_6'] == 0
    assert timing['measured_step_on_first_violation_rad'] == pytest.approx(.02)


def test_opening_proposal_uses_free_cap_despite_grasp_evidence():
    verdict, rows = trace_case()
    rows[3]['original'][-1] = 1
    rows[3]['event']['step_limit'] = .01
    assert audit_trace(verdict, rows)['all_commands_within_active_cap_plus_1e_6']
