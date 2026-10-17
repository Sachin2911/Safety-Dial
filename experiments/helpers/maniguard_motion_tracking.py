"""Audit commanded joint limits separately from measured physical joint motion."""

import hashlib
import json
from pathlib import Path

import numpy as np

from helpers.maniguard_results import read_episode

OUTCOMES = ('safe_completion', 'unsafe_completion', 'safe_unfinished', 'unsafe_unfinished')
CAP_TOLERANCE = 1e-6


def audit_trace(verdict, records):
    """Require aligned states and executed targets before describing tracking error."""
    config = verdict['adapter']
    if config['kind'] not in ('identity', 'motion'):
        raise ValueError('Tracking comparison requires identity or motion adapters')
    steps = verdict['steps']
    if steps < 1:
        raise ValueError('Tracking audit needs at least one executed action')
    proposed = [r for r in records if r['type'] == 'proposal']
    executed = [r for r in records if r['type'] == 'executed']
    observed = [r for r in records if r['type'] == 'observed']
    for rows, start in ((proposed, 0), (executed, 0), (observed, 1)):
        if [r['step'] for r in rows] != list(range(start, steps + start)):
            raise ValueError('Trace actions and observations must be complete and ordered')
    q = np.asarray([r['frame']['joints'] for r in proposed], dtype=float)
    qn = np.asarray([r['state'][:7] for r in observed], dtype=float)
    adapted = np.asarray([r['adapted'][:7] for r in proposed], dtype=float)
    command = np.asarray([r['action'][:7] for r in executed], dtype=float)
    if any(a.shape != (steps, 7) or not np.isfinite(a).all()
           for a in (q, qn, adapted, command)):
        raise ValueError('Each action and state must contain seven finite arm joints')
    if not np.array_equal(qn[:-1], q[1:]):
        raise ValueError('Post-action joint states differ from the next proposal frame')
    if not np.array_equal(adapted, command):
        raise ValueError('Adapted and executed arm targets differ')
    target_step = np.max(np.abs(command - q), axis=1)
    physical_step = np.max(np.abs(qn - q), axis=1)
    peak = int(np.argmax(physical_step))
    first_violation = next((r['monitor']['observation'] for r in observed
                            if r['monitor'] and r['monitor']['valid']
                            and r['monitor']['observation']['doomed']), None)
    result = {
        'scene': verdict['scene'], 'seed': verdict['seed'], 'adapter': config,
        'outcome': next(k for k in OUTCOMES if verdict[k]), 'actions': steps,
        'adjacent_measured_states_match_exactly': True,
        'adapted_and_executed_arm_targets_match_exactly': True,
        'command_step_max_rad': float(target_step.max()),
        'measured_step_max_rad': float(physical_step.max()),
        'measured_step_p95_rad': float(np.quantile(physical_step, .95)),
        'command_to_post_action_state_max_error_rad': float(np.max(np.abs(command - qn))),
        'peak_measured_step_observation': peak + 1,
        'command_step_at_measured_peak_rad': float(target_step[peak]),
        'first_violation_observation': first_violation,
        'changed_actions': sum(bool(r['event']['changed']) for r in proposed),
    }
    if config['kind'] == 'motion':
        caps = np.asarray([r['event']['step_limit'] for r in proposed], dtype=float)
        expected = np.asarray([
            config['grasp_step_limit'] if r['event']['grasp_evidence']
            and r['original'][-1] < 0 else config['free_step_limit'] for r in proposed])
        if not np.array_equal(caps, expected):
            raise ValueError('Logged active caps differ from configuration and grasp/opening events')
        if not np.all(target_step <= caps + CAP_TOLERANCE):
            raise ValueError('Executed target displacement exceeds the active command cap')
        result.update({
            'command_cap_max_excess_rad': float((target_step - caps).max()),
            'all_commands_within_active_cap_plus_1e_6': True,
            'measured_steps_above_active_cap_plus_1e_6': int(
                np.sum(physical_step > caps + CAP_TOLERANCE)),
        })
        if first_violation is not None:
            t = first_violation['step']
            before = physical_step[:t - 1]
            result['violation_timing'] = {
                'measured_step_on_first_violation_rad': float(physical_step[t - 1]),
                'active_command_cap_on_first_violation_rad': float(caps[t - 1]),
                'strictly_before_violation_max_measured_step_rad': (
                    float(before.max()) if before.size else None),
                'strictly_before_violation_peak_observation': (
                    int(np.argmax(before)) + 1 if before.size else None),
                'strictly_before_violation_steps_above_active_cap_plus_1e_6': int(
                    np.sum(before > caps[:t - 1] + CAP_TOLERANCE)),
                'largest_measured_step_is_after_first_violation': peak + 1 > t,
            }
    return result


def audit_episode(output):
    """Read only a validated completed episode and retain its raw trace hash."""
    output = Path(output)
    verdict = read_episode(output)
    trace = output / 'trace.jsonl'
    raw = trace.read_bytes()
    result = audit_trace(verdict, [json.loads(line) for line in raw.splitlines()])
    if result['changed_actions'] != verdict['ledger']['changed_steps']:
        raise ValueError('Changed-action count differs from the physical ledger')
    return {'episode': str(output), 'trace_sha256': hashlib.sha256(raw).hexdigest(), **result}
