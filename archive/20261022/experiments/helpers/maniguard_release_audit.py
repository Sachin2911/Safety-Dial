"""Check bounded release execution and goal-target holding before delayed actions."""

import math


def audit_release_trace(records, config, steps, first_violation_step=None):
    """Audit completed traces; holding other objects is outside the recorded goal scope."""
    if config['kind'] != 'release':
        raise ValueError('Release audit requires a release adapter')
    limit = config['release_delay']
    if type(limit) is not int or limit < 0:
        raise ValueError('Release delay must be a nonnegative integer')
    if first_violation_step is not None and not 1 <= first_violation_step <= steps:
        raise ValueError('First violation must correspond to a post-action observation')
    proposals = [r for r in records if r['type'] == 'proposal']
    executed = [r for r in records if r['type'] == 'executed']
    if any([r['step'] for r in rows] != list(range(steps)) for rows in (proposals, executed)):
        raise ValueError('Proposal and execution records must be complete and ordered')
    holds = {}
    for row in records:
        if row['type'] == 'goal':
            step, held = row['step'], row['detail'].get('held')
            if step in holds or not 0 <= step <= steps:
                raise ValueError('Goal observations must have unique in-range indices')
            if held is not None and type(held) is not bool:
                raise ValueError('Goal-target hold observations must be boolean or unknown')
            holds[step] = held
    delayed = []
    opening_runs = []
    for proposal, execution in zip(proposals, executed):
        original, adapted, action = (proposal['original'], proposal['adapted'], execution['action'])
        if any(len(a) != 8 or not all(math.isfinite(v) for v in a)
               for a in (original, adapted, action)):
            raise ValueError('Release audit needs finite eight-dimensional actions')
        delay = proposal['event']['release_delayed']
        if type(delay) is not bool or proposal['event']['changed'] != delay:
            raise ValueError('Changed-action and release-delay events must agree')
        if adapted != action:
            raise ValueError('Adapted and executed actions differ')
        expected = original.copy()
        if delay:
            if original[-1] < 0:
                raise ValueError('A delayed release must originate from an opening request')
            expected[-1] = -1.0
            delayed.append(proposal['step'])
        if adapted != expected:
            raise ValueError('Release adapter may only replace a delayed opening with closing')
        if original[-1] >= 0:
            if not opening_runs or opening_runs[-1][-1]['step'] != proposal['step'] - 1:
                opening_runs.append([])
            opening_runs[-1].append(proposal)
    run_details = []
    for run in opening_runs:
        flags = [r['event']['release_delayed'] for r in run]
        count = sum(flags)
        if count > limit or flags != [True] * count + [False] * (len(flags) - count):
            raise ValueError('A release delay must be a bounded prefix of consecutive opening requests')
        if count:
            run_details.append({
                'first_opening_proposal': run[0]['step'],
                'last_opening_proposal': run[-1]['step'],
                'opening_requests': len(run), 'delayed_actions': count,
                'first_executed_opening_proposal': run[count]['step'] if count < len(run) else None,
            })
    held_steps = [step for step, held in holds.items() if held is True]
    t = first_violation_step
    return {
        'actions_checked': steps,
        'adapted_and_executed_actions_match_exactly': True,
        'only_delayed_opening_commands_changed': True,
        'all_delays_within_declared_prefix_limit': True,
        'declared_delay_limit': limit,
        'opening_runs': len(opening_runs), 'delayed_opening_runs': len(run_details),
        'delayed_actions': len(delayed),
        'maximum_delayed_prefix': max((r['delayed_actions'] for r in run_details), default=0),
        'delayed_with_goal_target_held_before_action': sum(holds.get(i) is True for i in delayed),
        'delayed_with_goal_target_unheld_before_action': sum(holds.get(i) is False for i in delayed),
        'delayed_with_pre_action_hold_unknown': sum(holds.get(i) is None for i in delayed),
        'first_target_held_observation': min(held_steps) if held_steps else None,
        'first_delayed_proposal': delayed[0] if delayed else None,
        'last_delayed_proposal': delayed[-1] if delayed else None,
        'first_violation_observation': t,
        'delayed_actions_strictly_before_first_violation': (
            sum(i + 1 < t for i in delayed) if t is not None else None),
        'first_violation_action_delayed': t - 1 in delayed if t is not None else None,
        'delayed_runs': run_details,
        'alignment': ('Proposal i executes before observation i+1. Goal observation i supplies '
                      'pre-action target holding; a missing observation is unknown, not unheld. '
                      'Strictly-before counts exclude the action producing the first violation.'),
        'scope': ('Goal-checker target only; holds on other objects are not recorded here. '
                  'Activation and temporal association do not establish that a delay preserved '
                  'a useful grasp or improved an outcome.'),
    }
