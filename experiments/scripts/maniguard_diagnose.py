"""Summarize recorded mechanisms; event counts do not establish causal repairs."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_results import read_episode


def diagnose(output):
    verdict = read_episode(output)
    counts = Counter()
    first_violation = None
    prior_held = None
    last_command_open = None
    releases_before_loss = []
    proposals = 0
    previous_open_request = None
    last_release_request = None
    release_before_violation = None
    goal_target = None
    rows = [json.loads(line) for line in (Path(output) / 'trace.jsonl').read_text().splitlines()]
    query_steps = {row['step'] for row in rows if row['type'] == 'query'}
    chunk_limit = None
    for row in rows:
        if row['type'] == 'initial':
            goal_target = row['scene'].get('goal_region', {}).get('target_name')
        elif row['type'] == 'chunk':
            chunk_limit = min(verdict['adapter']['horizon'], len(row['actions']))
        elif row['type'] == 'proposal':
            proposals += 1
            event = row['event']
            for key in ('changed', 'guard_blocked', 'release_delayed', 'retry_started',
                        'retry_active', 'grasp_evidence'):
                counts[key] += int(event[key])
            opening = row['original'][-1] >= 0
            if opening and previous_open_request is False:
                last_release_request = {'proposal_step': row['step'],
                                        'grasp_evidence': bool(event['grasp_evidence']),
                                        'adapted_open': bool(row['adapted'][-1] >= 0)}
            previous_open_request = opening
            last_command_open = row['adapted'][-1] >= 0
            counts['opening_proposals'] += int(row['original'][-1] >= 0)
        elif row['type'] == 'executed':
            last_command_open = row['action'][-1] >= 0
        elif row['type'] == 'replan':
            counts['replan:' + row['reason']] += 1
            if chunk_limit is None:
                raise ValueError('Replan event has no preceding model chunk')
            if row['step'] not in query_steps:
                counts['replan_without_following_query:' + row['reason']] += 1
            elif row['executed_prefix'] < chunk_limit:
                counts['early_replan:' + row['reason']] += 1
            else:
                counts['boundary_replan:' + row['reason']] += 1
        elif row['type'] == 'goal':
            counts['goal_observations'] += 1
            held = row['detail'].get('held')
            if prior_held is True and held is False:
                counts['held_to_unheld'] += 1
                if last_command_open:
                    releases_before_loss.append(row['step'])
            prior_held = held
        elif row['type'] == 'observed':
            monitor = row['monitor']
            if (monitor and monitor['valid'] and monitor['observation']['doomed']
                    and first_violation is None):
                first_violation = monitor['observation']
                release_before_violation = last_release_request
    return {**verdict, 'proposal_count': proposals, 'events': dict(counts),
            'first_violation_observation': first_violation,
            'goal_target': goal_target,
            'last_release_request_before_first_violation': release_before_violation,
            'held_loss_scope': 'Goal-checker target only; intermediate object releases are not covered.',
            'held_loss_after_open_command_steps': (releases_before_loss
                if counts['goal_observations'] else None),
            'interpretation': 'Temporal associations only; causal repairs require matched episodes.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.episode), indent=2))


if __name__ == '__main__':
    main()
