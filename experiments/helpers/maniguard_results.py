"""Read complete physical evidence without treating failed monitoring as safety."""

import json
from pathlib import Path


def read_episode(output):
    output = Path(output)
    rows = [json.loads(line) for line in (output / 'episode/results.jsonl').read_text().splitlines()
            if line.strip()]
    if len(rows) != 1:
        raise ValueError('Expected exactly one physical episode result')
    result = rows[0]
    if result['status'] != 'completed':
        raise ValueError(f"Episode did not complete normally: {result['status']}")
    if result.get('ltl_monitored') is not True or type(result.get('ltl_violated')) is not bool:
        raise ValueError('Physical safety verdict is unavailable')
    if type(result.get('success')) is not bool:
        raise ValueError('Task completion verdict is unavailable')
    monitor = json.loads((output / 'episode' / (result['scene_name'] + '_ltl.json')).read_text())
    if monitor.get('error') is not None or monitor['total_steps_monitored'] != result['steps'] + 1:
        raise ValueError('Safety trace is incomplete')
    if [entry['step'] for entry in monitor['log']] != list(range(result['steps'] + 1)):
        raise ValueError('Safety observations are missing, duplicated or out of order')
    if monitor['violated'] != result['ltl_violated']:
        raise ValueError('Safety summary and episode verdict disagree')
    ledger = json.loads((output / 'ledger.json').read_text())
    if not ledger.get('physics_accounting_attached'):
        raise ValueError('Physics accounting was not attached')
    if ledger['policy_control_steps'] != result['steps']:
        raise ValueError('Executed-action accounting differs from the episode result')
    job = json.loads((output / 'job.json').read_text())
    success, violated = result['success'], result['ltl_violated']
    return {
        'family': job['family'], 'scene': result['scene_name'], 'seed': job['seed'],
        'adapter': job['adapter'], 'engineering_override': job['max_steps_override'],
        'safe_completion': success and not violated,
        'unsafe_completion': success and violated,
        'safe_unfinished': not success and not violated,
        'unsafe_unfinished': not success and violated,
        'violation': violated, 'ever_contacted': result['ever_contacted'],
        'ever_grasped': result['ever_grasped'], 'steps': result['steps'],
        'ledger': ledger, 'output': str(output),
    }
