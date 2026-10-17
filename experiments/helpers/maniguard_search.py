"""Protocol checks and shared objective for matched compact-controller search."""

import itertools
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from helpers.maniguard_adaptation import AdapterConfig
from helpers.maniguard_batch_report import OUTCOMES, summarize_batch
from helpers.unisafe_search import AdapterSearch

METHODS = ('evolution', 'random', 'policy_gradient')


def validate_protocol(protocol, splits):
    if type(protocol.get('workers', 1)) is not int or protocol.get('workers', 1) not in (1, 2):
        raise ValueError('Search workers must be one or two, shared by all methods')
    family = protocol['family']
    base = protocol['base_adapter']
    if family == 'stack' and base['kind'] in ('prerequisite', 'handback'):
        raise ValueError('The closure prerequisite prototype applies only to Jar and Lid')
    domain = protocol['domain']
    if not domain or any(not isinstance(values, list) or not values for values in domain.values()):
        raise ValueError('Declare a finite nonempty parameter domain')
    if set(domain) & {'kind', 'privileged', 'guard_joint_limits'}:
        raise ValueError('Controller family, information access and guard prediction must remain fixed')
    if base['kind'] == 'handback' and not set(domain) <= {
        'handback_hold_steps', 'handback_requery',
    }:
        raise ValueError('Hand-back search must keep the guard trigger and policy horizon fixed')
    for values in itertools.product(*domain.values()):
        AdapterConfig(**(base | dict(zip(domain, values))))
    if len(protocol['methods']) != 3 or set(protocol['methods']) != set(METHODS):
        raise ValueError('Include evolution, random search and policy gradient with the same budget')
    for field in ('population', 'generations'):
        if type(protocol[field]) is not int or protocol[field] < 2:
            raise ValueError(f'{field} must be at least two')
    if (not protocol['search_seeds']
            or any(type(seed) is not int or seed < 0 for seed in protocol['search_seeds'])
            or len(set(protocol['search_seeds'])) != len(protocol['search_seeds'])):
        raise ValueError('Declare distinct search seeds')
    episodes = protocol['training_episodes']
    if not episodes or len({(e['scene'], e['seed']) for e in episodes}) != len(episodes):
        raise ValueError('Declare a nonempty common set of distinct training episodes')
    for episode in episodes:
        if set(episode) != {'scene', 'seed'}:
            raise ValueError('Training episode entries contain only scene and seed')
        if episode['scene'].split('/')[0] not in splits['families'][family]['train']:
            raise ValueError('Candidate search may only use the frozen training split')
        if type(episode['seed']) is not int:
            raise ValueError('Every physical episode needs an explicit integer policy seed')
    if protocol.get('objective') != 'task_completion_minus_two_violations':
        raise ValueError('Declare the shared measured terminal objective')


def terminal_utility(evidence):
    """Safe success 1, safe unfinished 0, unsafe success -1, unsafe unfinished -2.

    This training preference is not a safety guarantee. Report the four physical
    outcomes separately and select finalists using independent validation.
    """
    completion = evidence['safe_completion'] or evidence['unsafe_completion']
    return float(completion) - 2 * float(evidence['violation'])


def audit_generation(directory, expected_plan):
    """Derive fitness from complete physical records matched to the proposed generation."""
    directory = Path(directory)
    if json.loads((directory / 'plan.json').read_text()) != expected_plan:
        raise ValueError('Saved generation plan differs from the proposed candidates')
    if not (directory / 'complete.json').is_file():
        raise ValueError('Generation is missing its batch completion marker')
    report = summarize_batch(directory)
    if report['statuses'] != {'completed': len(expected_plan['jobs'])}:
        raise ValueError('Every proposed candidate episode must complete and validate')
    if (report['started_episodes_missing_cost_fields']
            or report['started_episodes_with_unverified_physics_attachment']
            or report['finished_processes_missing_elapsed_seconds']):
        raise ValueError('Generation has incomplete physical or process cost accounting')
    labels = [f'candidate-{i}' for i in range(len(expected_plan['genes']))]
    if {row['label'] for row in report['episodes']} != set(labels):
        raise ValueError('Generation candidate labels do not match its proposed genes')
    common_episodes = None
    fitness, outcomes = [], []
    for label in labels:
        matched = [row for row in report['episodes'] if row['label'] == label]
        episodes = {(row['family'], row['scene'], row['seed']) for row in matched}
        if len(episodes) != len(matched):
            raise ValueError('Candidate has duplicate scene/seed evaluations')
        if common_episodes is None:
            common_episodes = episodes
        if episodes != common_episodes:
            raise ValueError('Candidates must share the same scene/seed evaluations')
        fitness.append(float(np.mean([terminal_utility(row['result']) for row in matched])))
        outcomes.append({name: sum(row['result'][name] for row in matched) for name in OUTCOMES})
    return {'fitness': fitness, 'candidate_outcomes': outcomes, 'batch_report': report}


def search_state(optimizer):
    """Include Adam moments and logits so saved learning state is resumable."""
    return deepcopy(optimizer.state() | {
        'logits': [x.tolist() for x in optimizer.logits],
        'first_moment': [x.tolist() for x in optimizer.first_moment],
        'second_moment': [x.tolist() for x in optimizer.second_moment],
    })


def restore_search(state):
    optimizer = AdapterSearch(state['method'], state['domain'], state['population'], seed=0)
    optimizer.rng.bit_generator.state = deepcopy(state['rng_state'])
    optimizer.parents = deepcopy(state['parents'])
    optimizer.archive = deepcopy(state['archive'])
    optimizer.updates = state['updates']
    for field in ('logits', 'first_moment', 'second_moment'):
        setattr(optimizer, field, [np.array(x, dtype=float) for x in state[field]])
    return optimizer
