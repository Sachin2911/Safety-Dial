"""Compile and audit the bounded phase-two pilot without launching any episodes."""

import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from helpers.maniguard_phase2_families import CONFIG, ROOT, validate_registry
from helpers.maniguard_phase2_model import DEFAULT_SPEC
from helpers.maniguard_phase2_model import SCHEMA as MODEL_SCHEMA
from helpers.maniguard_phase2_optim import PPOConfig, SearchConfig, WeightSearch, record_hash

SCHEMA = 'maniguard-phase2-pilot-v1'
FAMILIES = ('jar', 'stack', 'clutter')
METHODS = ('evolution', 'random', 'ppo')
SEARCH_SEEDS = (17, 29)
REQUIRED_CHECKS = (
    'preparation_plan', 'measured_costing', 'feature_contract', 'identity_equivalence',
    'completion_semantics', 'reference_selection', 'runtime_execution_audit',
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(path=CONFIG / 'pilot.json'):
    return json.loads(Path(path).read_text())


def _read_inputs(config_dir):
    config_dir = Path(config_dir)
    return {name: json.loads((config_dir / f'{name}.json').read_text())
            for name in ('families', 'splits', 'sources', 'encoder')}


def _panels(splits, registry):
    return [{'panel_id': f'panel-{index}', 'index': index, 'cases': [
        {'family': family, 'scene': f"{splits['families'][family]['train'][index]}/base",
         'policy_seed': index, 'max_steps': registry[family]['max_steps']}
        for family in FAMILIES]} for index in range(3)]


def _validation_cases(splits, registry):
    return [{'family': family, 'scene': f'{task}/base', 'policy_seed': 10,
             'max_steps': registry[family]['max_steps']}
            for family in FAMILIES for task in splits['families'][family]['validation'][:2]]


def _case_fields(case):
    return {'family': case['family'], 'scene': case['scene'], 'seed': case['policy_seed'],
            'max_steps': case['max_steps']}


def compile_schedule(protocol):
    """Return fixed episode slots; the coordinator must attach measured controller weights.

    Search candidate genomes depend only on earlier training results. PPO collection is
    sequential, with each consecutive family triplet forming a balanced update batch.
    Validation controller references remain unresolved until training nomination finishes.
    This function is not authorization to execute an incomplete or changed protocol.
    """
    jobs = []
    for seed in protocol['optimizer_seeds']:
        for panel in protocol['training_panels']:
            index = panel['index']
            for method in ('evolution', 'random'):
                for slot in range(protocol['search_config']['population']):
                    for case in panel['cases']:
                        jobs.append({'job_id': f'train/{method}/s{seed}/p{index}/c{slot}/'
                                              f"{case['family']}",
                                     'stage': 'training', 'split': 'train', 'method': method,
                                     'optimizer_seed': seed, 'panel_id': panel['panel_id'],
                                     'round': index, 'candidate_slot': slot,
                                     'controller_ref': f'{method}/s{seed}/round{index}/slot{slot}',
                                     **_case_fields(case)})
            for repetition in range(8):
                for family_index, case in enumerate(panel['cases']):
                    ordinal = index * 24 + repetition * 3 + family_index + 1
                    jobs.append({'job_id': f'train/ppo/s{seed}/episode{ordinal:02d}',
                                 'stage': 'training', 'split': 'train', 'method': 'ppo',
                                 'optimizer_seed': seed, 'panel_id': panel['panel_id'],
                                 'round': index, 'repetition': repetition,
                                 'ppo_episode': ordinal, 'ppo_batch': index * 8 + repetition,
                                 'controller_ref': f'ppo/s{seed}/collection-batch{index * 8 + repetition}',
                                 **_case_fields(case)})
    for method in METHODS:
        for seed in protocol['optimizer_seeds']:
            for finalist in range(2):
                reference = (f'ppo/s{seed}/episodes{(48, 72)[finalist]}' if method == 'ppo'
                             else f'{method}/s{seed}/finalist{finalist}')
                for case_index, case in enumerate(protocol['validation_cases']):
                    jobs.append({'job_id': f'validation/{method}/s{seed}/f{finalist}/case{case_index}',
                                 'stage': 'finalist_validation', 'split': 'validation',
                                 'method': method, 'optimizer_seed': seed, 'finalist': finalist,
                                 'controller_ref': reference, **_case_fields(case)})
    for reference in protocol['references']:
        for case_index, case in enumerate(protocol['validation_cases']):
            jobs.append({'job_id': f'validation/reference/{reference["name"]}/case{case_index}',
                         'stage': 'reference_validation', 'split': 'validation',
                         'method': reference['name'], 'controller_ref': reference['name'],
                         **_case_fields(case)})
    return jobs


def resource_envelope(jobs):
    """Action-bearing upper bounds; initialization/failed queries require separate metering."""
    scheduled_actions = sum(job['max_steps'] for job in jobs)
    preparation_actions = 24 * 4200
    short_actions = 12 * 32
    actions = scheduled_actions + preparation_actions + short_actions
    normal_queries = sum((job['max_steps'] + 3) // 4 for job in jobs) + (
        preparation_actions + short_actions) // 4
    return {
        'scheduled_full_episodes': len(jobs), 'scheduled_policy_actions_max': scheduled_actions,
        'preparation_full_episodes_max': 24, 'preparation_policy_actions_max': preparation_actions,
        'short_checks_max': 12, 'short_check_actions_max_each': 32,
        'short_policy_actions_max': short_actions,
        'all_full_episodes_max': len(jobs) + 24, 'policy_actions_max': actions,
        'control_hz': 20, 'physics_hz': 120, 'control_physics_ticks_max': actions * 6,
        'normal_interval_successful_queries_max': normal_queries,
        'exhaustion_safe_successful_queries_max': actions,
        'queries_note': 'Normal minimum interval four actions; short-chunk exhaustion can query '
                        'earlier. At most one successful query per policy action; reject empty '
                        'chunks. Failed/inference-only queries are separately counted.',
        'excluded_from_action_bounds': ['initialization steps and physics ticks',
                                        'failed or inference-only policy queries'],
        'gpu_hours_ceiling': 72, 'gpu_hour_definition': 'Sum of allocated GPU-hours across devices; '
                                                     'not the sum of overlapping process durations.',
        'gpu_hours_status': 'provisional_until_preparation_costing',
        'measured_wall_hours_estimate': None,
        'failure_rule': 'Count every attempted interaction and stop affected work on infrastructure '
                        'failure. No unbudgeted replacement episodes or free retries.',
    }


def build_pilot(config_dir=CONFIG, root=ROOT):
    """Prepare the default immutable allocations and explicit incomplete-freeze record."""
    config_dir, root = Path(config_dir), Path(root)
    inputs = _read_inputs(config_dir)
    registry = inputs['families']['families']
    panels = _panels(inputs['splits'], registry)
    protocol = {
        'schema': SCHEMA, 'status': 'freeze_incomplete',
        'purpose': 'Bounded comparative feasibility pilot; no reserved/transfer outcomes.',
        'families': list(FAMILIES), 'methods': list(METHODS),
        'optimizer_seeds': list(SEARCH_SEEDS), 'training_panels': panels,
        'panel_rule': 'First three frozen training base tasks in manifest order; '
                      'panel index is the shared VLA policy seed.',
        'validation_cases': _validation_cases(inputs['splits'], registry),
        'validation_rule': 'First two validation base tasks per development family, policy seed 10.',
        'model': {'schema': MODEL_SCHEMA, 'spec': asdict(DEFAULT_SPEC),
                  'parameter_count': DEFAULT_SPEC.parameter_count,
                  'runtime_decoding': 'argmax-first-index',
                  'ppo_training_decoding': 'independent categorical heads, joint log probability'},
        'search_config': asdict(SearchConfig()), 'ppo_config': asdict(PPOConfig()),
        'initialization': {
            'identity': 'Model identity_genome(seed=20261017,bias=0.25); exact deterministic identity.',
            'ppo': 'Same identity-biased actor with finite logits and stochastic collection.',
            'shared_first_panel_sha256': {
                str(seed): [candidate['sha256'] for candidate in WeightSearch('random', seed).ask()]
                for seed in SEARCH_SEEDS},
        },
        'search_schedule': {
            'round0': 'Identity plus seven random initializations, common to both methods.',
            'round1': 'Previous within-panel winner reevaluated plus seven new candidates.',
            'round2': 'Round-zero winner and round-one winner reevaluated plus six new candidates.',
            'duplicates': 'Both retained slots are paid even for identical winners; rank duplicate '
                          'genomes by their mean final-panel utility.',
            'nomination': 'Two distinct genomes ranked on the final common panel only.',
        },
        'ppo_schedule': {
            'episodes_per_panel': 24, 'episodes_per_family_per_panel': 8,
            'episodes_per_run': 72, 'balanced_batch_episodes': 3,
            'family_order': list(FAMILIES), 'checkpoints': [48, 72],
            'return': 'Undiscounted terminal utility at every decision; fixed collection-critic '
                      'Monte Carlo advantages; sum decision scores inside each episode, then '
                      'mean episodes within family and mean families.',
        },
        'objective': {'name': 'required_safe_completion_minus_two_violations_v1',
                      'formula': 'S_req - 2 * V', 'safe_complete': 1, 'safe_unfinished': 0,
                      'violating': -2, 'aggregation': 'equal-family macro mean',
                      'guarantee': False, 'completion_checklists': 'pending_preparation_audit'},
        'references': [
            {'name': 'identity', 'choices': [0, 0, 0, 0], 'selection': 'defined comparator'},
            {'name': 'fixed', 'proposed_choices': [0, 0, 1, 0],
             'selection': 'pending training-only preparation; proposed smoothing alpha 0.75'},
            {'name': 'contextual', 'proposed_rule': {
                'feature_slice': [12, 19], 'comparison': 'max(abs(slice)) > 0.5',
                'true_choices': [1, 0, 0, 0], 'false_choices': [0, 0, 0, 0],
                'meaning': 'Normalized tracking error exceeds 0.5, equivalent to 0.1 rad.'},
             'selection': 'pending training-only preparation; freeze selected rule before validation'},
        ],
        'selection': {'finalists_per_run': 2, 'criterion': 'validation family-macro utility',
                      'ties': ['lower violation rate', 'lower query cost', 'earlier nominated index'],
                      'ppo_checkpoints': [48, 72], 'selection_episodes': 72,
                      'reference_episodes': 18, 'statistical_claim': 'screening only'},
        'execution_order': {'seed': 20261017,
                            'rule': 'Coordinator randomizes method/candidate order within valid '
                                    'round dependencies. Preserve consecutive PPO family triplets '
                                    'and audit the actual schedule; no future panel feedback.'},
        'preparation': {'full_episode_ceiling': 24, 'short_check_ceiling': 12,
                        'max_short_policy_actions': 32, 'allowed_families': list(FAMILIES),
                        'assignment_status': 'pending explicit preparation plan',
                        'selection_partition': 'train', 'integration_partition': 'engineering'},
        'freeze_checks': {name: {'passed': False, 'evidence_path': None, 'sha256': None}
                          for name in REQUIRED_CHECKS},
        'input_files': {str((config_dir / f'{name}.json').relative_to(root)):
                        _sha(config_dir / f'{name}.json') for name in inputs},
        'implementation_files': {name: _sha(root / name) for name in (
            'experiments/helpers/maniguard_phase2_model.py',
            'experiments/helpers/maniguard_phase2_optim.py',
            'experiments/helpers/maniguard_phase2_features.py',
            'experiments/helpers/maniguard_phase2_protocol.py',
        )},
    }
    jobs = compile_schedule(protocol)
    protocol['schedule_sha256'] = record_hash(jobs)
    protocol['resource_envelope'] = resource_envelope(jobs)
    protocol['sha256'] = record_hash(protocol)
    return protocol


def _verify_file_map(files, root):
    root = Path(root).resolve()
    for relative, digest in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _sha(path) != digest:
            raise ValueError(f'Pinned protocol input/source changed: {relative}')


def validate_protocol(protocol, config_dir=CONFIG, root=ROOT, *, require_frozen=False,
                      verify_upstream=True):
    """Verify scopes, counts and source pins; refuse execution until evidence closes freeze."""
    raw = {key: value for key, value in protocol.items() if key != 'sha256'}
    if protocol.get('schema') != SCHEMA or protocol.get('sha256') != record_hash(raw):
        raise ValueError('Protocol schema or content hash mismatch')
    validate_registry(config_dir, root, verify_upstream=verify_upstream)
    _verify_file_map(protocol['input_files'], root)
    _verify_file_map(protocol['implementation_files'], root)
    expected = build_pilot(config_dir, root)
    # Allow the freeze evidence and measured wall-time estimate to be completed, but no
    # silent change in allocation, feature/model pins, search or selection semantics.
    mutable = {'sha256', 'status', 'freeze_checks', 'resource_envelope'}
    for key, value in expected.items():
        if key not in mutable and record_hash(protocol.get(key)) != record_hash(value):
            raise ValueError(f'Frozen pilot field changed: {key}')
    jobs = compile_schedule(protocol)
    if record_hash(jobs) != protocol['schedule_sha256']:
        raise ValueError('Compiled episode schedule hash mismatch')
    registry = _read_inputs(config_dir)['families']['families']
    splits = _read_inputs(config_dir)['splits']['families']
    ids = set()
    for job in jobs:
        family = job['family']
        if family not in FAMILIES or not registry[family]['pilot']:
            raise ValueError('Pilot cannot expose withheld-family outcomes')
        if job['job_id'] in ids:
            raise ValueError('Duplicate episode slot')
        ids.add(job['job_id'])
        scene, condition = job['scene'].split('/')
        if condition != 'base' or scene not in splits[family][job['split']]:
            raise ValueError('Pilot scene is outside its frozen partition')
        if job['max_steps'] != registry[family]['max_steps']:
            raise ValueError('Pilot full episode horizon differs from pinned family horizon')
    counts = Counter(job['stage'] for job in jobs)
    if dict(counts) != {'training': 432, 'finalist_validation': 72, 'reference_validation': 18}:
        raise ValueError('Pilot episode accounting differs from the 522-episode allocation')
    for seed in SEARCH_SEEDS:
        for panel in protocol['training_panels']:
            for method in METHODS:
                rows = [job for job in jobs if job['stage'] == 'training' and
                        job['method'] == method and job['optimizer_seed'] == seed and
                        job['panel_id'] == panel['panel_id']]
                if Counter(job['family'] for job in rows) != dict.fromkeys(FAMILIES, 8):
                    raise ValueError('Method scene/family exposure is not matched')
    envelope = protocol['resource_envelope']
    frozen_envelope = resource_envelope(jobs)
    for field, value in frozen_envelope.items():
        if field in ('gpu_hours_status', 'measured_wall_hours_estimate'):
            continue
        if envelope.get(field) != value:
            raise ValueError(f'Resource bound changed: {field}')
    if (envelope['scheduled_policy_actions_max'] != 1270200 or
            envelope['policy_actions_max'] != 1371384):
        raise ValueError('Resource action envelope is inconsistent')
    checks = protocol['freeze_checks']
    if set(checks) != set(REQUIRED_CHECKS):
        raise ValueError('Missing readiness/freeze requirement')
    blockers = []
    for name, check in checks.items():
        if check.get('passed') is not True:
            blockers.append(name)
            continue
        if (not isinstance(check.get('evidence_path'), str) or
                not re.fullmatch('[0-9a-f]{64}', str(check.get('sha256')))):
            raise ValueError(f'Freeze requirement needs hashed evidence: {name}')
        _verify_file_map({check['evidence_path']: check['sha256']}, root)
    if protocol['status'] not in ('freeze_incomplete', 'frozen'):
        raise ValueError('Invalid protocol freeze status')
    if protocol['status'] == 'frozen' and blockers:
        raise ValueError('Cannot mark protocol frozen with unresolved requirements')
    if protocol['status'] == 'frozen' and (
            envelope['gpu_hours_status'] != 'frozen_after_preparation_costing' or
            not isinstance(envelope['measured_wall_hours_estimate'], (int, float)) or
            envelope['measured_wall_hours_estimate'] <= 0):
        raise ValueError('Frozen protocol needs measured wall-time costing')
    ready = protocol['status'] == 'frozen' and not blockers
    if require_frozen and not ready:
        raise ValueError('Pilot freeze incomplete: ' + ', '.join(blockers))
    return {'schema': SCHEMA, 'protocol_sha256': protocol['sha256'], 'ready_to_run': ready,
            'unresolved_freeze_checks': blockers, 'episode_counts': dict(counts),
            'scheduled_full_episodes': len(jobs), 'families': list(FAMILIES),
            'optimizer_seeds': list(SEARCH_SEEDS), 'resource_envelope': deepcopy(envelope),
            'held_out_outcomes_loaded': False, 'simulator_executed': False}


def pilot_report(protocol=None, **validation_options):
    """Return a JSON-compatible review report, never an episode-execution command."""
    return validate_protocol(load_protocol() if protocol is None else protocol,
                             **validation_options)
