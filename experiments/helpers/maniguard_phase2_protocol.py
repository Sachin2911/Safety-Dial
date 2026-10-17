"""Compile and audit the bounded phase-two pilot without launching any episodes."""

import hashlib
import json
import math
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
INPUT_NAMES = ('families', 'splits', 'sources', 'encoder', 'completion', 'preparation')
IMPLEMENTATION_FILES = (
    'experiments/helpers/maniguard_phase2_model.py',
    'experiments/helpers/maniguard_phase2_optim.py',
    'experiments/helpers/maniguard_phase2_features.py',
    'experiments/helpers/maniguard_phase2_adapter.py',
    'experiments/helpers/maniguard_phase2_families.py',
    'experiments/helpers/maniguard_phase2_runtime.py',
    'experiments/helpers/maniguard_phase2_source.py',
    'experiments/helpers/maniguard_phase2_outcomes.py',
    'experiments/helpers/maniguard_phase2_results.py',
    'experiments/helpers/maniguard_phase2_audit.py',
    'experiments/helpers/maniguard_phase2_protocol.py',
    'experiments/helpers/maniguard_phase2_preparation_report.py',
    'experiments/helpers/maniguard_phase2_coordinator.py',
    'experiments/helpers/maniguard_phase2_services.py',
    'experiments/helpers/maniguard_runtime.py',
    'experiments/helpers/maniguard_source.py',
    'experiments/helpers/maniguard_adaptation.py',
    'experiments/helpers/maniguard_results.py',
    'experiments/helpers/maniguard_client.py',
    'experiments/helpers/maniguard_policy.py',
    'experiments/helpers/maniguard_policy_provenance.py',
    'experiments/helpers/maniguard_parallel_policy.py',
    'experiments/scripts/maniguard_phase2_episode.py',
    'experiments/scripts/maniguard_phase2_train.py',
    'experiments/scripts/maniguard_phase2_prepare.py',
    'experiments/scripts/maniguard_phase2_preparation_report.py',
    'experiments/scripts/maniguard_serve.py',
    'scripts/maniguard/serve.sh',
)
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
    names = [*INPUT_NAMES]
    if (config_dir / 'preparation_amendment.json').is_file():
        names.append('preparation_amendment')
    return {name: json.loads((config_dir / f'{name}.json').read_text()) for name in names}


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


def resource_envelope(jobs, preparation=None):
    """Action-bearing upper bounds; initialization/failed queries require separate metering."""
    scheduled_actions = sum(job['max_steps'] for job in jobs)
    preparation = preparation or _read_inputs(CONFIG)['preparation']
    preparation_actions = sum(job['full_horizon'] for job in preparation['full_jobs'])
    short_actions = sum(job['max_steps'] for job in preparation['short_jobs'])
    actions = scheduled_actions + preparation_actions + short_actions
    normal_queries = sum((job['max_steps'] + 3) // 4 for job in jobs) + (
        preparation_actions + short_actions) // 4
    return {
        'scheduled_full_episodes': len(jobs), 'scheduled_policy_actions_max': scheduled_actions,
        'preparation_full_episodes_max': len(preparation['full_jobs']),
        'preparation_policy_actions_max': preparation_actions,
        'preparation_action_basis': 'Sum of the frozen full-job family horizons and short-job caps.',
        'short_checks_max': len(preparation['short_jobs']), 'short_check_actions_max_each': 32,
        'short_policy_actions_max': short_actions,
        'all_full_episodes_max': len(jobs) + len(preparation['full_jobs']), 'policy_actions_max': actions,
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
        'estimate_scope': 'Prior continuous reservation plus projected pilot episode process time; '
                          'future idle and unmeasured policy lifecycle overhead are excluded. '
                          'This is a feasibility estimate, not a hard runtime bound. Service startup '
                          'and idle time still consume the enforced allocated-GPU ceiling.',
        'failure_rule': 'Count every attempted interaction and stop affected work on infrastructure '
                        'failure. No unbudgeted replacement episodes or free retries.',
    }


def build_pilot(config_dir=CONFIG, root=ROOT):
    """Prepare the default immutable allocations and explicit incomplete-freeze record."""
    config_dir, root = Path(config_dir), Path(root)
    inputs = _read_inputs(config_dir)
    registry = inputs['families']['families']
    allocation = effective_preparation(inputs['preparation'], inputs.get('preparation_amendment'), root=root)
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
                        'assignment_status': inputs['preparation']['status'],
                        'manifest': 'configs/maniguard/phase2/preparation.json',
                        'effective_allocation_sha256': allocation['sha256'],
                        'amendment_sha256': allocation['amendment_sha256'],
                        'selection_partition': 'train', 'integration_partition': 'engineering'},
        'freeze_provenance': None,
        'freeze_checks': {name: {'passed': False, 'evidence_path': None, 'sha256': None}
                          for name in REQUIRED_CHECKS},
        'input_files': {str((config_dir / f'{name}.json').relative_to(root)):
                        _sha(config_dir / f'{name}.json') for name in inputs},
        'implementation_files': {name: _sha(root / name) for name in IMPLEMENTATION_FILES},
    }
    jobs = compile_schedule(protocol)
    protocol['schedule_sha256'] = record_hash(jobs)
    protocol['resource_envelope'] = resource_envelope(jobs, inputs['preparation'])
    protocol['sha256'] = record_hash(protocol)
    return protocol


def _verify_file_map(files, root):
    root = Path(root).resolve()
    for relative, digest in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _sha(path) != digest:
            raise ValueError(f'Pinned protocol input/source changed: {relative}')


REVIEW_CHECKS = ('identity_equivalence', 'completion_semantics', 'runtime_execution_audit')
REVIEW_SCHEMA = 'maniguard-phase2-freeze-evidence-v1'


def _artifact(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Freeze evidence must be an existing artifact inside the repository root')
    return {'evidence_path': str(path.relative_to(root)), 'sha256': _sha(path)}


def _read_hashed_json(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != record_hash({key: item for key, item in value.items() if key != 'sha256'}):
        raise ValueError('Freeze evidence content hash mismatch: ' + str(path))
    return value


def _verify_sources(sources, root, *, within=None):
    if not isinstance(sources, list) or not sources:
        raise ValueError('Freeze evidence needs immutable source files')
    result = {}
    for source in sources:
        path = Path(source['path'])
        path = path.resolve() if path.is_absolute() else (Path(root) / path).resolve()
        if within is not None and not path.is_relative_to(Path(within).resolve()):
            raise ValueError('Preparation report source escapes its immutable report directory')
        reference = _artifact(path, root)
        if reference['sha256'] != source['sha256'] or str(path) in result:
            raise ValueError('Changed or duplicate freeze evidence source: ' + str(path))
        result[str(path)] = source['sha256']
    return result


def _validate_preparation(manifest, inputs):
    if manifest.get('sha256') != record_hash({key: value for key, value in manifest.items()
                                             if key != 'sha256'}):
        raise ValueError('Preparation assignment content hash mismatch')
    if (manifest['status'] != 'assignments_frozen_before_new_behavior' or
            manifest['families'] != list(FAMILIES) or len(manifest['short_jobs']) != 12 or
            len(manifest['full_jobs']) != 24):
        raise ValueError('Preparation assignments differ from the declared allocation')
    ids = set()
    for short, jobs in ((True, manifest['short_jobs']), (False, manifest['full_jobs'])):
        if Counter(job['family'] for job in jobs) != dict.fromkeys(FAMILIES, 4 if short else 8):
            raise ValueError('Preparation family exposure changed')
        for job in jobs:
            family, purpose = job['family'], 'engineering' if short else 'train'
            scene, condition = job['scene'].split('/')
            if (job['job_id'] in ids or job['purpose'] != purpose or condition != 'base' or
                    scene not in inputs['splits']['families'][family][purpose]):
                raise ValueError('Preparation assignment is duplicate or outside its declared split')
            ids.add(job['job_id'])
            if short:
                if type(job['max_steps']) is not int or not 1 <= job['max_steps'] <= 32:
                    raise ValueError('Preparation integration check horizon is invalid')
            elif (job['max_steps'] is not None or
                  job['full_horizon'] != inputs['families']['families'][family]['max_steps']):
                raise ValueError('Preparation full episode horizon was shortened')
    actions = sum(job['full_horizon'] for job in manifest['full_jobs']) + sum(
        job['max_steps'] for job in manifest['short_jobs'])
    if actions != manifest['budget']['policy_action_ceiling'] or actions != 58784:
        raise ValueError('Preparation action allocation differs from its declared cap')


AMENDMENT_SCHEMA = 'maniguard-phase2-preparation-amendment-v1'
EFFECTIVE_SCHEMA = 'maniguard-phase2-effective-preparation-v1'
RECOVERY_FAILED = 'short-clutter-identity'
RECOVERY_RETIRED = 'short-clutter-composition'
RECOVERY_REPLACEMENT = 'short-clutter-identity-recovery'
RECOVERY_OMISSION = 'Clutter composition engineering check; Jar and Stack composition coverage retained'


def _bound_artifact(reference, root):
    path = Path(reference['path'])
    path = path.resolve() if path.is_absolute() else (Path(root) / path).resolve()
    if _artifact(path, root)['sha256'] != reference['sha256']:
        raise ValueError('Preparation amendment evidence file changed: ' + str(path))
    return path, _read_hashed_json(path)


def effective_preparation(base, amendment=None, *, root=ROOT):
    """Validate one prospective paid recovery and return a separate derived allocation.

    The original manifest and all its existing receipts retain their original hashes.
    The failed initialization remains an invalid paid attempt; only its source-reconciled
    counters can resolve accounting. Exactly one still-unclaimed composition slot may
    become deterministic identity recovery, preserving all counts, caps and full jobs.
    No amendment or artifact is written by this function.
    """
    from helpers.maniguard_phase2_services import PRIOR_COUNTS

    if (base.get('schema') != 'maniguard-phase2-preparation-v1' or
            base.get('sha256') != record_hash({key: value for key, value in base.items()
                                               if key != 'sha256'})):
        raise ValueError('Original preparation manifest schema/hash mismatch')
    allocation = {'schema': EFFECTIVE_SCHEMA, 'base_preparation_sha256': base['sha256'],
                  'amendment_sha256': None, 'amendment_created_at': None,
                  'short_jobs': deepcopy(base['short_jobs']), 'full_jobs': deepcopy(base['full_jobs']),
                  'budget': deepcopy(base['budget']), 'families': deepcopy(base['families']),
                  'reference_selection': deepcopy(base['reference_selection']),
                  'retired_job_ids': [], 'resolved_failures': {}, 'omitted_checks': [],
                  'preparation_output': None, 'pre_recovery_receipts': {}}
    if amendment is not None:
        if (amendment.get('schema') != AMENDMENT_SCHEMA or
                amendment.get('sha256') != record_hash({key: value for key, value in amendment.items()
                                                        if key != 'sha256'}) or
                amendment.get('base_preparation_sha256') != base['sha256']):
            raise ValueError('Preparation amendment is not bound to the immutable original manifest')
        expected_keys = {'schema', 'created_at', 'base_preparation_sha256', 'failed_job_id',
                         'retired_unspent_job_id', 'replacement_job', 'budget', 'pre_recovery_report',
                         'failure_accounting', 'omitted_check', 'sha256'}
        if set(amendment) != expected_keys:
            raise ValueError('Unexpected preparation amendment field/domain')
        created = amendment['created_at']
        if type(created) not in (int, float) or not math.isfinite(created) or created <= 0:
            raise ValueError('Preparation amendment needs a finite prospective creation time')
        jobs = {job['job_id']: job for job in base['short_jobs']}
        replacement = {**jobs[RECOVERY_FAILED], 'job_id': RECOVERY_REPLACEMENT}
        if (amendment['failed_job_id'] != RECOVERY_FAILED or
                amendment['retired_unspent_job_id'] != RECOVERY_RETIRED or
                amendment['replacement_job'] != replacement or amendment['budget'] != base['budget'] or
                amendment['omitted_check'] != RECOVERY_OMISSION):
            raise ValueError('Only the declared unused Clutter slot can become identical-setting recovery')
        if (base['budget']['short_attempts'] != 12 or base['budget']['short_actions_per_attempt'] != 32 or
                base['budget']['integration_gpu_hour_ceiling'] != 4 or len(base['full_jobs']) != 24):
            raise ValueError('Recovery cannot increase the preparation allocation')
        report_path, before = _bound_artifact(amendment['pre_recovery_report'], root)
        if (before.get('schema') != 'maniguard-phase2-preparation-report-v1' or
                before.get('pending_receipts') or before.get('pending_setup_lifecycle') or
                created < before['timestamp']):
            raise ValueError('Recovery must follow a terminal immutable preparation checkpoint')
        prior_sources = _verify_sources(before['sources'], root, within=report_path.parent)
        original_path = report_path.parent / 'inputs/preparation/preparation.json'
        if str(original_path) not in prior_sources or json.loads(original_path.read_text()) != base:
            raise ValueError('Pre-recovery report is not bound to the original preparation manifest')
        prior_rows = {row['job_id']: row for row in before['terminal_attempts']}
        if (len(prior_rows) != len(before['terminal_attempts']) or RECOVERY_FAILED not in prior_rows or
                prior_rows[RECOVERY_FAILED]['valid'] is not False or
                any(row['valid'] is not True for job_id, row in prior_rows.items()
                    if job_id != RECOVERY_FAILED) or
                RECOVERY_RETIRED in prior_rows or
                RECOVERY_RETIRED not in before['absent_or_nonterminal_jobs']):
            raise ValueError('Recovery checkpoint must retain the one failure and an unused replacement slot')
        audit_path, audit = _bound_artifact(amendment['failure_accounting'], root)
        if (audit.get('schema') != 'maniguard-phase2-terminal-cost-audit-v1' or
                audit.get('status') != 'source_reconciled_terminal_counts' or
                audit.get('job_id') != RECOVERY_FAILED or created < audit['created_at'] or
                audit.get('source_audit', {}).get('valid') is not True):
            raise ValueError('Recovery needs an exact source-backed terminal failure accounting audit')
        audit_sources = _verify_sources(audit['sources'], root, within=audit_path.parent)
        receipt_ref = audit['subject']['receipt']
        receipt_path = Path(receipt_ref['path']).resolve()
        if audit_sources.get(str(receipt_path)) != receipt_ref['sha256']:
            raise ValueError('Failure accounting does not bind its immutable original receipt')
        receipt = json.loads(receipt_path.read_text())
        prior_receipt_path = report_path.parent / 'inputs/preparation/attempts' / (RECOVERY_FAILED + '.json')
        if (str(prior_receipt_path) not in prior_sources or
                record_hash(json.loads(prior_receipt_path.read_text())) != record_hash(receipt) or
                receipt.get('status') != 'invalid' or receipt.get('job') != jobs[RECOVERY_FAILED] or
                receipt.get('output') != audit['output'] or
                receipt.get('protocol_sha256') != base['sha256']):
            raise ValueError('Accounting must preserve the same invalid failed physical attempt')
        counts = audit.get('exact_counts', {})
        if set(counts) != set(PRIOR_COUNTS) or any(type(value) is not int or value < 0
                                                  for value in counts.values()):
            raise ValueError('Failure accounting requires every exact attempted-work counter')
        if any(value != 0 for key, value in counts.items() if key != 'physics_steps'):
            raise ValueError('Recovery exception is restricted to initialization before controller/inference work')
        if counts['physics_steps'] != receipt.get('ledger', {}).get('physics_steps'):
            raise ValueError('Audited failure physics must equal the preserved terminal ledger')
        preparation_output = str(Path(audit['output']).resolve().parent)
        all_jobs = {job['job_id']: job for job in base['short_jobs'] + base['full_jobs']}
        prior_receipts = {}
        for job_id in prior_rows:
            path = report_path.parent / 'inputs/preparation/attempts' / (job_id + '.json')
            if str(path) not in prior_sources:
                raise ValueError('Recovery checkpoint lacks a bound historical receipt: ' + job_id)
            historical = json.loads(path.read_text())
            if (historical.get('job') != all_jobs.get(job_id) or
                    historical.get('protocol_sha256') != base['sha256'] or
                    str(Path(historical['output']).resolve().parent) != preparation_output or
                    historical.get('status') not in ('valid', 'invalid')):
                raise ValueError('Historical recovery receipt differs from its original assignment')
            prior_receipts[job_id] = record_hash(historical)
        allocation['short_jobs'] = [replacement if job['job_id'] == RECOVERY_RETIRED else job
                                    for job in allocation['short_jobs']]
        allocation.update(amendment_sha256=amendment['sha256'], amendment_created_at=created,
                          retired_job_ids=[RECOVERY_RETIRED], omitted_checks=[RECOVERY_OMISSION],
                          preparation_output=preparation_output, pre_recovery_receipts=prior_receipts,
                          resolved_failures={RECOVERY_FAILED: {
                              'replacement_job_id': RECOVERY_REPLACEMENT,
                              'exact_counts': deepcopy(counts),
                              'failure_accounting': deepcopy(amendment['failure_accounting']),
                              'failed_output': audit['output'],
                              'failed_receipt_content_sha256': record_hash(receipt)}})
    allocation['sha256'] = record_hash(allocation)
    return allocation


def validate_preparation_receipt(receipt, allocation):
    """Bind original and amended receipts without rewriting historical protocol hashes."""
    jobs = {job['job_id']: job for job in allocation['short_jobs'] + allocation['full_jobs']}
    job = receipt.get('job', {})
    job_id = job.get('job_id')
    if (receipt.get('protocol_sha256') != allocation['base_preparation_sha256'] or
            jobs.get(job_id) != job):
        raise ValueError('Receipt differs from its effective preparation assignment')
    amendment = receipt.get('amendment_sha256')
    if amendment is not None and amendment != allocation['amendment_sha256']:
        raise ValueError('Receipt amendment hash differs from the effective allocation')
    replacements = {value['replacement_job_id'] for value in allocation['resolved_failures'].values()}
    if job_id in replacements and amendment != allocation['amendment_sha256']:
        raise ValueError('Replacement receipt lacks its prospective amendment hash')
    if amendment is not None:
        started = receipt.get('started_at')
        if (type(started) not in (int, float) or not math.isfinite(started) or
                started < allocation['amendment_created_at']):
            raise ValueError('Amended receipt was claimed before amendment creation')
    if job_id in allocation['resolved_failures']:
        failed = allocation['resolved_failures'][job_id]
        if (receipt.get('status') != 'invalid' or receipt.get('output') != failed['failed_output'] or
                record_hash(receipt) != failed['failed_receipt_content_sha256']):
            raise ValueError('Resolved infrastructure failure must remain the original invalid paid attempt')
    return True


def build_preparation_amendment(base, pre_recovery_report, failure_accounting, *, created_at, root=ROOT):
    """Return the narrowly scoped prospective amendment without writing or launching it."""
    failed = next(job for job in base['short_jobs'] if job['job_id'] == RECOVERY_FAILED)
    evidence = []
    for path in (pre_recovery_report, failure_accounting):
        reference = _artifact(path, root)
        evidence.append({'path': str(Path(path).resolve()), 'sha256': reference['sha256']})
    amendment = {'schema': AMENDMENT_SCHEMA, 'created_at': created_at,
                 'base_preparation_sha256': base['sha256'], 'failed_job_id': RECOVERY_FAILED,
                 'retired_unspent_job_id': RECOVERY_RETIRED,
                 'replacement_job': {**failed, 'job_id': RECOVERY_REPLACEMENT},
                 'budget': deepcopy(base['budget']), 'pre_recovery_report': evidence[0],
                 'failure_accounting': evidence[1], 'omitted_check': RECOVERY_OMISSION}
    amendment['sha256'] = record_hash(amendment)
    effective_preparation(base, amendment, root=root)
    return amendment


def _recovery_snapshot_evidence(amendment_path, snapshots, sources, root):
    """Bind report-local recovery copies to every retained immutable original."""
    expected = {}

    def visit(path, digest=None):
        path = Path(path)
        path = path.resolve() if path.is_absolute() else (Path(root) / path).resolve()
        actual = _artifact(path, root)['sha256']
        if digest is not None and actual != digest:
            raise ValueError('Original recovery evidence changed: ' + str(path))
        if str(path) in expected:
            return
        expected[str(path)] = actual
        try:
            value = json.loads(path.read_bytes())
        except (ValueError, UnicodeDecodeError):
            return
        if isinstance(value, dict):
            for reference in value.get('sources', []):
                visit(reference['path'], reference['sha256'])
            if value.get('schema') == AMENDMENT_SCHEMA:
                for key in ('pre_recovery_report', 'failure_accounting'):
                    visit(value[key]['path'], value[key]['sha256'])

    if amendment_path is not None:
        visit(amendment_path)
    if set(snapshots) != set(expected):
        raise ValueError('Recovery snapshot inventory differs from its immutable evidence chain')
    for original, digest in expected.items():
        copy = snapshots[original]
        if copy.get('sha256') != digest or sources.get(copy.get('path')) != digest:
            raise ValueError('Recovery snapshot differs from its preserved original: ' + original)


def _preparation_counts(allocation, measured, receipts, jobs, captured):
    """Reproduce raw costs and the single explicit failure overlay from saved rows."""
    from helpers.maniguard_phase2_services import PRIOR_COUNTS

    counts, coverage, unknown, reconciliations = Counter(), Counter(), [], []
    count = lambda value: type(value) is int and value >= 0
    for job_id in sorted(jobs, key=lambda name: name + '.json'):
        job, row, receipt = jobs[job_id], measured[job_id], receipts[job_id]
        baseline = job['baseline_executor']
        complete = row.get('ledger', receipt.get('ledger', {})).get('phase') == 'completed'
        for key in PRIOR_COUNTS:
            value = row.get('costs', {}).get(key, 0 if key == 'actor_rpc_attempts' else None)
            absent = baseline and key in ('encoder_image_calls', 'encoder_images',
                                          'encoder_text_calls', 'adapter_decisions')
            if absent:
                value = 0
            if count(value):
                counts[key] += value
                coverage[key] += 1
                if not complete and key != 'actor_rpc_attempts' and not absent:
                    unknown.append({'job_id': job_id, 'count': key,
                                    'reason': 'incomplete ledger is a lower bound'})
                if row.get('cost_provenance', {}).get(key) == 'trace_lower_bound':
                    unknown.append({'job_id': job_id, 'count': key, 'reason': 'trace lower bound'})
            else:
                unknown.append({'job_id': job_id, 'count': key})
        resolution = allocation['resolved_failures'].get(job_id)
        if resolution is not None:
            raw = {key: row.get('costs', {}).get(key, 0 if key == 'actor_rpc_attempts' else None)
                   for key in PRIOR_COUNTS}
            exact = resolution['exact_counts']
            if row['valid'] is not False or any(value is not None and value != exact[key]
                                               for key, value in raw.items()):
                raise ValueError('Recovery costs disagree with the original invalid episode')
            reconciliations.append({'job_id': job_id, 'output': receipt['output'],
                                    'original_episode_valid': False, 'raw_costs': raw,
                                    'exact_counts': exact,
                                    'failure_accounting': resolution['failure_accounting']})
    smoke = captured('inputs/setup/encoder-smoke.json')
    failure = captured('inputs/setup/encoder-loader-failure-v1.json')
    if smoke.get('finite') is not True or smoke.get('trainable_parameters') != 0:
        raise ValueError('Encoder preparation smoke lacks finite/frozen evidence')
    setup = Counter()
    for key in PRIOR_COUNTS:
        values = (smoke.get('costs', {}).get(key, 0), failure.get('counts', {}).get(key))
        if not all(count(value) for value in values):
            raise ValueError('Setup accounting lacks every exact attempted-work counter')
        setup[key] = sum(values)
    if {key: smoke.get('costs', {}).get(key) for key in (
            'encoder_image_calls', 'encoder_images', 'encoder_text_calls')} != {
                'encoder_image_calls': 5, 'encoder_images': 10, 'encoder_text_calls': 1}:
        raise ValueError('Encoder smoke diagnostic interaction allocation changed')
    reconciled, raw_unknown = counts.copy(), list(unknown)
    for resolution in reconciliations:
        for key, exact in resolution['exact_counts'].items():
            raw = resolution['raw_costs'][key]
            reconciled[key] += exact - (raw if count(raw) else 0)
        unknown = [row for row in unknown if row['job_id'] != resolution['job_id']]
    if unknown:
        raise ValueError('Preparation has unresolved lower-bound or unknown costs')
    return {'episode_counts_known': dict(counts), 'episode_count_coverage': dict(coverage),
            'episode_counts_reconciled': dict(reconciled), 'setup_counts_known': dict(setup),
            'raw_total_counts_known': {key: counts[key] + setup[key] for key in PRIOR_COUNTS},
            'total_counts_known': {key: reconciled[key] + setup[key] for key in PRIOR_COUNTS},
            'raw_unknown_counts': raw_unknown, 'unknown_counts': unknown,
            'cost_reconciliations': reconciliations}


def _preparation_evidence(report_path, draft, config_dir, root):
    # Local imports avoid the report -> protocol -> report cycle at module import time.
    from helpers.maniguard_phase2_audit import audit_preparation
    from helpers.maniguard_phase2_preparation_report import (
        activation_diagnostic,
        audit_process,
        integration_coverage,
        select_references,
        throughput_estimate,
    )
    from helpers.maniguard_phase2_services import PRIOR_SCHEMA, ReservationBudget

    report_path, root = Path(report_path).resolve(), Path(root).resolve()
    _artifact(report_path, root)
    report = _read_hashed_json(report_path)
    if report.get('schema') != 'maniguard-phase2-preparation-report-v1':
        raise ValueError('Unsupported preparation report schema')
    if (not all(report.get(key) is True for key in (
            'preparation_complete', 'prior_accounting_ready', 'ready_for_freeze_review')) or
            any(report.get(key) for key in ('invalid_reasons', 'absent_or_nonterminal_jobs',
                                            'pending_receipts', 'pending_setup_lifecycle', 'unknown_counts',
                                            'count_totals_are_lower_bounds'))):
        raise ValueError('Preparation report is incomplete or contains invalid/unknown work')
    directory = report_path.parent
    sources = _verify_sources(report['sources'], root, within=directory)

    def captured(relative):
        path = directory / relative
        if str(path) not in sources:
            raise ValueError('Preparation report does not bind required snapshot: ' + relative)
        return json.loads(path.read_text())

    inputs = _read_inputs(config_dir)
    manifest = captured('inputs/preparation/preparation.json')
    _validate_preparation(manifest, inputs)
    if record_hash(manifest) != record_hash(inputs['preparation']):
        raise ValueError('Reported preparation assignments differ from current frozen assignments')
    amendment = inputs.get('preparation_amendment')
    amendment_path = None
    if amendment is not None:
        if captured('inputs/preparation/preparation_amendment.json') != amendment:
            raise ValueError('Reported recovery amendment differs from its frozen input')
        amendment_path = Path(config_dir) / 'preparation_amendment.json'
    allocation = effective_preparation(manifest, amendment, root=root)
    if (report.get('effective_preparation') != allocation or
            draft['preparation']['effective_allocation_sha256'] != allocation['sha256']):
        raise ValueError('Preparation report effective allocation differs from the frozen assignment')
    _recovery_snapshot_evidence(amendment_path, report.get('recovery_snapshots', {}), sources, root)
    saved_pilot = captured('inputs/pilot.json')
    if saved_pilot.get('sha256') != record_hash({key: value for key, value in saved_pilot.items()
                                               if key != 'sha256'}):
        raise ValueError('Captured pilot content hash mismatch')
    # Report diagnostics must use the same model/search allocation; sources are pinned
    # separately below because execution may legitimately have preceded audit-only fixes.
    for key in ('model', 'search_config', 'ppo_config', 'initialization', 'training_panels',
                'validation_cases', 'optimizer_seeds', 'schedule_sha256'):
        if record_hash(saved_pilot.get(key)) != record_hash(draft[key]):
            raise ValueError('Preparation report used a different pilot setting: ' + key)
    audit_code = [name for name in draft['implementation_files']
                  if name.startswith('experiments/helpers/maniguard_phase2_')]
    audit_code += ['experiments/scripts/maniguard_phase2_prepare.py',
                   'experiments/scripts/maniguard_phase2_preparation_report.py']
    for relative in audit_code:
        path = directory / 'audit_source' / relative
        if sources.get(str(path)) != draft['implementation_files'][relative]:
            raise ValueError('Preparation/selection audit implementation changed: ' + relative)
    required = {job['job_id']: job for job in allocation['short_jobs'] + allocation['full_jobs']}
    terminal = report['terminal_attempts']
    if len(terminal) != len(required) or {row['job_id'] for row in terminal} != set(required):
        raise ValueError('Preparation report lacks a complete valid allocation/cost audit')
    measured, receipts, reproduced_terminal, engineering = {}, {}, [], []
    for job_id in sorted(required, key=lambda name: name + '.json'):
        job = required[job_id]
        row = captured('audited_episodes/' + job_id + '.json')
        receipt = captured('inputs/preparation/attempts/' + job_id + '.json')
        validate_preparation_receipt(receipt, allocation)
        original = allocation['pre_recovery_receipts'].get(job_id)
        if original is not None and record_hash(receipt) != original:
            raise ValueError('Pre-recovery receipt was changed or replaced: ' + job_id)
        recorded_job = captured('inputs/jobs/' + job_id + '.json')
        binding = ('family', 'scene', 'purpose', 'seed', 'sampling_seed', 'mode',
                   'fixed_choices', 'stochastic')
        if (any(recorded_job.get(key) != job.get(key) for key in binding) or
                recorded_job.get('max_steps_override') != job['max_steps'] or
                (job['baseline_executor'] and recorded_job.get('phase') != 'phase2_baseline_check')):
            raise ValueError('Captured episode job differs from its preparation assignment')
        process = audit_process(receipt)
        if row.get('process_audit') != process:
            raise ValueError('Captured process audit does not reproduce from its receipt')
        expected_valid = job_id not in allocation['resolved_failures']
        if row.get('valid') is not expected_valid or (expected_valid and not process['valid']):
            raise ValueError('Preparation requires valid coverage except the declared paid failure')
        measured[job_id], receipts[job_id] = row, receipt
        reproduced_terminal.append({
            'job_id': job_id, 'family': job['family'], 'valid': row['valid'],
            'receipt_status': receipt['status'], 'audit_history': receipt.get('audit_history', []),
            'process_audit': process, 'short_check': receipt['short_check'],
            'baseline_executor': job['baseline_executor'], 'steps': row.get('steps'),
            'elapsed_seconds': receipt.get('elapsed_seconds'),
            'invalid_reasons': row.get('invalid_reasons', [])})
        if row['valid'] and job['purpose'] == 'engineering' and not job['baseline_executor']:
            engineering.append({'job_id': job_id,
                                'features': [decision['features'] for decision in row['decisions']]})
    if terminal != reproduced_terminal:
        raise ValueError('Preparation terminal summary differs from captured receipt/episode evidence')
    accounting = audit_preparation(directory / 'inputs/preparation', receipt_ledgers_only=True,
                                   allocation=allocation)
    if (accounting != report['accounting'] or accounting.get('valid') is not True or
            accounting.get('within_all_declared_ceilings') is not True):
        raise ValueError('Preparation report lacks a complete valid allocation/cost audit')
    counts = _preparation_counts(allocation, measured, receipts, required, captured)
    for key, value in counts.items():
        if report.get(key) != value:
            raise ValueError('Preparation cost accounting does not reproduce: ' + key)
    coverage = integration_coverage(allocation, measured, counts['cost_reconciliations'])
    if coverage != report.get('integration_coverage') or not coverage['passed']:
        raise ValueError('Preparation integration coverage does not reproduce')
    if captured('engineering-features.json') != engineering:
        raise ValueError('Engineering features differ from valid captured episode decisions')
    selection = select_references(manifest, measured)
    if not selection['ready'] or record_hash(selection) != record_hash(report['reference_selection']):
        raise ValueError('Reference selection does not reproduce from the captured training cases')
    if selection['contextual_threshold_changed'] or selection['contextual'] != manifest[
            'reference_selection']['contextual']:
        raise ValueError('The prespecified contextual reference cannot be tuned during freeze')
    activation = activation_diagnostic(draft, captured('engineering-features.json'))
    if not activation['passed'] or record_hash(activation) != record_hash(report['activation']):
        raise ValueError('Feature/controller activation evidence does not reproduce')
    throughput = throughput_estimate(draft, terminal, report['reservation'])
    if (record_hash(throughput) != record_hash(report['throughput']) or
            throughput.get('ready_for_freeze_review') is not True or
            throughput.get('all_families_have_full_measurements') is not True or
            throughput.get('ceiling_changed') is not False):
        raise ValueError('Measured costing needs complete reproducible full-family evidence')
    estimate = throughput['total_gpu_hours_estimate']
    if (type(estimate) not in (int, float) or not math.isfinite(estimate) or estimate <= 0 or
            estimate > draft['resource_envelope']['gpu_hours_ceiling']):
        raise ValueError('Measured preparation/pilot estimate does not fit the fixed GPU ceiling')
    prior_path = directory / 'prior-accounting.json'
    prior = _read_hashed_json(prior_path)
    if (prior.get('schema') != PRIOR_SCHEMA or prior.get('counts') != report['total_counts_known'] or
            any(prior.get(key) != value for key, value in report['reservation'].items())):
        raise ValueError('Prior accounting differs from the immutable preparation report')
    ReservationBudget(prior, draft['resource_envelope'],
                      reserved_gpus=report['reservation']['reserved_gpus'],
                      wall=lambda: report['timestamp']).check()
    prior_sources = _verify_sources(prior['sources'], root, within=directory)
    if prior_sources != {**sources, str(report_path): _sha(report_path)}:
        raise ValueError('Prior accounting does not bind exactly the report and its evidence')
    return report, selection, throughput, _artifact(prior_path, root)


def _review_evidence(path, name, report, draft, root):
    evidence = _read_hashed_json(path)
    if (evidence.get('schema') != REVIEW_SCHEMA or evidence.get('check') != name or
            evidence.get('passed') is not True or
            evidence.get('preparation_report_sha256') != report['sha256'] or
            evidence.get('input_files') != draft['input_files'] or
            evidence.get('implementation_files') != draft['implementation_files']):
        raise ValueError('Readiness review is not bound to this report and implementation: ' + name)
    _verify_sources(evidence.get('sources'), root)
    findings = evidence.get('findings', {})
    required = {
        'identity_equivalence': {'same_frame_transformation_preserved': True,
                                 'eight_action_query_cadence_preserved': True,
                                 'trajectory_identity_claimed': False},
        'completion_semantics': {'requested_sequence_checklists_reviewed':
                                 sorted(json.loads((Path(root) / next(name for name in draft['input_files']
                                     if name.endswith('/completion.json'))).read_text())['families']),
                                 'benchmark_monitor_unchanged': True,
                                 'completion_signals_excluded_from_actor_critic': True},
        'runtime_execution_audit': {'tests_passed': True,
                                    'fresh_process_and_resume_audited': True,
                                    'cost_and_invalid_result_paths_audited': True},
    }[name]
    if any(findings.get(key) != value for key, value in required.items()):
        raise ValueError('Readiness review lacks required scoped findings: ' + name)
    return {'passed': True, **_artifact(path, root)}


def _derive_frozen(draft, report_path, reviews, config_dir, root):
    if set(reviews) != set(REVIEW_CHECKS):
        raise ValueError('Freeze requires explicit identity, completion and runtime review evidence')
    report, selection, throughput, prior = _preparation_evidence(report_path, draft, config_dir, root)
    frozen = deepcopy(draft)
    reference = _artifact(report_path, root)
    for name in ('preparation_plan', 'measured_costing', 'feature_contract', 'reference_selection'):
        frozen['freeze_checks'][name] = {'passed': True, **reference}
    for name, path in reviews.items():
        frozen['freeze_checks'][name] = _review_evidence(path, name, report, draft, root)
    chosen = selection['fixed']
    fixed = frozen['references'][1]
    fixed.update(selected_choices=chosen['choices'], selected_candidate=chosen['name'],
                 selection='Reproduced equal-family training-only preparation comparison; '
                           'U descending, V ascending, attempted queries ascending, name ascending.')
    contextual = frozen['references'][2]
    contextual.update(selected_rule=deepcopy(contextual['proposed_rule']),
                      selection='Prespecified preparation rule; threshold unchanged.')
    frozen['objective']['completion_checklists'] = 'source_and_preparation_audited'
    frozen['resource_envelope'].update(
        gpu_hours_status='frozen_after_preparation_costing',
        measured_wall_hours_estimate=throughput['total_gpu_hours_estimate'] /
                                     report['reservation']['reserved_gpus'])
    frozen['freeze_provenance'] = {'preparation_report': reference,
                                   'preparation_report_content_sha256': report['sha256'],
                                   'prior_accounting': prior,
                                   'review_evidence': {name: _artifact(path, root)
                                                       for name, path in reviews.items()}}
    frozen['status'] = 'frozen'
    frozen['sha256'] = record_hash({key: value for key, value in frozen.items() if key != 'sha256'})
    return frozen


def freeze_pilot(report_path, review_evidence, config_dir=CONFIG, root=ROOT, *, verify_upstream=True):
    """Return a reviewable frozen candidate; never write a config or launch work.

    The completed immutable preparation report derives the fixed comparator, untouched
    contextual rule and measured estimate. Three separate content-hashed JSON reviews
    (REVIEW_SCHEMA) must supply scoped identity/completion/runtime findings, hashed sources,
    this report's content hash, and the draft input_files/implementation_files maps. Hashes
    establish integrity, not external authentication. No caller-supplied optimizer changes
    or arbitrary reference choices are accepted. Future service/idle overhead remains an
    explicit forecast limitation and consumes the unchanged enforced GPU reservation cap.
    """
    draft = build_pilot(config_dir, root)
    frozen = _derive_frozen(draft, report_path, review_evidence, config_dir, root)
    validate_protocol(frozen, config_dir, root, require_frozen=True, verify_upstream=verify_upstream)
    return frozen


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
    mutable = {'sha256', 'status', 'freeze_checks', 'resource_envelope', 'freeze_provenance'}
    if protocol['status'] == 'frozen':
        mutable.update(('references', 'objective'))
    for key, value in expected.items():
        if key not in mutable and record_hash(protocol.get(key)) != record_hash(value):
            raise ValueError(f'Frozen pilot field changed: {key}')
    jobs = compile_schedule(protocol)
    if record_hash(jobs) != protocol['schedule_sha256']:
        raise ValueError('Compiled episode schedule hash mismatch')
    _validate_preparation(_read_inputs(config_dir)['preparation'], _read_inputs(config_dir))
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
    frozen_envelope = resource_envelope(jobs, _read_inputs(config_dir)['preparation'])
    for field, value in frozen_envelope.items():
        if field in ('gpu_hours_status', 'measured_wall_hours_estimate'):
            continue
        if envelope.get(field) != value:
            raise ValueError(f'Resource bound changed: {field}')
    if (envelope['scheduled_policy_actions_max'] != 1270200 or
            envelope['policy_actions_max'] != 1328984):
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
    if protocol['status'] == 'frozen':
        provenance = protocol.get('freeze_provenance')
        if not isinstance(provenance, dict):
            raise ValueError('Frozen pilot requires evidence-derived freeze provenance')
        evidence = [provenance['preparation_report'], provenance['prior_accounting'],
                    *provenance['review_evidence'].values()]
        for item in evidence:
            _verify_file_map({item['evidence_path']: item['sha256']}, root)
        derived = _derive_frozen(expected, Path(root) / provenance['preparation_report']['evidence_path'],
                                {name: Path(root) / value['evidence_path'] for name, value in
                                 provenance['review_evidence'].items()}, config_dir, root)
        if record_hash(protocol) != record_hash(derived):
            raise ValueError('Frozen pilot differs from its reproducible preparation/review evidence')
    elif protocol.get('freeze_provenance') is not None:
        raise ValueError('An incomplete pilot cannot carry completed freeze provenance')
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
