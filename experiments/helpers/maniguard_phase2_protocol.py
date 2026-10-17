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
    return {name: json.loads((config_dir / f'{name}.json').read_text())
            for name in INPUT_NAMES}


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


def _preparation_evidence(report_path, draft, config_dir, root):
    # Local imports avoid the report -> protocol -> report cycle at module import time.
    from helpers.maniguard_phase2_preparation_report import (
        activation_diagnostic,
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
    required = {job['job_id']: job for job in manifest['short_jobs'] + manifest['full_jobs']}
    terminal = report['terminal_attempts']
    if (len(terminal) != len(required) or {row['job_id'] for row in terminal} != set(required) or
            not all(row.get('valid') is True for row in terminal) or
            report['accounting'].get('valid') is not True or
            report['accounting'].get('within_all_declared_ceilings') is not True):
        raise ValueError('Preparation report lacks a complete valid allocation/cost audit')
    measured = {job_id: captured('audited_episodes/' + job_id + '.json') for job_id in required}
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
