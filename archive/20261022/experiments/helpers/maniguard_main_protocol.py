"""Pure, explicit main/transfer evaluation planning; no launcher or outcome access.

Callers supply concrete post-pilot cases, selected models, analyses and costs. There
are no sample-size, safety-margin or GPU-budget defaults. A draft cannot authorize
an episode. Even a frozen bundle is useful only after the live launcher calls
``authorize_job`` for every exact job, archives its bindings and meters all attempts.
Content hashes establish integrity, not independent authentication of review claims.
"""

import hashlib
import json
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

from helpers.maniguard_phase2_aac import load_config as load_aac_config
from helpers.maniguard_phase2_families import CONFIG, ROOT
from helpers.maniguard_phase2_model import genome_sha256, load_model
from helpers.maniguard_phase2_optim import record_hash

SCHEMA = 'maniguard-main-evaluation-v1'
SELECTION_SCHEMA = 'maniguard-main-selected-actor-v1'
REVIEW_SCHEMA = 'maniguard-main-freeze-review-v1'
INPUTS = ('families', 'splits', 'sources', 'encoder', 'completion')
REQUIRED_REVIEWS = {
    'selection_provenance': ('one_actor_per_run', 'validation_only_selection', 'training_runs_independent',
                             'transfer_families_excluded_from_optimization'),
    'exposure': ('exact_cases_not_used_for_selection', 'reserved_families_not_used_for_design',
                 'lid_design_exposure_disclosed'),
    'cost_precision': ('development_only_precision_basis', 'sample_sizes_and_risk_margin_justified',
                       'all_prior_and_projected_costs_included'),
    'runtime_binding': ('exact_job_gate_integrated', 'attempts_durable_and_metered',
                        'checkpoint_and_preprocessing_verified'),
    'analysis_completion': ('analysis_implementation_reviewed', 'completion_semantics_reviewed',
                            'scene_and_run_dependence_retained'),
}
SPEC_KEYS = {'study_id', 'cases', 'training_runs', 'selected_actors', 'comparators',
             'preprocessing_files', 'implementation_files', 'deployment', 'analysis', 'budget'}
WORK_COUNTS = ('policy_control_steps', 'policy_queries', 'physics_steps', 'encoder_image_calls',
               'encoder_images', 'encoder_text_calls', 'encoder_image_attempts',
               'encoder_text_attempts', 'encoder_load_attempts', 'adapter_decisions', 'actor_rpc_attempts')
BUDGET_KEYS = {name: ('policy_actions_max' if name == 'policy_control_steps' else name + '_max')
               for name in WORK_COUNTS}


def _signed(value):
    value = deepcopy(value)
    value['sha256'] = record_hash({key: item for key, item in value.items() if key != 'sha256'})
    return value


def _check_signed(value):
    if value.get('sha256') != _signed(value)['sha256']:
        raise ValueError('Content hash mismatch')


def _file(reference, root):
    if set(reference) != {'path', 'sha256'}:
        raise ValueError('A source binding requires exactly path and sha256')
    path = Path(reference['path'])
    path = path.resolve() if path.is_absolute() else (Path(root) / path).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('Bound source must be an existing file inside the study root')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != reference['sha256']:
        raise ValueError('Bound source changed: ' + str(path))
    return path


def source_reference(path, *, root=ROOT):
    """Return a relative byte-hash binding; read only the requested artifact."""
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact is outside the study root')
    return {'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _record(reference, root):
    value = json.loads(_file(reference, root).read_text())
    _check_signed(value)
    return value


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
        raise ValueError('Study, method and run identifiers must be simple path components')
    return value


def _integer(value):
    return type(value) is int and value >= 0


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def controller_resource_contract(profile, *, choices=None):
    """Return a supported implementation contract, never an empirical cost estimate.

    Image/text counts refer to the supplementary frozen CLIP encoder. The VLA's
    internal encoder/denoising work is paid through every policy RPC and its timing.
    The benchmark_original comparator retains native eight-action execution without
    supplementary CLIP or adapter decisions. Planning profiles do not authorize
    their live execution; each backend needs its own reviewed integration.
    """
    profiles = {'phase2_learned', 'phase2_identity', 'phase2_fixed', 'phase2_contextual',
                'aac_joint20', 'aac_fixed8_joint20', 'benchmark_original'}
    if profile not in profiles:
        raise ValueError('Unsupported controller resource profile')
    if profile == 'phase2_fixed':
        if (not isinstance(choices, list) or len(choices) != 4 or
                any(type(value) is not int or not 0 <= value <= 2 for value in choices)):
            raise ValueError('Fixed resource profile requires four supported choice indices')
        interval = (8, 4, 16)[choices[0]]
    else:
        if choices is not None:
            raise ValueError('Only phase2_fixed declares choices in its resource profile')
        interval = {'phase2_learned': 4, 'phase2_identity': 8, 'phase2_contextual': 4,
                    'aac_joint20': 2, 'aac_fixed8_joint20': 8, 'benchmark_original': 8}[profile]
    phase2 = profile.startswith('phase2_')
    aac = profile.startswith('aac_')
    return {'schema': 'maniguard-controller-resources-v1', 'profile': profile,
            'normal_minimum_actions_between_query_groups': interval,
            'normal_bound_requires_native_chunk_actions_at_least': interval,
            'exhaustion_minimum_actions_between_query_groups': 1,
            'short_chunk_exhaustion': True,
            'accessible_action_prefix': 8 if profile == 'benchmark_original' else 16,
            'policy_proposals_per_query_group': 20 if aac else 1,
            'policy_rpcs_per_proposal': 1,
            'proposal_rng': 'independent_per_episode_streams' if aac else 'single_policy_stream',
            'encoder_execution': 'clip_two_views_every_four_actions' if phase2 else 'none',
            'encoder_action_interval': 4 if phase2 else None,
            'encoder_image_calls_per_execution': 1 if phase2 else 0,
            'encoder_images_per_execution': 2 if phase2 else 0,
            'encoder_text_calls_per_episode': 1 if phase2 else 0,
            'encoder_load_attempts_per_episode': 1 if phase2 else 0,
            'adapter_decisions_per_encoder_execution': 1 if phase2 else 0,
            'actor_rpc_attempts_per_episode': 0}


def _validate_resources(declared, profile, *, choices=None):
    expected = controller_resource_contract(profile, choices=choices)
    if record_hash(declared) != record_hash(expected):
        raise ValueError('Controller resource declaration differs from its supported execution contract')
    return expected


def _comparator_resources(controller, root):
    kind = controller.get('kind')
    if kind == 'benchmark_original':
        expected_keys = {'kind', 'deterministic', 'resource_contract', 'sha256'}
        profile, contract_choices = 'benchmark_original', None
    elif kind in ('identity', 'fixed'):
        expected_keys = {'kind', 'choices', 'deterministic', 'resource_contract', 'sha256'}
        choices = controller.get('choices')
        if kind == 'identity' and record_hash(choices) != record_hash([0, 0, 0, 0]):
            raise ValueError('Identity comparator must preserve every original choice')
        profile, contract_choices = 'phase2_' + kind, choices if kind == 'fixed' else None
    elif kind == 'contextual':
        expected_keys = {'kind', 'rule', 'deterministic', 'resource_contract', 'sha256'}
        if record_hash(controller.get('rule')) != record_hash({
                'feature_slice': [12, 19], 'threshold': 0.5,
                'true_choices': [1, 0, 0, 0], 'false_choices': [0, 0, 0, 0]}):
            raise ValueError('Only the existing phase-two contextual comparator is supported')
        profile, contract_choices = 'phase2_contextual', None
    elif kind in ('aac_joint20', 'aac_fixed8_joint20'):
        expected_keys = {'kind', 'aac_config', 'deterministic', 'resource_contract', 'sha256'}
        config = load_aac_config(_file(controller['aac_config'], root))
        if config.sample_count != 20 or config.max_prefix != 16:
            raise ValueError('AAC planning requires the declared twenty-proposal, sixteen-action contract')
        profile, contract_choices = kind, None
    else:
        raise ValueError('Unsupported comparator execution/resource contract')
    if set(controller) != expected_keys:
        raise ValueError('Unsupported comparator fields cannot silently alter resource assumptions')
    return _validate_resources(controller.get('resource_contract'), profile, choices=contract_choices)


def episode_resource_envelope(action_horizon, contract):
    """Worst successful work for one full slot, including singleton chunk exhaustion."""
    if type(action_horizon) is not int or action_horizon <= 0:
        raise ValueError('A resource envelope requires a positive full action horizon')
    interval = contract['normal_minimum_actions_between_query_groups']
    groups = (action_horizon + interval - 1) // interval
    proposals = contract['policy_proposals_per_query_group']
    per_group = proposals * contract['policy_rpcs_per_proposal']
    image_interval = contract['encoder_action_interval']
    executions = ((action_horizon + image_interval - 1) // image_interval
                  if image_interval is not None else 0)
    counts = dict.fromkeys(WORK_COUNTS, 0)
    counts.update(policy_control_steps=action_horizon, policy_queries=action_horizon * per_group,
                  physics_steps=6 * action_horizon,
                  encoder_image_calls=executions * contract['encoder_image_calls_per_execution'],
                  encoder_images=executions * contract['encoder_images_per_execution'],
                  encoder_text_calls=contract['encoder_text_calls_per_episode'],
                  encoder_load_attempts=contract['encoder_load_attempts_per_episode'],
                  adapter_decisions=executions * contract['adapter_decisions_per_encoder_execution'],
                  actor_rpc_attempts=contract['actor_rpc_attempts_per_episode'])
    counts['encoder_image_attempts'] = counts['encoder_image_calls']
    counts['encoder_text_attempts'] = counts['encoder_text_calls']
    return {'query_groups_normal_max': groups, 'query_groups_exhaustion_max': action_horizon,
            'normal_interval_successful_queries_max': groups * per_group,
            'exhaustion_safe_successful_queries_max': action_horizon * per_group,
            'successful_policy_proposals_max': action_horizon * proposals,
            'supplementary_encoder_executions_max': executions, 'successful_work': counts}


def _actors(spec, root):
    roster = {}
    for run in spec['training_runs']:
        if set(run) != {'run_id', 'method', 'optimizer_seed', 'training_protocol_sha256'}:
            raise ValueError('Training roster must bind run, method, optimizer seed and training protocol')
        run_id = _identifier(run['run_id'])
        _identifier(run['method'])
        if (run_id in roster or not _integer(run['optimizer_seed']) or
                not re.fullmatch('[0-9a-f]{64}', run['training_protocol_sha256'])):
            raise ValueError('Duplicate or malformed independent training run')
        roster[run_id] = run
    pairs = [(run['method'], run['optimizer_seed']) for run in roster.values()]
    if len(pairs) != len(set(pairs)):
        raise ValueError('Repeated optimizer seeds cannot be labelled independent runs of one method')
    learned_resources = _validate_resources(spec['deployment'].get('trained_resource_contract'),
                                             'phase2_learned')
    controllers, seen = [], set()
    for selected in spec['selected_actors']:
        if set(selected) != {'run_id', 'actor', 'actor_sha256', 'selection'}:
            raise ValueError('Selected actor must bind one model and one selection record')
        run_id = selected['run_id']
        if run_id in seen or run_id not in roster:
            raise ValueError('Exactly one selected actor is allowed per declared training run')
        seen.add(run_id)
        run = roster[run_id]
        genome, model_spec, _ = load_model(_file(selected['actor'], root))
        actor_hash = genome_sha256(genome, model_spec)
        selection = _record(selected['selection'], root)
        if (actor_hash != selected['actor_sha256'] or selection.get('schema') != SELECTION_SCHEMA or
                any(selection.get(key) != value for key, value in run.items()) or
                selection.get('selected_actor_sha256') != actor_hash or
                selection.get('selection_partition') != 'validation' or
                not selection.get('selection_families') or
                not set(selection['selection_families']) <= {'jar', 'stack', 'clutter'} or
                not selection.get('training_families') or
                not set(selection['training_families']) <= {'jar', 'stack', 'clutter'}):
            raise ValueError('Selected model differs from its development-validation provenance')
        controllers.append({**run, 'run_kind': 'trained', 'actor_sha256': actor_hash,
                            'controller_sha256': actor_hash, 'actor': selected['actor'],
                            'selection': selected['selection'],
                            'resource_contract': deepcopy(learned_resources),
                            'resource_contract_sha256': record_hash(learned_resources)})
    if seen != set(roster) or not roster:
        raise ValueError('Every declared training run requires exactly one selected actor')
    methods = {value['method'] for value in controllers}
    for comparator in spec['comparators']:
        if set(comparator) != {'method', 'controller'}:
            raise ValueError('Fixed comparator requires method and hashed controller specification')
        method = _identifier(comparator['method'])
        controller = _record(comparator['controller'], root)
        if method in methods or controller.get('deterministic') is not True:
            raise ValueError('Fixed comparator methods must be distinct and deterministic')
        resources = _comparator_resources(controller, root)
        methods.add(method)
        controllers.append({'method': method, 'run_id': method + '-fixed', 'run_kind': 'fixed',
                            'actor_sha256': None, 'controller_sha256': controller['sha256'],
                            'controller': comparator['controller'], 'resource_contract': resources,
                            'resource_contract_sha256': record_hash(resources)})
    return sorted(controllers, key=lambda value: (value['method'], value['run_id']))


def _cases(spec, inputs):
    registry, splits = inputs['families']['families'], inputs['splits']['families']
    cases, seen = [], set()
    for case in spec['cases']:
        if set(case) != {'family', 'scene', 'seed', 'sampling_seed'}:
            raise ValueError('Every exact case needs family, scene, policy seed and sampling seed')
        family, scene = case['family'], case['scene']
        if family not in registry or not isinstance(scene, str) or len(scene.split('/')) != 2:
            raise ValueError('Unknown family or noncanonical task/condition scene')
        base, condition = scene.split('/')
        family_spec = registry[family]
        if (base not in family_spec['task_ids'] or condition not in family_spec['conditions'] or
                not _integer(case['seed']) or not _integer(case['sampling_seed'])):
            raise ValueError('Case lies outside the pinned metadata or seed domain')
        partitions = [name for name, tasks in splits[family].items() if base in tasks]
        if len(partitions) != 1:
            raise ValueError('Case does not have exactly one frozen task partition')
        role = family_spec['role']
        if role == 'development' and partitions[0] != 'test':
            raise ValueError('Main development-family evaluation requires reserved test tasks')
        if role not in ('development', 'design_exposed_transfer', 'reserved_transfer'):
            raise ValueError('Unrecognized exposure role')
        key = (family, scene, case['seed'], case['sampling_seed'])
        if key in seen:
            raise ValueError('Duplicate evaluation case cannot create a fresh paid replicate')
        seen.add(key)
        policy = inputs['sources']['policies'][family_spec['policy_key']]
        cases.append({**case, 'base_scene': base, 'condition': condition, 'partition': partitions[0],
                      'purpose': 'test' if role == 'development' else 'transfer', 'exposure_role': role,
                      'action_horizon': family_spec['max_steps'],
                      'checkpoint': {key: policy[key] for key in ('repo', 'revision', 'step', 'config')},
                      'eval_config_sha256': family_spec['eval_config_sha256']})
    if not cases:
        raise ValueError('Supply an explicit nonempty case list; no sample size is chosen automatically')
    return sorted(cases, key=lambda value: (value['family'], value['scene'], value['seed'],
                                           value['sampling_seed']))


def compile_jobs(protocol):
    """Common case grid for every selected run and each single-run fixed comparator."""
    jobs = []
    for controller in protocol['controllers']:
        for case in protocol['cases']:
            job_id = '/'.join((protocol['study_id'], controller['method'], controller['run_id'],
                               case['family'], case['scene'], f"p{case['seed']}",
                               f"s{case['sampling_seed']}"))
            jobs.append({**case, **controller, 'episode_id': job_id, 'job_id': job_id,
                         'stochastic': False,
                         'resource_envelope': episode_resource_envelope(case['action_horizon'],
                                                                       controller['resource_contract']),
                         'preprocessing_sha256': protocol['preprocessing_sha256'],
                         'deployment_sha256': record_hash(protocol['spec']['deployment']),
                         'implementation_sha256': record_hash(protocol['spec']['implementation_files']),
                         'analysis_sha256': record_hash(protocol['spec']['analysis']),
                         'exposure_sha256': record_hash(protocol['exposure'])})
    return jobs


def expected_cases(protocol):
    """Return planner jobs directly consumable as the main analyzer's expected-case mappings."""
    return compile_jobs(protocol)


def action_envelope(jobs):
    counts = {key: sum(job['resource_envelope']['successful_work'][key] for job in jobs)
              for key in WORK_COUNTS}
    groups = {}
    for job in jobs:
        key = job['method'] + '/' + job['run_id']
        if key not in groups:
            groups[key] = {'profile': job['resource_contract']['profile'], 'episode_attempts': 0,
                           'normal_interval_successful_queries_max': 0,
                           'successful_work': dict.fromkeys(WORK_COUNTS, 0)}
        groups[key]['episode_attempts'] += 1
        groups[key]['normal_interval_successful_queries_max'] += job['resource_envelope'][
            'normal_interval_successful_queries_max']
        for name in WORK_COUNTS:
            groups[key]['successful_work'][name] += job['resource_envelope']['successful_work'][name]
    return {'episode_attempts': len(jobs), 'policy_actions_max': counts['policy_control_steps'],
            'normal_interval_successful_queries_max': sum(job['resource_envelope'][
                'normal_interval_successful_queries_max'] for job in jobs),
            'exhaustion_safe_successful_queries_max': counts['policy_queries'],
            'control_physics_ticks_max': counts['physics_steps'], 'successful_work': counts,
            'per_controller': groups,
            'family_episode_counts': dict(sorted(Counter(job['family'] for job in jobs).items())),
            'limits': ['Each policy query is one RPC proposal; AAC pays all twenty proposals per group.',
                       'Normal cadence bounds require native chunks meeting each controller minimum; short chunks use the exhaustion bound.',
                       'Singleton chunk exhaustion permits one query group per control action.',
                       'Supplementary CLIP counts exclude VLA-internal encoders, which are paid by policy RPC/time.',
                       'Budget separate_caps must cover prior work, startup and failed work beyond successful slots.',
                       'Failed/startup caps cannot add policy actions or replacement episode slots.',
                       'Matched policy seeds do not establish identical physical starts.']}


def build_main_protocol(spec, *, config_dir=CONFIG, root=ROOT):
    """Build a deterministic draft from explicit settings; never write or authorize it."""
    root, config_dir = Path(root).resolve(), Path(config_dir).resolve()
    if set(spec) != SPEC_KEYS:
        raise ValueError('Main specification fields differ from the explicit planner contract')
    spec = json.loads(json.dumps(spec, allow_nan=False))
    _identifier(spec['study_id'])
    inputs = {name: json.loads((config_dir / (name + '.json')).read_text()) for name in INPUTS}
    for collection in ('preprocessing_files', 'implementation_files'):
        if not spec[collection] or len({row['path'] for row in spec[collection]}) != len(spec[collection]):
            raise ValueError('Preprocessing and implementation require distinct explicit source bindings')
        for reference in spec[collection]:
            _file(reference, root)
    deployment = spec['deployment']
    if (set(deployment) != {'decoding', 'memory_reset', 'control_hz', 'physics_hz',
                            'trained_resource_contract'} or
            deployment.get('decoding') != 'deterministic' or deployment.get('memory_reset') != 'episode' or
            deployment.get('control_hz') != 20 or deployment.get('physics_hz') != 120):
        raise ValueError('Planner envelope requires the declared phase-two deployment timing contract')
    exposure = {family: {'role': row['role'], 'phase1_outcomes_exposed': row['phase1_outcomes_exposed']}
                for family, row in inputs['families']['families'].items()}
    if (exposure['lid'] != {'role': 'design_exposed_transfer', 'phase1_outcomes_exposed': True} or
            any(exposure[name] != {'role': 'reserved_transfer', 'phase1_outcomes_exposed': False}
                for name in ('cabinet', 'dusty'))):
        raise ValueError('Lid design exposure and Cabinet/Dusty reserved status must be preserved')
    protocol = {'schema': SCHEMA, 'status': 'draft', 'study_id': spec['study_id'], 'spec': spec,
                'input_files': [source_reference(config_dir / (name + '.json'), root=root) for name in INPUTS],
                'planner_source': source_reference(root / 'experiments/helpers/maniguard_main_protocol.py', root=root),
                'controllers': _actors(spec, root), 'cases': _cases(spec, inputs), 'exposure': exposure,
                'preprocessing_sha256': record_hash({'encoder': inputs['encoder'],
                                                     'files': spec['preprocessing_files']}),
                'freeze': None}
    jobs = compile_jobs(protocol)
    protocol.update(grid_sha256=record_hash(jobs), envelope=action_envelope(jobs))
    return _signed(protocol)


def _ready_settings(protocol):
    analysis, budget = protocol['spec']['analysis'], protocol['spec']['budget']
    if analysis is None or budget is None:
        raise ValueError('Main analysis and measured budget remain unresolved')
    if not isinstance(analysis, dict) or not isinstance(budget, dict):
        raise TypeError('Main analysis and budget must be explicit mappings')
    if analysis.get('implementation') not in protocol['spec']['implementation_files']:
        raise ValueError('Analysis implementation must be one of the frozen source bindings')
    reference = analysis.get('reference_method')
    if not any(row['method'] == reference and row['run_kind'] == 'fixed' for row in protocol['controllers']):
        raise ValueError('Analysis requires a declared fixed reference method')
    if (analysis.get('primary_metric') != 'S_req' or analysis.get('cluster_unit') != 'family/base_scene' or
            analysis.get('optimizer_run_variability_retained') is not True or
            analysis.get('variants_jointly_clustered') is not True or
            not _positive(analysis.get('meaningful_completion_effect')) or
            not _positive(analysis.get('confidence_level')) or analysis['confidence_level'] >= 1 or
            type(analysis.get('violation_risk_margin')) not in (int, float) or
            not math.isfinite(analysis['violation_risk_margin']) or analysis['violation_risk_margin'] < 0 or
            not analysis.get('multiple_comparison_rule') or not analysis.get('inconclusive_safety_rule') or
            analysis.get('interval_method') != 'crossed_scene_and_training_run_percentile_bootstrap' or
            not _integer(analysis.get('bootstrap_replicates')) or
            not 2 <= analysis['bootstrap_replicates'] <= 100000 or
            not _integer(analysis.get('bootstrap_seed'))):
        raise ValueError('Analysis lacks explicit effects, safety margin and dependence/uncertainty rules')
    envelope = protocol['envelope']
    caps = budget.get('separate_caps')
    if not isinstance(caps, dict) or set(caps) != {'prior', 'startup', 'failed_work'}:
        raise ValueError('Budget requires separate prior, startup and failed-work caps')
    for category, values in caps.items():
        if (not isinstance(values, dict) or set(values) != set(WORK_COUNTS) or
                any(not _integer(value) for value in values.values())):
            raise ValueError('Every separate work cap must explicitly cover every resource counter')
        if category != 'prior' and values['policy_control_steps'] != 0:
            raise ValueError('Startup and failed work cannot create extra policy-action allocations')
    required = {'episode_attempts_max': envelope['episode_attempts']}
    for count, name in BUDGET_KEYS.items():
        required[name] = envelope['successful_work'][count] + sum(values[count] for values in caps.values())
    if any(not _integer(budget.get(key)) or budget[key] < lower for key, lower in required.items()):
        raise ValueError('Budget lacks attempted-work ceilings covering every controller and separate cap')
    if budget['episode_attempts_max'] != envelope['episode_attempts']:
        raise ValueError('Undeclared replacement episodes cannot be added through the budget')
    if (not _positive(budget.get('allocated_gpu_hours_max')) or
            not _positive(budget.get('estimated_total_allocated_gpu_hours')) or
            budget['estimated_total_allocated_gpu_hours'] > budget['allocated_gpu_hours_max'] or
            set(budget.get('estimate_includes', [])) != {'prior_work', 'episodes', 'initialization',
                'policy_startup_and_switches', 'encoder', 'analysis', 'idle_reservation'}):
        raise ValueError('Costed GPU estimate must include all declared work and fit its explicit ceiling')
    if {row['family'] for row in protocol['cases']} != set(protocol['exposure']):
        raise ValueError('The main study must predeclare reporting for all six families')


def freeze_main_protocol(draft, evidence, *, created_at, config_dir=CONFIG, root=ROOT):
    """Return a hashed freeze candidate only after explicit cost/analysis/exposure reviews."""
    validate_main_protocol(draft, config_dir=config_dir, root=root)
    if draft['status'] != 'draft' or not _positive(created_at) or set(evidence) != set(REQUIRED_REVIEWS):
        raise ValueError('Freeze requires a draft, explicit timestamp and every required review')
    _ready_settings(draft)
    for name, reference in evidence.items():
        review = _record(reference, root)
        if (review.get('schema') != REVIEW_SCHEMA or review.get('check') != name or
                review.get('draft_sha256') != draft['sha256'] or review.get('passed') is not True or
                not _positive(review.get('created_at')) or review['created_at'] > created_at or
                any(review.get('findings', {}).get(key) is not True for key in REQUIRED_REVIEWS[name]) or
                not review.get('sources')):
            raise ValueError('Freeze review is missing, incomplete or bound to a different draft: ' + name)
        for source in review['sources']:
            _file(source, root)
    return _signed({**draft, 'status': 'frozen', 'freeze': {
        'created_at': created_at, 'draft_sha256': draft['sha256'], 'evidence': deepcopy(evidence)}})


def validate_main_protocol(protocol, *, config_dir=CONFIG, root=ROOT, require_frozen=False):
    _check_signed(protocol)
    if protocol.get('schema') != SCHEMA or protocol.get('status') not in ('draft', 'frozen'):
        raise ValueError('Unsupported main evaluation protocol')
    draft = build_main_protocol(protocol['spec'], config_dir=config_dir, root=root)
    if protocol['status'] == 'frozen':
        freeze = protocol.get('freeze') or {}
        expected = freeze_main_protocol(draft, freeze.get('evidence', {}),
                                         created_at=freeze.get('created_at'), config_dir=config_dir, root=root)
    else:
        expected = draft
    if protocol != expected:
        raise ValueError('Main protocol differs from its exact source/case/actor/evidence bindings')
    if require_frozen and protocol['status'] != 'frozen':
        raise ValueError('Main evaluation authorization requires an explicit costed freeze')
    return {'ready_for_exact_job_authorization': protocol['status'] == 'frozen',
            'grid_sha256': protocol['grid_sha256'], 'envelope': deepcopy(protocol['envelope']),
            'outcomes_loaded': False, 'simulator_executed': False,
            'live_runtime_integration': 'Caller must enforce authorize_job before every attempt.'}


def authorize_job(protocol, job, *, config_dir=CONFIG, root=ROOT):
    """Fail closed on any changed/undeclared case, actor, seed or deployment binding.

    This checks identity, not whether a job was already claimed. A future durable
    launcher must also reject duplicate claims and enforce cumulative actual costs.
    """
    validate_main_protocol(protocol, config_dir=config_dir, root=root, require_frozen=True)
    matched = [expected for expected in compile_jobs(protocol) if expected['episode_id'] == job.get('episode_id')]
    if len(matched) != 1 or matched[0] != job:
        raise ValueError('Requested episode is not an exact member of the frozen evaluation grid')
    return {'authorized': True, 'protocol_sha256': protocol['sha256'],
            'episode_id': job['episode_id'], 'job_sha256': record_hash(job)}
