"""Structural protocol checks, exact budgets and launch prerequisites for readiness.

The existing review draft stays disabled. A locked executable protocol must be
supported by the recorded scientific decision and completed prelaunch evidence.
"""
import hashlib
import math
from pathlib import Path
import re

from helpers.evoReadinessBanks import CollectionRole, validate_collection_roles
from helpers.evoReadinessCMA import ReadinessCMAConfig
from helpers.evoReadinessStudy import SearchSpec


PENALTIES={'zero_penalty_matched_termination':'zero','rosarl_style':'rosarl_style'}



# These fields describe fixed implementation behavior, not selectable alternatives.
# Reject a changed scientific declaration rather than silently running the old method.
_SUPPORTED = {
    'policy.initial_residual': 'zero',
    'policy.clipping': [-1, 1],
    'policy.coordinate_scale': 'inverse_RMS_of_projected_features_on_existing_noise_fit_roots',
    'search.optimizer': 'separable_CMA_ES',
    'search.paired_seed_streams_across_arms': True,
    'search.noise_streams': 'separate_fitness_selection_evaluation_common_across_candidates_within_role',
    'search.scoring.shared_task_return': 'progress_strictly_before_first_imagined_violation_or_full_horizon_if_safe',
    'search.scoring.zero_penalty': 'mean_shared_task_return',
    'search.scoring.rosarl_style': 'mean_shared_task_return_plus_violation_times_V_MIN_minus_V_MAX',
    'search.scoring.bounds': 'cumulative_min_max_shared_task_returns_from_fitness_batches_only_per_run',
    'search.scoring.update_bounds': 'once_on_full_generation_before_ranking',
    'search.scoring.stop_compute_at_violation': False,
    'search.scoring.tie_break': 'chronological_candidate_id',
    'search.nominee': 'best_candidate_of_each_generation_on_that_generations_fitness_batch',
    'search.incumbent_rescoring': 'rescore_all_cached_nominees_with_current_fitness_only_bounds',
    'search.final_test_used_for_selection': False,
    'evaluation.evaluate_distribution_mean': False,
    'evaluation.real_steps_per_episode_always': 100,
    'evaluation.truth': 'dense_health_every_step',
    'evaluation.real_baseline_once': True,
    'evaluation.exact_recorded_tape_reference_once': True,
    'analysis.gap': 'real_violation_minus_mean_imagined_violation_at_declared_audit_noise',
    'analysis.primary_family': ['unpenalised_k0_gap_high_minus_low_gt_zero',
        'own_noise_gap_amplification_k1_rosarl_minus_k0_unpenalised_lt_zero'],
    'analysis.primary_intervals': 'two_sided_97.5pct_each_Bonferroni_family_95pct',
    'analysis.secondary_intervals': 'descriptive_95pct',
    'analysis.resampling': 'independent_search_seed_and_evaluation_episode_axes_with_all_arms_paired',
    'budget.no_auto_extension': True,
}


def _validate_execution_contract(config):
    for path, expected in _SUPPORTED.items():
        value = config
        for part in path.split('.'):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f'missing implemented protocol setting: {path}')
            value = value[part]
        if value != expected or (isinstance(expected, bool) and value is not expected):
            raise ValueError(f'unsupported protocol setting: {path}')

    def integer(path, value, minimum=1, maximum=None):
        if (not isinstance(value, int) or isinstance(value, bool) or value < minimum
                or (maximum is not None and value > maximum)):
            raise ValueError(f'invalid integer protocol setting: {path}')

    b, s, a = config['bank'], config['search'], config['analysis']
    for key in ['fitness_episodes', 'selection_episodes', 'evaluation_episodes',
                'max_attempts_per_required_root', 'max_steps_per_source_episode']:
        integer('bank.' + key, b[key])
    for key in ['environment_seed_bases', 'actor_noise_seed_bases']:
        for role, value in b[key].items():
            integer(f'bank.{key}.{role}', value, 0, 2**32 - 1)
    integer('search.generations', s['generations'])
    integer('search.fitness_roots_per_generation', s['fitness_roots_per_generation'])
    for value in s['population_sizes']:
        integer('search.population_sizes', value, 2)
    for value in s['search_seeds']:
        integer('search.search_seeds', value, 0, 2**32 - 2)
    for value in s['checkpoints']:
        integer('search.checkpoints', value, 0)
    for label, samples in [('fitness_samples', s['fitness_samples']),
                           ('selection_samples', s['selection_samples']),
                           ('audit_samples', config['evaluation']['audit_samples'])]:
        if set(samples) != {'zero_noise', 'positive_noise'}:
            raise ValueError(f'invalid sample roles: {label}')
        for value in samples.values():
            integer(label, value)
    integer('analysis.bootstrap_replicates', a['bootstrap_replicates'], 2)
    integer('analysis.bootstrap_seed', a['bootstrap_seed'], 0)
    for label in ['low_pressure', 'high_pressure']:
        point = a[label]
        if point['population'] not in s['population_sizes'] or point['generation'] not in s['checkpoints'] or point['generation'] == 0:
            raise ValueError(f'analysis endpoint is absent from search grid: {label}')
    for label, value in [('initial_CMA_sigma', config['policy']['initial_CMA_sigma']),
                         ('coordinate_RMS_floor', config['policy']['coordinate_RMS_floor']),
                         ('maximum_GPU_hours_proposed', config['budget']['maximum_GPU_hours_proposed'])]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'finite positive protocol setting required: {label}')
    loss = a['proposed_allowable_relative_progress_loss']
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or not 0 <= loss < 1:
        raise ValueError('progress loss must be finite and in [0, 1)')


def protocol_plan(config):
    _validate_execution_contract(config)
    bank,search,evaluation=config['bank'],config['search'],config['evaluation']
    if set(search['penalty_modes'])!=set(PENALTIES) or sorted(search['noise_levels'])!=[0.,1.]:
        raise ValueError('the declared main study requires both noise levels and both penalty modes')
    seeds=list(search['search_seeds'])
    populations=list(search['population_sizes'])
    checkpoints=list(search['checkpoints'])
    if len(seeds)<2 or len(set(seeds))!=len(seeds) or len(set(populations))!=len(populations):
        raise ValueError('multiple unique seeds and unique populations required')
    if checkpoints!=sorted(set(checkpoints)) or checkpoints[0]!=0 or checkpoints[-1]!=search['generations']:
        raise ValueError('ordered checkpoints must include zero and the final generation')
    if len(checkpoints)-1!=evaluation['selected_policy_count_per_run']:
        raise ValueError('selected slot count differs from nonbaseline checkpoints')
    if evaluation['horizon_model_blocks']!=10 or evaluation['horizon_sim_steps']!=100:
        raise ValueError('the frozen Walker interface requires ten blocks and one hundred real steps')
    if config['policy']['evolved_parameters']!=1020 or config['policy']['base_gain']!=1.:
        raise ValueError('the frozen policy interface requires 1020 residual coefficients and fixed gain one')
    if config['inputs']['new_model_or_controller_training'] or not search['diagonal_covariance']:
        raise ValueError('no retraining and diagonal CMA covariance are declared')
    if search['candidates_per_imagination_call']!=1:
        raise ValueError('one candidate per model call preserves the checked kernel layout')
    if evaluation['real_encode_batch']!=64 or evaluation['real_max_envs']!=64:
        raise ValueError('the validated real executor uses 64-image batches and at most 64 environments')
    if bank['root_step']!='uniform_integer_20_to_length_minus_100_exclusive' or not bank['retain_healthy_future_conditioning']:
        raise ValueError('source eligibility and root sampling must match the declared readiness distribution')
    if min(bank['fitness_episodes'],bank['selection_episodes'],bank['evaluation_episodes'])<2:
        raise ValueError('each role needs independent source episodes')
    if search['fitness_roots_per_generation']>bank['fitness_episodes']:
        raise ValueError('fitness subsample exceeds the fitness bank')
    if any(search[key]['zero_noise']!=1 for key in ['fitness_samples','selection_samples']) or evaluation['audit_samples']['zero_noise']!=1:
        raise ValueError('zero-noise queries require one sample')
    max_batch=config.get('implementation',{}).get('max_imagined_batch',16384)
    shapes=dict(fitness=search['fitness_roots_per_generation']*search['fitness_samples']['positive_noise'],
        selection=bank['selection_episodes']*search['selection_samples']['positive_noise'],
        evaluation=bank['evaluation_episodes']*evaluation['audit_samples']['positive_noise'])
    if min(shapes.values())<1 or max(shapes.values())>max_batch:
        raise ValueError('declared candidate batch shapes exceed the supported maximum')
    roles=[CollectionRole(name,int(bank[f'{name}_episodes']),int(bank['environment_seed_bases'][name]),
        int(bank['actor_noise_seed_bases'][name]),int(bank['max_attempts_per_required_root']),
        int(bank['max_steps_per_source_episode'])) for name in ['fitness','selection','evaluation']]
    validate_collection_roles(roles)
    specs=[]
    for seed in seeds:
        for population in populations:
            for k in search['noise_levels']:
                sample_key='zero_noise' if k==0 else 'positive_noise'
                for mode in search['penalty_modes']:
                    c=ReadinessCMAConfig(population=population,generations=search['generations'],
                        sigma0=config['policy']['initial_CMA_sigma'],seed=seed,
                        n_fitness_roots=bank['fitness_episodes'],fitness_roots_per_generation=search['fitness_roots_per_generation'],
                        fitness_samples=search['fitness_samples'][sample_key],selection_roots=bank['selection_episodes'],
                        selection_samples=search['selection_samples'][sample_key],penalty_mode=PENALTIES[mode],
                        horizon_blocks=evaluation['horizon_model_blocks'],checkpoints=tuple(checkpoints))
                    specs.append(SearchSpec(float(k),c))
    evaluation_plan=dict(audit_samples=evaluation['audit_samples'],noise_levels=list(evaluation['imaginary_audits_for_every_pick']),
                         horizon_blocks=evaluation['horizon_model_blocks'])
    if sorted(evaluation_plan['noise_levels'])!=[0.,1.]:
        raise ValueError('both audit noise levels are required for every selected policy')
    picks=len(specs)*(len(checkpoints)-1)
    fitness=sum(s.config.population*s.config.generations*s.config.fitness_roots_per_generation*s.config.fitness_samples*10 for s in specs)
    selection=sum((s.config.generations+1)*s.config.selection_roots*s.config.selection_samples*10 for s in specs)
    audit=(picks+len(seeds))*bank['evaluation_episodes']*10*sum(evaluation['audit_samples'].values())
    real=(picks+2)*bank['evaluation_episodes']*100
    max_collection=sum(s.roots*s.attempts_per_root*s.max_steps for s in roles)
    counts=dict(search_runs=len(specs),selected_policy_slots=picks,new_source_roots=sum(s.roots for s in roles),
        fitness_predictor_rows=fitness,selection_predictor_rows=selection,final_audit_predictor_rows=audit,
        total_predictor_rows=fitness+selection+audit,final_real_steps=real,
        maximum_source_collection_steps=max_collection,maximum_new_real_steps=real+max_collection,
        initial_history_renders_and_encodes=3*sum(s.roots for s in roles),
        fingerprint_renders_per_process=64,gradient_updates=0)
    return dict(collection_roles=roles,search_specs=specs,evaluation_plan=evaluation_plan,
                batch_shapes=shapes,counts=counts)



def scientific_decision_blockers(approval, repository):
    """Recognize explicit user authorization without asserting supervisor approval."""
    authority = approval.get('scientific_decision_authority', 'supervisor')
    if authority == 'supervisor':
        return [] if approval.get('supervisor_scientific_decision_recorded') is True else [
            'supervisor decision on readiness extension and ROSARL adaptation is not recorded']
    if authority != 'user' or approval.get('user_scientific_decision_recorded') is not True:
        return ['a recognized explicit scientific decision is not recorded']
    proof = approval.get('user_decision_evidence', {})
    path = Path(repository) / str(proof.get('path', ''))
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != proof.get('sha256'):
        return ['explicit user decision evidence is missing or changed']
    import json
    try:
        record = json.loads(path.read_text())
    except (ValueError, OSError):
        return ['explicit user decision evidence is unreadable']
    if (record.get('kind') != 'explicit_user_authorization'
            or record.get('readiness_extension_accepted') is not True
            or record.get('matched_termination_rosarl_adaptation_accepted') is not True
            or record.get('supervisor_approval_claim') is not False
            or not record.get('user_messages')):
        return ['explicit user decision does not cover the proposed scientific extension']
    return []


def launch_blockers(config,repository):
    """Read-only preflight. Return concrete unmet prerequisites; never enable a draft."""
    repository=Path(repository)
    blockers=[]
    if config.get('status')!='locked_for_execution' or config.get('execution_enabled') is not True:
        blockers.append('protocol is a disabled review draft, not locked for execution')
    approval=config.get('approval',{})
    blockers.extend(scientific_decision_blockers(approval, repository))
    required=approval.get('required_before_execution',[])
    completed=approval.get('completed_checks',{})
    evidence=approval.get('evidence',{})
    for name in required:
        if completed.get(name) is not True:
            blockers.append(f'prelaunch requirement incomplete: {name}')
            continue
        proof=evidence.get(name,{})
        path=repository/str(proof.get('path',''))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=proof.get('sha256'):
            blockers.append(f'missing or changed evidence: {name}')
    if config['bank'].get('evaluation_count_is_provisional_pending_power_review') is not False:
        blockers.append('final episode and search-seed counts remain provisional')
    revision=config['inputs'].get('noise_archive_revision')
    if not isinstance(revision,str) or not re.fullmatch(r'[0-9a-f]{40}',revision):
        blockers.append('noise calibration has no pinned private archive revision')
    inputs=config['inputs']
    source=repository/inputs['frozen_model_probe_data_config']
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=inputs['frozen_model_probe_data_config_sha256']:
        blockers.append('frozen model/probe/data input configuration is missing or changed')
    return blockers
