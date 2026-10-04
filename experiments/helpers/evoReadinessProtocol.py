"""Structural protocol checks, exact budgets and launch prerequisites for readiness.

The existing review draft stays disabled. A locked executable protocol must be
supported by the recorded scientific decision and completed prelaunch evidence.
"""
import hashlib
from pathlib import Path
import re

from helpers.evoReadinessBanks import CollectionRole, validate_collection_roles
from helpers.evoReadinessCMA import ReadinessCMAConfig
from helpers.evoReadinessStudy import SearchSpec


PENALTIES={'zero_penalty_matched_termination':'zero','rosarl_style':'rosarl_style'}


def protocol_plan(config):
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


def launch_blockers(config,repository):
    """Read-only preflight. Return concrete unmet prerequisites; never enable a draft."""
    repository=Path(repository)
    blockers=[]
    if config.get('status')!='locked_for_execution' or config.get('execution_enabled') is not True:
        blockers.append('protocol is a disabled review draft, not locked for execution')
    approval=config.get('approval',{})
    if approval.get('supervisor_scientific_decision_recorded') is not True:
        blockers.append('supervisor decision on readiness extension and ROSARL adaptation is not recorded')
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
