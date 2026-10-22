"""Pure main-planning tests; synthetic selection/cost evidence, no reserved outcomes."""

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_main_protocol import (
    BUDGET_KEYS,
    REQUIRED_REVIEWS,
    REVIEW_SCHEMA,
    SELECTION_SCHEMA,
    WORK_COUNTS,
    _signed,
    authorize_job,
    build_main_protocol,
    compile_jobs,
    controller_resource_contract,
    episode_resource_envelope,
    expected_cases,
    freeze_main_protocol,
    source_reference,
    validate_main_protocol,
)
from helpers.maniguard_phase2_model import genome_sha256, identity_genome, save_model


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


@pytest.fixture
def planned(tmp_path):
    root = tmp_path / 'repo'
    config = root / 'configs/maniguard/phase2'
    shutil.copytree(ROOT / 'configs/maniguard/phase2', config)
    planner = root / 'experiments/helpers/maniguard_main_protocol.py'
    planner.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / 'experiments/helpers/maniguard_main_protocol.py', planner)
    implementation = root / 'synthetic_runtime.py'
    implementation.write_text('# Synthetic test implementation; no simulator invocation.\n')
    splits = json.loads((config / 'splits.json').read_text())['families']
    cases = [{'family': family, 'scene': splits[family]['test'][0] + '/' + condition,
              'seed': 3, 'sampling_seed': 7}
             for family in sorted(splits) for condition in ('base', 'language')]
    runs, selected = [], []
    for seed in (17, 29):
        run = {'run_id': 'evolution-s' + str(seed), 'method': 'evolution', 'optimizer_seed': seed,
               'training_protocol_sha256': 'a' * 64}
        runs.append(run)
        genome = identity_genome(seed=seed)
        actor = root / 'models' / (run['run_id'] + '.json')
        actor.parent.mkdir(exist_ok=True)
        save_model(actor, genome)
        selection = root / 'selection' / (run['run_id'] + '.json')
        write(selection, _signed({'schema': SELECTION_SCHEMA, **run,
            'selected_actor_sha256': genome_sha256(genome), 'selection_partition': 'validation',
            'selection_families': ['jar', 'stack', 'clutter'],
            'training_families': ['jar', 'stack', 'clutter']}))
        selected.append({'run_id': run['run_id'], 'actor': source_reference(actor, root=root),
                         'actor_sha256': genome_sha256(genome),
                         'selection': source_reference(selection, root=root)})
    comparator = root / 'identity.json'
    write(comparator, _signed({'kind': 'identity', 'choices': [0, 0, 0, 0], 'deterministic': True,
                               'resource_contract': controller_resource_contract('phase2_identity')}))
    spec = {'study_id': 'synthetic-main', 'cases': cases, 'training_runs': runs,
            'selected_actors': selected,
            'comparators': [{'method': 'identity', 'controller': source_reference(comparator, root=root)}],
            'preprocessing_files': [source_reference(config / 'encoder.json', root=root)],
            'implementation_files': [source_reference(implementation, root=root)],
            'deployment': {'decoding': 'deterministic', 'memory_reset': 'episode',
                           'control_hz': 20, 'physics_hz': 120,
                           'trained_resource_contract': controller_resource_contract('phase2_learned')},
            'analysis': None, 'budget': None}
    return {'root': root, 'config': config, 'spec': spec}


def build(fixture):
    return build_main_protocol(fixture['spec'], config_dir=fixture['config'], root=fixture['root'])


def freeze_fixture(fixture):
    """Synthetic values exercise validation only; they are not a proposed study design."""
    envelope = build(fixture)['envelope']
    fixture['spec']['analysis'] = {
        'implementation': fixture['spec']['implementation_files'][0],
        'reference_method': 'identity', 'primary_metric': 'S_req', 'cluster_unit': 'family/base_scene',
        'optimizer_run_variability_retained': True, 'variants_jointly_clustered': True,
        'meaningful_completion_effect': 0.1, 'violation_risk_margin': 0.02, 'confidence_level': 0.95,
        'interval_method': 'crossed_scene_and_training_run_percentile_bootstrap',
        'bootstrap_replicates': 20, 'bootstrap_seed': 11,
        'multiple_comparison_rule': 'Synthetic prespecified comparison',
        'inconclusive_safety_rule': 'Report inconclusive when interval exceeds the margin'}
    caps = {name: dict.fromkeys(WORK_COUNTS, 0) for name in ('prior', 'startup', 'failed_work')}
    caps['startup']['physics_steps'] = 1000
    caps['failed_work']['policy_queries'] = 10
    fixture['spec']['budget'] = {
        'episode_attempts_max': envelope['episode_attempts'], 'separate_caps': caps,
        **{key: envelope['successful_work'][counter] + sum(part[counter] for part in caps.values())
           for counter, key in BUDGET_KEYS.items()},
        'allocated_gpu_hours_max': 10, 'estimated_total_allocated_gpu_hours': 5,
        'estimate_includes': ['prior_work', 'episodes', 'initialization', 'policy_startup_and_switches',
                             'encoder', 'analysis', 'idle_reservation']}
    draft, evidence = build(fixture), {}
    for name, findings in REQUIRED_REVIEWS.items():
        path = fixture['root'] / 'reviews' / (name + '.json')
        write(path, _signed({'schema': REVIEW_SCHEMA, 'check': name, 'draft_sha256': draft['sha256'],
                            'passed': True, 'created_at': 1000,
                            'findings': dict.fromkeys(findings, True),
                            'sources': fixture['spec']['implementation_files']}))
        evidence[name] = source_reference(path, root=fixture['root'])
    frozen = freeze_main_protocol(draft, evidence, created_at=1001,
                                  config_dir=fixture['config'], root=fixture['root'])
    return draft, evidence, frozen


def test_dry_grid_is_deterministic_common_and_has_exact_action_envelopes(planned):
    draft = build(planned)
    assert draft == build(planned)
    jobs = compile_jobs(draft)
    assert expected_cases(draft) == jobs and len(jobs) == 36
    panels = {}
    for job in jobs:
        panels.setdefault(job['run_id'], []).append((job['family'], job['scene'], job['seed'], job['sampling_seed']))
    assert len(panels) == 3 and all(panel == next(iter(panels.values())) for panel in panels.values())
    assert len({job['episode_id'] for job in jobs}) == 36
    assert draft['envelope']['policy_actions_max'] == 6 * (2000 + 4200 + 1900 + 1100 + 5700 + 5100)
    assert draft['envelope']['control_physics_ticks_max'] == 6 * draft['envelope']['policy_actions_max']
    assert draft['envelope']['normal_interval_successful_queries_max'] == sum(
        (job['action_horizon'] + (8 if job['method'] == 'identity' else 4) - 1) //
        (8 if job['method'] == 'identity' else 4) for job in jobs)
    assert {job['run_kind'] for job in jobs} == {'trained', 'fixed'}
    assert all(job['actor_sha256'] is None for job in jobs if job['run_kind'] == 'fixed')


def test_unresolved_draft_never_authorizes_reserved_outcomes(planned):
    draft = build(planned)
    audit = validate_main_protocol(draft, config_dir=planned['config'], root=planned['root'])
    assert not audit['ready_for_exact_job_authorization'] and not audit['outcomes_loaded']
    with pytest.raises(ValueError, match='explicit costed freeze'):
        authorize_job(draft, compile_jobs(draft)[0], config_dir=planned['config'], root=planned['root'])
    with pytest.raises(ValueError, match='every required review'):
        freeze_main_protocol(draft, {}, created_at=1001, config_dir=planned['config'], root=planned['root'])


def test_exposure_roles_and_checkpoint_configs_are_bound_without_reading_outcomes(planned):
    draft = build(planned)
    lid = next(case for case in draft['cases'] if case['family'] == 'lid')
    assert lid['exposure_role'] == 'design_exposed_transfer'
    assert lid['checkpoint']['config'].startswith('pi0-base_')
    for family in ('cabinet', 'dusty'):
        case = next(case for case in draft['cases'] if case['family'] == family)
        assert case['exposure_role'] == 'reserved_transfer' and case['purpose'] == 'transfer'
    registry = planned['config'] / 'families.json'
    value = json.loads(registry.read_text())
    value['families']['lid']['role'] = 'reserved_transfer'
    write(registry, value)
    with pytest.raises(ValueError, match='Lid design exposure'):
        build(planned)


@pytest.mark.parametrize('mutation', ['duplicate_run', 'missing_actor', 'two_actors', 'same_seed', 'actor_hash'])
def test_exactly_one_selected_actor_per_independent_run(planned, mutation):
    spec = planned['spec']
    if mutation == 'duplicate_run':
        spec['training_runs'].append(deepcopy(spec['training_runs'][0]))
    elif mutation == 'missing_actor':
        spec['selected_actors'].pop()
    elif mutation == 'two_actors':
        spec['selected_actors'].append(deepcopy(spec['selected_actors'][0]))
    elif mutation == 'same_seed':
        spec['training_runs'][1]['optimizer_seed'] = spec['training_runs'][0]['optimizer_seed']
    else:
        spec['selected_actors'][0]['actor_sha256'] = 'b' * 64
    with pytest.raises(ValueError):
        build(planned)


@pytest.mark.parametrize('mutation', ['training_case', 'duplicate_case', 'unknown_condition', 'reserved_training'])
def test_no_silent_expansion_or_reserved_family_training(planned, mutation):
    if mutation == 'reserved_training':
        selected = planned['spec']['selected_actors'][0]
        path = planned['root'] / selected['selection']['path']
        value = json.loads(path.read_text())
        value['training_families'].append('cabinet')
        write(path, _signed(value))
        selected['selection'] = source_reference(path, root=planned['root'])
    elif mutation == 'duplicate_case':
        planned['spec']['cases'].append(deepcopy(planned['spec']['cases'][0]))
    else:
        case = next(case for case in planned['spec']['cases'] if case['family'] == 'jar')
        if mutation == 'training_case':
            splits = json.loads((planned['config'] / 'splits.json').read_text())
            case['scene'] = splits['families']['jar']['train'][0] + '/base'
        else:
            case['scene'] = case['scene'].split('/')[0] + '/invented'
    with pytest.raises(ValueError):
        build(planned)


def test_frozen_exact_job_authorization_and_source_tampering(planned):
    _, _, frozen = freeze_fixture(planned)
    job = compile_jobs(frozen)[0]
    result = authorize_job(frozen, job, config_dir=planned['config'], root=planned['root'])
    assert result['authorized'] and result['protocol_sha256'] == frozen['sha256']
    source = planned['root'] / planned['spec']['implementation_files'][0]['path']
    source.write_text('# Changed code after freeze\n')
    with pytest.raises(ValueError, match='Bound source changed'):
        authorize_job(frozen, job, config_dir=planned['config'], root=planned['root'])


@pytest.mark.parametrize('field,value', [('seed', 9), ('sampling_seed', 2), ('actor_sha256', 'e' * 64),
                                        ('action_horizon', 32), ('preprocessing_sha256', 'f' * 64),
                                        ('analysis_sha256', '1' * 64), ('scene', 'task_9999/base')])
def test_frozen_authorization_requires_the_whole_exact_case(planned, field, value):
    _, _, frozen = freeze_fixture(planned)
    job = compile_jobs(frozen)[0]
    job[field] = value
    with pytest.raises(ValueError, match='exact member'):
        authorize_job(frozen, job, config_dir=planned['config'], root=planned['root'])


@pytest.mark.parametrize('mutation', ['risk_unset', 'extra_attempt', 'under_queries', 'over_gpu', 'missing_family'])
def test_freeze_requires_explicit_justified_analysis_and_costed_grid(planned, mutation):
    _, evidence, _ = freeze_fixture(planned)
    if mutation == 'risk_unset':
        planned['spec']['analysis']['violation_risk_margin'] = None
    elif mutation == 'extra_attempt':
        planned['spec']['budget']['episode_attempts_max'] += 1
    elif mutation == 'under_queries':
        planned['spec']['budget']['policy_queries_max'] = 1
    elif mutation == 'over_gpu':
        planned['spec']['budget']['estimated_total_allocated_gpu_hours'] = 11
    else:
        planned['spec']['cases'] = [row for row in planned['spec']['cases'] if row['family'] != 'dusty']
    draft = build(planned)
    with pytest.raises(ValueError):
        freeze_main_protocol(draft, evidence, created_at=1001, config_dir=planned['config'], root=planned['root'])


def test_rehashed_protocol_edit_cannot_reuse_older_review_evidence(planned):
    _, _, frozen = freeze_fixture(planned)
    frozen['spec']['analysis']['violation_risk_margin'] = 0.5
    with pytest.raises(ValueError, match='different draft'):
        validate_main_protocol(_signed(frozen), config_dir=planned['config'], root=planned['root'])


def test_planner_grid_feeds_analyzer_without_reference_duplication(planned):
    from helpers.maniguard_main_analysis import analyze

    jobs = expected_cases(build(planned))
    rows = []
    for job in jobs:
        success = int(job['run_kind'] == 'trained')
        rows.append({**job, 'valid': True, 'full_episode': True, 'short_check': False,
                     'outcome': {'valid': True, 'S_req': success, 'C_req': success, 'V': 0, 'U': success},
                     'costs': {'policy_queries': 4}})
    report = analyze(jobs, rows, reference_method='identity', bootstrap_replicates=0)
    assert report['complete'] and report['coverage']['expected'] == 36
    assert report['coverage']['recorded_cost_totals']['policy_queries'] == 144
    assert report['estimates']['identity']['unique_episode_count'] == 12
    assert report['estimates']['evolution']['unique_episode_count'] == 24
    assert report['risk_inference']['status'] == 'not_assessed'


def add_aac_comparator(fixture, kind):
    path = fixture['root'] / (kind + '.json')
    record = _signed({'kind': kind, 'deterministic': True,
                      'resource_contract': controller_resource_contract(kind),
                      'aac_config': source_reference(fixture['config'] / 'aac.json', root=fixture['root'])})
    write(path, record)
    fixture['spec']['comparators'].append({'method': kind, 'controller': source_reference(path, root=fixture['root'])})
    return path


def test_aac_twenty_proposals_and_fixed8_control_get_distinct_normal_query_envelopes(planned):
    add_aac_comparator(planned, 'aac_joint20')
    add_aac_comparator(planned, 'aac_fixed8_joint20')
    draft = build(planned)
    jobs = compile_jobs(draft)
    for job in jobs:
        envelope = job['resource_envelope']
        actions = job['action_horizon']
        if job['method'].startswith('aac_'):
            interval = 2 if job['method'] == 'aac_joint20' else 8
            assert envelope['normal_interval_successful_queries_max'] == 20 * ((actions + interval - 1) // interval)
            assert envelope['exhaustion_safe_successful_queries_max'] == 20 * actions
            assert envelope['successful_policy_proposals_max'] == 20 * actions
            assert envelope['successful_work']['encoder_image_attempts'] == 0
            assert envelope['successful_work']['encoder_load_attempts'] == 0
        else:
            decisions = (actions + 3) // 4
            work = envelope['successful_work']
            assert work['encoder_image_calls'] == work['encoder_image_attempts'] == decisions
            assert work['encoder_images'] == decisions * 2
            assert work['encoder_text_calls'] == work['encoder_text_attempts'] == 1
            assert work['encoder_load_attempts'] == 1
            assert work['adapter_decisions'] == decisions
    assert draft['envelope']['exhaustion_safe_successful_queries_max'] > draft['envelope']['policy_actions_max']
    _, _, frozen = freeze_fixture(planned)
    assert authorize_job(frozen, compile_jobs(frozen)[0], root=planned['root'],
                         config_dir=planned['config'])['authorized']


@pytest.mark.parametrize('profile,normal_interval,proposals', [
    ('phase2_learned', 4, 1), ('phase2_identity', 8, 1),
    ('aac_joint20', 2, 20), ('aac_fixed8_joint20', 8, 20), ('benchmark_original', 8, 1)])
def test_singleton_exhaustion_and_encoder_boundaries_are_conservative(profile, normal_interval, proposals):
    contract = controller_resource_contract(profile)
    assert contract['normal_bound_requires_native_chunk_actions_at_least'] == normal_interval
    for actions in (1, 3, 4, 5, 7, 8, 9, 17):
        envelope = episode_resource_envelope(actions, contract)
        assert envelope['normal_interval_successful_queries_max'] == proposals * ((actions + normal_interval - 1) // normal_interval)
        assert envelope['exhaustion_safe_successful_queries_max'] == proposals * actions
        assert envelope['successful_work']['encoder_image_calls'] == ((actions + 3) // 4 if profile.startswith('phase2_') else 0)


@pytest.mark.parametrize('choice,interval', [(0, 8), (1, 4), (2, 16)])
def test_fixed_query_head_controls_its_own_normal_cadence(choice, interval):
    contract = controller_resource_contract('phase2_fixed', choices=[choice, 0, 0, 0])
    assert contract['normal_minimum_actions_between_query_groups'] == interval
    envelope = episode_resource_envelope(17, contract)
    assert envelope['normal_interval_successful_queries_max'] == (17 + interval - 1) // interval
    assert envelope['exhaustion_safe_successful_queries_max'] == 17


@pytest.mark.parametrize('mutation', ['unknown_kind', 'twenty_to_one', 'encoder_disabled', 'unknown_field', 'aac_samples'])
def test_unsupported_or_understated_execution_declarations_are_rejected(planned, mutation):
    path = add_aac_comparator(planned, 'aac_joint20')
    comparator = planned['spec']['comparators'][-1]
    value = json.loads(path.read_text())
    if mutation == 'unknown_kind':
        value['kind'] = 'arbitrary_deterministic_code'
    elif mutation == 'twenty_to_one':
        value['resource_contract']['policy_proposals_per_query_group'] = 1
    elif mutation == 'unknown_field':
        value['hidden_extra_policy_queries'] = 100
    elif mutation == 'encoder_disabled':
        planned['spec']['deployment']['trained_resource_contract']['encoder_image_calls_per_execution'] = 0
    else:
        config_path = planned['config'] / 'aac.json'
        config = json.loads(config_path.read_text())
        config['settings']['sample_count'] = 21
        write(config_path, config)
        value['aac_config'] = source_reference(config_path, root=planned['root'])
    write(path, _signed(value))
    comparator['controller'] = source_reference(path, root=planned['root'])
    with pytest.raises(ValueError, match='Unsupported|resource declaration|twenty-proposal'):
        build(planned)


@pytest.mark.parametrize('counter', ['policy_queries', 'encoder_image_calls', 'encoder_images',
                                     'encoder_text_calls', 'encoder_image_attempts',
                                     'encoder_text_attempts', 'encoder_load_attempts', 'adapter_decisions'])
def test_freeze_rejects_caps_below_per_controller_resource_envelope(planned, counter):
    add_aac_comparator(planned, 'aac_joint20')
    draft, evidence, _ = freeze_fixture(planned)
    required = draft['envelope']['successful_work'][counter]
    assert required > 0
    planned['spec']['budget'][BUDGET_KEYS[counter]] = required - 1
    with pytest.raises(ValueError, match='every controller and separate cap'):
        freeze_main_protocol(build(planned), evidence, created_at=1001,
                              root=planned['root'], config_dir=planned['config'])


@pytest.mark.parametrize('mutation', ['missing', 'unknown_counter', 'unfunded_prior', 'startup_actions', 'unfunded_failure'])
def test_separate_prior_startup_and_failure_caps_cannot_hide_extra_work(planned, mutation):
    _, evidence, _ = freeze_fixture(planned)
    budget = planned['spec']['budget']
    if mutation == 'missing':
        del budget['separate_caps']
    elif mutation == 'unknown_counter':
        del budget['separate_caps']['prior']['physics_steps']
    elif mutation == 'unfunded_prior':
        budget['separate_caps']['prior']['policy_queries'] = 100
    elif mutation == 'startup_actions':
        budget['separate_caps']['startup']['policy_control_steps'] = 1
    else:
        budget['separate_caps']['failed_work']['encoder_image_attempts'] = 1
    with pytest.raises(ValueError, match='separate|resource counter|extra policy-action'):
        freeze_main_protocol(build(planned), evidence, created_at=1001,
                              root=planned['root'], config_dir=planned['config'])


def test_identity_choice_indices_reject_boolean_substitution(planned):
    comparator = planned['spec']['comparators'][0]
    path = planned['root'] / comparator['controller']['path']
    record = json.loads(path.read_text())
    record['choices'] = [False, False, False, False]
    write(path, _signed(record))
    comparator['controller'] = source_reference(path, root=planned['root'])
    with pytest.raises(ValueError, match='Identity comparator'):
        build(planned)


def test_unchanged_policy_is_distinct_from_identity_adapter_and_has_no_clip_cost(planned):
    path = planned['root'] / 'original.json'
    write(path, _signed({'kind': 'benchmark_original', 'deterministic': True,
                        'resource_contract': controller_resource_contract('benchmark_original')}))
    planned['spec']['comparators'].append({'method': 'original',
        'controller': source_reference(path, root=planned['root'])})
    jobs = compile_jobs(build(planned))
    original = [job for job in jobs if job['method'] == 'original']
    identity = [job for job in jobs if job['method'] == 'identity']
    assert len(original) == len(identity) == len(planned['spec']['cases'])
    for job in original:
        assert job['actor_sha256'] is None and job['run_kind'] == 'fixed'
        contract = job['resource_contract']
        assert contract['accessible_action_prefix'] == 8
        assert contract['proposal_rng'] == 'single_policy_stream'
        assert contract['policy_proposals_per_query_group'] == 1
        costs = job['resource_envelope']['successful_work']
        assert costs['policy_queries'] == job['action_horizon']
        assert costs['adapter_decisions'] == 0
        assert all(costs[name] == 0 for name in WORK_COUNTS if name.startswith('encoder_'))
    assert all(job['resource_envelope']['successful_work']['encoder_load_attempts'] == 1
               for job in identity)
    _, _, frozen = freeze_fixture(planned)
    job = next(job for job in compile_jobs(frozen) if job['method'] == 'original')
    assert authorize_job(frozen, job, root=planned['root'], config_dir=planned['config'])['authorized']


@pytest.mark.parametrize('change', ['choices', 'cadence', 'encoder'])
def test_unchanged_policy_cannot_silently_become_an_adapter(planned, change):
    path = planned['root'] / 'original.json'
    value = {'kind': 'benchmark_original', 'deterministic': True,
             'resource_contract': controller_resource_contract('benchmark_original')}
    if change == 'choices':
        value['choices'] = [0, 1, 0, 0]
    elif change == 'cadence':
        value['resource_contract']['normal_minimum_actions_between_query_groups'] = 16
    else:
        value['resource_contract']['encoder_load_attempts_per_episode'] = 1
    write(path, _signed(value))
    planned['spec']['comparators'].append({'method': 'original',
        'controller': source_reference(path, root=planned['root'])})
    with pytest.raises(ValueError, match='Unsupported comparator fields|resource declaration'):
        build(planned)
