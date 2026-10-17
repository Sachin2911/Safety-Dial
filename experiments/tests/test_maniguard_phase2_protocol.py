"""Verify actual schedule arithmetic, exposure isolation and readiness enforcement."""

import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_phase2_model import genome_sha256
from helpers.maniguard_phase2_optim import WeightSearch, record_hash
from helpers.maniguard_phase2_protocol import (
    FAMILIES,
    METHODS,
    SEARCH_SEEDS,
    build_pilot,
    compile_schedule,
    pilot_report,
    resource_envelope,
    validate_protocol,
)


def resign(protocol):
    protocol['sha256'] = record_hash({k: v for k, v in protocol.items() if k != 'sha256'})
    return protocol


def test_compiled_schedule_has_exact_counts_and_cost_envelope():
    protocol = build_pilot()
    jobs = compile_schedule(protocol)
    assert len(jobs) == 522
    assert Counter(job['stage'] for job in jobs) == {
        'training': 432, 'finalist_validation': 72, 'reference_validation': 18}
    assert len({job['job_id'] for job in jobs}) == 522
    assert {job['family'] for job in jobs} == set(FAMILIES)
    assert {job['split'] for job in jobs} == {'train', 'validation'}
    resources = resource_envelope(jobs)
    assert resources['scheduled_policy_actions_max'] == 1270200
    assert resources['preparation_policy_actions_max'] == 58400
    assert resources['short_policy_actions_max'] == 384
    assert resources['policy_actions_max'] == 1328984
    assert resources['control_physics_ticks_max'] == 7973904
    assert resources['normal_interval_successful_queries_max'] == 332246
    assert resources['all_full_episodes_max'] == 546
    assert resources['gpu_hours_status'] == 'provisional_until_preparation_costing'


def test_every_method_gets_identical_scene_seed_exposure_on_each_panel():
    protocol = build_pilot()
    jobs = compile_schedule(protocol)
    for seed in SEARCH_SEEDS:
        for panel_index, panel in enumerate(protocol['training_panels']):
            assert all(case['policy_seed'] == panel_index for case in panel['cases'])
            exposures = []
            for method in METHODS:
                rows = [job for job in jobs if job['stage'] == 'training' and
                        job['optimizer_seed'] == seed and job['method'] == method and
                        job['round'] == panel_index]
                assert len(rows) == 24
                exposures.append(Counter((j['family'], j['scene'], j['seed']) for j in rows))
            assert exposures[0] == exposures[1] == exposures[2]
            assert list(exposures[0].values()) == [8, 8, 8]
    for family in FAMILIES:
        scene_ids = [next(c['scene'] for c in p['cases'] if c['family'] == family)
                     for p in protocol['training_panels']]
        assert len(set(scene_ids)) == 3


def test_ppo_order_balances_each_update_and_nomination_is_48_72():
    jobs = compile_schedule(build_pilot())
    for seed in SEARCH_SEEDS:
        rows = [job for job in jobs if job['stage'] == 'training' and
                job['method'] == 'ppo' and job['optimizer_seed'] == seed]
        assert [job['ppo_episode'] for job in rows] == list(range(1, 73))
        for start in range(0, 72, 3):
            batch = rows[start:start + 3]
            assert [job['family'] for job in batch] == list(FAMILIES)
            assert len({job['ppo_batch'] for job in batch}) == 1
            assert len({job['controller_ref'] for job in batch}) == 1
        references = {job['controller_ref'] for job in jobs if
                      job['stage'] == 'finalist_validation' and job['method'] == 'ppo' and
                      job['optimizer_seed'] == seed}
        assert references == {f'ppo/s{seed}/episodes48', f'ppo/s{seed}/episodes72'}


def test_initial_candidate_hashes_match_both_actual_search_methods():
    protocol = build_pilot()
    for seed in SEARCH_SEEDS:
        wanted = protocol['initialization']['shared_first_panel_sha256'][str(seed)]
        for method in ('evolution', 'random'):
            candidates = WeightSearch(method, seed).ask()
            assert [row['sha256'] for row in candidates] == wanted
            assert [genome_sha256(row['genome']) for row in candidates] == wanted


def test_validation_cases_identical_for_every_finalist_and_reference():
    jobs = compile_schedule(build_pilot())
    groups = {}
    for job in jobs:
        if job['split'] == 'validation':
            groups.setdefault(job['controller_ref'], []).append(
                (job['family'], job['scene'], job['seed']))
    assert len(groups) == 15  # twelve finalists plus three references
    first = next(iter(groups.values()))
    assert len(first) == 6 and all(seed == 10 for _, _, seed in first)
    assert all(cases == first for cases in groups.values())


def test_read_only_report_explicitly_refuses_unfinished_freeze():
    protocol = build_pilot()
    report = pilot_report(protocol)
    assert report['ready_to_run'] is False
    assert report['simulator_executed'] is False
    assert report['held_out_outcomes_loaded'] is False
    assert 'measured_costing' in report['unresolved_freeze_checks']
    with pytest.raises(ValueError, match='freeze incomplete'):
        validate_protocol(protocol, require_frozen=True)
    protocol['status'] = 'frozen'
    resign(protocol)
    with pytest.raises(ValueError, match='unresolved requirements'):
        validate_protocol(protocol)


def test_transfer_exposure_reserved_scenes_and_shortened_horizons_rejected():
    original = build_pilot()
    changed = deepcopy(original)
    changed['training_panels'][0]['cases'][0]['family'] = 'cabinet'
    with pytest.raises(ValueError, match='training_panels'):
        validate_protocol(resign(changed))
    changed = deepcopy(original)
    changed['training_panels'][0]['cases'][0]['scene'] = 'task_0023/base'
    with pytest.raises(ValueError, match='training_panels'):
        validate_protocol(resign(changed))
    changed = deepcopy(original)
    changed['training_panels'][0]['cases'][0]['max_steps'] = 32
    with pytest.raises(ValueError, match='training_panels'):
        validate_protocol(resign(changed))


def test_cost_override_and_unsubstantiated_readiness_are_rejected():
    protocol = build_pilot()
    changed = deepcopy(protocol)
    changed['resource_envelope']['gpu_hours_ceiling'] = 720
    with pytest.raises(ValueError, match='Resource bound changed'):
        validate_protocol(resign(changed))
    changed = deepcopy(protocol)
    changed['freeze_checks']['feature_contract']['passed'] = True
    with pytest.raises(ValueError, match='hashed evidence'):
        validate_protocol(resign(changed))


def test_changed_inputs_or_source_pins_fail_before_execution():
    protocol = build_pilot()
    path = next(iter(protocol['input_files']))
    protocol['input_files'][path] = '0' * 64
    with pytest.raises(ValueError, match='input/source changed'):
        validate_protocol(resign(protocol))
    protocol = build_pilot()
    protocol['optimizer_seeds'] = [17, 30]
    with pytest.raises(ValueError, match='optimizer_seeds'):
        validate_protocol(resign(protocol))


def test_freeze_covers_scoring_action_execution_and_launch_sources():
    protocol = build_pilot()
    for name in ("completion", "preparation"):
        assert f"configs/maniguard/phase2/{name}.json" in protocol["input_files"]
    for name in ("adapter", "runtime", "source", "outcomes", "results", "coordinator", "services"):
        key = f"experiments/helpers/maniguard_phase2_{name}.py"
        assert key in protocol["implementation_files"]
        changed = deepcopy(protocol)
        changed["implementation_files"][key] = "0" * 64
        with pytest.raises(ValueError, match="input/source changed"):
            validate_protocol(resign(changed))
    for key in ("experiments/scripts/maniguard_phase2_episode.py",
                "experiments/scripts/maniguard_phase2_train.py",
                "experiments/helpers/maniguard_runtime.py",
                "experiments/scripts/maniguard_serve.py"):
        assert key in protocol["implementation_files"]


@pytest.fixture
def completed_freeze_evidence(tmp_path, monkeypatch):
    """Real immutable-report/freeze pipeline over synthetic audited episode rows."""
    import json
    import shutil

    import numpy as np

    from helpers import maniguard_phase2_preparation_report as reporting
    from helpers.maniguard_phase2_protocol import IMPLEMENTATION_FILES, REVIEW_SCHEMA, ROOT, _sha
    from helpers.maniguard_phase2_services import PRIOR_COUNTS

    root = tmp_path / 'repo'
    config = root / 'configs/maniguard/phase2'
    shutil.copytree(ROOT / 'configs/maniguard', root / 'configs/maniguard')
    for name in IMPLEMENTATION_FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    draft = build_pilot(config, root)
    preparation, setup, directory = root / 'preparation', root / 'setup', root / 'completed-report'

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + '\n')

    manifest = json.loads((config / 'preparation.json').read_text())
    write(preparation / 'preparation.json', manifest)
    write(root / 'draft.json', draft)
    write(setup / 'reservation.json', {'start_unix_seconds': 1000, 'allocated_gpus': 1})
    write(setup / 'encoder-smoke.json', {'finite': True, 'trainable_parameters': 0,
          'costs': {'encoder_image_calls': 5, 'encoder_images': 10, 'encoder_text_calls': 1}})
    write(setup / 'encoder-loader-failure-v1.json', {'counts': dict.fromkeys(PRIOR_COUNTS, 0)})
    measured = {}
    features = np.random.default_rng(13).uniform(-1, 1, size=(8, 64)).tolist()
    for job in manifest['short_jobs'] + manifest['full_jobs']:
        short = job['max_steps'] is not None
        output = preparation / job['job_id']
        value = int(job['job_id'].endswith('cap006'))
        costs = dict.fromkeys(PRIOR_COUNTS, 1)
        costs.update(policy_control_steps=32 if short else 100, physics_steps=200,
                     policy_queries=4, actor_rpc_attempts=0)
        ledger = {**costs, 'phase': 'completed', 'elapsed_seconds': 10,
                  'physics_accounting_attached': True}
        row = {'valid': True, 'family': job['family'], 'scene': job['scene'], 'seed': job['seed'],
               'purpose': job['purpose'], 'mode': job['mode'], 'fixed_choices': job['fixed_choices'],
               'full_episode': not short, 'short_check': short, 'steps': costs['policy_control_steps'],
               'costs': costs, 'ledger': ledger, 'outcome': {'U': value, 'V': False, 'S_req': bool(value)},
               'decisions': [{'features': feature} for feature in features]}
        measured[str(output)] = row
        write(output / 'job.json', {**job, 'max_steps_override': job['max_steps'],
              'phase': 'phase2_baseline_check' if job['baseline_executor'] else 'phase2'})
        write(preparation / 'attempts' / (job['job_id'] + '.json'), {
            'job': job, 'output': str(output), 'short_check': short,
            'protocol_sha256': manifest['sha256'], 'status': 'valid', 'returncode': 0,
            'ledger': ledger, 'elapsed_seconds': 10, 'timeout_seconds': 900})
    monkeypatch.setattr(reporting, 'ROOT', root)
    monkeypatch.setattr(reporting, 'read_episode', lambda path: deepcopy(measured[str(path)]))
    monkeypatch.setattr(reporting, 'read_original', lambda path: deepcopy(measured[str(path)]))
    monkeypatch.setattr(reporting, 'audit_sources', lambda path: {'valid': True})
    report = reporting.build_report(preparation, setup, directory, pilot_path=root / 'draft.json',
                                    timestamp=1100)
    assert report['ready_for_freeze_review'] and report['throughput']['ready_for_freeze_review']
    artifact = root / 'reviews/findings.txt'
    artifact.parent.mkdir()
    artifact.write_text('Synthetic fixture evidence; no simulator work or rollout equality claim.\n')
    findings = {
        'identity_equivalence': {'same_frame_transformation_preserved': True,
                                 'eight_action_query_cadence_preserved': True,
                                 'trajectory_identity_claimed': False,
                                 'recorded_initial_images_match': False},
        'completion_semantics': {'requested_sequence_checklists_reviewed':
                                 ['cabinet', 'clutter', 'dusty', 'jar', 'lid', 'stack'],
                                 'benchmark_monitor_unchanged': True,
                                 'completion_signals_excluded_from_actor_critic': True},
        'runtime_execution_audit': {'tests_passed': True,
                                    'fresh_process_and_resume_audited': True,
                                    'cost_and_invalid_result_paths_audited': True},
    }
    reviews = {}
    for name, details in findings.items():
        evidence = {'schema': REVIEW_SCHEMA, 'check': name, 'passed': True,
                    'preparation_report_sha256': report['sha256'],
                    'input_files': draft['input_files'], 'implementation_files': draft['implementation_files'],
                    'findings': details, 'sources': [{'path': str(artifact), 'sha256': _sha(artifact)}]}
        resign(evidence)
        reviews[name] = root / 'reviews' / (name + '.json')
        write(reviews[name], evidence)
    return {'root': root, 'config': config, 'draft': draft, 'report': report,
            'path': directory / 'report.json', 'reviews': reviews, 'write': write}


def freeze_fixture(fixture):
    from helpers.maniguard_phase2_protocol import freeze_pilot

    return freeze_pilot(fixture['path'], fixture['reviews'], fixture['config'], fixture['root'],
                        verify_upstream=False)


def test_freeze_derives_selected_reference_and_preserves_optimizer_settings(completed_freeze_evidence):
    fixture = completed_freeze_evidence
    from helpers.maniguard_phase2_protocol import _sha

    before = _sha(fixture['config'] / 'pilot.json')
    frozen = freeze_fixture(fixture)
    assert frozen['status'] == 'frozen'
    assert frozen['references'][1]['selected_candidate'] == 'cap006'
    assert frozen['references'][1]['selected_choices'] == [0, 2, 0, 0]
    assert frozen['references'][2]['selected_rule'] == fixture['draft']['references'][2]['proposed_rule']
    assert frozen['objective']['completion_checklists'] == 'source_and_preparation_audited'
    for key in ('model', 'search_config', 'ppo_config', 'initialization', 'selection',
                'training_panels', 'validation_cases', 'schedule_sha256'):
        assert frozen[key] == fixture['draft'][key]
    assert frozen['resource_envelope']['policy_actions_max'] == 1328984
    assert frozen['resource_envelope']['gpu_hours_ceiling'] == 72
    assert 'overhead are excluded' in frozen['resource_envelope']['estimate_scope']
    assert _sha(fixture['config'] / 'pilot.json') == before  # Pure builder never writes a freeze.
    assert validate_protocol(frozen, fixture['config'], fixture['root'], require_frozen=True,
                             verify_upstream=False)['ready_to_run'] is True


@pytest.mark.parametrize('field,value', [('optimizer_seeds', [17, 31]),
                                        ('reference', [0, 0, 1, 0]),
                                        ('objective', 'invented_completion')])
def test_frozen_candidate_cannot_override_optimization_or_evidence_choice(
        completed_freeze_evidence, field, value):
    fixture = completed_freeze_evidence
    frozen = freeze_fixture(fixture)
    if field == 'reference':
        frozen['references'][1]['selected_choices'] = value
    elif field == 'objective':
        frozen['objective']['completion_checklists'] = value
    else:
        frozen[field] = value
    with pytest.raises(ValueError, match='Frozen pilot'):
        validate_protocol(resign(frozen), fixture['config'], fixture['root'], verify_upstream=False)


def test_freeze_rejects_forged_summary_despite_rehashed_report(completed_freeze_evidence):
    fixture = completed_freeze_evidence
    report = deepcopy(fixture['report'])
    report['reference_selection']['fixed']['name'] = 'smooth075'
    report['reference_selection']['fixed']['choices'] = [0, 0, 1, 0]
    fixture['write'](fixture['path'], resign(report))
    with pytest.raises(ValueError, match='Reference selection does not reproduce'):
        freeze_fixture(fixture)


def test_freeze_rejects_missing_review_and_overstated_identity_claim(completed_freeze_evidence):
    import json

    from helpers.maniguard_phase2_protocol import freeze_pilot

    fixture = completed_freeze_evidence
    with pytest.raises(ValueError, match='explicit identity'):
        freeze_pilot(fixture['path'], {}, fixture['config'], fixture['root'], verify_upstream=False)
    path = fixture['reviews']['identity_equivalence']
    evidence = json.loads(path.read_text())
    evidence['findings']['trajectory_identity_claimed'] = True
    fixture['write'](path, resign(evidence))
    with pytest.raises(ValueError, match='scoped findings: identity_equivalence'):
        freeze_fixture(fixture)


def test_freeze_rejects_changed_captured_source_and_incomplete_work(completed_freeze_evidence):
    fixture = completed_freeze_evidence
    source = fixture['path'].parent / 'engineering-features.json'
    source.write_text('[]\n')
    with pytest.raises(ValueError, match='Changed or duplicate freeze evidence source'):
        freeze_fixture(fixture)
    report = deepcopy(fixture['report'])
    report['preparation_complete'] = False
    fixture['write'](fixture['path'], resign(report))
    with pytest.raises(ValueError, match='incomplete or contains invalid'):
        freeze_fixture(fixture)


def test_freeze_does_not_raise_the_gpu_ceiling_to_fit_slow_measurements(completed_freeze_evidence):
    from helpers.maniguard_phase2_preparation_report import throughput_estimate

    fixture = completed_freeze_evidence
    report = deepcopy(fixture['report'])
    for attempt in report['terminal_attempts']:
        attempt['elapsed_seconds'] *= 100
    report['throughput'] = throughput_estimate(fixture['draft'], report['terminal_attempts'],
                                              report['reservation'])
    assert report['throughput']['ready_for_freeze_review']
    assert report['throughput']['total_gpu_hours_estimate'] > 72
    fixture['write'](fixture['path'], resign(report))
    with pytest.raises(ValueError, match='does not fit the fixed GPU ceiling'):
        freeze_fixture(fixture)


def test_freeze_rejects_changed_selection_code_even_with_updated_snapshot_hash(
        completed_freeze_evidence):
    from helpers.maniguard_phase2_protocol import _sha

    fixture = completed_freeze_evidence
    report = deepcopy(fixture['report'])
    path = fixture['path'].parent / 'audit_source/experiments/helpers/maniguard_phase2_preparation_report.py'
    path.write_text('# changed selection implementation\n')
    for source in report['sources']:
        if source['path'] == str(path):
            source['sha256'] = _sha(path)
    fixture['write'](fixture['path'], resign(report))
    with pytest.raises(ValueError, match='audit implementation changed'):
        freeze_fixture(fixture)


def test_frozen_contextual_threshold_cannot_be_tuned_by_editing_the_candidate(completed_freeze_evidence):
    fixture = completed_freeze_evidence
    frozen = freeze_fixture(fixture)
    frozen['references'][2]['selected_rule']['comparison'] = 'max(abs(slice)) > 0.2'
    with pytest.raises(ValueError, match='differs from its reproducible'):
        validate_protocol(resign(frozen), fixture['config'], fixture['root'], verify_upstream=False)


def test_freeze_checks_exact_prior_schema_even_when_report_and_prior_agree(completed_freeze_evidence):
    import json

    from helpers.maniguard_phase2_protocol import _sha

    fixture = completed_freeze_evidence
    report = deepcopy(fixture['report'])
    del report['total_counts_known']['encoder_images']
    fixture['write'](fixture['path'], resign(report))
    prior_path = fixture['path'].parent / 'prior-accounting.json'
    prior = json.loads(prior_path.read_text())
    del prior['counts']['encoder_images']
    for source in prior['sources']:
        if source['path'] == str(fixture['path']):
            source['sha256'] = _sha(fixture['path'])
    fixture['write'](prior_path, resign(prior))
    for path in fixture['reviews'].values():
        review = json.loads(path.read_text())
        review['preparation_report_sha256'] = report['sha256']
        fixture['write'](path, resign(review))
    with pytest.raises(ValueError, match='every nonnegative attempted-work counter'):
        freeze_fixture(fixture)
