"""Synthetic saved-state comparisons; no simulator, services, or reserved outcomes."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_phase2_coordinator import _metrics
from helpers.maniguard_phase2_model import genome_hash, identity_genome, load_model, random_genome
from helpers.maniguard_phase2_optim import record_hash
from helpers.maniguard_phase2_protocol import build_pilot, compile_schedule
from helpers.maniguard_pilot_report import build_report, export_selected


def resign(value):
    value['sha256'] = record_hash({key: item for key, item in value.items() if key != 'sha256'})
    return value


def pilot_state(rule=None):
    protocol = json.loads(json.dumps(build_pilot()))
    protocol['status'] = 'frozen'
    protocol['references'][1]['selected_choices'] = [0, 0, 1, 0]
    resign(protocol)
    state = {'version': 'maniguard-phase2-coordinator-v1', 'protocol': protocol,
             'source_sha256': protocol['implementation_files']['experiments/helpers/maniguard_phase2_coordinator.py'],
             'ledger': [], 'claims': {}, 'blocked_reasons': [], 'nominees': {}, 'selected': {},
             'searches': {}, 'learners': {}, 'stage_index': 1, 'history': [], 'reference_metrics': {}}
    for method in protocol['methods']:
        for seed in protocol['optimizer_seeds']:
            key = f'{method}/s{seed}'
            state['nominees'][key] = [{'genome': random_genome(np.random.default_rng(seed * 10 + finalist)).tolist()}
                                       for finalist in range(2)]
            state['learners' if method == 'ppo' else 'searches'][key] = {'synthetic_archive': key}
    for template in compile_schedule(protocol):
        job = copy.deepcopy(template)
        training = job['stage'] == 'training'
        reference = job['stage'] == 'reference_validation'
        genome = identity_genome()
        if not training and not reference:
            genome = state['nominees'][f"{job['method']}/s{job['optimizer_seed']}"][job['finalist']]['genome']
        mode = 'contextual' if reference and job['method'] == 'contextual' else 'fixed' if reference else 'learned'
        choices = [0, 0, 1, 0] if reference and job['method'] == 'fixed' else [0, 0, 0, 0] if reference and job['method'] == 'identity' else None
        job.update(actor_sha256=genome_hash(genome), mode=mode, fixed_choices=choices,
                   stochastic=training and job['method'] == 'ppo',
                   sampling_seed=(job['optimizer_seed'] * 1000 + job['ppo_episode']
                                  if training and job['method'] == 'ppo' else 0), purpose=job['split'])
        complete = not training and not reference and job['finalist'] == 0 and job['family'] in ('jar', 'stack')
        violation, queries = False, 4
        if rule is not None:
            complete, violation, queries = rule(job, complete, violation, queries)
        row = {key: copy.deepcopy(job[key]) for key in ('family', 'scene', 'seed', 'sampling_seed',
               'actor_sha256', 'mode', 'fixed_choices', 'stochastic', 'purpose')}
        row.update(output='/synthetic/' + job['job_id'], episode_id='episode:' + job['job_id'],
                   action_horizon=job['max_steps'], full_episode=True, short_check=False, valid=True,
                   invalid_reasons=[], benchmark_result={'ever_contacted': complete},
                   costs={'policy_queries': queries, 'physics_steps': 48, 'elapsed_seconds': 2.0},
                   observed_work={'policy_control_steps': 8}, missing_cost_fields=[], cost_issues=[],
                   outcome={'valid': True, 'C_req': bool(complete), 'V': bool(violation),
                            'S_req': bool(complete and not violation),
                            'U': int(complete and not violation) - 2 * int(violation)})
        row['measured_row_sha256'] = record_hash(row)
        state['ledger'].append({'job_id': job['job_id'], 'job': job, 'result': row, 'accepted': True})
    reselect(state)
    return resign(state)


def reselect(state):
    protocol = state['protocol']
    for key, nominees in state['nominees'].items():
        method, seed = key.split('/s')
        candidates = []
        for finalist in range(2):
            rows = [entry['result'] for entry in state['ledger']
                    if entry['job']['stage'] == 'finalist_validation' and entry['job']['method'] == method
                    and entry['job']['optimizer_seed'] == int(seed) and entry['job']['finalist'] == finalist]
            candidates.append({'finalist': finalist, 'metrics': _metrics(rows, protocol['families'])})
        winner = min(candidates, key=lambda candidate: (-candidate['metrics']['U'], candidate['metrics']['V'],
                     candidate['metrics']['policy_queries'], candidate['finalist']))
        genome = nominees[winner['finalist']]['genome']
        state['selected'][key] = {**winner, 'genome': copy.deepcopy(genome), 'actor_sha256': genome_hash(genome),
                                  'validation_candidates': candidates}


def test_complete_report_retains_every_run_seed_family_and_reference():
    state = pilot_state()
    original = copy.deepcopy(state)
    report = build_report(state)
    assert state == original  # Reporting never mutates optimizer archives or selections.
    assert report['valid'] and report['complete'], report['invalid_reasons']
    assert len(report['all_training_runs']) == 6
    assert all(run['training']['valid_episodes'] == 72 for run in report['all_training_runs'].values())
    assert all(len(run['candidates']) == 2 for run in report['all_training_runs'].values())
    assert all(set(run['selected']['metrics']['per_family']) == {'jar', 'stack', 'clutter'}
               for run in report['all_training_runs'].values())
    assert len(report['comparisons']) == 30  # Three references and two same-seed methods per run.
    assert all(row['physical_episodes'] == 6 for row in report['references'].values())
    assert report['screening']['decision'] == 'completion_semantics_review_required'
    assert not report['screening']['evolutionary_superiority_claim']
    assert all(row['observed_completion_and_utility_at_least_evolution_in_both_seeds']
               for row in report['screening']['non_evolutionary_comparisons'].values())
    assert not any(row['equivalence_established']
                   for row in report['screening']['non_evolutionary_comparisons'].values())
    assert not report['screening']['safety_improvement_claim']
    assert report['costs']['allocated_gpu_hours'] is None
    reviewed = build_report(state, completion_review={'passed': True, 'evidence': 'synthetic review'})
    assert reviewed['screening']['decision'] == 'proceed_to_larger_study_screening_only'


@pytest.mark.parametrize('criterion,winner', [('utility', 0), ('violation', 1), ('queries', 1), ('nominee', 0)])
def test_selection_reconstructs_coordinator_ties_in_the_frozen_order(criterion, winner):
    def rule(job, complete, violation, queries):
        if job['stage'] != 'finalist_validation':
            return False, False, queries
        # Cases follow two scenes per family. Four completions and one violation
        # beat one safe completion by utility, despite greater observed risk.
        case = int(job['job_id'].rsplit('case', 1)[1])
        if criterion == 'utility':
            return (case < 4, case == 4, queries) if job['finalist'] == 0 else (case == 0, False, queries)
        if criterion == 'violation':
            return (case < 2, case == 2, queries) if job['finalist'] == 0 else (False, False, queries)
        return False, False, 2 if criterion == 'queries' and job['finalist'] == 1 else 4
    report = build_report(pilot_state(rule))
    assert report['valid'], report['invalid_reasons']
    assert {run['selected']['finalist'] for run in report['all_training_runs'].values()} == {winner}
    if criterion == 'utility':
        assert any(row['metric'] == 'V' and row['delta'] > 0 for row in report['regressions'])


def test_best_seed_cannot_hide_a_second_seed_regression():
    def rule(job, complete, violation, queries):
        if job['stage'] == 'reference_validation':
            return job['family'] == 'clutter', False, queries
        if job['stage'] == 'finalist_validation':
            return job['optimizer_seed'] == 17 and job['finalist'] == 0, False, queries
        return complete, violation, queries
    report = build_report(pilot_state(rule), completion_review={'passed': True})
    assert report['valid'] and report['complete']
    assert len(report['all_training_runs']) == 6
    assert not any(row['numerical_screen_passed'] for row in report['screening']['methods'].values())
    assert any(row['optimizer_seed'] == 29 and row['metric'] == 'S_req' and row['delta'] < 0
               for row in report['regressions'])
    assert report['method_metrics_all_seeds']['evolution']['episodes'] == 12


def test_invalid_or_missing_work_preserves_costs_and_blocks_screening_and_export(tmp_path):
    state = pilot_state()
    invalid = state['ledger'][0]
    invalid['accepted'] = False
    invalid['result'].update(valid=False, invalid_reasons=['native crash'], missing_cost_fields=['encoder_seconds'])
    state['claims']['unresolved'] = {'output': '/synthetic/in-flight'}
    resign(state)
    report = build_report(state)
    assert not report['complete'] and not report['valid']
    assert report['screening']['decision'] == 'repair_experiment'
    assert invalid['job_id'] in report['missing_or_invalid_jobs']
    assert report['costs']['recorded_totals']['physics_steps'] == 522 * 48
    assert report['costs']['unknown_records'] and report['in_flight_claims']
    with pytest.raises(ValueError, match='complete consistent pilot'):
        export_selected(state, tmp_path / 'exports')
    assert not (tmp_path / 'exports').exists()


def test_missing_validation_row_never_becomes_a_zero_outcome():
    state = pilot_state()
    missing = next(entry for entry in state['ledger'] if entry['job']['stage'] == 'finalist_validation')
    state['ledger'].remove(missing)
    resign(state)
    report = build_report(state)
    assert not report['complete'] and missing['job_id'] in report['missing_or_invalid_jobs']
    key = f"{missing['job']['method']}/s{missing['job']['optimizer_seed']}"
    assert report['all_training_runs'][key]['selected'] is None
    assert report['screening']['decision'] == 'incomplete_pilot'


@pytest.mark.parametrize('tamper', ['state_hash', 'source_pin', 'selected_actor', 'selection', 'dispatch', 'outcome'])
def test_saved_state_tampering_is_detected(tamper):
    state = pilot_state()
    if tamper == 'state_hash':
        state['ledger'][0]['result']['costs']['physics_steps'] += 1
        with pytest.raises(ValueError, match='state content hash'):
            build_report(state)
        return
    if tamper == 'source_pin':
        state['source_sha256'] = '0' * 64
        resign(state)
        with pytest.raises(ValueError, match='source differs'):
            build_report(state)
        return
    if tamper == 'selected_actor':
        state['selected']['evolution/s17']['genome'][0] += 1
    elif tamper == 'selection':
        state['selected']['evolution/s17']['finalist'] = 1
    elif tamper == 'dispatch':
        state['ledger'][0]['job']['scene'] = 'wrong/base'
    else:
        state['ledger'][0]['result']['outcome']['U'] = 1
    report = build_report(resign(state))
    assert not report['valid'] and report['screening']['decision'] == 'repair_experiment'


def test_duplicate_physical_evidence_cannot_increase_fitness_or_double_cost():
    state = pilot_state()
    state['ledger'].append(copy.deepcopy(state['ledger'][0]))
    report = build_report(resign(state))
    assert not report['valid'] and not report['complete']
    assert report['costs']['recorded_totals']['physics_steps'] == 522 * 48
    assert any('Duplicated physical' in reason for reason in report['invalid_reasons'])


def test_all_six_selected_actors_export_and_round_trip_with_full_state(tmp_path):
    state = pilot_state()
    manifest = export_selected(state, tmp_path / 'exports')
    assert len(manifest['exports']) == 6 and manifest['all_training_runs_preserved']
    for item in manifest['exports']:
        genome, _, metadata = load_model(item['path'])
        assert genome_hash(genome) == item['actor_sha256']
        assert metadata['run'] == item['run'] and metadata['state_sha256'] == state['sha256']
    saved = json.loads((tmp_path / 'exports/source-state.json').read_text())
    assert saved == state
    assert len(saved['ledger']) == 522
    with pytest.raises(FileExistsError):
        export_selected(state, tmp_path / 'exports')


def test_completion_gain_with_more_violations_requires_explicit_tradeoff_review():
    def rule(job, complete, violation, queries):
        if job['stage'] == 'finalist_validation' and job['finalist'] == 0:
            case = int(job['job_id'].rsplit('case', 1)[1])
            return case < 4, case == 4, queries
        return False, False, queries
    report = build_report(pilot_state(rule), completion_review={'passed': True})
    assert report['screening']['decision'] == 'explicit_completion_violation_tradeoff_review_required'
    assert all(row['violation_increases'] for row in report['screening']['methods'].values())
    assert not report['screening']['safety_improvement_claim']


def test_activated_decisions_without_completion_gain_support_interface_review_only():
    state = pilot_state(lambda job, complete, violation, queries: (False, False, queries))
    report = build_report(state, mechanism_review={'decisions_activate': True, 'evidence': 'synthetic mechanism check'})
    assert report['screening']['decision'] == 'reconsider_interface_after_reviewing_mechanisms'
    assert not report['screening']['statistical_significance_claim']


def test_report_scope_is_saved_summary_consistency_not_independent_physics_audit():
    report = build_report(pilot_state(), completion_review={'passed': True})
    assert report['valid'] and report['screening']['decision'] == 'proceed_to_larger_study_screening_only'
    assert report['independent_audit'] == {
        'episode_artifacts_reaudited': False, 'protocol_freeze_evidence_revalidated': False,
        'optimizer_archives_replayed': False, 'screening_authorizes_execution': False}
    assert 'saved dispatch/result summaries' in report['validity_scope']


def test_conflicting_physical_ids_make_cost_deduplication_explicitly_uncertain():
    state = pilot_state()
    duplicate = copy.deepcopy(state['ledger'][0])
    duplicate['result']['episode_id'] += '-conflicting-label'
    state['ledger'].append(duplicate)
    report = build_report(resign(state))
    assert not report['valid'] and not report['complete']
    assert report['costs']['recorded_totals_may_double_count']
    assert any('deduplication' in item.get('reason', '') for item in report['costs']['unknown_records'])
    assert report['screening']['decision'] == 'repair_experiment'


def test_exported_selection_records_feed_the_main_planner_and_preserve_all_seeds(tmp_path):
    from test_maniguard_main_protocol import planned

    from helpers.maniguard_main_protocol import build_main_protocol, compile_jobs

    fixture = planned.__wrapped__(tmp_path)
    manifest = export_selected(pilot_state(), fixture['root'] / 'pilot-export')
    fixture['spec']['training_runs'] = manifest['training_runs']
    fixture['spec']['selected_actors'] = manifest['selected_actors']
    draft = build_main_protocol(fixture['spec'], root=fixture['root'], config_dir=fixture['config'])
    trained = [controller for controller in draft['controllers'] if controller['run_kind'] == 'trained']
    assert len(trained) == 6
    assert {(row['method'], row['optimizer_seed']) for row in trained} == {
        (method, seed) for method in ('evolution', 'random', 'ppo') for seed in (17, 29)}
    assert len(compile_jobs(draft)) == 7 * 12  # Six retained runs plus one separately paid reference.
    assert not manifest['main_evaluation_authorized'] and draft['status'] == 'draft'
    for selected in manifest['selected_actors']:
        record = json.loads(Path(selected['selection']['path']).read_text())
        assert record['selection_partition'] == 'validation'
        assert record['training_families'] == ['jar', 'stack', 'clutter']
        assert record['source_state'] == manifest['source_state']
        assert record['comparison_report'] == manifest['comparison_report']
        assert record['selected_actor_sha256'] == selected['actor_sha256']


def test_real_coordinator_serialization_reproduces_full_synthetic_pilot_selection(tmp_path, monkeypatch):
    from test_maniguard_phase2_coordinator import measured_row

    from helpers import maniguard_phase2_coordinator as coordination

    protocol = build_pilot()
    protocol['status'] = 'frozen'
    protocol['references'][1]['selected_choices'] = [0, 0, 1, 0]
    resign(protocol)
    # Isolate state serialization; this deliberately supplies no real freeze evidence.
    monkeypatch.setattr(coordination, 'validate_protocol', lambda *args, **kwargs: {'ready_to_run': True})
    coordinator = coordination.PilotCoordinator(protocol)
    while coordinator.status != 'complete':
        jobs = coordinator.next_jobs()
        assert jobs and coordinator.status != 'blocked'
        for job in jobs:
            output = tmp_path / 'synthetic' / job['job_id']
            coordinator.mark_started(job['job_id'], output)
            row = measured_row(job, output, utility=1, queries=4)
            coordinator.record_result(job['job_id'], row)
    path = tmp_path / 'coordinator.json'
    coordinator.save(path)
    report = build_report(path)
    assert report['valid'] and report['complete'], report['invalid_reasons']
    assert len(report['attempts']) == 522
    assert report['costs']['recorded_totals']['physics_steps'] == 522 * 48
    for key, selected in coordinator.selected.items():
        reported = report['all_training_runs'][key]['selected']
        assert reported['actor_sha256'] == selected['actor_sha256']
        assert reported['finalist'] == selected['finalist']
    assert not report['independent_audit']['optimizer_archives_replayed']
