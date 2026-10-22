"""Cost/performance history tests using synthetic saved coordinator summaries."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_maniguard_pilot_report import pilot_state, resign

from helpers.maniguard_pilot_history import AXES, build_history, write_figures


def state_with_costs():
    state = pilot_state()
    for entry in state['ledger']:
        entry['result']['costs']['policy_control_steps'] = 8
    return resign(state)


def training_groups(report, run='evolution/s17'):
    return [row for row in report['training_evaluations'] if row['run'] == run]


def test_complete_history_retains_all_evaluations_and_charges_both_nominees():
    state = state_with_costs()
    before = copy.deepcopy(state)
    report = build_history(state)
    assert state == before
    assert report['valid'] and report['complete']
    assert len(report['training_evaluations']) == 6 * 24
    assert all(row['valid'] and row['expected_episodes'] == 3 for row in report['training_evaluations'])
    assert len(report['selected_validation']) == 6
    for point in report['selected_validation']:
        assert point['expected_episodes'] == point['received_episodes'] == 84
        assert point['total_cost']['policy_control_steps'] == 84 * 8
        assert point['total_cost']['policy_queries'] == 84 * 4
        assert point['known_costs_by_stage']['training']['policy_control_steps'] == 72 * 8
        assert point['known_costs_by_stage']['finalist_validation']['policy_control_steps'] == 12 * 8
    assert report['shared_reference_costs']['exact']['policy_control_steps'] == 18 * 8
    assert sum(point['total_cost']['policy_control_steps'] for point in report['selected_validation']) + 18 * 8 == 522 * 8
    assert report['allocated_gpu_hours'] is None
    assert report['preparation_startup_idle_costs_included'] is False


def test_interleaved_families_charge_all_earlier_candidates_not_only_point_members():
    state = state_with_costs()
    chosen = [row for row in state['ledger'] if row['job']['stage'] == 'training'
              and row['job']['method'] == 'evolution' and row['job']['optimizer_seed'] == 17
              and row['job']['round'] == 0]
    other = [row for row in state['ledger'] if row not in chosen]
    chosen.sort(key=lambda row: (['jar', 'stack', 'clutter'].index(row['job']['family']),
                                row['job']['candidate_slot']))
    state['ledger'] = chosen + other
    report = build_history(resign(state))
    point = next(row for row in training_groups(report) if row['panel'] == 0 and row['slot_or_batch'] == 0)
    assert point['ledger_index'] == 16
    assert point['cumulative_cost']['policy_control_steps'] == 17 * 8
    assert point['cumulative_cost']['policy_queries'] == 17 * 4
    assert point['metrics']['episodes'] == 3
    last = next(row for row in training_groups(report) if row['panel'] == 0 and row['slot_or_batch'] == 7)
    assert last['cumulative_cost']['policy_control_steps'] == 24 * 8


def test_rotating_panels_and_ppo_sampling_are_explicit_without_best_so_far_fabrication():
    report = build_history(state_with_costs())
    for run in ['evolution/s17', 'random/s17', 'ppo/s17']:
        points = training_groups(report, run)
        assert {point['panel'] for point in points} == {0, 1, 2}
        assert all(point['stochastic_training'] == run.startswith('ppo') for point in points)
        assert all('best_so_far' not in point for point in points)
        assert len(points) == 24


@pytest.mark.parametrize('axis', AXES)
def test_missing_cost_does_not_become_zero_or_an_exact_later_coordinate(axis):
    state = state_with_costs()
    del state['ledger'][0]['result']['costs'][axis]
    report = build_history(resign(state))
    run = f"{state['ledger'][0]['job']['method']}/s{state['ledger'][0]['job']['optimizer_seed']}"
    assert not report['valid']
    assert any(row.get('axis') == axis for row in report['unknown_costs'])
    assert all(point['cumulative_cost'][axis] is None for point in training_groups(report, run))
    selected = next(row for row in report['selected_validation'] if row['run'] == run)
    assert selected['total_cost'][axis] is None
    assert selected['known_total_cost'][axis] > 0


def test_invalid_attempt_preserves_paid_costs_without_scoring_its_group(tmp_path):
    state = state_with_costs()
    state['ledger'][0]['accepted'] = False
    state['ledger'][0]['result']['valid'] = False
    report = build_history(resign(state))
    assert not report['valid']
    first = training_groups(report)[0]
    assert not first['valid'] and first['metrics'] is None
    assert first['cumulative_cost']['policy_control_steps'] == 3 * 8
    assert report['all_recorded_physical_costs']['recorded_totals']['policy_control_steps'] == 522 * 8
    with pytest.raises(ValueError, match='Resolve report bindings'):
        write_figures(report, tmp_path / 'not-created')
    assert not (tmp_path / 'not-created').exists()


def test_missing_family_prevents_partial_macro_or_selection_point():
    state = state_with_costs()
    state['ledger'].pop(0)
    report = build_history(resign(state))
    point = training_groups(report)[0]
    assert not point['complete'] and point['metrics'] is None
    assert point['received_episodes'] == 2
    assert all(value is None for value in point['cumulative_cost'].values())
    selected = next(row for row in report['selected_validation'] if row['run'] == point['run'])
    assert selected['metrics'] is None
    assert all(value is None for value in selected['total_cost'].values())


def test_duplicate_physical_episode_never_adds_a_second_charge():
    state = state_with_costs()
    state['ledger'].append(copy.deepcopy(state['ledger'][0]))
    report = build_history(resign(state))
    assert not report['valid']
    assert any(row.get('reason') == 'Repeated physical identity' for row in report['unknown_costs'])
    point = next(row for row in report['selected_validation'] if row['run'] == 'evolution/s17')
    assert point['known_total_cost']['policy_control_steps'] == 84 * 8


def test_figures_are_standalone_and_do_not_overwrite_output(tmp_path):
    report = build_history(state_with_costs())
    files = write_figures(report, tmp_path / 'figures', title='Synthetic validation only')
    assert len(files) == 6
    assert all(Path(path).stat().st_size > 1000 for path in files)
    assert {Path(path).suffix for path in files} == {'.png', '.pdf'}
    with pytest.raises(FileExistsError):
        write_figures(report, tmp_path / 'figures')
    report['complete'] = False
    with pytest.raises(ValueError, match='hash/schema mismatch'):
        write_figures(report, tmp_path / 'tampered')


def test_cli_writes_consistent_input_snapshot_and_refuses_unknown_cost_figures(tmp_path):
    state = state_with_costs()
    del state['ledger'][0]['result']['costs']['policy_control_steps']
    path = tmp_path / 'state.json'
    path.write_text(json.dumps(resign(state)))
    output = tmp_path / 'report'
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/maniguard_pilot_figures.py'),
               '--state', str(path), '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert (output / 'source-state.json').read_bytes() == path.read_bytes()
    assert not json.loads((output / 'history.json').read_text())['valid']
    assert not (output / 'figures').exists()


def test_inflight_claims_do_not_imply_fully_known_expenditure():
    state = state_with_costs()
    state['claims']['unresolved'] = {'output': '/synthetic/in-flight'}
    report = build_history(resign(state))
    assert not report['valid'] and not report['complete']
    assert any(row.get('reason') == 'In-flight claims have unresolved expenditure'
               for row in report['unknown_costs'])
    assert report['in_flight_claims'] == state['claims']


def test_one_candidate_cannot_pool_outcomes_from_different_actor_weights():
    state = state_with_costs()
    entry = state['ledger'][0]
    entry['job']['actor_sha256'] = entry['result']['actor_sha256'] = 'b' * 64
    report = build_history(resign(state))
    assert not report['valid']
    assert report['history_issues'][0]['reason'] == 'One evaluation group contains different actor hashes'
    assert training_groups(report)[0]['metrics'] is None


def test_duplicate_slot_with_new_physical_record_cannot_replace_accepted_fitness():
    state = state_with_costs()
    duplicate = copy.deepcopy(state['ledger'][0])
    duplicate['accepted'] = False
    duplicate['result']['output'] += '-duplicate'
    duplicate['result']['episode_id'] += '-duplicate'
    duplicate['result']['outcome'].update(C_req=True, S_req=True, V=False, U=1)
    state['ledger'].append(duplicate)
    report = build_history(resign(state))
    assert not report['valid']
    assert training_groups(report)[0]['metrics'] is None
    selected = next(row for row in report['selected_validation'] if row['run'] == 'evolution/s17')
    assert selected['known_total_cost']['policy_control_steps'] == 85 * 8
    assert any(row['reason'] == 'Repeated scheduled slot' for row in report['history_issues'])


def test_missing_prior_panel_poison_later_exact_cost_coordinates():
    state = state_with_costs()
    state['ledger'].pop(0)
    report = build_history(resign(state))
    assert not report['valid']
    assert any(row['reason'] == 'Earlier training dependency missing from ledger prefix'
               for row in report['history_issues'])
    later = [row for row in training_groups(report) if row['panel'] > 0]
    assert all(all(value is None for value in row['cumulative_cost'].values()) for row in later)
    assert all(row['known_cumulative_cost']['policy_control_steps'] > 0 for row in later)


def test_ppo_cannot_skip_an_earlier_training_triplet_within_panel():
    state = state_with_costs()
    first = next(row for row in state['ledger'] if row['job']['stage'] == 'training'
                 and row['job']['method'] == 'ppo' and row['job']['optimizer_seed'] == 17)
    state['ledger'].remove(first)
    report = build_history(resign(state))
    assert not report['valid']
    later = [row for row in training_groups(report, 'ppo/s17') if row['slot_or_batch'] > 0]
    assert all(row['cumulative_cost']['policy_queries'] is None for row in later)


def test_unattributed_attempt_prevents_later_exact_cost_coordinates():
    state = state_with_costs()
    unknown = copy.deepcopy(state['ledger'][0]); unknown['job_id'] = 'unscheduled'
    state['ledger'].insert(0, unknown)
    report = build_history(resign(state))
    assert not report['valid']
    assert all(all(value is None for value in row['cumulative_cost'].values())
               for row in report['training_evaluations'])
