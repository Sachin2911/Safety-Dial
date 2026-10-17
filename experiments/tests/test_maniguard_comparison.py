"""Keep unmatched trials and outcome tradeoffs visible in condition comparisons."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_maniguard_results import write_fixture

from helpers.maniguard_comparison import compare_batch

ENTRIES = [('identity', 0, True, False), ('motion', 0, False, True),
           ('identity', 1, True, True), ('motion', 1, True, False)]


def write_batch(path, entries=ENTRIES):
    jobs, processes = [], []
    for index, (label, seed, success, violation) in enumerate(entries):
        adapter = {'kind': 'identity'} if label == 'identity' else {'kind': 'motion'}
        jobs.append({'family': 'jar', 'scene': 'task_0000/base', 'seed': seed,
                     'label': label, 'adapter': adapter})
        episode = path / f'episode-{index:04d}'
        files = write_fixture(episode, success=success, violation=violation)
        (episode / 'job.json').write_text(json.dumps(files['job.json'] | {
            'seed': seed, 'adapter': adapter,
        }))
        (episode / 'ledger.json').write_text(json.dumps(files['ledger.json'] | {
            'physics_steps': 12, 'physics_seconds': 0.1, 'env_step_attempts': 2,
            'env_steps_returned': 2, 'policy_queries': 1, 'policy_query_seconds': 0.01,
            'elapsed_seconds': 1.0,
        }))
        processes.append({'index': index, 'returncode': 0, 'elapsed_seconds': 1.1})
    (path / 'plan.json').write_text(json.dumps({'split': 'engineering', 'jobs': jobs}))
    (path / 'processes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in processes))
    (path / 'complete.json').write_text(json.dumps({'episodes': len(jobs)}))


def test_equal_safe_completion_counts_retain_gains_losses_and_unsafe_outcomes(tmp_path):
    write_batch(tmp_path)
    comparison = compare_batch(tmp_path)['comparisons'][0]
    assert comparison['completed_pairs'] == comparison['planned_pairs'] == 2
    assert comparison['all_cases_complete']
    assert comparison['safe_completion_gains'] == comparison['safe_completion_losses'] == 1
    assert comparison['violation_count_difference'] == 0
    transitions = comparison['outcome_transitions']
    assert transitions['safe_completion']['unsafe_unfinished'] == 1
    assert transitions['unsafe_completion']['safe_completion'] == 1
    assert comparison['paired_outcome_counts']['candidate']['unsafe_completion'] == 0
    assert comparison['completed_pair_cost_totals']['control']['physics_steps'] == 24


def test_incomplete_episode_does_not_enter_the_completed_pair_denominator(tmp_path):
    write_batch(tmp_path)
    (tmp_path / 'complete.json').unlink()
    process_path = tmp_path / 'processes.jsonl'
    records = [json.loads(s) for s in process_path.read_text().splitlines()]
    process_path.write_text(''.join(json.dumps(r) + '\n' for r in records if r['index'] != 1))
    comparison = compare_batch(tmp_path)['comparisons'][0]
    assert comparison['planned_pairs'] == 2 and comparison['completed_pairs'] == 1
    assert not comparison['all_cases_complete']
    assert comparison['cases'][0]['candidate']['status'] == 'awaiting_process_record'
    assert comparison['safe_completion_losses'] == 0
    assert comparison['paired_outcome_counts']['control']['safe_completion'] == 0


def test_missing_planned_control_remains_visible(tmp_path):
    write_batch(tmp_path, [ENTRIES[0], ENTRIES[1], ENTRIES[3]])
    comparison = compare_batch(tmp_path)['comparisons'][0]
    assert comparison['case_slots'] == 2
    assert comparison['planned_pairs'] == comparison['completed_pairs'] == 1
    assert not comparison['all_cases_complete']
    assert comparison['cases'][1]['control'] == {'status': 'not_planned'}
    assert comparison['cases'][1]['candidate']['outcome'] == 'safe_completion'


def test_reused_controls_do_not_inflate_whole_batch_costs(tmp_path):
    write_batch(tmp_path, ENTRIES + [('other', 0, False, False), ('other', 1, False, False)])
    report = compare_batch(tmp_path)
    assert len(report['comparisons']) == 2
    assert report['batch_report']['recorded_cost_totals']['physics_steps'] == 72
    assert all(c['completed_pair_cost_totals']['control']['physics_steps'] == 24
               for c in report['comparisons'])


def test_duplicate_condition_case_is_rejected_instead_of_silently_overwriting(tmp_path):
    write_batch(tmp_path, ENTRIES + [ENTRIES[1]])
    with pytest.raises(ValueError, match='Duplicate condition/scene/seed'):
        compare_batch(tmp_path)


def test_missing_query_cost_does_not_become_a_zero_cost_claim(tmp_path):
    write_batch(tmp_path)
    path = tmp_path / 'episode-0001/ledger.json'
    ledger = json.loads(path.read_text())
    del ledger['policy_queries']
    path.write_text(json.dumps(ledger))
    report = compare_batch(tmp_path)
    comparison = report['comparisons'][0]
    assert comparison['completed_pairs'] == 2
    assert comparison['completed_pair_cost_totals'] is None
    assert comparison['cases'][0]['candidate']['costs']['policy_queries'] is None
    assert report['batch_report']['started_episodes_missing_cost_fields']['policy_queries'] == 1


def test_absent_control_cannot_be_inferred_from_another_label(tmp_path):
    write_batch(tmp_path)
    with pytest.raises(ValueError, match='Requested control'):
        compare_batch(tmp_path, 'missing')
