"""Reporting must retain failed work and reject mismatched or overstated completion."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_maniguard_results import write_fixture

from helpers.maniguard_batch_report import summarize_batch


def setup_batch(path):
    jobs = [{'family': 'jar', 'scene': 'task_0000/base', 'seed': 0,
             'adapter': {'kind': 'identity'}} for _ in range(3)]
    (path / 'plan.json').write_text(json.dumps({'split': 'engineering', 'jobs': jobs}))
    for i, physics in [(0, 12), (1, 9)]:
        output = path / f'episode-{i:04d}'
        files = write_fixture(output, success=True)
        files['ledger.json']['physics_steps'] = physics
        (output / 'ledger.json').write_text(json.dumps(files['ledger.json']))
    records = [{'index': 0, 'returncode': 0, 'elapsed_seconds': 4},
               {'index': 1, 'returncode': -9, 'elapsed_seconds': 3}]
    (path / 'processes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))


def test_failed_and_unstarted_jobs_stay_visible_and_failed_costs_count(tmp_path):
    setup_batch(tmp_path)
    report = summarize_batch(tmp_path)
    assert report['planned'] == 3
    assert report['statuses'] == {'completed': 1, 'process_or_validation_failed': 1,
                                  'not_started': 1}
    assert report['groups'][0]['outcomes']['safe_completion'] == 1
    assert report['recorded_cost_totals']['physics_steps'] == 21
    assert report['finished_process_elapsed_seconds_sum'] == 7
    assert report['started_episodes_missing_cost_fields']['policy_queries'] == 2


def test_success_with_wrong_seed_is_not_accepted(tmp_path):
    setup_batch(tmp_path)
    path = tmp_path / 'episode-0000/job.json'
    job = json.loads(path.read_text())
    job['seed'] = 123
    path.write_text(json.dumps(job))
    report = summarize_batch(tmp_path)
    assert report['statuses']['invalid_result'] == 1
    assert not report['groups'][0]['outcomes']
    assert report['recorded_cost_totals']['physics_steps'] == 21


def test_false_completion_marker_rejected(tmp_path):
    setup_batch(tmp_path)
    (tmp_path / 'complete.json').write_text(json.dumps({'episodes': 3}))
    with pytest.raises(ValueError, match='completion marker'):
        summarize_batch(tmp_path)


def test_started_job_without_a_process_record_is_not_assumed_live_or_safe(tmp_path):
    setup_batch(tmp_path)
    (tmp_path / 'processes.jsonl').unlink()
    report = summarize_batch(tmp_path)
    assert report['statuses'] == {'awaiting_process_record': 2, 'not_started': 1}
    assert not report['groups'][0]['outcomes']
    assert report['recorded_cost_totals']['physics_steps'] == 21


@pytest.mark.parametrize('invalid', [-1, float('nan'), 'unknown', True])
def test_invalid_recorded_cost_is_rejected(tmp_path, invalid):
    setup_batch(tmp_path)
    path = tmp_path / 'episode-0001/ledger.json'
    ledger = json.loads(path.read_text())
    ledger['physics_steps'] = invalid
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='Invalid recorded cost'):
        summarize_batch(tmp_path)
