"""Cost accounting must include failed work without counting summaries twice."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_accounting import inventory_costs


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_failed_physics_and_failed_diagnostic_count_but_derived_reports_do_not(tmp_path):
    ledger = {'physics_steps': 86, 'policy_queries': 1, 'phase': 'crashed',
              'physics_accounting_attached': True}
    write(tmp_path / 'failed/ledger.json', ledger)
    check = {'physics_steps': 0, 'policy_queries_attempted': 6, 'query_seconds': [1.0] * 6,
             'elapsed_seconds': 10, 'status': 'failed'}
    write(tmp_path / 'failed-replay/check.json', check)
    write(tmp_path / 'reports/summary.json', {'ledger': ledger, 'check': check})
    result = inventory_costs(tmp_path)
    assert result['record_types'] == {'inference_diagnostic': 1, 'physical_episode': 1}
    assert result['recorded_cost_totals']['physics_steps'] == 86
    assert result['recorded_cost_totals']['policy_queries'] == 7
    assert result['recorded_cost_totals']['policy_query_seconds'] == 6
    assert {r['recorded_status'] for r in result['records']} == {'failed', 'crashed'}
    assert any('Missing cost field' in s for r in result['records'] for s in r['issues'])


def test_missing_or_unattached_ledger_does_not_silently_establish_zero_work(tmp_path):
    write(tmp_path / 'missing/job.json', {})
    write(tmp_path / 'unattached/job.json', {})
    write(tmp_path / 'unattached/ledger.json', {'physics_steps': 0, 'phase': 'initialization'})
    result = inventory_costs(tmp_path)
    assert result['warnings'] == [{'path': 'missing/job.json',
                                   'issue': 'Episode job has no cost ledger'}]
    assert any('attachment unverified' in s for s in result['records'][0]['issues'])


def test_file_alias_does_not_double_count_costs(tmp_path):
    write(tmp_path / 'original/ledger.json', {'physics_steps': 17})
    (tmp_path / 'alias').mkdir()
    (tmp_path / 'alias/ledger.json').symlink_to(tmp_path / 'original/ledger.json')
    result = inventory_costs(tmp_path)
    assert result['recorded_cost_totals']['physics_steps'] == 17
    assert len(result['records']) == 1
    assert any(w['issue'] == 'Duplicate file alias excluded' for w in result['warnings'])


def test_attempted_query_with_missing_latency_is_retained_and_flagged(tmp_path):
    write(tmp_path / 'replay/check.json', {'physics_steps': 0, 'status': 'failed',
          'policy_queries_attempted': 2, 'query_seconds': [0.4], 'elapsed_seconds': 20})
    result = inventory_costs(tmp_path)
    assert result['recorded_cost_totals']['policy_queries'] == 2
    assert result['recorded_cost_totals']['policy_query_seconds'] == 0.4
    assert result['records'][0]['issues'] == ['Not every attempted query has a recorded latency']


@pytest.mark.parametrize('value', [-1, True, 1.5, float('nan')])
def test_invalid_interaction_count_rejected(tmp_path, value):
    write(tmp_path / 'episode/ledger.json', {'physics_steps': value})
    with pytest.raises(ValueError, match='Invalid recorded cost'):
        inventory_costs(tmp_path)
