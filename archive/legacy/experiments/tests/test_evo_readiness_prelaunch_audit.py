"""Regression cases discovered by the independent prelaunch audit."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.evoReadinessProtocol import protocol_plan
from helpers.evoReadinessQueries import QueryStore

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(('path', 'value'), [
    ('search.scoring.shared_task_return', 'full_horizon_progress_even_after_violation'),
    ('search.scoring.bounds', 'include_final_evaluation'),
    ('search.scoring.stop_compute_at_violation', True),
    ('policy.clipping', [-.5, .5]),
    ('search.optimizer', 'full_covariance_CMA_ES'),
    ('evaluation.exact_recorded_tape_reference_once', False),
    ('analysis.primary_intervals', 'unadjusted_95pct'),
    ('analysis.low_pressure', {'population': 17, 'generation': 1}),
    ('analysis.bootstrap_replicates', 1),
    ('search.fitness_samples.positive_noise', 1.5),
    ('search.population_sizes', [1, 64, 256]),
    ('search.search_seeds', [-1, 32]),
    ('policy.initial_CMA_sigma', float('nan')),
    ('budget.maximum_GPU_hours_proposed', float('nan')),
    ('budget.maximum_GPU_hours_proposed', -1),
    ('bank.fitness_episodes', 96.5),
    ('analysis.proposed_allowable_relative_progress_loss', 1),
])
def test_unsupported_or_invalid_protocol_fails_before_queries(path, value):
    config = yaml.safe_load((ROOT / 'configs/evo/stage5_main_candidate.yaml').read_text())
    target = config
    parts = path.split('.')
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(ValueError):
        protocol_plan(config)


def stored_query(tmp_path):
    store = QueryStore(tmp_path, study_identity='synthetic-receipt-audit')
    counter = {'predictor_rows': 0, 'real_steps': 0}
    def callback():
        counter['predictor_rows'] += 10
        return {'value': np.array([1.])}
    options = dict(request={'synthetic': True},
                   expected_costs={'predictor_rows': 10, 'real_steps': 0},
                   read_costs=lambda: counter)
    store.execute('q', callback=callback, **options)
    def forbidden():
        raise AssertionError('cached query must never execute again')
    return store, dict(options, callback=forbidden)


def rewrite_receipt(tmp_path, mutate):
    path = tmp_path / 'q.npz'
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    receipt = json.loads(str(arrays['__receipt__']))
    mutate(receipt)
    arrays['__receipt__'] = np.asarray(json.dumps(receipt))
    np.savez(path, **arrays)


@pytest.mark.parametrize('mutation', ['cost', 'negative', 'boolean', 'request', 'key', 'shape'])
def test_reused_query_and_accounting_reject_changed_receipts(tmp_path, mutation):
    store, options = stored_query(tmp_path)
    def mutate(r):
        if mutation in ['cost', 'negative', 'boolean']:
            r['costs']['predictor_rows'] = {'cost': 999, 'negative': -1, 'boolean': True}[mutation]
        elif mutation == 'request':
            r['request']['synthetic'] = False
        elif mutation == 'key':
            r['key'] = 'other'
        else:
            r['requested_costs']['predictor_rows'] = 999
    rewrite_receipt(tmp_path, mutate)
    with pytest.raises(ValueError):
        store.execute('q', **options)
    with pytest.raises(ValueError):
        store.accounting()


def test_legacy_receipts_reuse_without_queries_but_still_check_expected_cost(tmp_path):
    store, options = stored_query(tmp_path)
    def legacy(r):
        r.pop('requested_costs')
        r.pop('receipt_sha256')
    rewrite_receipt(tmp_path, legacy)
    _, receipt = store.execute('q', **options)
    assert receipt['costs']['predictor_rows'] == 10
    assert store.accounting()['completed_costs']['predictor_rows'] == 10
    changed = copy.deepcopy(receipt['costs'])
    changed['predictor_rows'] = 999
    rewrite_receipt(tmp_path, lambda r: r.update(costs=changed))
    with pytest.raises(ValueError, match='charged costs'):
        store.execute('q', **options)
