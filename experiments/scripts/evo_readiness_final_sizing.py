#!/usr/bin/env python3
"""Review final counts using a completed, hash-verified four-arm development pilot."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from helpers.evoReadinessProtocol import protocol_plan
from helpers.evoReadinessQueries import digest_json
from evo_readiness_sizing_review import normal_power


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(pilot, repository=ROOT):
    root, pilot = Path(repository), Path(pilot)
    hashes = {}
    def read(path, yaml_file=False):
        path = Path(path)
        hashes[str(path.resolve().relative_to(root.resolve()))] = sha(path)
        return yaml.safe_load(path.read_text()) if yaml_file else json.loads(path.read_text())
    config = read(root / 'configs/evo/stage5_main_candidate.yaml', True)
    old = read(root / 'docs/evoPlan/results/stage5-independent-audit/sizing_review.json')
    variance = read(pilot / 'variance.json')
    check = read(pilot / 'check.json')
    manifest = read(pilot / 'manifest.json')
    launch = read(pilot / 'launch.json')
    if (check['passed'] is not True or check['main_final_bank_touched'] is not False
            or check['development_only'] is not True
            or check['all_selections_frozen_before_assessment'] is not True
            or manifest['kind'] != 'evo-readiness-variance-pilot-v1'
            or manifest['costs'] != check['costs']):
        raise ValueError('complete development-only pilot with verified controls required')
    for name, expected in manifest['source_sha256'].items():
        if sha(pilot / 'source' / name) != expected:
            raise ValueError(f'pilot source snapshot changed: {name}')
    if config['bank']['evaluation_episodes'] != 320:
        raise ValueError('the measured batch shape requires 320 final episodes')
    costs = check['costs']
    completed = []
    query_count = 0
    for path in sorted((pilot / 'study/searches').glob('*/queries/*.npz')):
        with np.load(path, allow_pickle=False) as saved:
            receipt = json.loads(str(saved['__receipt__']))
        if receipt['receipt_sha256'] != digest_json({k: v for k, v in receipt.items() if k != 'receipt_sha256'}):
            raise ValueError('pilot timing receipt checksum mismatch')
        completed.append(receipt['completed_at'])
        query_count += 1
    if query_count != 32 * 33:
        raise ValueError('all 32 pilot searches require 16 fitness and 17 selection queries')
    search_s = max(completed) - launch['started_at']
    after_search_s = costs['wall_s'] - search_s
    if min(search_s, after_search_s) <= 0:
        raise ValueError('invalid completed pilot timing partition')
    pilot_cfg = read(pilot / 'config.yaml', True)
    pilot_counts = protocol_plan(pilot_cfg)['counts']
    if costs['predictor_rows'] != pilot_counts['total_predictor_rows'] or costs['real_steps'] != pilot_counts['final_real_steps']:
        raise ValueError('pilot measured costs differ from its declared query budget')
    rows = []
    for seeds in [10, 20]:
        cfg = copy.deepcopy(config)
        cfg['search']['search_seeds'] = list(range(20261101, 20261101 + seeds))
        plan = protocol_plan(cfg)
        historical = next(r for r in old['scenarios'] if r['search_seeds'] == seeds)
        if historical['counts'] != plan['counts']:
            raise ValueError('saved throughput sizing does not match the candidate')
        predictor_ratio = (plan['counts']['fitness_predictor_rows'] + plan['counts']['selection_predictor_rows']) / (
            pilot_counts['fitness_predictor_rows'] + pilot_counts['selection_predictor_rows'])
        real_ratio = plan['counts']['final_real_steps'] / pilot_counts['final_real_steps']
        empirical_hours = (search_s * predictor_ratio + after_search_s * real_ratio) / 3600 + 1
        stress_hours = (search_s * predictor_ratio + 2 * after_search_s * real_ratio) / 3600 + 1
        precision = {}
        for name, value in variance.items():
            comp = value['variance_components']
            scenarios = []
            for multiplier in [1, 2, 4]:
                parameters = dict(seed_variance=comp['seed'] * multiplier,
                    episode_variance=comp['episode'] * multiplier,
                    interaction_variance=comp['interaction'] * multiplier)
                if sum(parameters.values()) == 0:
                    result = dict(standard_error=None, approximate_two_sided_power=None,
                                  approximate_80pct_effect=None, reason='degenerate four-seed estimate is not a power guarantee')
                else:
                    result = normal_power(effect=.1, seeds=seeds, episodes=320, **parameters)
                scenarios.append(dict(variance_multiplier=multiplier, **result))
            precision[name] = dict(pilot_components=comp, sensitivity=scenarios)
        rows.append(dict(search_seeds=seeds, evaluation_episodes=320, counts=plan['counts'],
            historical_benchmark_stress_hours=historical['stress_total_hours_with_2x_nonfitness_and_1h_reserve'],
            pilot_scaled_total_hours_with_1h_reserve=empirical_hours,
            pilot_scaled_stress_hours_with_2x_postsearch_and_1h_reserve=stress_hours,
            precision=precision))
    feasible = [r for r in rows if max(r['historical_benchmark_stress_hours'],
                r['pilot_scaled_stress_hours_with_2x_postsearch_and_1h_reserve']) <= config['budget']['maximum_GPU_hours_proposed']]
    recommendation = max((r['search_seeds'] for r in feasible), default=None)
    return dict(status='completed development review; final configuration must be locked before collection',
        recommended_search_seeds=recommendation, evaluation_episodes=320,
        selection_rule='Largest of 10 and 20 seeds fitting both recorded 12-hour stress scenarios, with unchanged 320-root measured batches. Method and endpoints are unchanged regardless of effect estimates.',
        pilot_timing=dict(search_and_startup_seconds=search_s, postsearch_seconds=after_search_s,
                         verified_search_receipts=query_count, total_seconds=costs['wall_s']),
        scenarios=rows,
        limitations=[
            'Four development seeds and reused development episodes support unstable variance estimates, not guaranteed power.',
            'Negative random-effects components are truncated at zero. Multipliers 2 and 4 are illustrative sensitivity assumptions, not confidence bounds.',
            'Normal-approximation power does not simulate the declared crossed-bootstrap test.',
            'Search time scales with predictor rows; all post-search time scales with real steps. Batch scaling, host contention and collection costs can differ.',
            'The one-hour reserve and doubled post-search time are planning allowances, not upper bounds.',
            'Counts are never increased after inspecting final outcomes. Null results retain their intervals and attainable detection limits.'
        ], input_sha256=hashes, generator_sha256=sha(Path(__file__)),
        new_predictor_rows=0, new_real_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.pilot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(recommended_search_seeds=result['recommended_search_seeds'],
                         scenarios=result['scenarios']), indent=2))


if __name__ == '__main__':
    main()
