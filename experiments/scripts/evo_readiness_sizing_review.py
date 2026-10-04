#!/usr/bin/env python3
"""Query-free sizing scenarios from archived pilot timing and the main batch benchmark.

The ROSARL contrast has no measured variance. These scenarios do not lock counts,
prove power, authorize execution, or change the experiment's declared endpoints.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from helpers.evoReadinessProtocol import protocol_plan  # noqa: E402


def normal_power(*, effect, seed_variance, episode_variance, interaction_variance,
                 seeds, episodes, alpha=.025):
    values = [effect, seed_variance, episode_variance, interaction_variance, alpha]
    if not all(math.isfinite(x) for x in values) or min(values[1:4]) < 0:
        raise ValueError('finite effect and nonnegative variance components required')
    if not 0 < alpha < 1 or min(seeds, episodes) < 2:
        raise ValueError('valid alpha and at least two seeds/episodes required')
    variance = seed_variance / seeds + episode_variance / episodes + interaction_variance / (seeds * episodes)
    if variance <= 0:
        raise ValueError('positive total planning variance required')
    se = math.sqrt(variance)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    shift = abs(effect) / se
    power = 1 - NormalDist().cdf(z - shift) + NormalDist().cdf(-z - shift)
    return dict(standard_error=se, approximate_two_sided_power=power,
                approximate_80pct_effect=(z + NormalDist().inv_cdf(.8)) * se,
                variance_of_mean=variance)


def review(repository=ROOT):
    root = Path(repository)
    inputs = {}
    def read(name, kind='json'):
        path = root / name
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return yaml.safe_load(path.read_text()) if kind == 'yaml' else json.loads(path.read_text())
    config = read('configs/evo/stage5_main_candidate.yaml', 'yaml')
    pilot_path = 'runs/walker2d-evo-s5-power-pilot-20261004-1/'
    pilot = read(pilot_path + 'pilot.json')
    timing = read(pilot_path + 'timing.json')
    benchmark_path = 'runs/walker2d-evo-s5-batch-benchmark-20261004-1/benchmark.json'
    benchmark = read(benchmark_path)
    if benchmark['passed'] is not True or benchmark['main_study_executed'] is not False:
        raise ValueError('completed engineering benchmark required')
    candidate_name = 'configs/evo/stage5_main_candidate.yaml'
    if benchmark['source_sha256'].get(candidate_name) != inputs[candidate_name]:
        raise ValueError('planning candidate differs from the benchmarked configuration')
    if config['bank']['evaluation_episodes'] != 320:
        raise ValueError('this sizing review requires the benchmarked 320-root audit')
    base_count = len(config['search']['search_seeds'])
    nonfitness = pilot['costs']['wall_s'] - sum(t['seconds'] for t in timing)
    if nonfitness <= 0:
        raise ValueError('timing components do not leave positive nonfitness time')
    rows = []
    for seeds in [10, 20, 30]:
        cfg = copy.deepcopy(config)
        first = cfg['search']['search_seeds'][0]
        cfg['search']['search_seeds'] = list(range(first, first + seeds))
        plan = protocol_plan(cfg)
        # Only the number of runs changes. Root/sample shapes stay at their measured sizes.
        if plan['batch_shapes'] != protocol_plan(config)['batch_shapes']:
            raise ValueError('new batch shape needs a new benchmark')
        model_hours = benchmark['projected_model_only_hours'] * seeds / base_count
        nonfitness_scaled = nonfitness * plan['counts']['final_real_steps'] / pilot['costs']['real_steps'] / 3600
        pilot_components = pilot['contrasts']['gap_k0']['variance_components']
        plugin = normal_power(effect=.1, seed_variance=pilot_components['seed'],
                              episode_variance=pilot_components['episode'],
                              interaction_variance=pilot_components['interaction'], seeds=seeds, episodes=320)
        scenarios = []
        for seed_sd in [.05, .10, .15, .20]:
            for episode_variance in [.10, .35]:
                components = dict(seed_variance=seed_sd**2, episode_variance=episode_variance,
                                  interaction_variance=.35)
                scenarios.append(dict(assumed_seed_sd=seed_sd, **components,
                    **normal_power(effect=.1, **components, seeds=seeds, episodes=320)))
        rows.append(dict(search_seeds=seeds, evaluation_episodes=320,
            counts=plan['counts'], measured_model_only_hours=model_hours,
            scaled_pilot_nonfitness_hours=nonfitness_scaled,
            illustrative_total_hours_with_1h_reserve=model_hours + nonfitness_scaled + 1,
            stress_total_hours_with_2x_nonfitness_and_1h_reserve=model_hours + 2 * nonfitness_scaled + 1,
            k0_pilot_plugin_power=plugin,
            unmeasured_rosarl_contrast_scenarios=scenarios))
    return dict(status='review scenarios only; no counts locked or scientific approval recorded',
        assumptions=dict(effect=.10, per_primary_alpha=.025, target_power=.80,
            final_episodes=320, seed_count_candidates=[10, 20, 30],
            unknown_rosarl_variance='Scenario values are sensitivity assumptions, not estimates or confidence bounds.',
            normal_approximation='Does not simulate the declared crossed-bootstrap test or prove its finite-sample power.',
            runtime_reserve='One hour is an explicit planning allowance, not a measured collection/archive/analysis bound.',
            nonfitness_scaling='All pilot time outside measured fitness calls is scaled by final real steps. This includes selection/audits already in model timing, so overlaps them; it still cannot guarantee total runtime.',
            unchanged_batch_shapes=True),
        measured_pilot_nonfitness_seconds=nonfitness,
        scenarios=rows,
        decision=dict(recommendation='Review 20 paired seeds and 320 final episodes after the ROSARL scientific decision; retain a declared development variance pilot or explicitly accepted sensitivity range before locking counts.',
            why='Twenty seeds improve sensitivity to search variability while preserving measured batch shapes. Thirty seeds exceed the 12-hour ceiling under the stated stress scenario.',
            not_a_power_guarantee=True, current_candidate_unchanged=True,
            if_variance_too_large='Report attainable sensitivity or revise the protocol prospectively. Do not increase counts after final outcomes.'),
        input_sha256=inputs, analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        incremental_costs=dict(predictor_rows=0, real_steps=0, new_episodes=0, training_updates=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps([dict(seeds=r['search_seeds'], model_hours=r['measured_model_only_hours'],
        stress_hours=r['stress_total_hours_with_2x_nonfitness_and_1h_reserve'])
        for r in result['scenarios']], indent=2))


if __name__ == '__main__':
    main()
