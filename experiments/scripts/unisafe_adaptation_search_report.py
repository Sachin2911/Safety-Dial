"""Reconstruct search decisions and summarize training costs without test claims."""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_search import AdapterSearch
from helpers.unisafe_source import ROOT, sha256


def read(path):
    return json.loads(path.read_text())


def matching_generation(left, right):
    a, b = read(left / 'summary.json'), read(right / 'summary.json')
    assert len(a['episodes']) == len(b['episodes'])
    initial_equal, lengths_equal, control_equal, outcomes_equal = True, True, True, True
    counts, maximum_errors, first = Counter(), {}, None
    control_fields = {
        'raw_task_action', 'task_proposed_action', 'proposed_action', 'filtered_action',
        'executed_normalized_action', 'environment_action', 'before', 'after', 'camera_sha256',
        'intervened', 'would_intervene', 'value_trigger', 'uncertainty_trigger',
        'terminated', 'truncated', 'termination_terms',
    }
    for i in range(len(a['episodes'])):
        x, y = left / f'episode-{i:04d}', right / f'episode-{i:04d}'
        u, v = np.load(x / 'initial_state.npz'), np.load(y / 'initial_state.npz')
        initial_equal &= u.files == v.files and all(np.array_equal(u[k], v[k]) for k in u.files)
        rows_x = [json.loads(line) for line in (x / 'steps.jsonl').read_text().splitlines()]
        rows_y = [json.loads(line) for line in (y / 'steps.jsonl').read_text().splitlines()]
        same_length = len(rows_x) == len(rows_y)
        lengths_equal &= same_length
        control_equal &= same_length
        for step, (u, v) in enumerate(zip(rows_x, rows_y)):
            for key in set(u) | set(v):
                if key == 'inference_seconds' or u.get(key) == v.get(key):
                    continue
                counts[key] += 1
                if key in control_fields:
                    control_equal = False
                if first is None or (i, step, key) < (first['episode_index'], first['step'], first['field']):
                    first = {'episode_index': i, 'step': step, 'field': key}
                if isinstance(u.get(key), (int, float, list)) and isinstance(v.get(key), (int, float, list)):
                    delta = float(np.abs(np.asarray(u[key]) - np.asarray(v[key])).max())
                    maximum_errors[key] = max(maximum_errors.get(key, 0.), delta)
        for key in ('outcome', 'steps', 'physical_failure', 'boundary_exit', 'safe_success', 'interventions', 'utility'):
            outcomes_equal &= a['episodes'][i][key] == b['episodes'][i][key]
    return {
        'episodes_each': len(a['episodes']),
        'reference_control_steps': a['environment_steps'], 'candidate_control_steps': b['environment_steps'],
        'initial_states_equal': initial_equal, 'trajectory_lengths_equal': lengths_equal,
        'control_trajectories_equal': initial_equal and control_equal,
        'outcomes_and_fitness_equal': outcomes_equal,
        'all_recorded_fields_equal_except_timing': initial_equal and lengths_equal and not counts,
        'differing_field_step_counts': dict(sorted(counts.items())),
        'maximum_absolute_differences': maximum_errors, 'first_recorded_difference': first,
        'limits': 'Differences remain visible even when executed trajectories and outcomes agree. This checks recorded quantities, not hidden simulator or model state.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.search
    summary, protocol, settings = (read(root / f'{name}.json') for name in ('summary', 'protocol', 'settings'))
    assert summary['status'] == 'complete'
    assert sha256(root / 'implementation/experiments/helpers/unisafe_search.py') == sha256(ROOT / 'experiments/helpers/unisafe_search.py')
    grouped = defaultdict(list)
    for run in summary['runs']:
        grouped[run['method'], run['search_seed']].append(run)
    histories, costs, hashes = [], defaultdict(list), {}
    all_queries = Counter()
    for (method, seed), runs in sorted(grouped.items()):
        optimizer = AdapterSearch(method, protocol['initial_search_space'], settings['population'], seed)
        order_rng = np.random.default_rng(seed + 917)
        for generation, run in enumerate(sorted(runs, key=lambda r: r['generation'])):
            assert generation == run['generation']
            genes = optimizer.ask()
            assert genes == run['genes']
            directory = Path(run['directory'])
            measured = read(directory / 'summary.json')
            episodes = measured['episodes']
            expected = []
            for initial_seed in protocol['training_seeds']:
                for i in order_rng.permutation(len(genes)):
                    expected.append((initial_seed, int(i), optimizer.decode(genes[i])))
            observed = [(e['seed'], e['job']['candidate'], e['job']['adapter']) for e in episodes]
            assert observed == expected
            fitness = [float(np.mean([e['utility'] for e in episodes if e['job']['candidate'] == i])) for i in range(len(genes))]
            assert fitness == run['fitness']
            optimizer.tell(genes, fitness)
            stored = read(directory.parent / f'optimizer-{generation:02d}.json')
            assert optimizer.state() == stored
            assert measured['environment_steps'] == run['environment_steps']
            all_queries.update(measured['query_counts'])
            hashes[str(directory / 'summary.json')] = sha256(directory / 'summary.json')
            histories.append({
                'method': method, 'search_seed': seed, 'generation': generation,
                'cumulative_training_episodes_per_replica': (generation + 1) * len(expected),
                'mean_population_utility': float(np.mean(fitness)),
                'candidate_fitness': fitness, 'outcomes': run['outcomes'],
            })
            costs[method].append(run)
        assert optimizer.state() == read(root / f'{method}-{seed}/finalists.json')
    assert dict(all_queries) == summary['query_counts']
    common = []
    methods = settings['methods']
    for seed in settings['search_seeds']:
        for method in methods[1:]:
            result = matching_generation(root / f'{methods[0]}-{seed}/generation-00', root / f'{method}-{seed}/generation-00')
            common.append({'search_seed': seed, 'methods': [methods[0], method], **result})
    totals = {method: {
        'replicas': len(settings['search_seeds']),
        'episodes': sum(r['episodes'] for r in runs),
        'environment_steps': sum(r['environment_steps'] for r in runs),
        'physics_steps': sum(r['physics_steps'] for r in runs),
        'summed_worker_elapsed_seconds': sum(r['worker_elapsed_seconds'] for r in runs),
        'outcomes': dict(sum((Counter(r['outcomes']) for r in runs), Counter())),
        'updates': len(runs), 'pretrained_weight_updates': 0,
    } for method, runs in costs.items()}
    colors = {'evolution': '#316ab0', 'random': '#d67f28', 'policy_gradient': '#398a61'}
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for method in methods:
        curves = []
        for seed in settings['search_seeds']:
            items = [h for h in histories if h['method'] == method and h['search_seed'] == seed]
            x = [h['cumulative_training_episodes_per_replica'] for h in items]
            y = [h['mean_population_utility'] for h in items]
            curves.append(y)
            ax.plot(x, y, color=colors[method], alpha=.25, linewidth=1)
        ax.plot(x, np.mean(curves, axis=0), color=colors[method], marker='o', label=method.replace('_', ' '))
    ax.set(xlabel='Training episodes per search replica', ylabel='Mean utility of evaluated population', ylim=(-1.05, 1.05))
    ax.set_title('Training evaluations; thin lines are separate search seeds')
    ax.axhline(0, color='grey', linewidth=.7)
    ax.legend(loc='lower right')
    fig.savefig(args.output / 'training_utility.png', dpi=150)
    plt.close(fig)
    result = {
        'status': 'pass' if all(c['all_recorded_fields_equal_except_timing'] for c in common) else 'complete_with_repeatability_differences',
        'search': str(root), 'methods': totals, 'training_history': histories,
        'search_decisions_reconstructed_exactly': True, 'common_initial_population_checks': common,
        'analysis_script_sha256': sha256(__file__), 'analysis_elapsed_seconds': time.perf_counter() - started,
        'additional_environment_steps': 0, 'additional_model_queries': 0,
        'all_model_queries': dict(all_queries), 'source_summary_sha256': hashes,
        'environment_steps': summary['environment_steps'], 'physics_steps': summary['physics_steps'],
        'search_elapsed_seconds': summary['elapsed_seconds'],
        'limits': 'These are training outcomes across evaluated candidates, not held-out performance of final controllers. Every duplicate and elite reevaluation counts. Equal episode caps do not give equal realized step or wall-time costs. Exact reconstruction validates the recorded optimizer procedure, not its optimality or novelty. Training renderer history depends on preceding trials.',
    }
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Reconstructed all search decisions. Initial-population audit:', result['status'])
    for check in common:
        if not check['all_recorded_fields_equal_except_timing']:
            print('Repeatability difference:', check)
    for method, values in totals.items():
        print(method, values['outcomes'], 'control steps', values['environment_steps'])


if __name__ == '__main__':
    main()
