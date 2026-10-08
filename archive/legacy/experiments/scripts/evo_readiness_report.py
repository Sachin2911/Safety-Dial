#!/usr/bin/env python3
"""Render complete readiness results without re-estimating or selecting endpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

ARMS = [(0., 'zero'), (0., 'rosarl_style'), (1., 'zero'), (1., 'rosarl_style')]
COLORS = ['#3465a4', '#b15b22', '#23875e', '#91469b']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(k, mode):
    return f'k={k:g}, ' + ('zero penalty' if mode == 'zero' else 'ROSARL-style')


def interval(value, *, percent=True):
    if value.get('defined') is False:
        return 'undefined'
    scale = 100 if percent else 1
    suffix = ' pp' if percent else ''
    return f"{scale * value['point']:.2f} [{scale * value['lo']:.2f}, {scale * value['hi']:.2f}]{suffix}"


def render(run, output, ledger=None):
    run, output = Path(run), Path(output)
    names = ['analysis.json', 'manifest.json', 'config.yaml']
    names += ['costs.json'] if (run / 'costs.json').is_file() else ['check.json']
    inputs = {name: digest(run / name) for name in names}
    a = json.loads((run / 'analysis.json').read_text())
    costs = (json.loads((run / 'costs.json').read_text()) if (run / 'costs.json').is_file()
             else json.loads((run / 'check.json').read_text())['costs'])
    manifest = json.loads((run / 'manifest.json').read_text())
    config = yaml.safe_load((run / 'config.yaml').read_text())
    is_main = manifest['kind'] == 'evo-readiness-main-v1'
    if not is_main and manifest['kind'] != 'evo-readiness-variance-pilot-v1':
        raise ValueError('a completed readiness main study or variance pilot is required')
    if manifest['costs'] != costs:
        raise ValueError('cost record and final manifest differ')
    sampling = a['sampling']
    if sampling['search_seeds'] != config['search']['search_seeds']:
        raise ValueError('reported search seeds differ from the declared protocol')
    if sampling['independent_episode_count'] != config['bank']['evaluation_episodes']:
        raise ValueError('reported episode count differs from the declared protocol')
    expected = {(k, mode, p, g) for k, mode in ARMS
                for p in config['search']['population_sizes']
                for g in config['search']['checkpoints'] if g}
    observed = [(c['noise_k'], c['penalty_mode'], c['population'], c['generation'])
                for c in a['curves']]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('report requires the complete declared cell grid')
    ledger_data = json.loads(Path(ledger).read_text()) if ledger is not None else None
    if ledger_data is not None:
        if run.name not in {r['run_id'] for r in ledger_data['evolution_runs']}:
            raise ValueError('cost ledger must include this completed run')
        inputs['cost_ledger'] = digest(ledger)
    output.mkdir(parents=True, exist_ok=False)
    status = 'Fresh final episodes' if is_main else 'Development pilot: previously inspected episodes'
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'font.family': 'DejaVu Sans'})
    populations = config['search']['population_sizes']
    fig, axes = plt.subplots(1, len(populations), figsize=(5 * len(populations), 4.5),
                             sharey=True, squeeze=False)
    for ax, pop in zip(axes[0], populations):
        for (k, mode), color in zip(ARMS, COLORS):
            rows = sorted([c for c in a['curves'] if
                           (c['noise_k'], c['penalty_mode'], c['population']) == (k, mode, pop)],
                          key=lambda c: c['generation'])
            x = [c['offspring_evaluations'] for c in rows]
            real = np.array([[c['real_failure'][n] for n in ['point', 'lo', 'hi']] for c in rows]) * 100
            imagined = np.array([c['imagined_failure']['point'] for c in rows]) * 100
            ax.plot(x, real[:, 0], color=color, marker='o', label=label(k, mode))
            ax.fill_between(x, real[:, 1], real[:, 2], color=color, alpha=.10)
            ax.plot(x, imagined, color=color, linestyle='--', marker='x')
        ax.set(xscale='log', ylim=(-2, 102), title=f'Population {pop}',
               xlabel='Offspring evaluated (population × generation)')
        ticks = [pop * g for g in config['search']['checkpoints'] if g]
        ax.set_xticks(ticks, [str(v) for v in ticks])
        ax.minorticks_off()
        ax.grid(alpha=.2)
    axes[0, 0].set_ylabel('Violation rate (%)')
    axes[0, -1].legend(fontsize=8, loc='upper left')
    fig.suptitle(status + '\nSolid: real; dashed: own-noise imagination; shading: descriptive 95% intervals')
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'failure_curves.{extension}', bbox_inches='tight')
    plt.close(fig)
    high = a['pressure']['high']
    rows = [next(c for c in a['curves'] if
                 (c['noise_k'], c['penalty_mode'], c['population'], c['generation']) == (k, mode, *high))
            for k, mode in ARMS]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    metrics = [('real_failure', 100, 'Real violation rate (%)'),
               ('gap_own_noise', 100, 'Real minus imagined violation (pp)'),
               ('real_progress', 1, 'Real forward progress (m)')]
    for ax, (key, scale, title) in zip(axes, metrics):
        for i, (row, color) in enumerate(zip(rows, COLORS)):
            v = row[key]
            point, lo, hi = [scale * v[n] for n in ['point', 'lo', 'hi']]
            ax.plot([lo, hi], [i, i], color=color, lw=2)
            ax.plot(point, i, 'o', color=color)
        ax.set_yticks(range(4), [label(*arm) for arm in ARMS])
        ax.set_title(title)
        if key == 'real_failure':
            ax.set_xlim(-2, 102)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    fig.suptitle(f'{status}\nHigh pressure: population {high[0]}, generation {high[1]}; descriptive 95% intervals')
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'high_pressure.{extension}', bbox_inches='tight')
    plt.close(fig)
    fields = ['noise_k', 'penalty_mode', 'population', 'generation', 'offspring_evaluations']
    metrics = ['real_failure', 'imagined_failure', 'gap_own_noise', 'gap_common_k0', 'real_progress']
    columns = fields + [f'{metric}_{stat}' for metric in metrics for stat in ['point', 'lo', 'hi']]
    with (output / 'all_cells.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for cell in a['curves']:
            writer.writerow({**{k: cell[k] for k in fields},
                **{f'{metric}_{stat}': cell[metric][stat]
                   for metric in metrics for stat in ['point', 'lo', 'hi']}})
    p, practical = a['primary'], a['practical_benefit']
    baseline = a['baseline']['real_failure']
    primary_lines = [
        '| Declared contrast | Estimate [97.5% interval], pp | Direction supported |',
        '|---|---:|---|',
        f"| k=0 zero-penalty gap, high minus low pressure | {interval(p['k0_zero_gap_amplification'])} | {p['amplification_interval_above_zero']} |",
        f"| Combined method minus k=0 zero-penalty gap amplification | {interval(p['combined_method_minus_k0_zero_amplification'])} | {p['combined_amplification_difference_interval_below_zero']} |",
    ]
    text = [f'# Readiness results: {run.name}', '', status + '.', '',
        f"The study used {sampling['seed_count']} paired search seeds and {sampling['independent_episode_count']} independent source episodes. "
        'Uncertainty resamples search seeds and source episodes independently, retaining pairing across all arms. '
        'The product of seed and episode counts is not the number of independent episodes.', '',
        f"The shared frozen baseline's real violation rate was {100 * baseline['point']:.2f}% "
        f"[{100 * baseline['lo']:.2f}, {100 * baseline['hi']:.2f}] (descriptive 95% interval).", '',
        *primary_lines, '',
        'The two co-primary intervals use Bonferroni family coverage of 95%. '
        + ('These are the locked final-study contrasts.' if is_main else
           'These development results inform variance planning and do not provide final confirmation.'), '',
        f"At high pressure, the combined method's real failure difference is {interval(practical['real_failure_difference'])} "
        f"and its ratio of mean real progress is {interval(practical['ratio_of_mean_real_progress'], percent=False)} "
        '(descriptive paired 95% intervals). '
        f"Both declared practical conditions supported: {practical['both_practical_conditions_supported']}. "
        f"The practical criterion requires the real-failure difference's upper interval endpoint below zero and the progress ratio's lower endpoint at least {practical['minimum_progress_ratio']:.2f}. "
        'A smaller prediction gap alone does not establish a real safety improvement.', '',
        '![Real and imagined failures](failure_curves.png)', '',
        '![High-pressure comparisons](high_pressure.png)', '',
        f"Incremental run cost: {costs['predictor_rows']:,} predictor rows, {costs['real_steps']:,} real simulator steps, "
        f"{costs['gradient_updates']:,} gradient updates and {costs['wall_s'] / 3600:.3f} elapsed hours. "
        'These are incremental costs; original asset training, data collection and prior development runs are additional.', '',
        'Historical Gate 0 and the transfer screen remain failures. Inference is conditional on the frozen assets, '
        'fitness/selection banks and source distribution. The branches are 0.8 seconds long and do not establish safe '
        'full-episode control. A degenerate bootstrap interval does not prove zero population risk. Every declared cell is included in `all_cells.csv`; rates there are fractions, not percentages.', '']
    if ledger_data is not None:
        (output / 'cost_ledger.json').write_text(json.dumps(ledger_data, indent=2) + '\n')
        text += [f"The [scoped cost ledger](cost_ledger.json) records {ledger_data['recorded_evolution_real_steps']:,} real steps across preserved evolution runs, including this run. "
            f"The shared original set-A, probe and root data add {ledger_data['shared_original_collection_total']:,} collection steps. "
            f"The original world model reached optimizer index {ledger_data['original_LeWM_final_optimizer_index']:,}. "
            'PPO/PPO-Lagrangian training interactions and some older failed-run costs remain unquantified, so these records do not establish a complete lifetime cost.', '']
    (output / 'README.md').write_text('\n'.join(text))
    provenance = dict(input_directory=str(run.resolve()), input_sha256=inputs,
        generator_sha256=digest(__file__), new_model_queries=0, new_real_steps=0,
        result_status=status, outputs_sha256={p.name: digest(p) for p in sorted(output.iterdir())})
    (output / 'report_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ledger', type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.output, args.ledger), indent=2))


if __name__ == '__main__':
    main()
