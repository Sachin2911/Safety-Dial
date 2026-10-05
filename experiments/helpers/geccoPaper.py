"""Build GECCO paper figures and tables from frozen study results.

Run from repository root: uv run python experiments/helpers/geccoPaper.py [--render]
--render additionally restores saved MuJoCo poses for an illustrative image strip.
It never steps physics, queries a learned model, or changes experiment artifacts.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'walker2d-evo-s5-main-20261004-1'
SOURCE = ROOT / 'docs/evoPlan/results/stage5-main' / RUN_ID
DIAGNOSTIC = (ROOT / 'docs/evoPlan/results/stage6-candidate-quality'
              / 'walker2d-evo-s6-candidate-quality-20261005-1')
REPAIR = (ROOT / 'docs/evoPlan/results/stage6-repair'
          / 'walker2d-evo-s6-repair-20261005-1')
DIVERGENCE = (ROOT / 'docs/evoPlan/results/stage6-divergence'
              / 'walker2d-evo-s6-divergence-20261005-2')
OUT = ROOT / 'docs/geccoPaper'
ARMS = [(0., 'zero', 'Control', '#333333', 'o'),
        (0., 'rosarl_style', 'Penalty', '#D55E00', 's'),
        (1., 'zero', 'Noise', '#0072B2', '^'),
        (1., 'rosarl_style', 'Noise + penalty', '#009E73', 'D')]
# Match the manuscript's serif typography; sizes are specified at print width.
plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 8,
                     'mathtext.fontset': 'stix',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 7.5,
                     'axes.linewidth': .6, 'lines.linewidth': 1.1,
                     'xtick.major.width': .6, 'ytick.major.width': .6,
                     'xtick.major.size': 3, 'ytick.major.size': 3,
                     'axes.axisbelow': True, 'grid.color': '#DEDEDE',
                     'grid.linewidth': .5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'savefig.dpi': 300, 'savefig.pad_inches': .025})



def save(fig, name):
    for suffix in ['pdf', 'png']:
        fig.savefig(OUT / 'figures' / f'{name}.{suffix}', bbox_inches='tight',
                    metadata={'Creator': 'Matplotlib'})
    plt.close(fig)


def build():
    a = json.loads((SOURCE / 'analysis.json').read_text())
    rows = a['curves']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    for panel, (ax, pop) in enumerate(zip(axes, [16, 64, 256])):
        for k, penalty, label, color, marker in ARMS:
            rr = sorted([r for r in rows if r['noise_k'] == k
                         and r['penalty_mode'] == penalty and r['population'] == pop],
                        key=lambda r: r['generation'])
            assert len(rr) == 3
            y = np.array([r['real_failure']['point'] for r in rr]) * 100
            lo = np.array([r['real_failure']['lo'] for r in rr]) * 100
            hi = np.array([r['real_failure']['hi'] for r in rr]) * 100
            ax.plot([1, 4, 16], y, color=color, marker=marker, ms=3.5, label=label)
            ax.fill_between([1, 4, 16], lo, hi, color=color, alpha=.08, linewidth=0)
        ax.axhline(a['baseline']['real_failure']['point'] * 100,
                   ls='--', color='#777777', lw=1)
        ax.set(xscale='log', xticks=[1, 4, 16], xticklabels=['1', '4', '16'],
               xlabel='Generation', title=f'({chr(97 + panel)}) Population {pop}', ylim=(10, 40))
        ax.minorticks_off()
        ax.grid(axis='y')
    axes[0].set_ylabel('Actual failure (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    baseline_line = axes[0].lines[-1]
    fig.legend(handles + [baseline_line], labels + ['Starting controller'], ncol=5,
               loc='upper center', bbox_to_anchor=(.52, 1.01), frameon=False,
               handlelength=1.8, columnspacing=1.8)
    fig.subplots_adjust(left=.075, right=.985, bottom=.22, top=.74, wspace=.13)
    save(fig, 'failure_pressure')

    # A forest plot keeps effect size and uncertainty on the same scale.
    fig, ax = plt.subplots(figsize=(3.35, 1.8))
    keys = [('k0_zero_gap_amplification', 'Control amplification'),
            ('combined_method_minus_k0_zero_amplification', 'Combined minus control')]
    for y, (key, label), color in zip([1, 0], keys, ['#333333', '#009E73']):
        v = a['primary'][key]
        point, lo, hi = [100 * v[x] for x in ['point', 'lo', 'hi']]
        ax.errorbar(point, y, xerr=[[point-lo], [hi-point]], fmt='o', color=color,
                    capsize=2.5, ms=4, lw=1.2)
        ax.text(-16, y+.32, label, va='bottom', fontsize=8)
        ax.text(18, y+.32, f'{point:+.2f} [{lo:+.2f}, {hi:+.2f}]',
                ha='right', va='bottom', fontsize=7.5, color=color)
    ax.axvline(0, color='#888888', lw=.7, ls=(0, (3, 3)))
    ax.set(yticks=[], ylim=(-.45, 1.85), xlim=(-16, 18),
           xticks=[-10, 0, 10], xlabel='Change in failure gap (percentage points)')
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x')
    fig.subplots_adjust(left=.035, right=.97, bottom=.24, top=.99)
    save(fig, 'primary_contrasts')

    # Separate the development path from the withheld evaluation bank. The
    # paired audit shares initial states, not necessarily subsequent actions.
    fig, ax = plt.subplots(figsize=(7, 1.75))
    fig.subplots_adjust(left=.005, right=.995, bottom=.015, top=.985)
    ax.set(xlim=(0, 7), ylim=(0, 1.75))
    ax.axis('off')
    columns = [(0.05, 1.48, '01', 'Disjoint episode banks'),
               (1.92, 1.36, '02', 'Search in imagination'),
               (3.64, 1.23, '03', 'Freeze selections'),
               (5.22, 1.73, '04', 'Paired final audit')]
    for x, width, number, title in columns:
        ax.text(x, 1.61, number, color='#666666', fontsize=8, va='center')
        ax.text(x+.22, 1.61, title, fontsize=9, va='center')
        ax.plot([x, x+width], [1.43, 1.43], color='#444444', lw=.65)
    for x, lines in [(0.05, ['96 fitness episodes', '64 selection episodes']),
                     (1.92, ['4 arms × 20 seeds', '3 population sizes']),
                     (3.64, ['240 searches', '720 policy slots'])]:
        for y, line in zip([1.19, .93], lines):
            ax.text(x, y, line, fontsize=8.5, va='center')
    ax.text(1.92, .65, 'Generations 1, 4 and 16', fontsize=8, color='#555555')
    ax.text(3.64, .65, 'No final outcomes seen', fontsize=8, color='#555555')
    ax.text(5.22, 1.19, 'Imagination: 10 action blocks', fontsize=8.5, va='center')
    ax.text(5.22, .93, 'MuJoCo: 100 physics steps', fontsize=8.5, va='center')
    ax.text(5.22, .65, 'Same saved initial states', fontsize=8, color='#0072B2')
    for start, end in [(1.55, 1.83), (3.31, 3.55), (4.91, 5.13)]:
        ax.add_patch(FancyArrowPatch((start, 1.06), (end, 1.06), arrowstyle='-|>',
                                    mutation_scale=8, color='#444444', lw=.7))
    ax.text(.05, .28, '320 evaluation episodes', fontsize=8.5, color='#0072B2',
            va='center')
    ax.plot([1.54, 6.16, 6.16], [.28, .28, .47], color='#0072B2', lw=.8)
    ax.add_patch(FancyArrowPatch((6.16, .45), (6.16, .59), arrowstyle='-|>',
                                mutation_scale=8, color='#0072B2', lw=.8))
    ax.text(3.6, .34, 'Withheld until every selection is frozen', ha='center',
            va='bottom', fontsize=8, color='#0072B2')
    save(fig, 'protocol')

    # Split each high-pressure gap into missed between-block events, real-image
    # readout error and the remaining imagined-versus-real-readout difference.
    decomposition = json.loads((SOURCE / 'decomposition.json').read_text())
    parts = [('dense_minus_endpoint', 'Between-block events', '#B9B9B9'),
             ('endpoint_minus_real_readout', 'Readout on real frames', '#D55E00'),
             ('real_readout_minus_imagined', 'Imagined vs. real frames', '#0072B2')]
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    for y, (k, penalty, label, _, _) in enumerate(ARMS[::-1]):
        d = next(e for e in decomposition if e['condition'] == [k, penalty, 256, 16])
        pos = neg = 0.
        for key, part, color in parts:
            v = 100 * d[key]['point']
            ax.barh(y, v, left=pos if v >= 0 else neg, color=color, height=.6,
                    label=part if y == 0 else None)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        ax.plot(pos + neg, y, 'k|', ms=9, mew=1.4)
    ax.axvline(0, color='#888888', lw=.8)
    ax.set(yticks=range(4), yticklabels=[x[2] for x in ARMS[::-1]],
           xlabel='Contribution to failure gap (percentage points)', xlim=(-10, 40))
    ax.legend(ncol=1, loc='lower left', bbox_to_anchor=(-.37, 1.03), frameon=False,
              fontsize=7.5, handlelength=1.4, borderaxespad=0, labelspacing=.3)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x')
    fig.subplots_adjust(left=.28, right=.98, bottom=.20, top=.70)
    save(fig, 'gap_decomposition')

    # Exploratory fixed-pool diagnostic: development-check failure against how far
    # each candidate moves the starting controller's actions on the same inputs.
    pool = json.loads((DIAGNOSTIC / 'analysis.json').read_text())
    styles = {'baseline': ('Starting controller', '#000000', '*', 60),
              'imagined_winner': ('Imagined winner', '#D55E00', 'D', 16),
              'contracted_winner': ('Contracted winner', '#E69F00', 'v', 14),
              'capped_winner': ('Capped winner', '#CC79A7', 's', 12),
              'population_sample': ('Population sample', '#0072B2', 'o', 12),
              'random_direction': ('Random direction', '#009E73', '^', 12)}
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    for kind, (label, color, marker, size) in styles.items():
        members = [r for r in pool['candidates'] if r['origins'][0]['kind'] == kind]
        ax.scatter([r['action_delta_rms'] for r in members],
                   [100 * r['check_failure'] for r in members], s=size, marker=marker,
                   color=color, alpha=.85, lw=0, label=label, zorder=3 if kind == 'baseline' else 2)
    ax.axhline(100 * pool['baseline_failure'], ls='--', color='#777777', lw=1)
    ax.set(xlabel='Action change from starting controller (RMS)',
           ylabel='Check failure (%)', ylim=(0, 32))
    ax.legend(ncol=2, loc='lower left', bbox_to_anchor=(-.15, 1.03), frameon=False,
              fontsize=7.5, handletextpad=.3, columnspacing=1.2, borderaxespad=0,
              labelspacing=.35, markerscale=.9)
    ax.grid(axis='y')
    fig.subplots_adjust(left=.14, right=.98, bottom=.18, top=.69)
    save(fig, 'candidate_quality')

    repair = json.loads((REPAIR / 'analysis.json').read_text())
    repair_arms = ['baseline', 'probe', 'dynamics', 'combined']
    labels = ['Original', 'Probe only', 'Predictor only', 'Both']
    colors = ['#555555', '#0072B2', '#009E73', '#D55E00']
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.95), sharey=True)
    for ax, key, title in zip(axes, ['event_detection', 'false_alarm'],
                             ['(a) Detection (%)', '(b) False alarms (%)']):
        vals = [repair['audit'][arm]['teacher_forced'][key] for arm in repair_arms]
        for y, value, color in zip(range(3, -1, -1), vals, colors):
            point, lo, hi = [100 * value[k] for k in ['point', 'lo', 'hi']]
            ax.errorbar(point, y, xerr=[[point-lo], [hi-point]], fmt='o',
                        color=color, capsize=2, ms=3.5, lw=1)
        ax.set_title(title, fontsize=8, pad=9)
        ax.set(ylim=(-.5, 3.5), yticks=range(3, -1, -1), yticklabels=labels)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x')
    axes[0].set(xlim=(-3, 100), xticks=[0, 50, 100])
    axes[1].set(xlim=(-.04, 1.5), xticks=[0, .5, 1.0, 1.5])
    fig.subplots_adjust(left=.26, right=.98, top=.83, bottom=.16, wspace=.36)
    save(fig, 'predictor_repair')
    compose_example()

    table = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Method & Actual (\%) & Imagined (\%) & Progress (m) \\', r'\midrule']
    values = []
    for k, penalty, label, _, _ in ARMS:
        r = next(r for r in rows if r['noise_k'] == k and r['penalty_mode'] == penalty
                 and r['population'] == 256 and r['generation'] == 16)
        vals = (r['real_failure']['point']*100, r['imagined_failure']['point']*100,
                r['real_progress']['point'])
        table.append(f'{label} & {vals[0]:.2f} & {vals[1]:.2f} & {vals[2]:.3f} '+r'\\')
        values.append({'method': label, 'actual_count_over_seed_episode_pairs':
                       int(round(r['real_failure']['point']*6400)), 'seed_episode_pairs': 6400,
                       'independent_source_episodes': 320, 'values': vals})
    b = a['baseline']
    baseline_row = (f"Frozen baseline & {100*b['real_failure']['point']:.2f} & "
                    f"{100*b['imagined_failure']['0.0']['point']:.2f} & "
                    f"{b['real_progress']['point']:.3f} " + r'\\')
    table += [r'\midrule', baseline_row,
              r'\bottomrule', r'\end{tabular}']
    (OUT/'tables').mkdir(exist_ok=True)
    (OUT/'tables/high_pressure.tex').write_text('\n'.join(table)+'\n')
    (OUT/'tables/high_pressure_values.json').write_text(json.dumps(values, indent=2)+'\n')
    return a


def compose_example():
    """Lay out the archived simulator images without rerendering or stepping."""
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.22))
    for panel, (ax, step) in enumerate(zip(axes, [0, 50, 100])):
        ax.imshow(plt.imread(OUT/'figures'/f'walker_t{step:03d}.png'))
        ax.axis('off')
        ax.set_title(f'({chr(97 + panel)}) {step*.008:.2f} s', fontsize=8, pad=3)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=.84, wspace=.025)
    save(fig, 'walker_example')


def render_example():
    os.environ['MUJOCO_GL'] = 'egl'
    sys.path.insert(0, str(ROOT/'experiments'))
    from helpers.locoData import RenderContext
    from helpers.walkerRules import health_clearance, rule_unsafe

    run = ROOT/'runs'/RUN_ID
    picks = json.loads((run/'study/frozen_selections.json').read_text())['picks']
    pick = next(p for p in picks if p['noise_k'] == 0 and p['penalty_mode'] == 'zero'
                and p['population'] == 256 and p['generation'] == 16
                and p['seed'] == 20261101)
    real_path = run/f"study/evaluation/pick{pick['slot']:04d}-real.npz"
    imag_path = run/f"study/evaluation/pick{pick['slot']:04d}-k0.npz"
    with np.load(real_path, allow_pickle=False) as f:
        real = {k: f[k] for k in f.files if k != '__receipt__'}
    with np.load(imag_path, allow_pickle=False) as f:
        ro = f['readout']
        predicted = rule_unsafe('health', health_clearance(ro[..., 0], ro[..., 1]))
        predicted = predicted.reshape(320, -1).any(axis=1)
    eligible = np.flatnonzero(real['dense_violated'] & ~predicted)
    assert len(eligible)
    idx = int(eligible[0])
    ctx = RenderContext(width=448, height=448)
    times = [0, 50, 100]
    for t in times:
        frame = ctx.render_state(real['qpos'][idx, t], real['qvel'][idx, t])
        assert frame.std() > 5
        plt.imsave(OUT/'figures'/f'walker_t{t:03d}.png', frame)
    ctx.close()
    compose_example()
    first = int(real['dense_first_step'][idx])
    provenance = dict(selection='First final-bank row with actual failure and no k0 imagined '
                      'violation for the first search seed, control, population 256, generation 16. '
                      'Outcome-selected illustration, not a representative effect estimate.',
                      pick=pick, root_index=idx, pair=real['pairs'][idx].tolist(),
                      dense_first_step_zero_based=first, first_violation_completed_step=first+1,
                      first_violation_time_s=(first+1)*.008, saved_state_indices=times,
                      actual_failed=True, imagined_k0_failed=False, simulator_steps=0,
                      learned_model_queries=0, rendered_frames=3,
                      sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [real_path, imag_path]})
    (OUT/'figure_example_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'example_root': idx, 'first_violation_completed_step': first+1}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    (OUT/'figures').mkdir(parents=True, exist_ok=True)
    build()
    if args.render:
        render_example()
    provenance = {'run_id': RUN_ID, 'sources': {
        str(path.relative_to(ROOT)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [SOURCE/'analysis.json', SOURCE/'decomposition.json',
                     DIAGNOSTIC/'analysis.json', DIVERGENCE/'analysis.json',
                     REPAIR/'analysis.json', REPAIR/'paired_changes.json',
                     *sorted((OUT/'figures').glob('walker_t*.png'))]},
        'generator': str(Path(__file__).relative_to(ROOT)),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_model_queries': 0, 'new_simulator_steps': 0,
        'figure_style': 'STIX serif, vector PDF, 300 dpi PNG; nominal widths 3.35 and 7 inches.',
        'walker_layout': 'Composed from the three saved frame PNGs; original selection unchanged.',
        'curve_intervals': 'Descriptive pointwise 95% crossed seed/episode bootstrap.',
        'primary_intervals': '97.5% each; Bonferroni family coverage 95%.',
        'baseline_imagined_table': 'k0 audit; noisy baseline audit is reported in source analysis.',
        'candidate_quality': 'Exploratory Stage 6 fixed pool on reused development episodes.',
        'predictor_repair': 'Exploratory reused audit: 24 source episodes, 89 correlated candidates, three fitting seeds; 95% whole-episode bootstrap.'}
    (OUT/'figure_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
