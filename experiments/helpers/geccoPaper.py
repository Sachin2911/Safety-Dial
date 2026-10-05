"""Build GECCO short-paper figures and tables from frozen study results.

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'walker2d-evo-s5-main-20261004-1'
SOURCE = ROOT / 'docs/evoPlan/results/stage5-main' / RUN_ID
OUT = ROOT / 'docs/geccoPaper'
ARMS = [(0., 'zero', 'Control', '#333333', 'o'),
        (0., 'rosarl_style', 'Penalty', '#D55E00', 's'),
        (1., 'zero', 'Noise', '#0072B2', '^'),
        (1., 'rosarl_style', 'Noise + penalty', '#009E73', 'D')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 7,
                     'pdf.fonttype': 42, 'savefig.dpi': 220})


def save(fig, name):
    for suffix in ['pdf', 'png']:
        fig.savefig(OUT / 'figures' / f'{name}.{suffix}', bbox_inches='tight',
                    metadata={'Creator': 'Matplotlib'})
    plt.close(fig)


def build():
    a = json.loads((SOURCE / 'analysis.json').read_text())
    rows = a['curves']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95), sharey=True)
    for ax, pop in zip(axes, [16, 64, 256]):
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
               xlabel='Generation', title=f'Population {pop}', ylim=(10, 40))
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Actual failure (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.09),
               frameon=False)
    fig.tight_layout(w_pad=.6)
    save(fig, 'failure_pressure')

    fig, ax = plt.subplots(figsize=(3.35, 1.55))
    keys = [('k0_zero_gap_amplification', 'Control amplification'),
            ('combined_method_minus_k0_zero_amplification', 'Change in amplification')]
    for y, (key, label) in zip([1, 0], keys):
        v = a['primary'][key]
        p, lo, hi = [100 * v[x] for x in ['point', 'lo', 'hi']]
        ax.errorbar(p, y, xerr=[[p-lo], [hi-p]], fmt='o', color=['#009E73', '#333333'][y],
                    capsize=3, ms=4)
        ax.text(p, y+.21, f'{p:+.2f} [{lo:+.2f}, {hi:+.2f}]', ha='center', fontsize=7)
    ax.axvline(0, color='#888888', lw=.8, ls='--')
    ax.set(yticks=[1, 0], yticklabels=[x[1] for x in keys], ylim=(-.45, 1.65),
           xlim=(-16, 18), xlabel='Percentage points (97.5% intervals)')
    ax.grid(axis='x', alpha=.15)
    save(fig, 'primary_contrasts')

    fig, ax = plt.subplots(figsize=(7, 1.25))
    ax.set(xlim=(0, 10), ylim=(0, 2))
    ax.axis('off')
    boxes = [(0, 2.3, 'Fresh source\nepisodes', '96 fitness / 64 selection\n320 final evaluation'),
             (2.65, 2.1, 'Evolve in\nimagination', '4 arms / 20 seeds\n3 population sizes'),
             (5.1, 1.85, 'Freeze\nselections', '240 searches\n720 policy slots'),
             (7.3, 2.65, 'Paired final\naudit', 'Same saved start states\n10 blocks / 100 steps')]
    for x, w, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, .3), w, 1.35, boxstyle='round,pad=0.02',
                                   facecolor='#EEF3F5', edgecolor='#59717F', lw=.8))
        ax.text(x+w/2, 1.3, title, ha='center', va='center', weight='bold', fontsize=8)
        ax.text(x+w/2, .68, body, ha='center', va='center', fontsize=7.5)
    for x1, x2 in [(2.32, 2.63), (4.77, 5.08), (6.97, 7.28)]:
        ax.add_patch(FancyArrowPatch((x1, .97), (x2, .97), arrowstyle='-|>',
                                    mutation_scale=10, color='#59717F'))
    save(fig, 'protocol')

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
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.3))
    for ax, t in zip(axes, times):
        frame = ctx.render_state(real['qpos'][idx, t], real['qvel'][idx, t])
        assert frame.std() > 5
        ax.imshow(frame)
        ax.axis('off')
        ax.set_title(f'{t*.008:.1f} s', fontsize=8, pad=1)
        plt.imsave(OUT/'figures'/f'walker_t{t:03d}.png', frame)
    ctx.close()
    fig.subplots_adjust(wspace=.015)
    save(fig, 'walker_example')
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
        str((SOURCE/'analysis.json').relative_to(ROOT)):
        hashlib.sha256((SOURCE/'analysis.json').read_bytes()).hexdigest()},
        'generator': str(Path(__file__).relative_to(ROOT)),
        'new_model_queries': 0, 'new_simulator_steps': 0,
        'curve_intervals': 'Descriptive pointwise 95% crossed seed/episode bootstrap.',
        'primary_intervals': '97.5% each; Bonferroni family coverage 95%.',
        'baseline_imagined_table': 'k0 audit; noisy baseline audit is reported in source analysis.'}
    (OUT/'figure_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
