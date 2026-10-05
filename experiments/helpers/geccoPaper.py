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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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

    # Split each high-pressure gap into missed between-block events, real-image
    # readout error and the remaining imagined-versus-real-readout difference.
    decomposition = json.loads((SOURCE / 'decomposition.json').read_text())
    parts = [('dense_minus_endpoint', 'Between-block events', '#BBBBBB'),
             ('endpoint_minus_real_readout', 'Readout on real frames', '#D55E00'),
             ('real_readout_minus_imagined', 'Imagined vs. real frames', '#0072B2')]
    fig, ax = plt.subplots(figsize=(3.35, 1.45))
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
           xlabel='Contribution to actual-minus-imagined failure (pp)', xlim=(-10, 40))
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.42, 1.3), frameon=False,
              fontsize=6.5, handlelength=1)
    ax.grid(axis='x', alpha=.15)
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
    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    for kind, (label, color, marker, size) in styles.items():
        members = [r for r in pool['candidates'] if r['origins'][0]['kind'] == kind]
        ax.scatter([r['action_delta_rms'] for r in members],
                   [100 * r['check_failure'] for r in members], s=size, marker=marker,
                   color=color, alpha=.85, lw=0, label=label, zorder=3 if kind == 'baseline' else 2)
    ax.axhline(100 * pool['baseline_failure'], ls='--', color='#777777', lw=1)
    ax.set(xlabel='Same-input action change from starting controller (RMS)',
           ylabel='Check failure (%)', ylim=(0, 32))
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(.45, 1.0), frameon=False,
              fontsize=6.5, handletextpad=.2, columnspacing=.8)
    ax.grid(alpha=.15)
    save(fig, 'candidate_quality')

    repair = json.loads((REPAIR / 'analysis.json').read_text())
    repair_arms = ['baseline', 'probe', 'dynamics', 'combined']
    labels = ['Original', 'Probe', 'Predictor', 'Both']
    colors = ['#777777', '#0072B2', '#009E73', '#D55E00']
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.85))
    for ax, key, title in zip(axes, ['event_detection', 'false_alarm'],
                             ['Failure-endpoint detection', 'Healthy-endpoint false alarms']):
        vals = [repair['audit'][arm]['teacher_forced'][key] for arm in repair_arms]
        points = np.array([v['point'] for v in vals])*100
        bounds = np.array([[v['lo'], v['hi']] for v in vals])*100
        ax.bar(np.arange(4), points, color=colors, width=.7)
        ax.errorbar(np.arange(4), points, yerr=np.stack([points-bounds[:, 0], bounds[:, 1]-points]),
                    fmt='none', ecolor='black', capsize=2, lw=.8)
        ax.set_xticks(np.arange(4), labels, rotation=40, ha='right', fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.set_ylabel('%', fontsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', alpha=.15)
    fig.tight_layout(w_pad=.8)
    save(fig, 'predictor_repair')

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
        str(path.relative_to(ROOT)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [SOURCE/'analysis.json', SOURCE/'decomposition.json',
                     DIAGNOSTIC/'analysis.json', DIVERGENCE/'analysis.json',
                     REPAIR/'analysis.json', REPAIR/'paired_changes.json']},
        'generator': str(Path(__file__).relative_to(ROOT)),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_model_queries': 0, 'new_simulator_steps': 0,
        'curve_intervals': 'Descriptive pointwise 95% crossed seed/episode bootstrap.',
        'primary_intervals': '97.5% each; Bonferroni family coverage 95%.',
        'baseline_imagined_table': 'k0 audit; noisy baseline audit is reported in source analysis.',
        'candidate_quality': 'Exploratory Stage 6 fixed pool on reused development episodes.',
        'predictor_repair': 'Exploratory reused audit: 24 source episodes, 89 correlated candidates, three fitting seeds; 95% whole-episode bootstrap.'}
    (OUT/'figure_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
