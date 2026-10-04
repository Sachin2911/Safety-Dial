#!/usr/bin/env python3
"""Post-hoc decomposition and figures for the completed fixed transfer pilot.

No selection, fitting, simulator interaction or retrospective change to gates.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.evoRanking import segment_metrics
from helpers.evoTransferPilot import rank_audit

ROOT = Path(__file__).resolve().parents[2]


def confusion(truth, prediction):
    t, p = np.asarray(truth, bool), np.asarray(prediction, bool)
    tp, fn = int((t & p).sum()), int((t & ~p).sum())
    fp, tn = int((~t & p).sum()), int((~t & ~p).sum())
    return dict(n=int(t.size), true_positive=tp, false_negative=fn,
                false_positive=fp, true_negative=tn,
                sensitivity=None if tp + fn == 0 else tp / (tp + fn),
                specificity=None if tn + fp == 0 else tn / (tn + fp))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run_id
    dest = ROOT / 'docs/evoPlan/results/stage4-transfer-pilot' / args.run_id
    assert not (dest / 'posthoc_decomposition.json').exists()
    pilot = json.loads((run / 'pilot.json').read_text())
    meta = json.loads((run / 'candidate_plan.json').read_text())
    with np.load(run / 'scores.npz') as f:
        scores = {k: f[k] for k in f.files}
    endpoint, dense, readout, imagined, margin = [], [], [], [], []
    for j in range(len(meta)):
        with np.load(run / f'real_{j:03d}.npz') as f:
            d = (f['dense_clearance'] <= 0).any(-1)
            e = (f['endpoint_clearance'] <= 0).any(-1)
            r = segment_metrics(f['readout'])['violated']
            assert np.array_equal(d, scores['real_viol'][j])
            assert np.array_equal(d, f['dense_violated'])
            assert np.array_equal(r, scores['readout_viol'][j])
            assert not (e & ~d).any()
        with np.load(run / f'imagined_{j:03d}.npz') as f:
            m = segment_metrics(f['readout'][0, :, 0])
            assert np.array_equal(m['violated'], scores['imag_viol'][j])
        endpoint.append(e)
        dense.append(d)
        readout.append(r)
        imagined.append(m['violated'])
        margin.append(m['min_clearance'])
    endpoint, dense, readout, imagined, margin = map(
        np.stack, [endpoint, dense, readout, imagined, margin])
    residual = np.array([j for j, m in enumerate(meta) if m['kind'] != 'attenuation'])
    winner = pilot['imagined_winner']
    groups = {'all_candidates': np.arange(len(meta)), 'residual_candidates_only': residual,
              'baseline': np.array([0]), 'imagined_winner': np.array([winner])}
    analysis = dict(status='post-hoc descriptive analysis; no gate or selection changed',
                    independent_source_episodes=64, candidate_population_seeds=1,
                    pooled_counts_are_repeated_candidate_root_pairs=True,
                    new_simulator_steps=0, optimizer_updates=0, groups={})
    for name, idx in groups.items():
        analysis['groups'][name] = dict(
            candidates=len(idx), candidate_root_pairs=int(dense[idx].size),
            dense_violations=int(dense[idx].sum()), endpoint_violations=int(endpoint[idx].sum()),
            real_readout_violations=int(readout[idx].sum()), imagined_violations=int(imagined[idx].sum()),
            endpoint_vs_dense=confusion(dense[idx], endpoint[idx]),
            real_readout_vs_endpoint=confusion(endpoint[idx], readout[idx]),
            real_readout_vs_dense=confusion(dense[idx], readout[idx]),
            imagined_vs_dense=confusion(dense[idx], imagined[idx]),
            real_failure_count_range=[int(dense[idx].sum(-1).min()), int(dense[idx].sum(-1).max())])
    for name, idx in [('all_candidates', groups['all_candidates']), ('residual_candidates_only', residual)]:
        # A negative margin is used only as a descriptive risk ordering, never as a new gate.
        analysis['groups'][name]['real_readout_rank'] = rank_audit(
            readout[idx], dense[idx], scores['episodes'], n_boot=2000, seed=20261021)
        analysis['groups'][name]['imagined_negative_margin_rank'] = rank_audit(
            -margin[idx], dense[idx], scores['episodes'], n_boot=2000, seed=20261021)
    analysis['assessment_counts'] = {
        label: dict(real=int(dense[j, 32:].sum()), imagined=int(imagined[j, 32:].sum()),
                    real_readout=int(readout[j, 32:].sum()), n=32)
        for label, j in [('baseline', 0), ('imagined_winner', winner)]}
    for group in analysis['groups'].values():
        # These post-hoc rankings must not be mistaken for the preregistered transfer screen.
        for key in ['real_readout_rank', 'imagined_negative_margin_rank']:
            if key in group:
                group[key].pop('transfer_screen_pass', None)
    for path in [dest / 'posthoc_decomposition.json', run / 'posthoc_decomposition.json']:
        path.write_text(json.dumps(analysis, indent=2, allow_nan=False) + '\n')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    colors = {'baseline': '#183153', 'residual': '#15847b', 'attenuation': '#a75a23'}
    for kind in colors:
        idx = [j for j, m in enumerate(meta) if m['kind'] == kind]
        for ax, x, y in [
            (axes[0], scores['imag_ret'].mean(1), scores['real_ret'].mean(1)),
            (axes[1], imagined.mean(1)*100, dense.mean(1)*100),
            (axes[2], readout.mean(1)*100, dense.mean(1)*100),
        ]:
            ax.scatter(x[idx], y[idx], s=27, alpha=.72, c=colors[kind], label=kind)
    for ax, x, y in [
        (axes[0], scores['imag_ret'].mean(1), scores['real_ret'].mean(1)),
        (axes[1], imagined.mean(1)*100, dense.mean(1)*100),
        (axes[2], readout.mean(1)*100, dense.mean(1)*100),
    ]:
        ax.scatter(x[winner], y[winner], marker='*', s=150, color='#cc2345', label='imagined winner', zorder=4)
        ax.grid(alpha=.18)
    axes[0].set(xlabel='Imagined progress (m)', ylabel='Real progress (m)', title='Progress transfer')
    axes[1].set(xlabel='Imagined violations (%)', ylabel='Dense real violations (%)', title='Safety in imagination')
    axes[2].set(xlabel='Readout on real images (%)', ylabel='Dense real violations (%)', title='Safety from real images')
    for ax in axes[1:]:
        ax.plot([0, 100], [0, 100], '--', color='.6', lw=1, zorder=0)
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle('Fixed Walker2d pilot: 101 controllers on the same 64 source episodes')
    for suffix in ['png', 'pdf']:
        path = dest / f'transfer_pilot.{suffix}'
        fig.savefig(path, dpi=180)
        shutil.copy2(path, run / path.name)
    plt.close(fig)
    source = Path(__file__).resolve()
    copy = run / 'source/experiments/scripts' / source.name
    shutil.copy2(source, copy)
    provenance = dict(analysis='post-hoc decomposition and plot', new_simulator_steps=0,
        code_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scores_sha256=hashlib.sha256((run / 'scores.npz').read_bytes()).hexdigest(),
        bootstrap_seed=20261021, bootstrap_replicates=2000,
        note='Fixed candidate population; uncertainty resamples episodes, not candidates or search seeds.')
    for path in [dest / 'report_provenance.json', run / 'report_provenance.json']:
        path.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
