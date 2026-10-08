#!/usr/bin/env python3
"""Descriptive report tables and plot from a completed, immutable power pilot."""
import argparse
import json
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = root/'runs'/args.run_id
    results = root/'docs/evoPlan/results/stage5-power-pilot'/args.run_id
    assert (run/'manifest.json').is_file() and not (run/'failure.json').exists()
    assessments = json.loads((run/'assessments.json').read_text())
    timing = json.loads((run/'timing.json').read_text())
    pilot = json.loads((run/'pilot.json').read_text())
    groups = []
    for k in [0,1]:
        for population in [16,256]:
            for generation in [1,4,16]:
                rows = [r for r in assessments if r['noise_k']==k and r['population']==population and r['generation']==generation]
                assert len(rows)==4 and len({r['seed'] for r in rows})==4
                real = statistics.mean(r['real_violations']/r['n_roots'] for r in rows)
                imagined = statistics.mean(r[f'imagined_k{k}'] for r in rows)
                groups.append(dict(k=k,population=population,generation=generation,
                    seed_real_counts=[r['real_violations'] for r in rows],real_rate=real,
                    imagined_rate=imagined,gap=real-imagined,
                    real_progress=statistics.mean(r['real_progress'] for r in rows)))
    times=[]
    for k in [0,1]:
        for population in [16,256]:
            rows=[r for r in timing if r['search'].startswith(f'k{k}-pop{population}-') and any(str(s) in r['search'] for s in [20261212,20261213])]
            times.append(dict(k=k,population=population,generations=len(rows),
                median_seconds=statistics.median(r['seconds'] for r in rows),
                summed_seconds=sum(r['seconds'] for r in rows),predictor_rows=sum(r['rows'] for r in rows)))
    summary=dict(groups=groups,timing_seeds_2_and_3=times,
        timing_warning='Excludes first/final seed with known full-suite overlap. This was not a dedicated isolated main-shape benchmark.',
        independent_episodes=96,search_seeds=4,nominal_interval=.95,
        intervals='Exploratory crossed seed/episode bootstrap from the declared pilot analysis, not multiplicity-adjusted confirmation.',
        pilot_costs=pilot['costs'])
    for dest in [run,results]:
        (dest/'report_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    colors={0:'#3569ad',1:'#b6681f'}
    for ax,population in zip(axes,[16,256]):
        for k in [0,1]:
            rows=[r for r in groups if r['k']==k and r['population']==population]
            x=[r['generation'] for r in rows]
            ax.plot(x,[100*r['real_rate'] for r in rows],color=colors[k],marker='o',label=f'k={k}: real')
            ax.plot(x,[100*r['imagined_rate'] for r in rows],color=colors[k],marker='s',linestyle='--',label=f'k={k}: imagined')
        ax.set_xscale('log',base=4)
        ax.set_xticks([1,4,16],['1','4','16'])
        ax.set_xlabel('Completed generations')
        ax.set_title(f'Population {population}')
        ax.grid(axis='y',alpha=.2)
        ax.set_ylim(0,40)
    axes[0].set_ylabel('Mean violation rate across four searches (%)')
    axes[1].legend(fontsize=9)
    fig.suptitle('Development pilot: real risk rises while imagined risk changes little')
    fig.text(.5,.01,'Same 96 assessment episodes across searches; previously inspected development data. No ROSARL comparison.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94))
    fig.savefig(results/'pilot_curve.png',dpi=180)
    plt.close(fig)
    shutil.copy2(results/'pilot_curve.png',run/'pilot_curve.png')
    dest=run/'source/experiments/scripts/evo_power_pilot_report.py'
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,dest)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
