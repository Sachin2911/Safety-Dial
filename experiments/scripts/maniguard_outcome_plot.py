"""Plot a fully validated batch's physical outcomes and actual policy-query costs."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_batch_report import OUTCOMES, summarize_batch

COLORS = ('#208568', '#c85252', '#a9b9cc', '#d7a44d')
NAMES = ('Safe complete', 'Unsafe complete', 'Safe unfinished', 'Unsafe unfinished')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.figure.exists() or args.report.exists():
        raise ValueError('Use new figure and report paths to preserve existing evidence')
    report = summarize_batch(args.batch)
    if report['statuses'] != {'completed': report['planned']}:
        raise ValueError('Plot requires every planned episode to be completed and validated')
    groups = sorted(report['groups'], key=lambda g: (
        g['family'], g['label'] not in ('identity', 'baseline_repeat'), g['label']))
    fig, (outcomes, queries) = plt.subplots(
        1, 2, figsize=(12, max(4.5, len(groups) * 0.55 + 1.5)),
        gridspec_kw={'width_ratios': [1.6, 1]}, sharey=True)
    labels, query_totals = [], []
    for y, group in enumerate(groups):
        labels.append(group['family'].title() + ': ' + group['label'].replace('_', ' '))
        left = 0
        for name, color in zip(OUTCOMES, COLORS):
            count = group['outcomes'][name]
            outcomes.barh(y, count, left=left, color=color, height=0.68)
            if count:
                outcomes.text(left + count / 2, y, str(count), ha='center', va='center',
                              color='white' if color in COLORS[:2] else '#17202b')
            left += count
        episodes = [r['result'] for r in report['episodes']
                    if r['family'] == group['family'] and r['label'] == group['label']]
        query_totals.append(sum(r['ledger']['policy_queries'] for r in episodes))
    outcomes.set_yticks(range(len(groups)), labels)
    outcomes.invert_yaxis()
    outcomes.set_xlabel('Episodes')
    outcomes.set_xlim(0, max(g['planned'] for g in groups))
    outcomes.xaxis.set_major_locator(MaxNLocator(integer=True))
    queries.barh(range(len(groups)), query_totals, color='#42566e', height=0.68)
    for y, count in enumerate(query_totals):
        queries.text(count, y, f' {count:,}', ha='left', va='center', fontsize=10)
    queries.set_xlim(0, max(query_totals) * 1.22)
    queries.set_xlabel('Policy queries, summed over episodes')
    queries.tick_params(axis='y', left=False, labelleft=False)
    for ax in (outcomes, queries):
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=0.15)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS]
    fig.legend(handles, NAMES, loc='lower center', ncol=4, frameon=False)
    fig.suptitle(f'ManiGuard {report["split"]} screen: {report["planned"]} completed episodes',
                 fontsize=15)
    fig.text(0.5, 0.92, 'Descriptive physical outcomes; query totals include early terminations',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    with args.report.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    fig.savefig(args.figure, dpi=150)
    plt.close(fig)
    print(f'Saved {args.figure} and audited evidence {args.report}')


if __name__ == '__main__':
    main()
