"""Plot observed Jar prerequisites and executed guard/hand-back activity."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'experiments/scripts')]

from maniguard_diagnose import diagnose

LABELS = {'identity': 'Unchanged policy', 'prerequisite': 'Prerequisite guard',
          'handback': 'Guard with settling and fresh queries'}
LANES = ('Post-guard hold', 'Guard blocks arm', 'Target held', 'Jar closed')
COLORS = ('#bf8500', '#b63b36', '#3767a8', '#278168')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, action='append', required=True)
    parser.add_argument('--figure', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.figure.exists() or args.report.exists():
        raise ValueError('Use new output paths to preserve earlier evidence')
    episodes = [diagnose(path) for path in args.episode]
    if len({(r['family'], r['scene'], r['seed']) for r in episodes}) != 1:
        raise ValueError('Compare one family, scene and root seed')
    if any(r['family'] != 'jar' or r['adapter']['kind'] not in LABELS
           or r['engineering_override'] is not None for r in episodes):
        raise ValueError('Requires completed full-horizon Jar identity/guard/hand-back episodes')
    fig, axes = plt.subplots(len(episodes), 1, sharex=True, squeeze=False,
                             figsize=(11, 2.35 * len(episodes) + 1.2))
    evidence = []
    for ax, path, result in zip(axes[:, 0], args.episode, episodes):
        trace = path / 'trace.jsonl'
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        lanes = [[] for _ in LANES]
        previous_blocked, previous_remaining = False, 0
        for row in rows:
            if row['type'] == 'observed' and row['monitor']['observation']['ap']['jar_closed']:
                lanes[3].append(row['step'])
            elif row['type'] == 'goal' and row['detail']['held']:
                lanes[2].append(row['step'])
            elif row['type'] == 'proposal':
                event = row['event']
                blocked, remaining = event['guard_blocked'], event.get('handback_remaining', 0)
                if blocked:
                    lanes[1].append(row['step'])
                if result['adapter']['kind'] == 'handback' and not blocked and (
                        remaining > 0 or (previous_remaining > 0 and not previous_blocked)):
                    lanes[0].append(row['step'])
                previous_blocked, previous_remaining = blocked, remaining
        if len(lanes[0]) != result['events'].get('handback_hold_inferred', 0):
            raise ValueError('Hold timeline disagrees with episode diagnosis')
        if len(lanes[1]) != result['ledger']['guard_steps']:
            raise ValueError('Blocked timeline disagrees with physical ledger')
        for y, (steps, color) in enumerate(zip(lanes, COLORS)):
            ax.hlines(y, 0, result['steps'], color='#e5e9ee', linewidth=7, zorder=0)
            ax.scatter(steps, [y] * len(steps), marker='|', s=105, color=color, linewidths=1)
        first = result['first_violation_observation']
        if first is not None:
            ax.axvline(first['step'], color='#b63b36', linestyle='--', linewidth=1)
            ax.text(first['step'] + 15, 3.55, 'First safety violation', fontsize=8,
                    color='#a12929', va='bottom')
        outcome = next(k for k in ('safe_completion', 'unsafe_completion',
                                  'safe_unfinished', 'unsafe_unfinished') if result[k])
        ax.set_title(f"{LABELS[result['adapter']['kind']]}: {outcome.replace('_', ' ')}; "
                     f"{result['steps']:,} actions, {result['ledger']['policy_queries']:,} queries",
                     loc='left', fontsize=10, pad=15)
        ax.set_yticks(range(4), LANES)
        ax.set_ylim(-0.5, 4.1)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', alpha=0.15)
        evidence.append({'episode': str(path), 'trace_sha256': hashlib.sha256(
            trace.read_bytes()).hexdigest(), 'lane_counts': dict(zip(LANES, map(len, lanes))),
            'diagnostic': result})
    axes[-1, 0].set_xlabel('Policy action / observation index')
    axes[-1, 0].set_xlim(0, max(r['steps'] for r in episodes))
    fig.suptitle(f"Jar {episodes[0]['scene']}, root seed {episodes[0]['seed']}: "
                 'prerequisite and intervention timelines', fontsize=13)
    fig.text(0.5, 0.018, 'Gray denotes an inactive lane during the recorded episode. '
             'Separate physical starts; these are not deterministic counterfactuals.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    report = {'scope': 'Completed engineering episodes; descriptive temporal evidence only.',
              'alignment': 'Jar closed and target held are measured after the indexed action count. '
              'Guard and hold activity use the proposal index before its action executes. '
              'The hold countdown is stored after the action, so its final zero is included '
              'and intervening guard-blocked proposals are excluded.',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'episodes': evidence}
    with args.report.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    fig.savefig(args.figure, dpi=150)
    plt.close(fig)
    print(f'Saved {args.figure} and {args.report}')


if __name__ == '__main__':
    main()
