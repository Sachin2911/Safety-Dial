"""Account for all recorded adaptation interactions without counting aggregates twice."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_source import ROOT, sha256


def read(path):
    return json.loads(path.read_text())


def phase(directory):
    first = directory.relative_to(ROOT / 'runs/unisafe').parts[0]
    labels = {
        'adaptation-identity-20261015': 'identity_acceptance',
        'adaptation-pilot-20261015': 'command_history_development',
        'adaptation-grasp-acceptance-20261015': 'grasp_acceptance',
        'adaptation-grasp-pilot-20261015': 'grasp_development',
        'adaptation-paircheck-serial-20261015': 'fresh_process_acceptance',
        'adaptation-search-dev-20261015': 'initial_search_development',
        'adaptation-search-replicated-20261015': 'replicated_training',
        'adaptation-validation-20261015': 'selection_validation',
        'adaptation-evaluation-20261015': 'reserved_test',
    }
    if first not in labels:
        raise ValueError(f'Unclassified adaptation worker: {directory}')
    return labels[first]


def compact_worker(directory, summary):
    assert summary['status'] == 'complete'
    assert summary['environment_steps'] == sum(e['steps'] for e in summary['episodes'])
    assert summary['physics_steps'] == 5 * summary['environment_steps']
    assert summary['training_updates'] == 0
    assert summary['query_counts'] == dict(sum((Counter(e['query_counts']) for e in summary['episodes']), Counter()))
    return {
        'directory': str(directory.relative_to(ROOT)), 'phase': phase(directory),
        'episodes': len(summary['episodes']), 'environment_steps': summary['environment_steps'],
        'physics_steps': summary['physics_steps'], 'model_queries': summary['query_counts'],
        'outcomes': dict(Counter(e['outcome'] for e in summary['episodes'])),
        'worker_elapsed_seconds': summary['elapsed_seconds'],
        'summary_sha256': sha256(directory / 'summary.json'),
    }


def aggregate(items):
    return {
        'worker_runs': len(items), 'episodes': sum(x['episodes'] for x in items),
        'environment_steps': sum(x['environment_steps'] for x in items),
        'physics_steps': sum(x['physics_steps'] for x in items),
        'model_queries': dict(sum((Counter(x['model_queries']) for x in items), Counter())),
        'outcomes': dict(sum((Counter(x['outcomes']) for x in items), Counter())),
        'summed_worker_elapsed_seconds': sum(x['worker_elapsed_seconds'] for x in items),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    root = ROOT / 'runs/unisafe'
    workers, incomplete = [], []
    for study in sorted(root.glob('adaptation-*')):
        if not study.is_dir():
            continue
        for path in sorted(study.rglob('summary.json')):
            try:
                summary = read(path)
            except json.JSONDecodeError:
                continue  # A currently running writer will be recorded as incomplete below.
            if {'platform', 'script_sha256', 'query_counts', 'episodes'} <= summary.keys():
                workers.append(compact_worker(path.parent, summary))
        unfinished = defaultdict(list)
        for trace in study.rglob('steps.jsonl'):
            if not (trace.parent.parent / 'summary.json').exists():
                unfinished[trace.parent.parent].append(trace)
        for directory, traces in unfinished.items():
            rows = 0
            for path in traces:
                for line in path.read_text().splitlines():
                    try:
                        json.loads(line)
                        rows += 1
                    except json.JSONDecodeError:
                        pass
            incomplete.append({
                'directory': str(directory.relative_to(ROOT)), 'phase': phase(directory),
                'recorded_control_steps_lower_bound': rows,
                'limits': 'Incomplete worker is excluded from exact completed totals. Buffered or failed transitions and model calls may be unrecorded; do not interpret the lower bound as an exact total.',
            })
    assert len({r['directory'] for r in workers}) == len(workers)
    grouped = defaultdict(list)
    for worker in workers:
        grouped[worker['phase']].append(worker)
    phases = {k: aggregate(v) for k, v in grouped.items()}
    heads = read(root / 'release-heads-20261015/summary.json')
    stopped = read(root / 'adaptation-paircheck-20261015/stopped.json')
    assert stopped['environment_steps'] == stopped['model_queries'] == stopped['training_updates'] == 0
    handoff = read(root / 'adaptation-evaluation-20261015/handoff.json')
    complete = handoff['status'] == 'complete_pending_research_review' and not incomplete
    optimizers = []
    for name in ('adaptation-search-dev-20261015', 'adaptation-search-replicated-20261015'):
        directory = root / name
        path = directory / 'summary.json'
        search = read(path) if path.exists() else {'runs': read(directory / 'progress.json')}
        runs = search['runs']
        optimizers.append({
            'directory': str(directory.relative_to(ROOT)), 'complete': path.exists(),
            'selection_or_distribution_updates': len(runs),
            'policy_gradient_updates': sum(r['method'] == 'policy_gradient' for r in runs),
            'pretrained_weight_updates': 0,
            'methods': {m: {
                'episodes': sum(r['episodes'] for r in runs if r['method'] == m),
                'environment_steps': sum(r['environment_steps'] for r in runs if r['method'] == m),
                'worker_elapsed_seconds': sum(r['worker_elapsed_seconds'] for r in runs if r['method'] == m),
            } for m in ('evolution', 'random', 'policy_gradient')},
        })
    baseline = read(ROOT / 'docs/research/reports/UNISafe baseline pilot.results.json')['followup_diagnosis']
    causal = read(ROOT / 'docs/research/reports/UNISafe release and switching diagnosis.results.json')
    earlier_engineering = []
    for path in sorted(root.glob('environment-*/summary.json')) + [root / 'terminal-validation-20261015/summary.json']:
        value = read(path)
        earlier_engineering.append({
            'directory': str(path.parent.relative_to(ROOT)), 'summary_sha256': sha256(path),
            **{k: value[k] for k in ('environment_steps', 'physics_steps', 'learned_model_queries', 'elapsed_seconds') if k in value},
            'counts_instrumented': 'environment_steps' in value,
        })
    earlier_offline = []
    for name in ('inference-20261015T183740Z', 'repeat-20261015'):
        path = root / name / 'summary.json'
        value = read(path)
        earlier_offline.append({
            'directory': str(path.parent.relative_to(ROOT)), 'summary_sha256': sha256(path),
            'environment_steps': value['simulator_interactions'],
            'model_queries': dict(sum((Counter(e['query_counts']) for e in value['episodes']), Counter())),
            'elapsed_seconds': value['elapsed_seconds'],
        })
    completed = aggregate(workers)
    offline_queries = Counter(heads['query_counts_replay']) + Counter(heads['additional_model_queries'])
    result = {
        'status': 'complete' if complete else 'partial_completed_work_only',
        'recorded_at_utc': datetime.now(UTC).isoformat(),
        'scope': 'New adaptation study; all complete worker runs counted exactly once. Earlier shared foundations are separate and not free data.',
        'new_physical_completed': completed, 'new_physical_phases': phases, 'workers': workers,
        'incomplete_workers': incomplete, 'optimizer_accounting': optimizers,
        'new_offline_diagnostic': {
            'directory': 'runs/unisafe/release-heads-20261015', 'environment_steps': 0,
            'model_queries': dict(offline_queries), 'elapsed_seconds': heads['elapsed_seconds'],
            'summary_sha256': sha256(root / 'release-heads-20261015/summary.json'),
        },
        'new_total_completed_model_queries': dict(Counter(completed['model_queries']) + offline_queries),
        'stopped_startup_trial': stopped,
        'earlier_shared_foundation': {
            'baseline_autonomous': baseline['autonomous_totals_including_pilot_and_discarded_followups'],
            'baseline_autonomous_queries': baseline['model_query_totals'],
            'causal_autonomous': causal['new_physical_totals'],
            'causal_autonomous_queries': causal['physical_run_model_queries'],
            'causal_offline_probe': {k: causal['offline_probe'][k] for k in ('query_counts_replay', 'query_counts_additional_individual_samples', 'elapsed_seconds', 'environment_steps')},
            'environment_acceptance_runs': earlier_engineering,
            'recorded_data_inference': earlier_offline,
            'limits': 'The baseline report includes the final 60-step acceptance in its separate engineering record, not its autonomous total; it appears once here. Several initial migration attempts were not fully instrumented, including an initial environment run and a pre-step integration failure with unknown partial query count. These are not assigned zero cost. Original checkpoint pretraining and earlier unrelated projects are outside this study.',
        },
        'elapsed_time_scope': 'Worker timers include simulator/model startup and episode execution. Aggregate them as measured process time, not elapsed project wall time. Search, validation, batch, handoff waiting and worker timers overlap; never sum these levels. Offline diagnostics and stopped concurrent startup may overlap physical work. Video/analysis time is separate.',
        'query_units': 'Calls per named component; uncertainty_ensemble counts an ensemble invocation. Offline conditional samples count individual evaluations rather than batched kernels.',
    }
    if args.require_complete:
        assert complete, 'Evaluation handoff or physical runs remain incomplete'
        assert all(x['complete'] for x in optimizers)
        expected_counts = {
            'identity_acceptance': 3, 'command_history_development': 32,
            'grasp_acceptance': 3, 'grasp_development': 32, 'fresh_process_acceptance': 2,
            'initial_search_development': read(root / 'adaptation-search-dev-20261015/summary.json')['episodes'],
            'replicated_training': read(root / 'adaptation-search-replicated-20261015/summary.json')['episodes'],
            'selection_validation': read(root / 'adaptation-validation-20261015/summary.json')['episodes'],
            'reserved_test': read(root / 'adaptation-evaluation-20261015/test/summary.json')['completed_episodes'],
        }
        assert set(phases) == set(expected_counts)
        assert all(phases[k]['episodes'] == n for k, n in expected_counts.items())
        assert completed['environment_steps'] == sum(p['environment_steps'] for p in phases.values())
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(result['status'], 'completed episodes', completed['episodes'], 'control steps', completed['environment_steps'])
    for name, values in phases.items():
        print(name, values['episodes'], values['environment_steps'])
    print('Incomplete workers:', len(incomplete))


if __name__ == '__main__':
    main()
