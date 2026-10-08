#!/usr/bin/env python3
"""Present terminal S4 evidence without inference, simulation or new statistical claims.

Only JSON is decoded. Frozen banks/checkpoints are hashed, never deserialized.
The original artifacts are immutable; output must be a fresh sibling of results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from walker_s3_figures import REPO, check, file_hash, finite, flag, json_safe, metric_cells, number, output_hash, table

ARMS = ('random', 'boundary')
BANKS = ('test', 'stress')


def read(path):
    return json.loads(Path(path).read_text())


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def paired_cells(record):
    """Validate a saved interval's semantics without recomputing the bootstrap."""
    for key in ('reference', 'updated'):
        metric_cells(record[key])
    defined = finite(record['point'])
    check(type(record['defined']) is bool and record['defined'] == defined,
          'Paired defined flag differs from saved point')
    ref, updated = record['reference']['fsa'], record['updated']['fsa']
    check(defined == (finite(ref) and finite(updated)), 'Paired point has undefined constituent FSA')
    if defined:
        check(math.isclose(record['point'], updated - ref, abs_tol=1e-12),
              'Saved paired point disagrees with constituent FSA')
    lo, hi = record['lo'], record['hi']
    interval = finite(lo) and finite(hi)
    check(interval or (not finite(lo) and not finite(hi)), 'Only one paired interval endpoint is defined')
    if interval:
        check(defined and -1 <= lo <= hi <= 1, 'Undefined point cannot support a directional interval')
    direction = ('decrease' if interval and hi < 0 else 'increase' if interval and lo > 0 else
                 'inconclusive' if defined else 'undefined')
    check(record['direction'] == direction and record['interval_excludes_zero'] == (direction in ('decrease', 'increase')),
          'Saved paired direction contradicts the saved interval')
    reason = ('zero_acceptance' if not record['reference']['n_accepted'] or not record['updated']['n_accepted'] else
              'accepted_censored_futures' if not defined else None)
    check(record['undefined_reason'] == reason, 'Saved undefined reason contradicts the paired support')
    requested, usable, missing = (record[k] for k in ('n_boot_requested', 'n_boot', 'n_boot_undefined'))
    check(all(type(v) is int and v >= 0 for v in (requested, usable, missing)) and requested > 0
          and usable + missing == requested, 'Inconsistent saved bootstrap support')
    check(record['cluster_unit'] == 'source_episode', 'Unexpected paired cluster unit')
    return [number(record['point']), f'[{number(lo)}, {number(hi)}]', direction,
            reason or 'n/a', f'{usable}/{requested}', str(record['n_source_episodes'])]


def frozen_stage(workflow_state, stage_id, folders):
    state = read(workflow_state)
    stage = state.get('stages', {}).get(stage_id, {})
    check(stage.get('status') in ('completed', 'gate_stopped'), 'S4 stage is not finalized')
    attempts = stage.get('attempts', [])
    check(attempts and attempts[-1].get('finished_unix') is not None
          and attempts[-1].get('returncode') in (0, 2), 'S4 subprocess has not finalized')
    for folder in folders:
        check(str(folder) in stage.get('output_hashes', {}), f'Workflow did not freeze {folder}')
        check(output_hash(folder) == stage['output_hashes'][str(folder)], f'Finalized output changed: {folder}')
    return stage


def load_evidence(run, results, bank, state_path, stage_id):
    stage = frozen_stage(state_path, stage_id, (run, results, bank))
    hashes = {}
    def tracked(path):
        hashes[str(path)] = file_hash(path)
        return read(path)
    report, original = tracked(results / 'study.json'), tracked(run / 'study.json')
    core = {k: v for k, v in report.items() if k != 'bank_hf_revision'}
    check(same(core, original), 'Run/results scientific reports disagree')
    manifest = tracked(results / 'manifest.json')
    for folder in (run, bank):
        check(same(tracked(folder / 'manifest.json'), manifest), 'S4 manifest mirrors disagree')
    gate = tracked(run / 'decomposition_gate.json')
    check(file_hash(results / 'decomposition_gate.json') == hashes[str(run / 'decomposition_gate.json')],
          'Development gate mirrors disagree')
    hashes[str(results / 'decomposition_gate.json')] = file_hash(results / 'decomposition_gate.json')
    flag(gate['go'])
    check(gate['role'] == 'development', 'S4 prerequisite must use development evidence')
    identity = stage['gate']
    check(identity['key'] == 'go' and Path(identity['path']).resolve() == run / 'decomposition_gate.json'
          and identity['sha256'] == hashes[str(run / 'decomposition_gate.json')]
          and identity['passes'] is gate['go'], 'Workflow gate does not identify this saved S4 gate')
    check(report['run_id'] == manifest['run_id'] == run.name, 'S4 run identity mismatch')
    check(report['saved_evaluation_rows'] == manifest['data']['saved_evaluation_rows'], 'Row export identities disagree')
    def export(ref):
        name = ref['file']
        check(Path(name).name == name and name.endswith('.json'), 'Invalid mirrored row filename')
        for folder in (run, results, bank):
            path = folder / name
            check(file_hash(path) == ref['sha256'] and path.stat().st_size == ref['bytes'], 'Saved row export changed')
            hashes[str(path)] = ref['sha256']
    for ref in report['saved_evaluation_rows'].values():
        export(ref)
    if report['status'] == 'diagnostic_stop':
        check(stage['status'] == 'gate_stopped' and gate['go'] is False and same(report['gate'], gate)
              and report['final_test_evaluated'] is False and 'adaptation' not in report,
              'Diagnostic stop cannot claim final evaluation or acquisition completion')
        check(set(report['saved_evaluation_rows']) == {'development'}, 'Diagnostic rows must be development only')
        check(report['saved_evaluation_rows']['development']['file'] == 'development_rows.json', 'Wrong development export')
        dev = tracked(run / 'development_rows.json')
        check(dev and all(row['bank'] == 'dev' for row in dev), 'Diagnostic export contains non-development rows')
        check(all(v['n'] == len(dev) for v in gate['evidence'].values()), 'Development row count differs from gate')
        check(same(tracked(bank / 'diagnostic.json'), core), 'Bank diagnostic report differs')
        return report, gate, manifest, stage, hashes
    check(report['status'] == 'complete' and stage['status'] == 'completed' and gate['go'] is True
          and same(report['development_gate'], gate), 'S4 study is not a completed gated acquisition')
    check(set(report['saved_evaluation_rows']) == {'baseline_decomposition', 'no_update_evaluation'}, 'Both baseline measurements must be retained')
    check(report['saved_evaluation_rows']['baseline_decomposition']['file'] == 'baseline_decomposition_rows.json'
          and report['saved_evaluation_rows']['no_update_evaluation']['file'] == 'no_update_evaluation_rows.json', 'Baseline export roles disagree')
    check(same(tracked(bank / 'study.json'), core), 'Bank scientific report differs')
    check(report['repair_rules'] == gate['repair_rules'], 'Repair rules differ from saved gate')
    check(report['active_rules'] and set(report['active_rules']) <= {'health', 'speed'}
          and 'health' in report['active_rules'], 'Unexpected active rule set')
    paired = report['boundary_vs_random']
    check(paired['contrast'] == 'boundary minus random' and paired['cluster_unit'] == 'source_episode'
          and paired['confidence_level'] == .95, 'Unexpected saved contrast or uncertainty convention')
    budgets, seeds = report['planned_budgets'], report['acquisition_seeds']
    check(budgets and budgets == sorted(set(budgets)) and seeds and len(set(seeds)) == len(seeds), 'Invalid planned grid')
    check(set(report['adaptation']) == {'no_update', *ARMS}, 'Unexpected acquisition arms')
    check(set(paired['by_budget']) == {str(b) for b in budgets}, 'Incomplete paired budgets')
    for seed in seeds:
        for arm in ARMS:
            check(set(report['adaptation'][arm]) == {str(s) for s in seeds}, 'Incomplete acquisition seeds')
            curve = report['adaptation'][arm][str(seed)]
            check([p['budget'] for p in curve] == budgets, 'Incomplete acquisition budgets')
            for point in curve:
                path = run / point['evaluation_rows']['file']
                check(path.resolve().is_relative_to(run) and file_hash(path) == point['evaluation_rows']['sha256'], 'Adapted evaluation row identity changed')
                hashes[str(path)] = file_hash(path)
                check(point['charged_steps'] == point['root_generation_steps'] + point['common_seed_steps'] + point['acquired_steps'], 'Charged cost components disagree')
                for name in BANKS:
                    for rule in report['active_rules']:
                        metric = point['eval'][name][rule]
                        check(metric['margin'] == point['eval']['dev'][rule]['margin']
                              and metric['at_matched']['m'] == metric['margin'],
                              'Final evaluation margin differs from development')
                        metric_cells(metric['at_matched'])
        for budget in budgets:
            entry = paired['by_budget'][str(budget)]
            check(set(entry['by_seed']) == {str(s) for s in seeds}, 'Incomplete paired seeds')
            entry = entry['by_seed'][str(seed)]
            points = {a: next(p for p in report['adaptation'][a][str(seed)] if p['budget'] == budget) for a in ARMS}
            check(entry['charged_steps'] == points['random']['charged_steps'] == points['boundary']['charged_steps'], 'Paired costs are not equal')
            for arm in ARMS:
                check(entry['evaluation_rows_sha256'][arm] == points[arm]['evaluation_rows']['sha256'], 'Paired row hashes disagree')
            for name in BANKS:
                for rule in report['active_rules']:
                    item = entry[name][rule]
                    paired_cells(item)
                    check(same(item['reference'], points['random']['eval'][name][rule]['at_matched'])
                          and same(item['updated'], points['boundary']['eval'][name][rule]['at_matched']), 'Paired constituent statistics differ from their arms')
    return report, gate, manifest, stage, hashes


def gate_table(gate):
    t = gate['thresholds']
    return table(['Rule', 'Branches', 'Unsafe', 'Imagined false-safe', 'Imagination attributed', 'Share', 'Saved result'],
                 [[rule, e['n'], e['n_unsafe'], e['n_false_safe'], e['n_imagination'], number(e['imagination_share']), flag(e['pass'])]
                  for rule, e in gate['evidence'].items()]) + '\n\n' + (
        f"Recorded thresholds: unsafe ≥ {t['min_unsafe']}; false-safe ≥ {t['min_false_safe']}; "
        f"imagination-attributed ≥ {t['min_imagination']}; share ≥ {number(t['min_share'])}.")


def markdown(report, gate, manifest):
    stopped = report['status'] == 'diagnostic_stop'
    title = 'Development diagnostic stop' if stopped else 'Completed acquisition comparison'
    text = [f'# Walker S4: {title}', '', f"Run: `{report['run_id']}`.", '', '## Saved development gate', '',
            gate_table(gate), '', '![Saved gate evidence](development_gate.png)', '']
    if stopped:
        text += ['The development imagination-attribution prerequisite did not pass. No final-test measurements or acquisition comparison are reported. '
                 'Final banks may already have been constructed and charged; their existence does not establish that final-test predictions were evaluated.', '',
                 f"Recorded evaluation-bank construction cost: {number(manifest['costs']['generated_evaluation_steps'])} simulator steps.", '']
    else:
        text += ['This is the completed prospective random-versus-boundary acquisition grid. Each model uses its own saved development-calibrated margin, unchanged on test/stress. '
                 'The no-update rows are the separately evaluated no-update model, not a substitution of the baseline decomposition measurements.', '',
                 '## Paired acquisition estimates', '',
                 'Contrasts are boundary minus random FSA; negative values favor boundary. Intervals are the saved 95% bootstrap intervals clustered by source episode. '
                 'Seeds remain separate. No averages, intervals or new directional claims are estimated here. An undefined full-data point stays undefined even when some bootstrap replicates are usable.', '',
                 '![Saved paired estimates](paired_fsa.png)', '']
        for name in BANKS:
            for rule in report['active_rules']:
                text += [f'### {name.title()}: {rule}', '']
                paired_rows, absolute_rows = [], []
                base = report['adaptation']['no_update'][name][rule]
                metric_cells(base['at_matched'])
                check(base['margin'] == report['adaptation']['no_update']['dev'][rule]['margin']
                      and base['at_matched']['m'] == base['margin'], 'No-update margin differs from development')
                absolute_rows.append(['No update', 'n/a', 'n/a', number(base['margin'])] + metric_cells(base['at_matched']))
                for budget in report['planned_budgets']:
                    for seed in report['acquisition_seeds']:
                        entry = report['boundary_vs_random']['by_budget'][str(budget)]['by_seed'][str(seed)]
                        paired_rows.append([budget, seed, entry['charged_steps']] + paired_cells(entry[name][rule]))
                        for arm in ARMS:
                            point = next(p for p in report['adaptation'][arm][str(seed)] if p['budget'] == budget)
                            metric = point['eval'][name][rule]
                            absolute_rows.append([arm.title(), seed, budget, number(metric['margin'])] + metric_cells(metric['at_matched']))
                text += [table(['Additional branches', 'Seed', 'Charged steps', 'Δ FSA', '95% interval', 'Saved direction', 'Undefined reason', 'Usable bootstrap', 'Source episodes'], paired_rows), '',
                         table(['Model', 'Seed', 'Additional branches', 'Margin', 'Branches', 'Accepted', 'Unsafe accepted', 'Accepted unresolved', 'Acceptance rate', 'FSA'], absolute_rows), '']
        costs = [[arm, seed, p['budget'], p['root_generation_steps'], p['common_seed_steps'], p['acquired_steps'], p['charged_steps'], p['optimizer_steps'], p['model_candidate_queries'], number(p['training_time_s'])]
                 for arm in ARMS for seed in report['acquisition_seeds'] for p in report['adaptation'][arm][str(seed)]]
        text += ['## Recorded costs', '',
                 'Charged steps allocate source-pool generation plus paid common-seed branches plus acquired branches to each comparison. Budgets are cumulative, and shared costs recur in rows; '
                 'summing this table would double-count experience. These are comparison allocations, not a lifetime simulator total. Candidate queries and optimizer updates are separate costs.', '',
                 table(['Arm', 'Seed', 'Additional branches', 'Source steps', 'Seed steps', 'Acquired steps', 'Charged steps', 'Optimizer updates', 'Candidate queries', 'Training seconds'], costs), '',
                 f"Separately recorded evaluation-bank construction: {number(report['evaluation_steps'])} simulator steps.", '']
    text += ['## Scope and provenance', '',
             'FSA is unsafe accepted / all accepted. Zero acceptance and unresolved accepted futures do not acquire a zero estimate. '
             'Nonfinite values display as undefined and become null in presentation JSON. The saved scientific gate is unchanged. '
             'Input/source hashes and completed workflow output identities are in `manifest.json`. No model, simulator, network, new bootstrap or final-outcome selection was used. '
             'This presentation validates saved report consistency and identities; it does not independently reproduce the underlying simulator or model measurements.', '']
    return '\n'.join(text)


def plot_gate(gate, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rules = list(gate['evidence'])
    fig, axes = plt.subplots(1, len(rules), figsize=(6 * len(rules), 4), squeeze=False, constrained_layout=True)
    for ax, rule in zip(axes[0], rules, strict=True):
        e, t = gate['evidence'][rule], gate['thresholds']
        counts = [e['n_unsafe'], e['n_false_safe'], e['n_imagination']]
        limits = [t['min_unsafe'], t['min_false_safe'], t['min_imagination']]
        ax.bar(range(3), counts, color=['#777777', '#0072B2', '#D55E00'])
        ax.scatter(range(3), limits, marker='_', s=450, color='black', label='Required minimum', zorder=4)
        for i, v in enumerate(counts):
            ax.annotate(str(v), (i, v), xytext=(0, 6), textcoords='offset points', ha='center')
        ax.set_ylim(0, max([1, *counts, *limits]) * 1.25)
        ax.set_xticks(range(3), ['Unsafe', 'Imagined\nfalse-safe', 'Imagination\nattributed'])
        ax.set_ylabel('Development branches')
        ax.set_title(f"{rule.title()}: {flag(e['pass'])}\nShare {number(e['imagination_share'])}; required ≥ {number(t['min_share'])}")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, -.08))
    fig.suptitle('Recorded development attribution evidence')
    fig.savefig(output / 'development_gate.png', dpi=180, bbox_inches='tight')
    fig.savefig(output / 'development_gate.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_pairs(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rules = report['active_rules']
    fig, axes = plt.subplots(len(rules), 2, figsize=(14, 3.9 * len(rules)), squeeze=False, constrained_layout=True)
    for ri, rule in enumerate(rules):
        for bi, name in enumerate(BANKS):
            ax, labels = axes[ri, bi], []
            for i, (budget, seed) in enumerate((b, s) for b in report['planned_budgets'] for s in report['acquisition_seeds']):
                entry = report['boundary_vs_random']['by_budget'][str(budget)]['by_seed'][str(seed)]
                item = entry[name][rule]
                paired_cells(item)
                suffix = ' | undefined' if not finite(item['point']) else ' | CI undefined' if not finite(item['lo']) else ''
                labels.append(f"N={budget}, s={seed} | {entry['charged_steps']:,} steps{suffix}")
                if finite(item['point']):
                    if finite(item['lo']):
                        ax.hlines(i, item['lo'] * 100, item['hi'] * 100, color='#0072B2', linewidth=2)
                    ax.scatter(item['point'] * 100, i, color='#0072B2', s=30, zorder=3)
            ax.axvline(0, color='black', linestyle=':', linewidth=1)
            ax.set_yticks(range(len(labels)), labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f'{name.title()}: {rule}')
            ax.set_xlabel('Boundary minus random FSA (percentage points)')
            ax.grid(axis='x', alpha=.2)
    fig.suptitle('Saved paired estimates and 95% source-episode bootstrap intervals\nN = additional branches; each row shows allocated charged simulator steps')
    fig.savefig(output / 'paired_fsa.png', dpi=180, bbox_inches='tight')
    fig.savefig(output / 'paired_fsa.pdf', bbox_inches='tight')
    plt.close(fig)


def create_presentation(run_dir, results_dir, bank_dir, output_dir, workflow_state, *, stage_id='prospective_acquisition'):
    run, results, bank, output, state = map(lambda p: Path(p).resolve(), (run_dir, results_dir, bank_dir, output_dir, workflow_state))
    check(not output.exists(), 'Refusing to overwrite presentation output')
    check(output.parent == results.parent, 'Presentation must be a sibling of results')
    check(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in (run, results, bank)), 'Output must not overlap immutable inputs')
    report, gate, manifest, stage, hashes = load_evidence(run, results, bank, state, stage_id)
    sources = [Path(__file__).resolve(), REPO / 'experiments/scripts/walker_s3_figures.py',
               REPO / 'experiments/helpers/managedWorkflow.py', REPO / 'experiments/scripts/walker_s4_prospective.py']
    source_hashes = {str(p.relative_to(REPO)): file_hash(p) for p in sources}
    content = markdown(report, gate, manifest)
    output.mkdir(exist_ok=False)
    (output / 'README.md').write_text(content)
    plot_gate(gate, output)
    if report['status'] == 'complete':
        plot_pairs(report, output)
    (output / 'presentation.json').write_text(json.dumps(json_safe({'study_status': report['status'],
        'recorded_gate': gate, 'saved_report': report, 'gate_recomputed': False}), indent=2, allow_nan=False) + '\n')
    check(frozen_stage(state, stage_id, (run, results, bank)) == stage, 'Finalized stage changed during presentation')
    check(all(file_hash(Path(p)) == h for p, h in hashes.items()), 'Input evidence changed during presentation')
    check(all(file_hash(REPO / p) == h for p, h in source_hashes.items()), 'Presentation source changed')
    provenance = {'status': 'complete', 'study_status': report['status'], 'run_id': report['run_id'],
        'created_utc': datetime.now(timezone.utc).isoformat(), 'inputs_sha256': hashes, 'source_sha256': source_hashes,
        'workflow_state': str(state), 'workflow_stage': stage_id, 'completed_source_output_hashes': stage['output_hashes'],
        'input_artifacts_unchanged': True, 'gate_recomputed': False, 'bootstrap_recomputed': False,
        'model_calls': 0, 'simulator_steps': 0, 'network_calls': 0,
        'files_sha256': {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'results-dir', 'bank-dir', 'output-dir', 'workflow-state'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage-id', default='prospective_acquisition')
    args = parser.parse_args()
    result = create_presentation(args.run_dir, args.results_dir, args.bank_dir, args.output_dir,
                                 args.workflow_state, stage_id=args.stage_id)
    print(json.dumps({'status': result['status'], 'study_status': result['study_status'], 'output_dir': str(args.output_dir)}))


if __name__ == '__main__':
    main()
