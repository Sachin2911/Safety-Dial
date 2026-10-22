"""Write a read-only pilot comparison and optionally export every selected actor."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from helpers.maniguard_pilot_report import build_report, export_selected


def markdown(report):
    status = 'complete' if report['complete'] else 'incomplete'
    lines = ['# ManiGuard shared-controller pilot', '',
             f"Pilot coverage: **{status}**. Saved-summary consistency: **{report['valid']}**.", '',
             ('This report reconstructs saved validation selection. It does not independently '
              're-audit simulator artifacts or authorize the main study.'), '',
             f"Screening decision: `{report['screening']['decision']}`.", '',
             '| Method | Optimizer seed | Family | Safe requested completion | Requested completion | Violations | Mean utility |',
             '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for run in report['all_training_runs'].values():
        selected = run['selected']
        if selected is None:
            lines.append(f"| {run['method']} | {run['optimizer_seed']} | Selection incomplete | unavailable | unavailable | unavailable | unavailable |")
            continue
        for family, row in selected['metrics']['per_family'].items():
            counts, n = row['counts'], row['episodes']
            lines.append(f"| {run['method']} | {run['optimizer_seed']} | {family} | "
                         f"{counts['S_req']}/{n} | {counts['C_req']}/{n} | {counts['V']}/{n} | {row['U']:.3f} |")
    lines += ['', 'Reference validation records are reused across seed comparisons and counted once.', '',
              '| Reference | Family | Safe requested completion | Violations |',
              '| --- | --- | ---: | ---: |']
    for name, reference in report['references'].items():
        for family, row in reference['metrics']['per_family'].items():
            counts, n = row['counts'], row['episodes']
            lines.append(f"| {name} | {family} | {counts['S_req']}/{n} | {counts['V']}/{n} |")
    lines += ['', '## Interpretation limits', '']
    lines += ['- ' + value for value in report['limitations']]
    if report['invalid_reasons']:
        lines += ['', '## Inconsistent or missing evidence', '']
        lines += ['- ' + value for value in report['invalid_reasons']]
    lines += ['', ('The JSON report retains all comparisons, regressions, missing jobs, costs and '
                   'source-state hashes. Episode-time sums are not allocated GPU wall time.'), '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New report directory')
    parser.add_argument('--completion-review', type=Path)
    parser.add_argument('--mechanism-review', type=Path)
    parser.add_argument('--export-models', action='store_true', help='Export all six actors only after complete consistent selection')
    args = parser.parse_args()
    review = lambda path: json.loads(path.read_text()) if path is not None else None
    state_bytes = args.state.read_bytes()
    state = json.loads(state_bytes)
    report = build_report(state, completion_review=review(args.completion_review),
                          mechanism_review=review(args.mechanism_review))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output / 'report.md').write_text(markdown(report))
    paths = [Path(__file__), ROOT / 'experiments/helpers/maniguard_pilot_report.py']
    paths += [path for path in (args.completion_review, args.mechanism_review) if path is not None]
    sources = [{'path': str(args.state.resolve()), 'sha256': hashlib.sha256(state_bytes).hexdigest()}]
    sources += [{'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    (args.output / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    if args.export_models:
        export_selected(state, args.output / 'selected-actors')
    print(json.dumps({'report': str((args.output / 'report.md').resolve()),
                      'complete': report['complete'], 'valid': report['valid'],
                      'screening': report['screening']['decision'],
                      'selected_actors_exported': args.export_models}))


if __name__ == '__main__':
    main()
