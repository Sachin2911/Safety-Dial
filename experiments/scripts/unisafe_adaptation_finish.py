"""Advance a live adaptation study through declared validation and reserved evaluation."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_source import ROOT, save_implementation, sha256


def process_identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    save_implementation(args.output)
    started = time.perf_counter()
    identity = process_identity(args.wait_pid)
    record = {
        'status': 'waiting_for_validation', 'prerequisite_pid': args.wait_pid,
        'prerequisite_process_start_ticks': identity,
        'protocol_sha256': sha256(args.protocol), 'script_sha256': sha256(__file__),
        'automatic_retries': 0, 'stages': [],
    }
    status_file = args.output / 'handoff.json'

    def save():
        temporary = status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(status_file)

    def run(stage, script, *options):
        record['status'] = stage
        save()
        print('Starting', stage, flush=True)
        begin = time.perf_counter()
        with (args.output / f'{stage}.log').open('w') as log:
            result = subprocess.run(
                [sys.executable, '-u', str(ROOT / 'experiments/scripts' / script), *map(str, options)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        record['stages'].append({
            'stage': stage, 'returncode': result.returncode,
            'elapsed_seconds': time.perf_counter() - begin,
        })
        if result.returncode:
            record['status'] = 'failed'
            save()
            raise RuntimeError(f'{stage} failed; retained partial artifacts need inspection')
        save()
        print('Finished', stage, flush=True)

    save()
    try:
        prerequisite = args.validation / 'summary.json'
        while identity is not None and process_identity(args.wait_pid) == identity:
            print('Waiting for the recorded live search/validation process', args.wait_pid, flush=True)
            time.sleep(30)
        if not prerequisite.exists():
            raise RuntimeError('Prerequisite process exited without completed validation')
        validation = json.loads(prerequisite.read_text())
        assert validation['status'] == 'complete'
        assert validation['reserved_test_observations_used'] == 0
        assert sha256(args.protocol) == record['protocol_sha256']
        search = json.loads((args.search / 'summary.json').read_text())
        assert search['status'] == 'complete'
        protocol = json.loads(args.protocol.read_text())
        expected_episodes = 3 * len(protocol['search_seeds']) * protocol['generations'] * protocol['population'] * len(protocol['training_seeds'])
        assert search['episodes'] == expected_episodes
        assert {(r['method'], r['search_seed'], r['generation']) for r in search['runs']} == {
            (m, s, g) for m in ('evolution', 'random', 'policy_gradient')
            for s in protocol['search_seeds'] for g in range(protocol['generations'])
        }
        assert all(r['episodes'] == protocol['population'] * len(protocol['training_seeds']) for r in search['runs'])
        record['wait_elapsed_seconds'] = time.perf_counter() - started
        record['validation_sha256'] = sha256(prerequisite)
        record['selected'] = validation['selected']
        save()
        run('training_audit', 'unisafe_adaptation_analyze.py',
            *[r['directory'] for r in search['runs']], '--audit-only',
            '--output', args.output / 'training-audit')
        run('validation_audit', 'unisafe_adaptation_analyze.py', args.validation / 'rollouts',
            '--baseline', 'baseline', '--output', args.output / 'validation-audit')
        run('freeze_test', 'unisafe_adaptation_test_plan.py', '--protocol', args.protocol,
            '--validation', args.validation, '--output', args.output / 'test-plan.json')
        run('reserved_tests', 'unisafe_adaptation_batch.py', '--plan', args.output / 'test-plan.json',
            '--output', args.output / 'test')
        run('test_audit', 'unisafe_adaptation_evaluate.py', '--batch', args.output / 'test',
            '--output', args.output / 'analysis')
        analysis = json.loads((args.output / 'analysis/summary.json').read_text())
        assert analysis['prefix_audit_all_pass'], 'Investigate mismatched prefixes before illustrating effects'
        batch = json.loads((args.output / 'test/summary.json').read_text())
        cases = {}
        for job in batch['jobs']:
            directory = Path(job['directory'])
            episode = json.loads((directory / 'summary.json').read_text())['episodes'][0]
            cases[(episode['job']['name'], episode['seed'])] = (directory / 'episode-0000', episode)
        evolution = sorted(s for method, s in cases if method == 'evolution')
        examples = {
            'improvement': next((s for s in evolution if cases['evolution', s][1]['safe_success'] and not cases['baseline', s][1]['safe_success']), None),
            'failure': next((s for s in evolution if cases['evolution', s][1]['physical_failure'] or cases['evolution', s][1]['boundary_exit']), None),
        }
        record['video_selection'] = examples
        for label, seed in examples.items():
            if seed is not None:
                run(f'video_{label}', 'unisafe_adaptation_video.py',
                    '--baseline', cases['baseline', seed][0], '--candidate', cases['evolution', seed][0],
                    '--candidate-label', 'Selected evolutionary adapter',
                    '--selection-description', f'First held-out seed in numerical order satisfying {label}; not a representative-frequency claim.',
                    '--output', args.output / f'video-{label}')
        record['status'] = 'complete_pending_research_review'
        record['elapsed_seconds_including_wait'] = time.perf_counter() - started
        save()
        print('Reserved evaluation and audits complete; research interpretation remains required.', flush=True)
    except BaseException as error:
        record.update(status='failed', error=repr(error), elapsed_seconds_including_wait=time.perf_counter() - started)
        save()
        raise


if __name__ == '__main__':
    main()
