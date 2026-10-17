"""Run a predeclared single-family episode plan in fresh processes with bounded concurrency."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_adaptation import AdapterConfig
from helpers.maniguard_jobs import run_jobs
from helpers.maniguard_results import read_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=3600)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    jobs = plan['jobs']
    if not jobs or len({job['family'] for job in jobs}) != 1:
        raise ValueError('A batch must contain one policy family and at least one job')
    if args.timeout_seconds <= 0:
        raise ValueError('Timeout must be positive')
    splits = json.loads((ROOT / 'configs/maniguard/splits.json').read_text())['families']
    split = plan['split']
    if split not in ('engineering', 'train', 'validation', 'test'):
        raise ValueError('Declare the scene split for this batch')
    for job in jobs:
        if job.get('max_steps') is not None and split != 'engineering':
            raise ValueError('Shortened horizons are restricted to engineering diagnostics')
        AdapterConfig(**job['adapter'])
        if job['scene'].split('/')[0] not in splits[job['family']][split]:
            raise ValueError(f"Scene {job['scene']} is outside the declared {split} split")
    metadata = None
    if args.workers > 1:
        from helpers.maniguard_client import StudyClient
        client = StudyClient(host='127.0.0.1', port=18000)
        try:
            metadata = client.get_server_metadata()
        finally:
            client._ws.close()
        if metadata.get('rng_isolation') != 'per_episode':
            raise ValueError('Concurrent episodes require isolated server sampling streams')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    sources = [Path(__file__), ROOT / 'experiments/helpers/maniguard_jobs.py']
    (output / 'batch.json').write_text(json.dumps({
        'workers': args.workers, 'server_metadata': metadata,
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
    }, indent=2) + '\n')
    for source in sources:
        (output / source.name).write_bytes(source.read_bytes())
    processes = output / 'processes.jsonl'
    evidence = output / 'evidence.jsonl'
    environment = dict(os.environ, OMNI_KIT_ALLOW_ROOT='1', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4')

    def execute(index, job):
        job_output = output / f'episode-{index:04d}'
        adapter_path = output / f'adapter-{index:04d}.json'
        adapter_path.write_text(json.dumps(job['adapter'], indent=2) + '\n')
        command = [sys.executable, '-u', str(ROOT / 'experiments/scripts/maniguard_episode.py'),
                   '--family', job['family'], '--scene', job['scene'], '--seed', str(job['seed']),
                   '--adapter', str(adapter_path), '--output', str(job_output)]
        if job.get('max_steps') is not None:
            command += ['--max-steps', str(job['max_steps'])]
        started = time.monotonic()
        process = {'index': index, 'command': command, 'output': str(job_output)}
        print(f"Starting {index + 1}/{len(jobs)}: {job['scene']} {job['adapter']}", flush=True)
        with (output / f'episode-{index:04d}.log').open('x') as log:
            try:
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=args.timeout_seconds,
                                           check=False)
                process['returncode'] = completed.returncode
            except OSError as exc:
                process['returncode'] = None
                process['error'] = str(exc)
            except subprocess.TimeoutExpired:
                process['returncode'] = None
                process['error'] = 'timeout; subprocess killed and waited for'
        process['elapsed_seconds'] = time.monotonic() - started
        if process['returncode'] != 0:
            return process, None
        try:
            row = read_episode(job_output)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            process['validation_error'] = {'type': type(exc).__name__, 'message': str(exc)}
            return process, None
        row['label'] = job.get('label', job['adapter']['kind'])
        return process, row

    def record(process, row):
        with processes.open('a') as stream:
            stream.write(json.dumps(process) + '\n')
        if row is not None:
            with evidence.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(f"Finished: safe_completion={row['safe_completion']}, "
                  f"violation={row['violation']}, steps={row['steps']}", flush=True)
        else:
            print(f"Infrastructure failure: {process}", flush=True)

    if not run_jobs(jobs, execute, record, args.workers):
        raise RuntimeError('Batch failed; in-flight jobs drained and process records preserved')
    (output / 'complete.json').write_text(json.dumps({'episodes': len(jobs)}) + '\n')


if __name__ == '__main__':
    main()
