"""Run a predeclared single-family episode plan in fresh sequential processes."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_adaptation import AdapterConfig
from helpers.maniguard_results import read_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=3600)
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
        AdapterConfig(**job['adapter'])
        if job['scene'].split('/')[0] not in splits[job['family']][split]:
            raise ValueError(f"Scene {job['scene']} is outside the declared {split} split")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    processes = output / 'processes.jsonl'
    evidence = output / 'evidence.jsonl'
    environment = dict(os.environ, OMNI_KIT_ALLOW_ROOT='1', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4')
    for index, job in enumerate(jobs):
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
            except subprocess.TimeoutExpired:
                process['returncode'] = None
                process['error'] = 'timeout; subprocess killed and waited for'
        process['elapsed_seconds'] = time.monotonic() - started
        with processes.open('a') as stream:
            stream.write(json.dumps(process) + '\n')
        if process['returncode'] != 0:
            raise RuntimeError(f'Infrastructure failure; preserved {job_output}. See process record.')
        row = read_episode(job_output)
        row['label'] = job.get('label', job['adapter']['kind'])
        with evidence.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(f"Finished: safe_completion={row['safe_completion']}, "
              f"violation={row['violation']}, steps={row['steps']}", flush=True)
    (output / 'complete.json').write_text(json.dumps({'episodes': len(jobs)}) + '\n')


if __name__ == '__main__':
    main()
