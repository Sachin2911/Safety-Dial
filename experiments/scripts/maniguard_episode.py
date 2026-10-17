"""Run one fresh-process, instrumented ManiGuard episode at a pinned configuration."""

import argparse
import json
import os
import runpy
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'third_party/ManiGuard'))

from helpers.maniguard_adaptation import AdapterConfig
from helpers.maniguard_source import prepare_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('jar', 'lid', 'stack'), required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, help='Engineering diagnostic only; record override')
    parser.add_argument('--random-policy', action='store_true', help='Engineering diagnostic only')
    parser.add_argument('--port', type=int, default=18000)
    args = parser.parse_args()
    config = AdapterConfig(**json.loads(args.adapter.read_text()))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    job = {"family": args.family, "scene": args.scene, "seed": args.seed,
           "adapter": asdict(config), "output": str(output), "episode_id": str(output),
           "max_steps_override": args.max_steps, "random_policy": args.random_policy}
    job_path = output / 'job.json'
    job_path.write_text(json.dumps(job, indent=2) + '\n')
    os.environ['SAFETY_DIAL_MANIGUARD_JOB'] = str(job_path)
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    os.environ.setdefault('OMNIGIBSON_HEADLESS', '1')
    os.environ['OMNIGIBSON_DATA_PATH'] = str(ROOT / 'data/maniguard/datasets')
    family = {'jar': 'jar_transport', 'lid': 'lid_transport', 'stack': 'stack_retrieve'}[args.family]
    config_name = 'lid_transport_food_joint' if args.family == 'lid' else family + '_joint'
    runtime = prepare_runtime(output)
    sys.argv = [str(runtime), '--config',
                str(ROOT / 'third_party/ManiGuard/configs/eval' / (config_name + '.yaml')),
                '--benchmark-root', str(ROOT / 'data/maniguard/benchmark' / family),
                '--scenes', args.scene, '--seed', str(args.seed), '--host', '127.0.0.1',
                '--port', str(args.port), '--source-profile', 'legacy',
                '--output-dir', str(output), '--run-name', 'episode',
                '--execute-horizon', str(config.horizon)]
    if args.max_steps is not None:
        sys.argv += ['--max-steps', str(args.max_steps)]
    if args.random_policy:
        sys.argv += ['--random-policy']
    runpy.run_path(str(runtime), run_name='__main__')


if __name__ == '__main__':
    main()
