"""Compare evolution, uniform search and episodic categorical REINFORCE fairly."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_search import terminal_utility, validate_protocol
from helpers.unisafe_search import AdapterSearch


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol, json.loads((ROOT / 'configs/maniguard/splits.json').read_text()))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'protocol.json', protocol)
    implementation = output / 'search_source'
    implementation.mkdir()
    hashes = {}
    for relative in ('experiments/scripts/maniguard_search.py',
                     'experiments/helpers/maniguard_search.py',
                     'experiments/helpers/unisafe_search.py',
                     'experiments/scripts/maniguard_batch.py'):
        raw = (ROOT / relative).read_bytes()
        (implementation / Path(relative).name).write_bytes(raw)
        hashes[relative] = hashlib.sha256(raw).hexdigest()
    write(output / 'source_hashes.json', hashes)
    started = time.monotonic()
    progress = []
    for seed in protocol['search_seeds']:
        optimizers = {method: AdapterSearch(method, protocol['domain'], protocol['population'], seed)
                      for method in protocol['methods']}
        order_rng = np.random.default_rng(seed + 917)
        for generation in range(protocol['generations']):
            for method in order_rng.permutation(protocol['methods']).tolist():
                optimizer = optimizers[method]
                genes = optimizer.ask()
                directory = output / f'{method}-seed{seed}-generation{generation}'
                jobs = []
                for episode in protocol['training_episodes']:
                    for candidate in order_rng.permutation(len(genes)).tolist():
                        jobs.append({'family': protocol['family'], **episode,
                                     'label': f'candidate-{candidate}',
                                     'adapter': protocol['base_adapter'] | optimizer.decode(genes[candidate])})
                plan = {'split': 'train', 'description': 'Candidate evaluations, including elite reevaluations.',
                        'method': method, 'search_seed': seed, 'generation': generation,
                        'genes': genes, 'jobs': jobs}
                plan_path = output / (directory.name + '.plan.json')
                write(plan_path, plan)
                print(f'Starting {directory.name}: {len(jobs)} full episodes', flush=True)
                stage_started = time.monotonic()
                with (output / (directory.name + '.log')).open('x') as log:
                    result = subprocess.run([
                        sys.executable, '-u', str(ROOT / 'experiments/scripts/maniguard_batch.py'),
                        '--plan', str(plan_path), '--output', str(directory),
                    ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    write(output / 'failure.json', {'stage': directory.name,
                          'returncode': result.returncode, 'partial_artifacts': str(directory),
                          'elapsed_seconds': time.monotonic() - started,
                          'accounting': 'Count all preserved process records and episode ledgers, including failed work.'})
                    raise RuntimeError(f'Search stopped on infrastructure failure: {directory}')
                evidence = [json.loads(line) for line in (directory / 'evidence.jsonl').read_text().splitlines()]
                if len(evidence) != len(jobs):
                    raise ValueError('Candidate evidence is incomplete')
                fitness = []
                for candidate in range(len(genes)):
                    matched = [row for row in evidence if row['label'] == f'candidate-{candidate}']
                    if len(matched) != len(protocol['training_episodes']):
                        raise ValueError('Every candidate needs the same number of training episodes')
                    fitness.append(float(np.mean([terminal_utility(row) for row in matched])))
                optimizer.tell(genes, fitness)
                write(output / (directory.name + '.optimizer.json'), optimizer.state())
                totals = {key: sum(row['ledger'][key] for row in evidence) for key in
                          ('physics_steps', 'physics_seconds', 'env_step_attempts', 'env_steps_returned',
                           'policy_queries', 'policy_query_seconds', 'policy_control_steps')}
                progress.append({'method': method, 'search_seed': seed, 'generation': generation,
                                 'genes': genes, 'fitness': fitness, 'episodes': len(evidence),
                                 'counts': totals, 'elapsed_seconds': time.monotonic() - stage_started,
                                 'directory': str(directory)})
                write(output / 'progress.json', progress)
                print(f'Finished {directory.name}: fitness={fitness}', flush=True)
        for method, optimizer in optimizers.items():
            write(output / f'{method}-seed{seed}-finalists.json', {
                'family': protocol['family'], 'scope': 'Training finalists; validation and reserved testing remain.',
                'method': method, 'search_seed': seed,
                'adapters': [protocol['base_adapter'] | params for params in optimizer.finalists()],
                'optimizer': optimizer.state(),
            })
    write(output / 'complete.json', {'stages': progress, 'elapsed_seconds': time.monotonic() - started,
          'pretrained_weight_updates': 0,
          'comparison_scope': 'Equal candidate episode allocations; realized control steps and queries may differ.',
          'learning_scope': 'Categorical REINFORCE learns a distribution over the same compact adapter parameters; it does not fine-tune the VLA.'})


if __name__ == '__main__':
    main()
