"""Compare original and interleaved wrappers on one loaded frozen model, without physics."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'third_party/ManiGuard'))
os.environ.setdefault('MANIGUARD_SKIP_OMNIGIBSON_PATCH', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

from helpers.maniguard_parallel_policy import IsolatedEpisodePolicy
from helpers.maniguard_policy import EpisodeSeededPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    job = json.loads((args.episode / 'job.json').read_text())
    result = json.loads((args.episode / 'episode/results.jsonl').read_text().strip())
    rows = [json.loads(line) for line in (args.episode / 'trace.jsonl').read_text().splitlines()]
    initial = next(r for r in rows if r['type'] == 'initial')
    original = np.asarray(next(r['actions'] for r in rows if r['type'] == 'chunk'), dtype=np.float32)
    with np.load(args.episode / 'initial_observation.npz') as saved:
        observation = {'observation/image_left': saved['overview'],
                       'observation/wrist_image': saved['wrist'],
                       'observation/state': saved['state'], 'prompt': initial['scene']['prompt'],
                       'episode_seed': result['episode_seed']}
    spec = json.loads((ROOT / 'configs/maniguard/sources.json').read_text())['policies'][job['family']]
    folder = ROOT / 'data/maniguard/policies' / job['family']
    provenance = json.loads((folder / 'download-provenance.json').read_text())
    if provenance['revision'] != spec['revision'] or provenance['repo'] != spec['repo']:
        raise ValueError('Policy provenance does not match the study pin')
    import jax
    from maniguard.openpi_sft import train_configs
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    started = time.monotonic()
    record = {'status': 'running', 'physics_steps': 0, 'policy_queries_attempted': 0,
              'query_seconds': [], 'episode_seed': result['episode_seed'], 'source_episode': str(args.episode.resolve()),
              'checkpoint': spec, 'source_hashes': {}}
    for path in [Path(__file__), ROOT / 'experiments/helpers/maniguard_policy.py',
                 ROOT / 'experiments/helpers/maniguard_parallel_policy.py']:
        raw = path.read_bytes()
        (args.output / path.name).write_bytes(raw)
        record['source_hashes'][str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    try:
        train_configs.register()
        policy = policy_config.create_trained_policy(train_config.get_config(spec['config']),
                                                      folder / spec['step'])
        serial = EpisodeSeededPolicy(policy, jax.random.key)
        isolated = IsolatedEpisodePolicy(policy, jax.random.key)
        actions = []
        for wrapper, episode in [(serial, 'a'), (serial, 'a'), (isolated, 'b'),
                                 (isolated, 'c'), (isolated, 'b'), (isolated, 'c')]:
            record['policy_queries_attempted'] += 1
            query_started = time.monotonic()
            reply = wrapper.infer(observation | {'study_episode_id': episode})
            record['query_seconds'].append(time.monotonic() - query_started)
            actions.append(np.asarray(reply['actions'], dtype=np.float32))
        np.savez_compressed(args.output / 'actions.npz', **{f'query_{i}': a
                                                          for i, a in enumerate(actions)})
        record['wrapper_queries_equal'] = [np.array_equal(actions[a], actions[b])
                                            for a, b in [(0, 2), (0, 3), (1, 4), (1, 5)]]
        record['original_wrapper_matches_old_process'] = np.array_equal(actions[0], original)
        record['old_process_max_arm_difference'] = float(np.max(np.abs(actions[0][:, :7] - original[:, :7])))
        record['old_process_max_difference'] = float(np.max(np.abs(actions[0] - original)))
        if not all(record['wrapper_queries_equal']):
            raise ValueError('Wrappers differ on the same loaded model')
        record['status'] = 'passed'
    except (ValueError, RuntimeError, OSError) as exc:
        record['status'] = 'failed'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'check.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
