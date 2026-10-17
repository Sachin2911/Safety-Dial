"""Check real-policy episode isolation from a saved observation; use an idle server."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_client import StudyClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--interleaved', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    job = json.loads((args.episode / 'job.json').read_text())
    result = json.loads((args.episode / 'episode/results.jsonl').read_text().strip())
    initial = next(json.loads(line) for line in (args.episode / 'trace.jsonl').read_text().splitlines()
                   if json.loads(line)['type'] == 'initial')
    original_chunk = next(json.loads(line)['actions']
                          for line in (args.episode / 'trace.jsonl').read_text().splitlines()
                          if json.loads(line)['type'] == 'chunk')
    with np.load(args.episode / 'initial_observation.npz') as saved:
        observation = {'observation/image_left': saved['overview'],
                       'observation/wrist_image': saved['wrist'],
                       'observation/state': saved['state'], 'prompt': initial['scene']['prompt'],
                       'episode_seed': result['episode_seed']}
    client = StudyClient(host='127.0.0.1', port=18000)
    metadata = client.get_server_metadata()
    if metadata.get('checkpoint_revision') != job['policy']['revision']:
        raise ValueError('Server checkpoint does not match the saved episode')
    if args.interleaved and metadata.get('rng_isolation') != 'per_episode':
        raise ValueError('Interleaving requires independent episode RNG streams')
    record = {'source_episode': str(args.episode.resolve()), 'policy_queries_attempted': 0,
              'physics_steps': 0, 'metadata': metadata, 'query_seconds': [], 'status': 'running',
              'interleaved': args.interleaved}
    started = time.monotonic()
    actions = []
    try:
        order = [0, 0, 1, 2, 1, 2] if args.interleaved else [0, 0, 1, 1]
        clients = [client] + [StudyClient(host='127.0.0.1', port=18000)
                              for _ in range(max(order))]
        try:
            for episode in order:
                observation['study_episode_id'] = str(args.output.resolve()) + f'/repeat-{episode}'
                record['policy_queries_attempted'] += 1
                query_started = time.monotonic()
                reply = clients[episode].infer(observation)
                record['query_seconds'].append(time.monotonic() - query_started)
                action = np.asarray(reply['actions'], dtype=np.float32)
                if not np.isfinite(action).all():
                    raise ValueError('Nonfinite policy output')
                actions.append(action)
        finally:
            for connection in clients:
                connection._ws.close()
        pairs = [(0, 2), (0, 3), (1, 4), (1, 5)] if args.interleaved else [(0, 2), (1, 3)]
        record['query_order'] = order
        record['corresponding_queries_equal'] = [np.array_equal(actions[a], actions[b])
                                                 for a, b in pairs]
        record['first_matches_original_episode'] = np.array_equal(
            actions[0], np.asarray(original_chunk, dtype=np.float32),
        )
        record['action_hashes'] = [hashlib.sha256(a.tobytes()).hexdigest() for a in actions]
        np.savez_compressed(args.output / 'actions.npz', **{f'query_{i}': a
                                                          for i, a in enumerate(actions)})
        if not all(record['corresponding_queries_equal']) or not record['first_matches_original_episode']:
            raise ValueError('Real-policy same-seed replay differs; inspect recorded actions')
        record['status'] = 'passed'
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'check.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
