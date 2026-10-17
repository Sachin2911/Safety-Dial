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
    record = {'source_episode': str(args.episode.resolve()), 'policy_queries_attempted': 0,
              'physics_steps': 0, 'metadata': metadata, 'query_seconds': [], 'status': 'running'}
    started = time.monotonic()
    actions = []
    try:
        for episode in range(2):
            observation['study_episode_id'] = str(args.output.resolve()) + f'/repeat-{episode}'
            for _ in range(2):
                record['policy_queries_attempted'] += 1
                query_started = time.monotonic()
                reply = client.infer(observation)
                record['query_seconds'].append(time.monotonic() - query_started)
                action = np.asarray(reply['actions'], dtype=np.float32)
                if not np.isfinite(action).all():
                    raise ValueError('Nonfinite policy output')
                actions.append(action)
        record['corresponding_queries_equal'] = [np.array_equal(actions[i], actions[i + 2])
                                                 for i in range(2)]
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
