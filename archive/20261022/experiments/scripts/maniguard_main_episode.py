"""Execute one exact parent-claimed main job; unsupported enforcement fails closed."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_main_episode import execute_episode, prepare_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    command = request.get('command')
    if (not isinstance(command, list) or len(command) != 4
            or Path(command[0]).resolve() != Path(sys.executable).resolve()
            or Path(command[1]).resolve() != Path(__file__).resolve()
            or command[2] != '--request' or Path(command[3]).resolve() != args.request.resolve()):
        raise ValueError('CLI invocation differs from the signed parent command')
    execute_episode(prepare_episode(request))


if __name__ == '__main__':
    main()
