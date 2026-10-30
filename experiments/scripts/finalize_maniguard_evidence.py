"""Preserve a terminal evolution run without deleting its rented host.

This command performs file orchestration only. It never starts or resumes episodes,
changes the experiment deadline, or calls a provider deletion endpoint. Its receipt
remains outside the sealed project so verification cannot invalidate the seal.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from preserve_maniguard_evolution import atomic, digest, eligible, stage, upload_and_verify
from seal_maniguard_evolution import seal, verify_local


def read(path):
    return json.loads(Path(path).read_text())


def live(record, marker):
    proc = Path('/proc') / str(record['pid'])
    try:
        stat = (proc / 'stat').read_text().split()
        if stat[2] == 'Z' or stat[21] != str(record['start_ticks']):
            return False
        command = (proc / 'cmdline').read_bytes()
        if not command:
            # Linux clears mm/cmdline while exiting, before publishing zombie state.
            return False
        if marker.encode() not in command:
            raise ValueError('Live process command does not match its recorded identity')
        return True
    except FileNotFoundError:
        return False


def require_terminal(project, run_name, trainer):
    if live(trainer, 'run_maniguard_evolution.py'):
        return False
    state = read(project / run_name / 'status.json')
    if state['status'] in ('ready', 'running'):
        raise ValueError('Trainer disappeared without a terminal status; audit recovery required')
    for path in (project / run_name / 'episodes').glob('*/process.json'):
        if live(read(path), 'benchmark_adapter.py'):
            return False
    return True


def stop_mirror(record, state_path, timeout=120):
    """Freeze at a verified boundary, avoiding interruption during an upload."""
    end = time.monotonic() + timeout
    while live(record, 'mirror_maniguard_v3_evidence.py'):
        if time.monotonic() >= end:
            raise TimeoutError('Mirror did not reach a verified boundary; left running')
        os.kill(record['pid'], signal.SIGSTOP)
        try:
            # SIGSTOP delivery is asynchronous. Confirm the process has actually stopped.
            for _ in range(100):
                stat = (Path('/proc') / str(record['pid']) / 'stat').read_text().split()
                if stat[2] in ('T', 't'):
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError('Could not confirm stopped mirror')
            if read(state_path).get('pending') is None:
                os.kill(record['pid'], signal.SIGTERM)
                break
        finally:
            if live(record, 'mirror_maniguard_v3_evidence.py'):
                os.kill(record['pid'], signal.SIGCONT)
        time.sleep(1)
    wait_retired(record, 'mirror_maniguard_v3_evidence.py')


def wait_retired(record, marker, timeout=120):
    end = time.monotonic() + timeout
    while live(record, marker):
        if time.monotonic() >= end:
            raise TimeoutError('Exact process did not retire; refusing to seal changing evidence')
        time.sleep(1)


def audit(project, run_name, output, allow_incomplete):
    command = [sys.executable, str(project / 'experiments/scripts/analyze_maniguard_evolution.py'),
               str(project / run_name), '--out', str(output)]
    if allow_incomplete:
        command.append('--allow-incomplete')
    subprocess.run(command, check=True)
    result = read(output)
    if not result['final']:
        raise ValueError('Terminal audit did not produce a final record')
    return result


def mirror_records(project, base):
    destination = project / 'supervision/private-mirror-closeout'
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(base / 'state.json', destination / 'state.json')
    shutil.copytree(base / 'receipts', destination / 'receipts')
    for path in sorted((base / 'snapshots').glob('*/index.json')):
        target = destination / 'snapshot-indexes' / path.parent.name / 'index.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def check_secrets(project):
    secrets = [os.environ[k].encode() for k in ('HF_TOKEN', 'GITHUB_TOKEN', 'VAST_API_KEY')
               if os.environ.get(k)]
    forbidden = {'auth.json', 'credentials.json', 'omnigibson.key', 'handoff.env'}
    for path in project.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink in evidence project')
        if not path.is_file() or not eligible(path.relative_to(project)):
            continue
        if path.name in forbidden:
            raise ValueError('Credential-like file in evidence project')
        # Scan binary files as well, without loading large rollout archives into memory.
        tail = b''
        overlap = max(map(len, secrets), default=1) - 1
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                block = tail + block
                if any(secret in block for secret in secrets):
                    raise ValueError('Known credential in evidence project')
                tail = block[-overlap:] if overlap else b''


def finalize(project, run_name, mirror_base, output, trainer, mirror, server):
    if not require_terminal(project, run_name, trainer):
        raise ValueError('Experiment processes still live')
    output.mkdir(parents=True, exist_ok=True)
    seal_path = project / 'remote-seal.json'
    identity_path = output / 'seal-identity.json'
    release = output / 'release'
    if not identity_path.exists():
        stop_mirror(mirror, mirror_base / 'state.json')
        if read(mirror_base / 'state.json').get('pending') is not None:
            raise ValueError('Mirror has unverified pending evidence')
        if live(server, 'openpi_native'):
            os.kill(server['pid'], signal.SIGTERM)
        wait_retired(server, 'openpi_native')
        if not (project / 'supervision/private-mirror-closeout').exists():
            mirror_records(project, mirror_base)
        state = read(project / run_name / 'status.json')
        incomplete = state['status'] != 'completed'
        audit(project, run_name, project / 'supervision/final-audit.json', incomplete)
        predecessor = read(project / run_name / 'config.json').get('engineering_predecessor')
        if predecessor:
            predecessor = Path(predecessor['run'])
            if predecessor.resolve().parent != project:
                raise ValueError('Predecessor outside the authorized evidence project')
            audit(project, predecessor.name, project / 'supervision/predecessor-final-audit.json',
                  True)
        check_secrets(project)
        sealed = seal(project, run_name, seal_path, allow_incomplete=incomplete)
        atomic(identity_path, sealed)
    sealed = read(identity_path)
    if digest(seal_path) != sealed['remote_seal_sha256']:
        raise ValueError('Recorded seal changed; refusing to replace evidence')
    verify_local(project, seal_path, sealed['remote_seal_sha256'])
    if not release.exists():
        incomplete = not read(seal_path)['complete_schedule']
        stage(project, release, final=not incomplete, terminal=incomplete, run_name=run_name)
    name = output.name
    receipt = upload_and_verify(release, output / 'download', name, output / 'private-receipt.json')
    downloaded = output / 'download' / receipt['prefix']
    # The release adds an index and observation. Rebuild precisely the sealed inventory
    # from downloaded bytes, then apply the strict whole-tree seal verifier.
    independent = output / 'sealed-download'
    if independent.exists():
        shutil.rmtree(independent)
    independent.mkdir()
    manifest = read(downloaded / 'remote-seal.json')
    for relative in [*manifest['files'], 'remote-seal.json']:
        source = downloaded / relative
        target = independent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    verified = verify_local(independent, independent / 'remote-seal.json',
                            sealed['remote_seal_sha256'])
    result = {'status': 'preserved_teardown_pending', 'run': run_name,
              'complete_schedule': receipt['complete_schedule'], 'private': receipt,
              'independent_seal_verification': verified,
              'provider_absence_confirmed': False, 'deletion_requested': False,
              'deadline_guard_unchanged': True,
              'verified_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic(output / 'final-preservation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--mirror-base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trainer-record', type=Path, required=True)
    parser.add_argument('--mirror-record', type=Path, required=True)
    parser.add_argument('--server-record', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    if project == output or project in output.parents:
        raise ValueError('Output must be outside the evidence project')
    if args.run != Path(args.run).name or args.run in ('.', '..'):
        raise ValueError('Run must be one directory name')
    output.parent.mkdir(parents=True, exist_ok=True)
    # A separate lock avoids either duplicate finalizers or changing the sealed tree.
    with (output.parent / (output.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        trainer, mirror, server = [read(p) for p in
                                  (args.trainer_record, args.mirror_record, args.server_record)]
        while not require_terminal(project, args.run, trainer):
            if not args.wait:
                raise SystemExit('Trainer or worker still live; no preservation mutation performed')
            time.sleep(30)
        if args.wait and read(project / args.run / 'status.json')['status'] not in (
                'completed', 'budget_exhausted'):
            raise SystemExit('Technical failure requires diagnosis before terminal preservation')
        result = finalize(project, args.run, args.mirror_base.resolve(), output,
                          trainer, mirror, server)
        print(json.dumps({'status': result['status'],
                          'complete_schedule': result['complete_schedule']}), flush=True)


if __name__ == '__main__':
    main()
