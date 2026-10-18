"""Void zero-interaction setup failures under the approved pilot rule, then restore state.

Approved by Sachin on 18 October 2026: an attempt that fails before simulator
initialization and before any policy query may be voided and rerun with full
disclosure. Any failure after interaction begins still blocks and needs a decision.

Every attempt in the named deployment must prove zero interaction from its own trace
and ledger. Outputs, attempt records and the blocked coordinator are moved under
``void-attempts/<deployment>/``, never deleted, and the pinned pre-failure coordinator
replaces the blocked one only when the blocked ledger extends it by exactly those jobs.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_phase2_coordinator import PilotCoordinator
from helpers.maniguard_phase2_services import active_episode_processes, atomic_json

ALLOWED_TRACE = {'job', 'policy_metadata'}
INTERACTION_COUNTERS = ('policy_queries', 'physics_steps', 'env_step_attempts', 'env_steps_returned',
                        'policy_control_steps', 'encoder_image_calls')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interaction_evidence(output):
    """Return a zero-interaction summary, or raise if any interaction is recorded."""
    output = Path(output)
    trace = output / 'trace.jsonl'
    types = sorted({json.loads(line)['type'] for line in trace.read_text().splitlines()
                    if line.strip()}) if trace.exists() else []
    if not set(types) <= ALLOWED_TRACE:
        raise ValueError(f'{output}: trace records interaction ({types})')
    ledger = output / 'ledger.json'
    counters = {}
    if ledger.exists():
        values = json.loads(ledger.read_text())
        counters = {key: values.get(key) for key in INTERACTION_COUNTERS}
        if any(value not in (0, None) for value in counters.values()):
            raise ValueError(f'{output}: ledger records interaction ({counters})')
        counters['phase'] = values.get('phase')
    if (output / 'initial_observation.npz').exists():
        raise ValueError(f'{output}: an initial observation was recorded')
    return {'trace_types': types, 'ledger': counters,
            'files': sorted(path.name for path in output.iterdir())}


def void(run, deployment, restore, expected_content, cause, approval):
    run = Path(run).resolve()
    folder = run / 'deployments' / deployment
    destination = run / 'void-attempts' / deployment
    if (destination / 'void.json').exists():
        raise ValueError('This deployment was already voided')
    if active_episode_processes():
        raise RuntimeError('Episodes are running; stop the launcher and drain first')
    launch = json.loads((folder / 'launch.json').read_text())
    if launch.get('finished_at') is None:
        raise RuntimeError('The deployment launcher has not finished')
    restored = json.loads(Path(restore).read_text())
    if restored['sha256'] != expected_content:
        raise ValueError('Restore source differs from its expected content hash')
    blocked_path = run / 'coordinator.json'
    blocked = json.loads(blocked_path.read_text())
    if blocked['ledger'][:len(restored['ledger'])] != restored['ledger']:
        raise ValueError('Blocked ledger does not extend the restore point')
    if blocked['claims']:
        raise ValueError('Blocked state still has in-flight claims')
    attempts = sorted(folder.glob('attempts/**/*.json'))
    records = [json.loads(path.read_text()) for path in attempts]
    jobs = [record['job']['job_id'] for record in records]
    added = [entry['job_id'] for entry in blocked['ledger'][len(restored['ledger']):]]
    if sorted(jobs) != sorted(added) or not jobs:
        raise ValueError(f'Deployment attempts {jobs} differ from added ledger entries {added}')
    evidence = {}
    for record in records:
        process = record.get('process') or {}
        if process.get('returncode') == 0:
            raise ValueError('A clean process exit is not a setup failure: ' + record['job']['job_id'])
        evidence[record['job']['job_id']] = interaction_evidence(record['output'])
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {
        'schema': 'maniguard-pilot-void-attempts-v1', 'created_at': time.time(),
        'rule': __doc__.split('\n\n')[1].replace('\n', ' '), 'approval': approval, 'cause': cause,
        'job_ids': jobs, 'deployment': str(folder), 'interaction_evidence': evidence,
        'attempt_records': {str(path.relative_to(run)): sha256(path) for path in attempts},
        'blocked_coordinator': {'file': str((destination / 'coordinator-blocked.json').relative_to(run)),
                                'file_sha256': sha256(blocked_path),
                                'content_sha256': blocked['sha256'],
                                'physical_attempts_recorded': len(blocked['ledger']),
                                'blocked_reasons': blocked['blocked_reasons']},
        'restored_coordinator': {'source': str(restore), 'file_sha256': sha256(restore),
                                 'content_sha256': restored['sha256']},
        'moved_outputs': {},
    }
    shutil.copy2(blocked_path, destination / 'coordinator-blocked.json')
    for record in records:
        source = Path(record['output'])
        target = destination / 'episodes' / record['job']['job_id']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        receipt['moved_outputs'][str(source)] = str(target)
    atomic_json(destination / 'void.json', receipt)
    temporary = run / 'coordinator.json.restore.tmp'
    shutil.copy2(restore, temporary)
    temporary.replace(blocked_path)
    coordinator = PilotCoordinator.load(blocked_path)
    return {'restored_content_sha256': coordinator._last_saved_sha256, 'report': coordinator.report(),
            'voided': jobs, 'receipt': str(destination / 'void.json')}


def void_entries(run, deployment, signature, cause, approval):
    """Remove failed ledger entries of one deployment that share an approved failure signature.

    Used when later valid results prevent restoring an earlier file. Their measured
    interactions stay in the receipt and must be added to every reported cost total.
    Refuses unless these entries explain every blocked reason and no claim is open.
    """
    run = Path(run).resolve()
    folder = run / 'deployments' / deployment
    destination = run / 'void-attempts' / deployment
    if (destination / 'void.json').exists():
        raise ValueError('This deployment was already voided')
    if active_episode_processes():
        raise RuntimeError('Episodes are running; stop the launcher and drain first')
    if json.loads((folder / 'launch.json').read_text()).get('finished_at') is None:
        raise RuntimeError('The deployment launcher has not finished')
    state_path = run / 'coordinator.json'
    before_file = sha256(state_path)
    coordinator = PilotCoordinator.load(state_path)
    if coordinator.claims:
        raise ValueError('Open claims must be reconciled first')
    attempts = {}
    for path in sorted(folder.glob('attempts/**/*.json')):
        record = json.loads(path.read_text())
        attempts[record['job']['job_id']] = (path, record)
    voided, kept = [], []
    for entry in coordinator.ledger:
        job_id = entry['job_id']
        record = attempts.get(job_id, (None, None))[1]
        log = attempts[job_id][0].with_suffix('.log') if record else None
        failed = (record is not None and not entry['accepted'] and record.get('valid') is False
                  and entry['result']['output'] == record['output'] and log.exists()
                  and signature in log.read_text(errors='replace'))
        (voided if failed else kept).append(entry)
    if not voided:
        raise ValueError('No failed entry of this deployment carries the approved signature')
    expected = [reason for entry in coordinator.ledger if not entry['accepted']
                for reason in entry.get('binding_errors', [])]
    if coordinator.blocked_reasons != expected:
        raise ValueError('Blocked reasons are not exactly the recorded binding errors')
    remaining = [reason for entry in kept if not entry['accepted']
                 for reason in entry.get('binding_errors', [])]
    if remaining:
        raise ValueError('Other blocking failures remain; they need their own decision')
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(state_path, destination / 'coordinator-blocked.json')
    receipt = {
        'schema': 'maniguard-pilot-void-attempts-v1', 'mode': 'ledger_entries',
        'created_at': time.time(), 'approval': approval, 'cause': cause,
        'failure_signature': signature, 'deployment': str(folder),
        'job_ids': [entry['job_id'] for entry in voided],
        'voided_entries': voided,
        'voided_costs': {key: sum((entry['result'].get('costs') or {}).get(key, 0) for entry in voided)
                         for key in sorted({key for entry in voided
                                            for key in (entry['result'].get('costs') or {})})},
        'attempt_records': {str(attempts[entry['job_id']][0].relative_to(run)):
                            sha256(attempts[entry['job_id']][0]) for entry in voided},
        'blocked_coordinator': {'file': str((destination / 'coordinator-blocked.json').relative_to(run)),
                                'file_sha256': before_file,
                                'content_sha256': coordinator._last_saved_sha256,
                                'physical_attempts_recorded': len(coordinator.ledger),
                                'blocked_reasons': coordinator.blocked_reasons},
        'moved_outputs': {},
    }
    for entry in voided:
        source = Path(entry['result']['output'])
        target = destination / 'episodes' / entry['job_id']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        receipt['moved_outputs'][str(source)] = str(target)
    coordinator.ledger = kept
    coordinator.blocked_reasons = []
    coordinator.revision += 1
    if coordinator.active_jobs is not None and len(coordinator.active_results) == len(coordinator.active_jobs):
        coordinator._complete_stage()
    atomic_json(destination / 'void.json', receipt)
    receipt['restored_content_sha256'] = coordinator.save(state_path)
    atomic_json(destination / 'void.json', receipt)
    return {'restored_content_sha256': receipt['restored_content_sha256'],
            'report': coordinator.report(), 'voided': receipt['job_ids'],
            'voided_costs': receipt['voided_costs'], 'receipt': str(destination / 'void.json')}


def void_lane_entries(run, deployment, cause, approval):
    """Void zero-interaction failures recorded in live lane state under the standing rule."""
    from helpers.maniguard_pilot_lanes import LaneCoordinator

    run = Path(run).resolve()
    folder = run / 'deployments' / deployment
    destination = run / 'void-attempts' / deployment
    if (destination / 'void.json').exists():
        raise ValueError('This deployment already has a manual void receipt')
    if active_episode_processes():
        raise RuntimeError('Episodes are running; stop the launcher and drain first')
    if json.loads((folder / 'launch.json').read_text()).get('finished_at') is None:
        raise RuntimeError('The deployment launcher has not finished')
    state_path = run / 'lanes.json'
    before_file = sha256(state_path)
    lanes = LaneCoordinator.load(state_path)
    base = lanes.base
    if base.claims:
        raise ValueError('Open claims must be reconciled first')
    attempts = {}
    for path in sorted(folder.glob('attempts/**/*.json')):
        record = json.loads(path.read_text())
        attempts[record['job']['job_id']] = (path, record)
    voided, kept, evidence = [], [], {}
    for entry in base.ledger:
        path, record = attempts.get(entry['job_id'], (None, None))
        eligible = (record is not None and not entry['accepted'] and record.get('valid') is False
                    and entry['result']['output'] == record['output']
                    and (record.get('process') or {}).get('returncode') not in (0, None))
        if eligible:
            evidence[entry['job_id']] = interaction_evidence(record['output'])
            voided.append(entry)
        else:
            kept.append(entry)
    if not voided:
        raise ValueError('No zero-interaction failure of this deployment is recorded')
    expected = [reason for entry in base.ledger if not entry['accepted']
                for reason in entry.get('binding_errors', [])]
    if base.blocked_reasons != expected:
        raise ValueError('Blocked reasons are not exactly the recorded binding errors')
    if [reason for entry in kept if not entry['accepted'] for reason in entry.get('binding_errors', [])]:
        raise ValueError('Other blocking failures remain; they need their own decision')
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(state_path, destination / 'lanes-blocked.json')
    receipt = {'schema': 'maniguard-pilot-void-attempts-v1', 'mode': 'lane_ledger_entries',
               'created_at': time.time(), 'rule': __doc__.split('\n\n')[1].replace('\n', ' '),
               'approval': approval, 'cause': cause, 'deployment': str(folder),
               'job_ids': [entry['job_id'] for entry in voided], 'voided_entries': voided,
               'interaction_evidence': evidence, 'voided_costs': {},
               'attempt_records': {str(attempts[e['job_id']][0].relative_to(run)):
                                   sha256(attempts[e['job_id']][0]) for e in voided},
               'blocked_state': {'file': str((destination / 'lanes-blocked.json').relative_to(run)),
                                 'file_sha256': before_file, 'content_sha256': lanes._last_saved_sha256,
                                 'blocked_reasons': base.blocked_reasons},
               'moved_outputs': {}}
    for entry in voided:
        source = Path(entry['result']['output'])
        target = destination / 'episodes' / entry['job_id']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        receipt['moved_outputs'][str(source)] = str(target)
    base.ledger = kept
    base.blocked_reasons = []
    base.revision += 1
    completed = []
    for index, record in lanes.stages.items():
        if record['status'] == 'active' and len(record['active_results']) == len(record['active_jobs']):
            with lanes._stage(index) as frozen:
                frozen._complete_stage()
            completed.append(index)
    receipt['blocks_completed_after_unblocking'] = completed
    atomic_json(destination / 'void.json', receipt)
    receipt['restored_content_sha256'] = lanes.save(state_path)
    atomic_json(destination / 'void.json', receipt)
    return {'restored_content_sha256': receipt['restored_content_sha256'], 'report': lanes.report(),
            'voided': receipt['job_ids'], 'blocks_completed': completed,
            'receipt': str(destination / 'void.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/maniguard/phase2/pilot-v1')
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--restore', type=Path)
    parser.add_argument('--expected-content-sha256')
    parser.add_argument('--signature', help='Void failed ledger entries whose log has this text')
    parser.add_argument('--lanes', action='store_true',
                        help='Void zero-interaction failures recorded in live lane state')
    parser.add_argument('--cause', required=True)
    parser.add_argument('--approval', required=True)
    args = parser.parse_args()
    if args.lanes:
        result = void_lane_entries(args.run, args.deployment, args.cause, args.approval)
    elif args.signature:
        result = void_entries(args.run, args.deployment, args.signature, args.cause, args.approval)
    else:
        result = void(args.run, args.deployment, args.restore, args.expected_content_sha256,
                      args.cause, args.approval)
    report = result.pop('report')
    print(json.dumps(result | {key: report[key] for key in (
        'status', 'accepted_episode_slots', 'physical_attempts_recorded', 'in_flight_claims')},
        indent=2))


if __name__ == '__main__':
    main()
