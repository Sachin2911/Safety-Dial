"""Parallel pilot scheduling without changing optimizer updates or episode allocation.

One parent owns the existing coordinator and persists each claim before a worker
starts. Workers never mutate optimizer state. Multiple families may execute in a
single already-declared dependency block; the coordinator alone advances it.
"""

import math
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy


class AllocationClock:
    """Integrate overlapping GPU rentals, including idle time and migration."""

    def __init__(self, prior, allocations, envelope, *, clock=time.time):
        self.prior, self.envelope, self.clock = deepcopy(prior), deepcopy(envelope), clock
        self.allocations = deepcopy(allocations)
        ids = set()
        if not allocations:
            raise ValueError('Explicit continuing and added GPU allocations are required')
        for item in self.allocations:
            if set(item) != {'instance_id', 'gpus', 'started_at', 'ended_at'}:
                raise ValueError('Unexpected GPU allocation record')
            if item['instance_id'] in ids:
                raise ValueError('Duplicate GPU rental allocation')
            ids.add(item['instance_id'])
            if type(item['gpus']) is not int or item['gpus'] <= 0:
                raise ValueError('GPU allocation count must be a positive integer')
            for key in ['started_at', 'ended_at']:
                if item[key] is not None and (type(item[key]) not in (int, float)
                                              or not math.isfinite(item[key])):
                    raise ValueError('GPU allocation timestamps must be finite')
            if item['started_at'] is None or item['started_at'] > clock():
                raise ValueError('GPU allocation start must already be observed')
            if item['ended_at'] is not None and not item['started_at'] <= item['ended_at'] <= clock():
                raise ValueError('GPU allocation end must be observed after its start')
        # The prior report pays the old allocation up to its timestamp. Require
        # exact coverage there; additions may overlap after the same snapshot.
        covered = sum(item['gpus'] for item in allocations if item['started_at'] <= prior['timestamp']
                      and (item['ended_at'] is None or item['ended_at'] > prior['timestamp']))
        if covered != prior['reserved_gpus']:
            raise ValueError('Allocation history does not continue the prior GPU reservation exactly')

    def snapshot(self, costs=None):
        now = self.clock()
        if now < self.prior['timestamp']:
            raise ValueError('Clock predates audited prior accounting')
        extra = sum(item['gpus'] * max(0, min(now, item['ended_at'] or now)
                    - max(self.prior['timestamp'], item['started_at'])) for item in self.allocations)
        counts = dict(self.prior['counts'])
        for key, value in (costs or {}).items():
            if key in counts:
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid measured work counter')
                counts[key] += value
        return {'timestamp': now, 'allocated_gpu_seconds': self.prior['allocated_gpu_seconds'] + extra,
                'counts': counts, 'allocations': deepcopy(self.allocations)}

    def check(self, costs=None, *, next_actions=0, next_queries=0):
        value = self.snapshot(costs)
        if value['allocated_gpu_seconds'] >= self.envelope['gpu_hours_ceiling'] * 3600:
            raise RuntimeError('Aggregate GPU-hour ceiling reached')
        for count, extra, maximum in [
            ('policy_control_steps', next_actions, 'policy_actions_max'),
            ('policy_queries', next_queries, 'exhaustion_safe_successful_queries_max')]:
            if type(extra) is not int or extra < 0:
                raise ValueError('Pending work reservation must be a nonnegative integer')
            if value['counts'][count] + extra > self.envelope[maximum]:
                raise RuntimeError('Cannot reserve all concurrent attempts inside the ' + count + ' ceiling')
        return value

    def remaining_wall_seconds(self, costs=None):
        value = self.check(costs)
        active = sum(item['gpus'] for item in self.allocations if item['ended_at'] is None)
        if not active:
            raise RuntimeError('No active GPU reservation')
        return (self.envelope['gpu_hours_ceiling'] * 3600 - value['allocated_gpu_seconds']) / active


def validate_routes(routes, families):
    """Routes identify preloaded family servers; workers cannot switch them mid-run."""
    if not routes or {row['family'] for row in routes} != set(families):
        raise ValueError('Every pilot family needs at least one fixed policy route')
    ports, devices, slots = set(), {}, []
    for row in routes:
        if set(row) != {'gpu', 'family', 'port', 'slots'}:
            raise ValueError('Unexpected worker route field')
        if type(row['gpu']) is not int or row['gpu'] < 0:
            raise ValueError('Worker GPU index must be a nonnegative integer')
        if type(row['port']) is not int or not 1 <= row['port'] <= 65535 or row['port'] in ports:
            raise ValueError('Each policy server needs a unique valid port')
        if type(row['slots']) is not int or not 1 <= row['slots'] <= 2:
            raise ValueError('This deployment permits at most two episode workers per GPU')
        if row['gpu'] in devices:
            raise ValueError('Each GPU has exactly one preloaded family policy server')
        ports.add(row['port'])
        devices[row['gpu']] = row['family']
        slots.extend({**row, 'slot_id': f"gpu{row['gpu']}-worker{index}"} for index in range(row['slots']))
    return slots


def run_parallel(coordinator, routes, state_path, output_for, execute, accept, budget, *,
                 max_new_jobs=None, on_progress=None, timeout_reserve=30, stop_requested=None,
                 release=None):
    """Drain concurrent results before stopping; never resample/retry a claimed job.

    ``execute`` owns a fresh process and its bounded cleanup; ``accept`` independently
    reads and binds its result. Both are supplied by the managed deployment. All
    coordinator reads/writes and audits happen on this parent thread. A failed
    result prevents new launches while already running work is drained and paid.
    ``accept`` may return None only for an attempt voided under an approved
    zero-interaction rule; ``release`` then frees its claim so the same job reruns.
    """
    slots = validate_routes(routes, coordinator.protocol['families'])
    if coordinator.report()['in_flight_claims']:
        raise ValueError('Reconcile existing claims before parallel takeover; no automatic retry')
    if max_new_jobs is not None and (type(max_new_jobs) is not int or max_new_jobs <= 0):
        raise ValueError('max_new_jobs must be a positive launch limit')
    in_flight, launched = {}, 0
    stopped = False
    failure = None
    saved_revision = None
    with ThreadPoolExecutor(max_workers=len(slots), thread_name_prefix='pilot-episode') as pool:
        while True:
            pending = []
            try:
                # Any scheduling failure stops launches but still drains claimed work.
                if stop_requested is not None and stop_requested():
                    stopped = True
                if coordinator.report()['status'] in ('blocked', 'complete'):
                    stopped = True
                pending = [] if stopped else coordinator.next_jobs()
                revision = getattr(coordinator, 'revision', None)
                if revision is None or revision != saved_revision:
                    coordinator.save(state_path)
                    saved_revision = revision
            except Exception as error:  # noqa: BLE001 - drain already claimed physical work
                stopped, pending = True, []
                failure = failure or error
            busy = {record['slot']['slot_id'] for record in in_flight.values()}
            try:
                for slot in slots:
                    if slot['slot_id'] in busy or (max_new_jobs is not None and launched >= max_new_jobs):
                        continue
                    job = next((item for item in pending if item['family'] == slot['family']), None)
                    if job is None:
                        continue
                    reserved = sum(record['job']['max_steps'] for record in in_flight.values())
                    costs = coordinator.report()['cost_totals']
                    budget.check(costs, next_actions=reserved + job['max_steps'],
                                 next_queries=reserved + job['max_steps'])
                    if budget.remaining_wall_seconds(costs) <= timeout_reserve:
                        raise RuntimeError('Insufficient aggregate reservation for safe process cleanup')
                    output = output_for(job)
                    coordinator.mark_started(job['job_id'], output)
                    coordinator.save(state_path)
                    saved_revision = getattr(coordinator, 'revision', None)
                    # Submission follows the durable claim, never the reverse.
                    future = pool.submit(execute, deepcopy(job), deepcopy(slot), output)
                    in_flight[future] = {'job': job, 'slot': slot, 'output': output}
                    pending.remove(job)
                    launched += 1
            except Exception as error:  # noqa: BLE001 - drain already claimed physical work
                stopped, failure = True, failure or error
            if not in_flight:
                if failure is not None:
                    raise failure
                if pending and not stopped and (max_new_jobs is None or launched < max_new_jobs):
                    raise RuntimeError('Pending jobs have no usable worker route')
                break
            done, _ = wait(in_flight, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                record = in_flight.pop(future)
                try:
                    value = future.result()
                    error = None
                except Exception as worker_error:  # noqa: BLE001 - retain every paid worker failure
                    value, error = None, worker_error
                try:
                    row = accept(record['job'], record['slot'], record['output'], value, error)
                    if row is None:
                        if release is None:
                            raise RuntimeError('Accept voided an attempt without a release policy')
                        release(record['job']['job_id'])
                    else:
                        coordinator.record_result(record['job']['job_id'], row)
                    coordinator.save(state_path)
                    saved_revision = getattr(coordinator, 'revision', None)
                except Exception as audit_error:  # noqa: BLE001 - the claim stays for reconciliation
                    stopped, failure = True, failure or audit_error
                if coordinator.report()['status'] == 'blocked':
                    stopped = True
            if on_progress is not None:
                try:
                    on_progress(coordinator.report(), list(in_flight.values()))
                except Exception as progress_error:  # noqa: BLE001 - stop launching, keep draining
                    stopped, failure = True, failure or progress_error
    return coordinator.report()
