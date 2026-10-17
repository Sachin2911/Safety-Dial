"""Bounded subprocess scheduling; stop launching work after an observed failure."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def run_jobs(jobs, execute, record, workers):
    if workers not in (1, 2):
        raise ValueError('The supported worker limit is one or two')
    next_index = 0
    failed = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        while next_index < len(jobs) or pending:
            while not failed and next_index < len(jobs) and len(pending) < workers:
                pending[pool.submit(execute, next_index, jobs[next_index])] = next_index
                next_index += 1
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                error = future.exception()
                if error is None:
                    process, row = future.result()
                else:
                    process = {'index': index, 'returncode': None,
                               'scheduler_error': {'type': type(error).__name__,
                                                   'message': str(error)}}
                    row = None
                record(process, row)
                failed |= row is None
    return not failed
