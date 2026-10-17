"""Check interleaved RNG streams and bounded job scheduling before physical use."""

import sys
from pathlib import Path
from threading import Event

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_jobs import run_jobs
from helpers.maniguard_parallel_policy import IsolatedEpisodePolicy


class FakePolicy:
    def infer(self, observation):
        assert set(observation) == {'state'}
        result = self._rng
        self._rng += 1
        return result


def test_interleaved_streams_advance_independently():
    policy = IsolatedEpisodePolicy(FakePolicy(), int)
    a = {'study_episode_id': 'a', 'episode_seed': 42, 'state': [0]}
    b = dict(a, study_episode_id='b', episode_seed=7)
    assert [policy.infer(o) for o in (a, b, a, b)] == [42, 7, 43, 8]
    assert a == {'study_episode_id': 'a', 'episode_seed': 42, 'state': [0]}
    c = dict(a, study_episode_id='c')
    assert policy.infer(c) == 42
    assert policy.infer(a) == 44
    with pytest.raises(ValueError, match='cannot change'):
        policy.infer(dict(a, episode_seed=99))


@pytest.mark.parametrize('workers', [1, 2])
def test_all_jobs_recorded_once(workers):
    records = []
    assert run_jobs(list(range(8)), lambda i, j: ({'index': i}, {'job': j}),
                    lambda p, r: records.append((p['index'], r['job'])), workers)
    assert sorted(records) == [(i, i) for i in range(8)]
    if workers == 1:
        assert records == [(i, i) for i in range(8)]


def test_failure_stops_new_work_and_records_other_running_job():
    second_started = Event()
    failure_recorded = Event()
    records = []

    def execute(index, job):
        if index == 0:
            assert second_started.wait(5)
            return {'index': index}, None
        assert index == 1
        second_started.set()
        assert failure_recorded.wait(5)
        return {'index': index}, {'job': job}

    def record(process, row):
        records.append((process, row))
        if row is None:
            failure_recorded.set()

    assert not run_jobs([0, 1, 2, 3], execute, record, 2)
    assert [p['index'] for p, _ in records] == [0, 1]
    assert records[1][1] == {'job': 1}


def test_launch_exception_is_recorded_and_stops_serial_queue():
    records = []

    def execute(index, job):
        raise OSError('Cannot create episode output')

    assert not run_jobs([0, 1], execute, lambda p, r: records.append((p, r)), 1)
    assert len(records) == 1
    assert records[0][0]['scheduler_error']['type'] == 'OSError'
    assert records[0][1] is None
