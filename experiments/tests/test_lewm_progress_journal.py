"""NAV journal and safety-head parity checks, run only on a declared rented GPU."""

import datetime as dt
import json
import os
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from helpers.lewm_campaign_training import CampaignTraining, IncompleteJobError
from helpers.lewm_progress_journal import (
    JournalCapExceeded,
    JournalCorruption,
    JournalError,
    ProgressJournal,
    recover_progress_journals,
    replay_progress_journal,
)
from helpers.portable_motion import MotionChecker
from run_lewm_navigation_campaign_v2 import (
    AccountedGoalReadout,
    AccountedSafetyHead,
    ProgressCounter,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("RUNPOD_POD_ID"),
    reason="All experiment software tests belong on a declared rented GPU",
)


def journal(tmp_path, downstream=None, cap=3):
    return ProgressJournal(
        tmp_path / "progress-journal.jsonl",
        downstream or (lambda *_: None),
        max_actions=cap,
        metadata={"job_id": tmp_path.name},
    )


def test_full_cumulative_prefix_and_boundary_forwarding(tmp_path):
    forwarded = []
    j = journal(tmp_path, lambda a, r: forwarded.append((a, r)))
    j(0, {"environment_resets": 1})
    j(0, {"model_attempts": 1, "model_rows": 32})
    j(0, {"model_successes": 1})
    j(0, {"model_successes": 1})  # Identical snapshot is already durable.
    j(1, {"environment_actions": 1})
    summary = j.finish()
    j.close()
    replay = replay_progress_journal(j.path)
    assert replay["records"] == summary["fsync_calls"] == 6
    assert len(forwarded) == summary["downstream_forwards"] == 3
    assert [a for a, _ in forwarded] == [0, 1, 1]
    assert forwarded[0][1].get("model_attempts", 0) == 0
    assert replay["last"]["resources"] == {
        "environment_actions": 1,
        "environment_resets": 1,
        "model_attempts": 1,
        "model_rows": 32,
        "model_successes": 1,
    }
    lines = j.path.read_bytes().splitlines(keepends=True)
    for size in range(1, len(lines) + 1):
        prefix = tmp_path / f"prefix-{size}.jsonl"
        prefix.write_bytes(b"".join(lines[:size]))
        assert replay_progress_journal(prefix)["records"] == size


def test_unterminated_tail_only_is_ignored_and_hash_corruption_refused(tmp_path):
    j = journal(tmp_path)
    j(0, {"model_attempts": 1})
    j.close()
    original = j.path.read_bytes()
    with j.path.open("ab") as stream:
        stream.write(b'{"schema":"partial')
    result = replay_progress_journal(j.path)
    assert result["truncated_tail_bytes"] == len(b'{"schema":"partial')
    assert result["last"]["resources"]["model_attempts"] == 1
    j.path.write_bytes(original + b"not-json\n")
    with pytest.raises(JournalCorruption, match="malformed"):
        replay_progress_journal(j.path)
    rows = original.splitlines(keepends=True)
    row = json.loads(rows[-1])
    row["resources"]["model_attempts"] = 100
    j.path.write_bytes(b"".join(rows[:-1]) + json.dumps(row).encode() + b"\n")
    with pytest.raises(JournalCorruption, match="hash"):
        replay_progress_journal(j.path)


def test_monotonicity_cap_and_post_final_rejection(tmp_path):
    forwarded = []
    j = journal(tmp_path, lambda a, r: forwarded.append((a, r)), cap=1)
    j(1, {"model_attempts": 2})
    with pytest.raises(JournalError, match="regressed"):
        j(1, {"model_attempts": 1})
    with pytest.raises(JournalCapExceeded):
        j(2, {"model_attempts": 2})
    assert forwarded[-1][0] == 2
    assert replay_progress_journal(j.path)["last"]["environment_actions"] == 2
    j.finish(status="error")
    with pytest.raises(JournalError, match="finished"):
        j(2, {})
    j.close()


def test_fsync_failure_prevents_call_and_future_work(tmp_path, monkeypatch):
    forwarded, calls = [], []
    j = journal(tmp_path, lambda a, r: forwarded.append((a, r)))

    def failed_fsync(_):
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr("helpers.lewm_progress_journal.os.fsync", failed_fsync)
    with pytest.raises(OSError, match="synthetic fsync"):
        j(0, {"model_attempts": 1})
        calls.append("must never execute")
    assert not calls
    with pytest.raises(JournalError, match="failed"):
        j(0, {"model_attempts": 2})
    j.finish(status="error")
    j.close()
    assert j.summary()["failed"]
    assert forwarded[-1][1]["model_attempts"] == 1


def test_recovery_refuses_live_owner_and_does_not_adopt_episode(tmp_path):
    job = tmp_path / "jobs" / "j0"
    job.mkdir(parents=True)
    j = journal(job)
    j(1, {"model_attempts": 17})
    j.close()
    with pytest.raises(JournalError, match="remains live"):
        recover_progress_journals(tmp_path)
    result = recover_progress_journals(tmp_path, require_stopped=False)
    live = json.loads((job / "live.json").read_text())
    assert live["environment_actions"] == 1 and live["resources"]["model_attempts"] == 17
    assert not live["finalized"] and not (job / "result.json").exists()
    assert result["episode_outcomes_adopted"] == result["new_nn_calls"] == 0


def test_recovered_started_job_still_refuses_optimizer_replay(tmp_path):
    protocol = {
        "schema": "lewm-campaign-training-v1",
        "allowed_pod_ids": [os.environ["RUNPOD_POD_ID"]],
        "deadline_utc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat(),
        "method": "random",
        "repeat_id": 0,
        "optimizer_seed": 72,
        "initial_parameters": [0.1, 0.2],
        "training_panel": [{"case_id": "a", "seed": 101}],
        "max_environment_actions": 20,
        "max_episode_actions": 3,
        "max_episode_starts": 20,
        "max_candidate_panels": 1,
        "population": 2,
        "sigma": 0.1,
        "parameter_bounds": [-6, 6],
        "source_ids": {"purpose": "synthetic journal interruption, zero real interactions"},
        "reinforce": {
            "learning_rate": 0.05,
            "baseline_decay": 0.9,
            "temperature": 1.0,
            "max_grad_norm": 10.0,
        },
    }
    calls = []

    def interrupted(job, downstream):
        calls.append(job.job_id)
        j = journal(job.directory, downstream)
        j(1, {"model_attempts": 4})
        j(1, {"model_attempts": 9})  # Latest model work is not a legacy live refresh.
        j.close()  # Simulate an abrupt worker exit before episode finalization.
        raise RuntimeError("synthetic interrupted worker")

    root = tmp_path / "training"
    first = CampaignTraining(protocol, root, interrupted)
    with pytest.raises(RuntimeError, match="interrupted worker"):
        first.run()
    assert first.state["resource_totals"]["model_attempts"] == 4
    recover_progress_journals(root, require_stopped=False)
    resumed = CampaignTraining(protocol, root, interrupted, resume=True)
    assert resumed.state["resource_totals"]["model_attempts"] == 9
    assert resumed.state["actual_environment_actions"] == 1
    assert resumed.state["charged_environment_actions"] == 3
    assert not resumed.state["action_accounting_exact"]
    with pytest.raises(IncompleteJobError):
        resumed.run()
    assert len(calls) == 1


class ToyHead(torch.nn.Module):
    def __init__(self, hook=None, fail=False):
        super().__init__()
        self.hook, self.fail = hook, fail
        self.spec = {
            "action_dim": 2,
            "normalized_action_low": [-1, -1],
            "normalized_action_high": [1, 1],
            "body_radius": 7,
        }
        self.calibration = {"ready": True, "allowance": 0.0}

    def forward(self, initial, actions):
        if self.hook:
            self.hook()
        if self.fail:
            raise RuntimeError("synthetic head forward failure")
        return initial[:, None, -2:] + actions.cumsum(dim=-2)


def test_head_wrapper_identical_values_and_durable_before_forward(tmp_path):
    j = journal(tmp_path)
    new_c = ProgressCounter(j)

    def before_forward():
        saved = replay_progress_journal(j.path)["last"]["resources"]
        assert saved["actual_safety_call_attempts"] == 1
        assert saved["actual_safety_candidate_rollouts_attempted"] == 3
        assert saved["actual_safety_transition_rows_attempted"] == 30
        assert saved.get("actual_safety_call_successes", 0) == 0

    old_c = ProgressCounter(lambda *_: None)
    old = MotionChecker(ToyHead().cuda(), [], 3, old_c)
    new = MotionChecker(AccountedSafetyHead(ToyHead(before_forward).cuda(), new_c), [], 3, new_c)
    old.observe([40.0, 50.0])
    new.observe([40.0, 50.0])
    candidates = torch.linspace(-1.2, 1.2, 60, device="cuda").reshape(1, 3, 2, 10)
    expected = old.check(candidates.clone())
    actual = new.check(candidates.clone())
    new_c.publish()
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
    assert new_c["learned_candidate_checks"] == old_c["learned_candidate_checks"] == 3
    assert new_c["learned_transition_rows"] == old_c["learned_transition_rows"] == 30
    assert new_c["actual_safety_call_successes"] == 1
    j.finish()
    j.close()


def test_head_failure_is_charged_without_success(tmp_path):
    j = journal(tmp_path)
    counters = ProgressCounter(j)
    checker = MotionChecker(
        AccountedSafetyHead(ToyHead(fail=True).cuda(), counters), [], 3, counters
    )
    checker.observe([40.0, 50.0])
    with pytest.raises(RuntimeError, match="head forward failure"):
        checker.check(torch.zeros((1, 2, 1, 10), device="cuda"))
    counters.publish()
    j.finish(status="error")
    j.close()
    saved = replay_progress_journal(j.path)["last"]["resources"]
    assert saved["actual_safety_call_attempts"] == saved["actual_safety_call_failures"] == 1
    assert saved["actual_safety_candidate_rollouts_attempted"] == 2
    assert saved["actual_safety_transition_rows_attempted"] == 10
    assert saved.get("actual_safety_call_successes", 0) == 0
    assert counters["learned_candidate_checks"] == 0


def test_readout_wrapper_preserves_fp64_values_and_counts_before_call(tmp_path):
    j = journal(tmp_path)
    counters = ProgressCounter(j)

    class Readout:
        fingerprint = "synthetic-frozen-map"
        calls = 0

        def __call__(self, embedding):
            saved = replay_progress_journal(j.path)["last"]["resources"]
            assert saved["goal_readout_queries"] == saved["goal_readout_call_attempts"] == 1
            self.calls += 1
            return embedding.to(dtype=torch.float64)[..., :2] * 1.25 + 3.0

    original = Readout()
    wrapped = AccountedGoalReadout(original, counters)
    embedding = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32, device="cuda")
    actual = wrapped(embedding)
    expected = embedding.to(dtype=torch.float64)[:2] * 1.25 + 3.0
    assert actual.dtype == torch.float64 and torch.equal(actual, expected)
    assert wrapped.fingerprint == original.fingerprint and wrapped.calls == 1
    assert counters["goal_readout_call_successes"] == 1
    j.finish()
    j.close()
