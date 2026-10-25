"""Scientific boundary and restart-accounting tests for the frozen DSRL06 protocol."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from helpers.dsrl06_protocol import (
    CAPS,
    EpisodeScorer,
    ProtocolLedger,
    actor_query_seed,
    benchmark_episode_seed,
    check_return_identity,
    decoder_vectors,
    evaluate_authority,
    frozen_jobs,
    gripper_commands,
    reward_transition,
    select_probe_snapshots,
)


def observe(scorer, step, **changes):
    fields = dict(
        jar_closed=False,
        jar_on_support=True,
        grasped=False,
        instantaneous_goal=False,
        raw_violation=False,
        counted_violation=False,
    )
    fields.update(changes)
    return scorer.observe(step, **fields)


def test_closure_is_latched_and_equal_departure_is_valid():
    scorer = EpisodeScorer()
    assert observe(scorer, 0) == 0
    assert observe(scorer, 1, jar_closed=True, jar_on_support=False, grasped=True) == 0.75
    assert scorer.first_closed_step == scorer.first_departure_step == 1
    assert not scorer.order_failed
    assert observe(scorer, 2) == 0.20
    assert not scorer.jar_closed_now and scorer.closed_once
    assert scorer.potential(terminal=True) == 0


def test_early_carry_cannot_be_repaired_by_later_closure_or_goal():
    scorer = EpisodeScorer()
    observe(scorer, 0)
    assert observe(scorer, 1, jar_on_support=False) == 0
    for step in range(2, 12):
        assert (
            observe(
                scorer,
                step,
                jar_closed=True,
                jar_on_support=False,
                grasped=True,
                instantaneous_goal=True,
            )
            == 0
        )
    result = scorer.labels(True)
    assert result["order_failed"] and result["cost"] == 1
    assert not result["requested_success"] and not result["strict_safe_requested"]


def test_contact_gated_raw_violation_cannot_be_called_strictly_safe():
    scorer = EpisodeScorer()
    observe(scorer, 0, instantaneous_goal=True)  # Initial state is not a goal control.
    for step in range(1, 11):
        observe(scorer, step, jar_closed=True, instantaneous_goal=True, raw_violation=step == 3)
    result = scorer.labels(True, counted_violation=False)
    assert result["goal_hold_count"] == 10
    assert result["requested_success"] and result["safe_requested"]
    assert not result["strict_safe_requested"] and result["cost"] == 1
    assert scorer.potential() == 0
    result = scorer.labels(True, counted_violation=True)
    assert not result["safe_requested"]


def test_safe_unfinished_and_invalid_attempts_are_distinguished():
    scorer = EpisodeScorer()
    observe(scorer, 0)
    with pytest.raises(ValueError, match="at least one control"):
        scorer.labels(False)
    observe(scorer, 1)
    assert scorer.labels(False)["cost"] == 0
    assert scorer.labels(False)["unfinished"]
    invalid = scorer.labels(None, valid=False)
    assert invalid["strict_safe_requested"] is None and invalid["cost"] is None
    with pytest.raises(ValueError, match="Interrupted"):
        reward_transition(0, 0, 8, terminal=True, valid=False)


def test_goal_and_state_alignment_fail_closed():
    scorer = EpisodeScorer()
    with pytest.raises(ValueError, match="every control"):
        observe(scorer, 1)
    with pytest.raises(ValueError, match="boolean"):
        observe(scorer, 0, raw_violation=None)
    observe(scorer, 0)
    with pytest.raises(ValueError, match="without a raw"):
        observe(scorer, 1, counted_violation=True)
    for step in range(1, 10):
        observe(scorer, step, jar_closed=True, instantaneous_goal=True)
    with pytest.raises(ValueError, match="ten-control"):
        scorer.labels(True)
    observe(scorer, 10, jar_closed=True, instantaneous_goal=True)
    with pytest.raises(ValueError, match="ten-control"):
        scorer.labels(False)
    with pytest.raises(ValueError, match="every control"):
        observe(scorer, 12)


def test_scorer_resume_preserves_first_events_and_violation_history():
    scorer = EpisodeScorer()
    observe(scorer, 0)
    observe(scorer, 1, jar_closed=True)
    observe(scorer, 2, raw_violation=True)
    restored = EpisodeScorer.from_state_dict(json.loads(json.dumps(scorer.state_dict())))
    assert restored.state_dict() == scorer.state_dict()
    for step in range(3, 8):
        assert observe(restored, step, grasped=True) == observe(scorer, step, grasped=True) == 0
    with pytest.raises(ValueError, match="Invalid restored"):
        EpisodeScorer.from_state_dict({**scorer.state_dict(), "first_closed_step": 100})


@pytest.mark.parametrize("terminal_success", [False, True])
def test_complete_horizon_return_identity_and_terminal_mask(terminal_success):
    rows = []
    phi = np.r_[np.zeros(20), np.full(50, 0.2), np.full(50, 0.5), np.full(131, 0.75)]
    assert len(phi) == 251
    for i in range(250):
        rows.append(
            reward_transition(
                float(phi[i]),
                float(phi[i + 1]),
                8,
                terminal=i == 249,
                strict_safe_requested=terminal_success and i == 249,
            )
        )
    assert rows[-1]["phi_after_raw"] == 0.75 and rows[-1]["phi_after"] == 0
    assert rows[-1]["mask"] == 0 and rows[-2]["mask"] == 1
    result = check_return_identity(rows, atol=1e-12)
    assert abs(result["return_identity_error"]) < 1e-12
    assert result["discounted_base_return"] == pytest.approx(
        0.9998**1992 if terminal_success else 0
    )


def test_partial_chunk_initial_potential_and_closure_cycles_reconcile():
    rows = [
        reward_transition(0.2, 0.5, 8, terminal=False),
        reward_transition(0.5, 0.2, 8, terminal=False),
        reward_transition(0.2, 0.75, 3, terminal=True),
    ]
    result = check_return_identity(rows, atol=1e-12)
    assert result["discounted_shaped_return"] == pytest.approx(-0.05)
    assert result["corrected_discounted_shaped_return"] == pytest.approx(0)
    with pytest.raises(ValueError, match="eight controls"):
        reward_transition(0, 0.2, 3, terminal=False)
    broken = [dict(row) for row in rows]
    broken[-1]["phi_after"] = 0.75
    with pytest.raises(ValueError, match="phi_after"):
        check_return_identity(broken)
    broken = [dict(row) for row in rows]
    broken[1]["phi_before"] = 0
    with pytest.raises(ValueError):
        check_return_identity(broken)


def test_frozen_jobs_match_counts_schedules_and_pinned_seed_rule():
    jobs = frozen_jobs()
    assert list(map(len, jobs.values())) == [6, 24, 48]
    assert [j["mode"] for j in jobs["engineering_jobs"]] == ["native"] * 4 + ["random", "actor"]
    assert jobs["engineering_jobs"][3]["admission_requires"] == []
    assert jobs["engineering_jobs"][0]["admission_requires"] == ["E3_first_policy_response"]
    assert jobs["engineering_jobs"][4]["admission_requires"] == ["authority_gate_passed"]
    training = jobs["training_jobs"]
    assert Counter(j["task"] for j in training) == {t: 4 for t in (0, 3, 6, 9, 15, 17)}
    assert [j["lane"] for j in training] == [0, 1, 2] * 8
    assert sum(j["warmup_required"] for j in training) == 6
    evaluation = jobs["evaluation_jobs"]
    for arm in ("native", "fixed_zero", "random", "dsrl"):
        assert Counter(j["lane"] for j in evaluation if j["arm"] == arm) == dict.fromkeys(
            range(4), 3
        )
    assert benchmark_episode_seed(1, 390) == 1523358492  # Recorded task-5 provenance.
    assert all(j["simulator_reset_seed"] is None for v in jobs.values() for j in v)
    all_ids = [j["job_id"] for v in jobs.values() for j in v]
    assert len(set(all_ids)) == 78
    query_seeds = [actor_query_seed(j, q) for j in all_ids for q in range(250)]
    assert query_seeds == [actor_query_seed(j, q) for j in all_ids for q in range(250)]
    assert len(set(query_seeds)) == len(query_seeds)


def test_probe_vectors_match_frozen_receipt_and_do_not_mutate_global_rng():
    before = np.random.get_state()
    labels, vectors = decoder_vectors()
    assert vectors.shape == (14, 32) and vectors.dtype == np.dtype("<f4")
    assert np.unique(vectors, axis=0).shape[0] == 14 and abs(vectors).max() == 1
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads((root / "progress/receipts/06-dsrl-decoder-probe.json").read_text())
    assert labels == receipt["candidate_labels"]
    assert (
        hashlib.sha256(vectors.tobytes()).hexdigest() == receipt["candidate_float32_bytes_sha256"]
    )
    for a, b in zip(before, np.random.get_state(), strict=True):
        np.testing.assert_array_equal(a, b)


def probe_panel():
    panel = []
    for task in (1, 4, 10, 11):
        for slot in range(2):
            reference = np.zeros((16, 8), dtype=np.float32)
            panel.append(
                {
                    "task": task,
                    "snapshot_id": f"task{task}_slot{slot}",
                    "reference": reference,
                    "native_replays": np.stack([reference, reference]),
                    "candidates": np.zeros((14, 16, 8), dtype=np.float32),
                }
            )
    return panel


def test_authority_requires_two_tasks_and_robust_executed_prefix_change():
    panel = probe_panel()
    panel[0]["candidates"][1, 0, 7] = 0.02
    panel[1]["candidates"][1, 0, 7] = 0.02
    result = evaluate_authority(panel)
    assert result["valid"] and not result["admit"] and result["sensitive_tasks"] == [1]
    panel[2]["candidates"][1, 0, 7] = 0.02
    assert evaluate_authority(panel)["admit"]
    panel[0]["candidates"][1, 0, 7] = 0.010001
    panel[1]["candidates"][1, 0, 7] = 0.010001
    assert not evaluate_authority(panel)["admit"]  # Binary flips without margin do not pass.
    panel[0]["candidates"][1, 8:, 7] = 1
    assert not evaluate_authority(panel)["admit"]  # Discarded half-chunk cannot pass.


def test_native_replay_cannot_cross_the_gripper_threshold_within_numeric_tolerance():
    panel = probe_panel()
    panel[0]["reference"][0, 7] = 0.009999
    panel[0]["native_replays"][:, 0, 7] = 0.010001
    result = evaluate_authority(panel)
    assert not result["valid"] and "converted gripper" in result["reason"]
    np.testing.assert_array_equal(
        gripper_commands(np.array([-0.1, 0, 0.01, 0.01001])), [-1, -1, -1, 1]
    )
    panel = probe_panel()
    panel[0]["candidates"][0, 0, 7] = np.nan
    assert not evaluate_authority(panel)["valid"]
    assert not evaluate_authority(panel[:7])["valid"]


def test_snapshot_selection_prefers_first_safe_closure_then_frozen_fallback():
    episodes = {}
    for i, task in enumerate((1, 4, 10, 11)):
        queries = [
            dict(
                step=s,
                closed_once=s >= 16 and task != 4,
                raw_violation=s >= 808,
                order_failed=False,
            )
            for s in (0, 8, 16, 800, 808)
        ]
        episodes[f"E{i}"] = {"task": task, "queries": queries}
    selected = select_probe_snapshots(episodes)
    assert [q["step"] for q in selected] == [0, 16, 0, 800, 0, 16, 0, 16]
    episodes["E0"]["queries"] = episodes["E0"]["queries"][:1]
    with pytest.raises(ValueError, match="Insufficient"):
        select_probe_snapshots(episodes)


def test_ledger_resumes_attempts_and_requires_recorded_repair(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = ProtocolLedger(path)
    ledger.reserve_start("E3", event_id="start-E3-1")
    ledger.reserve_start("E3", event_id="start-E3-1")  # Known reservation survives restart.
    assert ledger.summary()["usage"]["starts"] == 1
    with pytest.raises(ValueError, match="invalid attempt"):
        ledger.reserve_start("E3", attempt=2, repair_receipt="repair.md")
    ledger.finish_start("E3", 1, valid=False, reason="renderer_error")
    ledger = ProtocolLedger(path)
    with pytest.raises(ValueError, match="repair"):
        ledger.reserve_start("E3", attempt=2)
    ledger.reserve_start("E3", attempt=2, repair_receipt="repair.md")
    ledger.finish_start("E3", 2, valid=True)
    assert ledger.summary()["usage"]["starts"] == 2
    assert ledger.summary()["usage"]["replacements"] == 1
    with pytest.raises(ValueError):
        ledger.reserve_start("E3", attempt=3, repair_receipt="again.md")
    with pytest.raises(ValueError, match="frozen job"):
        ledger.reserve_start("new-outcome-driven-job")


def test_ledger_counter_caps_and_idempotent_accounting(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = ProtocolLedger(path)
    ledger.charge("model_queries", 0, event_id="startup-zero")
    ledger.charge("model_queries", 128, event_id="probe-attempts")
    ledger.charge("policy_controls", 2000, event_id="E3-controls")
    ledger.charge("learner_updates", 2000, phase="engineering", event_id="engineering")
    ledger = ProtocolLedger(path)
    ledger.charge("model_queries", 128, event_id="probe-attempts")
    assert ledger.summary()["usage"]["model_queries"] == 128
    with pytest.raises(ValueError, match="different data"):
        ledger.charge("model_queries", 127, event_id="probe-attempts")
    with pytest.raises(ValueError, match="engineering_updates"):
        ledger.charge("learner_updates", 1, phase="engineering")
    ledger.charge("learner_updates", 90000, phase="training")
    usage = ledger.summary()["usage"]
    assert usage["learner_updates"] == 92000 and usage["learning_updates"] == 90000
    with pytest.raises(ValueError, match="learning_updates"):
        ledger.charge("learner_updates", 1, phase="learning")
    with pytest.raises(ValueError, match="model_queries"):
        ledger.charge("model_queries", 21000)
    ledger.charge("warmup_controls", 800)
    with pytest.raises(ValueError, match="warmup_controls"):
        ledger.charge("warmup_controls", 1)
    with pytest.raises(TypeError):
        ledger.caps["model_queries"] = 999999
    assert CAPS["model_queries"] == 21000


def test_ledger_detects_truncation_and_changed_bytes(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = ProtocolLedger(path)
    ledger.charge("model_queries", 7)
    original = path.read_text()
    path.write_text(original.rstrip())
    with pytest.raises(ValueError, match="incomplete"):
        ProtocolLedger(path)
    path.write_text(original.replace('"count":7', '"count":6'))
    with pytest.raises(ValueError, match="modified"):
        ProtocolLedger(path)


def test_ledger_cannot_use_more_than_two_replacements(tmp_path):
    ledger = ProtocolLedger(tmp_path / "ledger.jsonl")
    for job in ("E0", "E1", "E2"):
        ledger.reserve_start(job)
        ledger.finish_start(job, 1, valid=False)
    for job in ("E0", "E1"):
        ledger.reserve_start(job, 2, repair_receipt="same-infrastructure-repair.md")
    with pytest.raises(ValueError, match="replacements"):
        ledger.reserve_start("E2", 2, repair_receipt="same-infrastructure-repair.md")
    assert ledger.summary()["usage"]["replacements"] == 2
