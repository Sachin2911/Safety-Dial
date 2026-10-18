"""Causal feedback and experiment-budget invariants, independent of the simulator."""

import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_followup_budget import clock
from helpers.maniguard_part2_controller import (
    Config,
    FeedbackController,
    conditions,
)
from helpers.maniguard_part2_protocol import draft, plan
from helpers.maniguard_part2_source import prepare_runtime
from helpers.maniguard_phase3_protocol import validate_frozen


def frame(q=0, fingers=0.04):
    return SimpleNamespace(joints=np.full(7, q, dtype=np.float32),
                           fingers=np.full(2, fingers, dtype=np.float32))


def execute(adapter, proposal, observation=None, actual=None):
    adapted, _ = adapter.act(proposal, frame() if observation is None else observation)
    adapter.applied(adapted if actual is None else actual)
    return adapted


def test_identity_is_bitwise_and_preserves_dtype_and_gripper():
    rng = np.random.default_rng(7)
    adapter = FeedbackController(conditions()["A"])
    for i in range(16):
        if i % 8 == 0:
            adapter.on_query(None)
        p = rng.normal(size=8).astype(np.float32)
        p[0] = -0.0
        result = execute(adapter, p)
        assert result.tobytes() == p.tobytes()
        assert adapter.correction == 0
        assert adapter.replan_reason(None) == ("maximum_interval" if i % 8 == 7 else None)


def test_restraint_identical_across_schedules_and_grasp_heuristic():
    adapters = [FeedbackController(conditions()[key]) for key in "BCD"]
    proposal = np.ones(8, dtype=np.float32)
    proposal[-1] = -1
    for step in range(3):
        outputs = [execute(a, proposal, frame(fingers=0.02)) for a in adapters]
        for output in outputs:
            np.testing.assert_array_equal(output, outputs[0])
            np.testing.assert_allclose(output[:7], 0.04 if step == 0 else 0.02)
            assert output[-1] == proposal[-1]


def test_actual_downstream_clipping_drives_correction_and_causal_history():
    adapter = FeedbackController(conditions()["A"])
    p = np.full(8, 0.04, dtype=np.float32)
    actual = np.zeros(8, dtype=np.float32)
    adapter.act(p, frame())
    assert adapter.correction == 0
    with pytest.raises(ValueError):
        adapter.replan_reason(None)
    with pytest.raises(ValueError):
        adapter.on_query(None)
    adapter.applied(actual)
    assert adapter.correction == pytest.approx(1)
    np.testing.assert_array_equal(adapter.last_executed, actual)
    adapter.on_query(None)
    assert adapter.correction == 0 and adapter.since_query == 0
    np.testing.assert_array_equal(adapter.last_executed, actual)


def test_threshold_accumulates_but_cannot_query_before_four():
    adapter = FeedbackController(conditions()["D"])
    p = np.full(8, 0.1, dtype=np.float32)
    for step in range(4):
        execute(adapter, p)
        assert adapter.replan_reason(None) == ("correction_threshold" if step == 3 else None)
    adapter.on_query(None)
    for step in range(8):
        execute(adapter, np.zeros(8, dtype=np.float32))
        assert adapter.replan_reason(None) == ("maximum_interval" if step == 7 else None)


def test_fixed_four_and_numerical_tolerance():
    adapter = FeedbackController(conditions()["C"])
    for step in range(4):
        execute(adapter, np.zeros(8, dtype=np.float32))
        assert adapter.replan_reason(None) == ("fixed_minimum" if step == 3 else None)
    identity = FeedbackController(conditions()["A"])
    execute(identity, np.full(8, 1e-7, dtype=np.float32), actual=np.zeros(8))
    assert identity.correction == 0


def test_schedule_has_only_four_matched_training_blocks_and_no_expansion():
    jobs = plan()["jobs"]
    assert len(jobs) == 16
    assert len({j["episode_id"] for j in jobs}) == 16
    for block in range(4):
        selected = [j for j in jobs if j["block"] == block]
        assert len({(j["scene"], j["seed"]) for j in selected}) == 1
        assert {j["condition"] for j in selected} == set("ABCD")
        assert all(j["action_horizon"] == 1100 for j in selected)
    assert draft()["engineering"]["attempts"] == 4
    assert draft()["engineering"]["max_successful_actions"] == 96
    assert {j["purpose"] for j in jobs} == {"train"}


def test_original_clock_and_explicit_extension():
    status = clock(now=1792371637)  # 2026-10-19 01:00:37 UTC
    assert status["elapsed_seconds"] == 9 * 3600
    assert status["allocated_gpu_hours"] == 9
    assert status["physical_remaining"] == 2 * 3600
    assert status["hard_remaining"] == 3 * 3600


def test_frozen_part1_and_checked_additive_source_remain_intact(tmp_path):
    root = Path(__file__).resolve().parents[2]
    validate_frozen(root / "runs/maniguard/phase3-part1/20261018-56578764/protocol.json")
    path = prepare_runtime(tmp_path)
    source = path.read_text()
    assert "from helpers.maniguard_part2_runtime import STUDY" in source
    assert "STUDY.applied(step_idx, action_clipped)" in source
    assert "STUDY.should_replan" in source
    compile(source, str(path), "exec")


@pytest.mark.parametrize("genome", [[0, 0, 4, 0], [3, 3, 1, 0], [1, 2, 3, 1]])
def test_part3_four_parameter_configs_round_trip(genome):
    cfg = Config.from_genome(genome)
    assert Config(**asdict(cfg)) == cfg


def test_entry_point_preloads_legacy_before_setting_active_job(tmp_path):
    import json
    import subprocess

    root = Path(__file__).resolve().parents[2]
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "phase": "phase3_followup", "output": str(tmp_path), "controller": asdict(Config()),
        "controller_sha256": __import__("helpers.maniguard_part2_controller", fromlist=["config_hash"]).config_hash(Config()),
    }))
    code = """
import runpy,os,sys
runpy.run_path(sys.argv[1],run_name='preload_check')
assert 'SAFETY_DIAL_MANIGUARD_JOB' not in os.environ
os.environ['SAFETY_DIAL_MANIGUARD_JOB']=sys.argv[2]
from helpers.maniguard_part2_runtime import STUDY
assert STUDY.config.kind == 'phase3_followup'
assert STUDY.config.privileged is False
STUDY.close()
"""
    subprocess.run([sys.executable, "-c", code,
                    str(root / "experiments/scripts/maniguard_part2_episode.py"), str(job)],
                   check=True, capture_output=True, text=True)
    assert (tmp_path / "ledger.json").exists()
