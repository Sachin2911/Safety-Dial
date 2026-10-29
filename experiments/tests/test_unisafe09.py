"""Scientific contracts for the fresh UNISafe mechanism study."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe09_adapter import Adapter
from helpers.unisafe09_controller import FrozenController, backup_arm_action
from helpers.unisafe09_environment import clone_observation, normalized_to_environment
from helpers.unisafe09_infra import FinancialBook
from helpers.unisafe09_outcomes import EpisodeScore
from helpers.unisafe09_protocol import condition, schedule, select_confirmation

ROOT = Path(__file__).resolve().parents[2]


def apply(adapter, grip=-1.0, aperture=0.012):
    proposal = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, grip]])
    original = proposal.clone()
    obs = {"gripper_pos": torch.tensor([[aperture, -aperture]])}
    changed, log = adapter(proposal, obs, None)
    assert torch.equal(proposal, original)
    return changed, log


@pytest.mark.parametrize("cid", ["d00_g100", "d15_g100", "d40_g100", "reference_random"])
def test_matches_archived_adapter_over_adversarial_sensor_sequences(cid):
    # Load the frozen historical implementation only as an independent oracle.
    path = ROOT / "archive/20261022/experiments/helpers/unisafe_adaptation.py"
    spec = importlib.util.spec_from_file_location("archived_adapter_oracle", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    cfg = condition(cid)
    reference = old.GraspReleaseDelay(cfg["hold_steps"], cfg["min_close_steps"], 0.005)
    current = Adapter(cfg)
    generator = torch.Generator().manual_seed(2026102509)
    for tick in range(1000):
        proposal = torch.rand((1, 7), generator=generator) * 2 - 1
        # Include long stable runs, empty fingers, motion and new closing sequences.
        aperture = (
            0.012
            if tick % 100 < 70
            else 0.0
            if tick % 100 < 80
            else float(torch.rand((), generator=generator)) * 0.04
        )
        obs = {"gripper_pos": torch.tensor([[aperture, -aperture]])}
        a, x = reference(proposal, obs, None)
        b, y = current(proposal, obs, None)
        assert torch.equal(a, b)
        assert all(y[k] == v for k, v in x.items())
        assert current.closed_run == reference.closed_run


def test_identity_does_not_allocate_or_consume_rng():
    adapter = Adapter(condition("d00_g100"))
    for grip in [-1.0] * 20 + [1.0] * 60:
        proposal = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, grip]])
        rng = torch.random.get_rng_state().clone()
        result, log = adapter(proposal, {"gripper_pos": torch.tensor([[0.012, -0.012]])}, None)
        assert result is proposal
        assert not log["adapter_gripper_override"]
        assert torch.equal(rng, torch.random.get_rng_state())


def test_translation_gate_current_sensor_only_and_no_rotation_change():
    adapter = Adapter(condition("d00_g050"))
    first, _ = apply(adapter)
    second, log = apply(adapter)
    assert torch.equal(first[:, :3] * 0.5, second[:, :3])
    assert torch.equal(first[:, 3:], second[:, 3:])
    assert log["adapter_translation_active"] and not log["adapter_gripper_override"]
    for aperture in (0.0, 0.04, 0.006, 0.012):
        _, log = apply(adapter, aperture=aperture)
        assert not log["adapter_translation_active"]


def test_delay_is_bounded_and_grasp_loss_does_not_cancel_committed_countdown():
    adapter = Adapter(condition("d15_g100"))
    for _ in range(4):
        apply(adapter)
    for tick in range(15):
        result, log = apply(adapter, grip=1.0, aperture=0.012 if tick == 0 else 0.0)
        assert result[0, -1] == -1
        assert log["adapter_trigger"] == (tick == 0)
    result, _ = apply(adapter, grip=1.0, aperture=0.0)
    assert result[0, -1] == 1
    for _ in range(4):
        apply(adapter)
    _, log = apply(adapter, grip=1.0)
    assert log["adapter_trigger"] and adapter.events == 2


def test_history_ignores_aperture_and_latch_survives_grasp_loss_and_extension():
    history = Adapter(condition("history_only"))
    for _ in range(3):
        apply(history, aperture=0.0)
    assert apply(history, grip=1.0, aperture=0.0)[1]["adapter_trigger"]
    latch = Adapter(condition("grasp_latch"))
    for _ in range(4):
        apply(latch)
    assert apply(latch, grip=1.0)[1]["adapter_trigger"]
    for _ in range(400):
        result, log = apply(latch, grip=1.0, aperture=0.0)
        assert result[0, -1] == -1 and log["adapter_latched"]
    latch.reset()
    assert apply(latch, grip=1.0)[0][0, -1] == 1


def advance(score, success=False, failure=False, boundary=False):
    return score.advance(success=success, physical_failure=failure, boundary=boundary)


@pytest.mark.parametrize("failure,boundary", [(True, False), (False, True), (True, True)])
def test_violation_beats_coincident_success(failure, boundary):
    score = EpisodeScore(True)
    assert advance(score, True, failure, boundary)
    result = score.summary()
    assert not result["safe_success"] and result["violation"] and result["J"] == 15.0


def test_continuation_can_cross_timeout_and_retains_benchmark():
    score = EpisodeScore(True)
    for _ in range(299):
        assert not advance(score)
    assert not advance(score, True)
    for _ in range(39):
        assert not advance(score, True)
    assert advance(score, True)
    result = score.summary()
    assert result["steps"] == 300 and result["controls"] == 340
    assert result["sustained_safe_success"] and result["J"] == 15.0


def test_continuation_loss_is_sticky_and_post_success_failure_is_separate():
    score = EpisodeScore(True)
    advance(score, True)
    advance(score, False)
    assert advance(score, True, True)
    result = score.summary()
    assert result["safe_success"] and not result["violation"]
    assert result["post_success_predicate_loss"] and result["post_success_violation"]
    assert not result["sustained_safe_success"]


def test_timeout_does_not_look_fast_and_stage_b_has_no_continuation():
    score = EpisodeScore(False, horizon=3)
    advance(score)
    advance(score)
    assert advance(score)
    assert score.summary()["J"] == pytest.approx(0.15)
    assert score.summary()["timeout"]
    score = EpisodeScore(False)
    assert advance(score, True)
    assert score.summary()["sustained_safe_success"] is None


def test_snapshots_own_buffers_and_action_mapping():
    obs = {"x": {"y": torch.ones(3)}}
    snapshot = clone_observation(obs)
    obs["x"]["y"].zero_()
    assert snapshot["x"]["y"].sum() == 3
    x = torch.tensor([[2.0, -2.0, 0.5, -0.5, 0.0, 0.1, -1.0]])
    torch.testing.assert_close(
        normalized_to_environment(x),
        torch.tensor([[0.15, -0.15, 0.075, -0.075, 0.0, 0.015, -0.15]]),
    )


def test_schedules_complete_paired_disjoint_and_repeatable():
    a, b = schedule("A"), schedule("B")
    c = schedule("C", ["d00_g100", "d15_g100", "d40_g100", "reference_random"])
    assert b == schedule("B") and len(a) == 10 and len(b) == 192 and len(c) == 256
    assert len({j["id"] for j in b}) == 192
    assert not ({j["seed"] for j in a} & {j["seed"] for j in b})
    assert not ({j["seed"] for j in b} & {j["seed"] for j in c})
    for seed in {j["seed"] for j in b}:
        assert len([j for j in b if j["seed"] == seed]) == 12


def panel(counts):
    rows = []
    for job in schedule("B"):
        cid, seed = job["condition"], job["seed"]
        successes, violations, duration = counts.get(cid, (8, 2, 4.0))
        index = seed - 9200000
        success = index < successes
        rows.append(
            {
                "condition": cid,
                "seed": seed,
                "safe_success": success,
                "violation": successes <= index < successes + violations,
                "J": duration if success else 15.0,
            }
        )
    return rows


def test_selection_negative_and_ablation_gate_do_not_admit_arbitrary_endpoints():
    rows = panel({})
    assert not select_confirmation(rows)["admit_signal"]
    result = select_confirmation(panel({"history_only": (3, 2, 4.0)}))
    assert result["admit_signal"]
    assert result["conditions"][-1] == "history_only" and len(result["conditions"]) == 5
    assert result["primary_contrasts"] == [["history_only", "d15_g100"]]
    with pytest.raises(ValueError):
        select_confirmation(rows[:-1])


def test_safety_screen_and_two_condition_limit():
    result = select_confirmation(
        panel(
            {
                "d15_g050": (14, 0, 3.0),
                "d40_g050": (13, 3, 2.0),
                "history_only": (1, 5, 4.0),
                "grasp_latch": (3, 4, 4.0),
            }
        )
    )
    assert "d15_g050" in result["conditions"]
    assert "d40_g050" not in result["conditions"]
    assert len(result["conditions"]) == 6


def test_performance_ablation_retains_both_required_primary_comparators():
    result = select_confirmation(panel({"grasp_latch": (14, 0, 3.0)}))
    assert ["grasp_latch", "reference_random"] in result["primary_contrasts"]
    assert ["grasp_latch", "d15_g100"] in result["primary_contrasts"]


def test_two_individually_qualifying_endpoints_receive_a_primary_pair_contrast():
    result = select_confirmation(panel({"d15_g050": (13, 0, 2.0), "d40_g075": (14, 0, 4.0)}))
    assert ["d40_g075", "d15_g050"] in result["primary_contrasts"]


def test_task9_finance_is_separate_and_hard_capped(tmp_path):
    book = FinancialBook(tmp_path / "finance.json")
    assert book.limits()["total_usd"] == 10 and book.limits()["aggregate_hours"] == 15
    offer = {
        "dph_total": 0.65,
        "driver_version": "580.126.09",
        "gpu_name": "RTX 3090",
        "num_gpus": 1,
        "cpu_ram": 64000,
        "cpu_cores_effective": 12,
        "disk_space": 200,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    }
    book.reserve("unisafe09", "2026-10-25T10:00:00+00:00", offer)
    book.attach("unisafe09", 99)
    from datetime import datetime

    snapshot = book.snapshot(datetime.fromisoformat("2026-10-25T22:00:00+00:00").timestamp())
    assert not snapshot["stop"]
    snapshot = book.snapshot(datetime.fromisoformat("2026-10-25T22:15:00+00:00").timestamp())
    assert snapshot["stop"]
    cheaper = FinancialBook(tmp_path / "cheaper-finance.json")
    cheaper.reserve("unisafe09-cheaper", "2026-10-25T10:00:00+00:00", {**offer, "dph_total": 0.4})
    cheaper.attach("unisafe09-cheaper", 100)
    assert not cheaper.snapshot(
        datetime.fromisoformat("2026-10-25T22:00:00+00:00").timestamp()
    )["stop"]
    assert cheaper.snapshot(datetime.fromisoformat("2026-10-26T01:00:00+00:00").timestamp())["stop"]
    with pytest.raises(ValueError):
        book.validate_offer({**offer, "dph_total": 0.66})


def test_filter_assesses_scaled_proposal_and_backup_is_not_scaled_again():
    histories, assessed = [], []
    feature = torch.zeros(1, 4)

    def posterior(latent, previous, embed, first, sample):
        histories.append(previous.clone() if previous is not None else None)
        return feature, None

    def uncertainty(value):
        assessed.append(value.clone())
        return torch.tensor([9.0])

    dynamics = SimpleNamespace(
        obs_step=posterior, get_feat=lambda x: x, img_step=lambda *a, **k: feature
    )
    wm = SimpleNamespace(
        preprocess=lambda x: x,
        encoder=lambda x: feature,
        dynamics=dynamics,
        heads={"failure": lambda x: SimpleNamespace(mean=torch.tensor([0.1]))},
    )
    raw = torch.tensor([[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5]])
    agent = SimpleNamespace(
        _wm=wm,
        _task_behavior=SimpleNamespace(actor=lambda x: SimpleNamespace(mode=lambda: raw)),
        _disag_ensemble=SimpleNamespace(intrinsic_reward_penn=uncertainty),
    )
    backup = torch.tensor([[0.6, -0.6, 0.2, 0.3, 0.4, -0.2]])
    safety = SimpleNamespace(
        select_action=lambda *a, **k: (backup, None),
        critic1=lambda *a: torch.tensor([0.5]),
        critic2=lambda *a: torch.tensor([0.5]),
    )
    settings = {
        "uncertainty_threshold": 4.3,
        "reachability_threshold": 0.1,
        "backup_postprocess": "released",
        "action_denom": 1.0,
    }
    controller = FrozenController(agent, safety, settings, Adapter(condition("d00_g050")))
    obs = {"is_first": torch.tensor([False])}
    sensor = {"gripper_pos": torch.tensor([[0.012, -0.012]])}
    first, _ = controller.act(obs, "filtered", adapter_observation=sensor)
    second, row = controller.act(obs, "filtered", adapter_observation=sensor)
    torch.testing.assert_close(assessed[1][:, -7:-4], torch.full((1, 3), 0.2))
    torch.testing.assert_close(second[:, :6], backup_arm_action(backup, "released") * 0.15)
    assert torch.equal(first, second)
    assert row["adapter_translation_active"] and row["intervened"]
    assert torch.equal(histories[2], first / 0.15) and torch.equal(histories[3], first / 0.15)


def test_terminal_capture_suppresses_success_reset_but_stops_on_violation(monkeypatch):
    from helpers.unisafe09_environment import make_environment

    class Base:
        def __init__(self, cfg):
            self.reset_count = 0
            self.physical = False
            self.observation_manager = SimpleNamespace(
                compute=lambda: {
                    "subtask_terms": {
                        "success": torch.tensor([True]),
                        "failure": torch.tensor([self.physical]),
                    }
                }
            )
            self.termination_manager = SimpleNamespace(
                active_terms=[
                    "success",
                    "ee_position",
                    "failure",
                    "time_out",
                    "cube_2_dropping",
                    "cube_3_dropping",
                ],
                get_term=lambda name: torch.tensor(
                    [name == "success" or (name == "failure" and self.physical)]
                ),
            )
            data = SimpleNamespace(root_link_state_w=SimpleNamespace(torch=torch.ones(1, 13)))
            self.scene = {
                k: SimpleNamespace(data=data) for k in ("base", "cube_1", "cube_2", "cube_3")
            }
            self.scene["robot"] = SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos=SimpleNamespace(torch=torch.ones(1, 9)),
                    joint_vel=SimpleNamespace(torch=torch.zeros(1, 9)),
                )
            )

        def _reset_idx(self, ids):
            self.reset_count += 1

        def step(self, action):
            self._reset_idx(torch.tensor([0]))
            return None, None, torch.tensor([True]), torch.tensor([False]), {}

    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "isaaclab.envs", SimpleNamespace(ManagerBasedRLEnv=Base))
    env = make_environment(None, continuation=True)
    result = env.step(None)
    assert env.reset_count == 0 and env.suppressed_resets == 1
    assert result[4]["final_obs"]["subtask_terms"]["success"].item()
    env.physical = True
    result = env.step(None)
    assert env.reset_count == 1 and env.automatic_resets == 1
    assert result[4]["final_obs"]["subtask_terms"]["failure"].item()


def test_incomplete_attempts_reserve_full_control_allowance(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "unisafe09_runner_test", ROOT / "experiments/scripts/run_unisafe09.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import json

    folder = tmp_path / "attempts" / "0000-engineering-00"
    folder.mkdir(parents=True)
    intent = {"maximum_controls": 340, "identity_shadow": True}
    (folder / "intent.json").write_text(json.dumps(intent))
    (folder / "accounting.json").write_text(json.dumps({"completed_controls": 7}))
    row = module.inventory(tmp_path)[0]
    assert not row["valid"] and row["charged_controls"] == 340 and row["controls_uncertain"]
    (folder / "summary.json").write_text(
        json.dumps({"status": "complete", "controls": 50, "identity_checks": 50})
    )
    row = module.inventory(tmp_path)[0]
    assert row["valid"] and row["charged_controls"] == 50 and not row["controls_uncertain"]
