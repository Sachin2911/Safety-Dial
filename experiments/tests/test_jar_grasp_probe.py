import copy
import json
from pathlib import Path

import numpy as np

from experiments.helpers.jar_grasp_probe import GraspController

PLAN = json.loads(Path("configs/jar_grasp_probe_20261029/plan.json").read_text())


def fixture():
    settings = copy.deepcopy(PLAN["controller"])
    settings["reference"]["eef_position_in_jar_frame_m"] = [0.1, 0, 0]
    settings["reference"]["eef_quaternion_in_jar_frame_xyzw"] = [0, 0, 0, 1]
    observation = {
        "valid": True, "eef_pos": [0.1, 0, 0], "eef_quat": [0, 0, 0, 1],
        "jar_pos": [0, 0, 0], "jar_quat": [0, 0, 0, 1], "grasp": -1,
        "ap": {"jar_closed": True, "jar_on_support": True, "jar_upright": True,
               "jar_dropped": False}, "previous_violation": False,
    }
    native = np.array([0.1, -0.3, 0.5, -1, 0.7, 0.9, -0.2, 1], dtype=np.float32)
    return settings, observation, native


def test_native_arm_and_gripper_are_exact_identity():
    settings, observation, native = fixture()
    controller = GraspController(settings, False)
    for step in range(50):
        output, trace = controller.apply(step, native, observation)
        assert np.array_equal(output, native)
        assert output.dtype == native.dtype and not trace["override"]


def test_two_observations_and_exact_forty_control_acquisition():
    settings, observation, native = fixture()
    controller = GraspController(settings, True)
    outputs = [controller.apply(step, native, observation)[0] for step in range(50)]
    assert [i for i, value in enumerate(outputs) if value[7] == -1] == list(range(1, 41))
    assert all(np.array_equal(value[:7], native[:7]) for value in outputs)
    assert controller.mode == "finished"


def test_grasp_confirmation_holds_and_two_losses_end_attempt():
    settings, observation, native = fixture()
    controller = GraspController(settings, True)
    for step in range(2):
        controller.apply(step, native, observation)
    observation["grasp"] = 1
    controller.apply(2, native, observation)
    controller.apply(3, native, observation)
    assert controller.mode == "holding"
    observation["grasp"] = -1
    assert controller.apply(4, native, observation)[0][7] == -1
    assert controller.apply(5, native, observation)[0][7] == 1
    assert controller.mode == "finished"
    for step in range(6, 12):
        assert controller.apply(step, native, observation)[0][7] == 1


def test_invalid_idle_data_reset_streak_and_invalid_active_data_end_attempt():
    settings, observation, native = fixture()
    controller = GraspController(settings, True)
    controller.apply(0, native, observation)
    observation["grasp"] = 0
    assert controller.apply(1, native, observation)[0][7] == 1
    observation["grasp"] = -1
    assert controller.apply(2, native, observation)[0][7] == 1
    assert controller.apply(3, native, observation)[0][7] == -1
    observation["valid"] = False
    assert controller.apply(4, native, observation)[0][7] == 1
    assert controller.mode == "finished"


def test_no_activation_before_closure_or_after_violation():
    settings, observation, native = fixture()
    for key in ("jar_closed", "jar_on_support", "jar_upright"):
        changed = copy.deepcopy(observation)
        changed["ap"][key] = False
        controller = GraspController(settings, True)
        for step in range(4):
            assert controller.apply(step, native, changed)[0][7] == 1
    observation["previous_violation"] = True
    controller = GraspController(settings, True)
    for step in range(4):
        assert controller.apply(step, native, observation)[0][7] == 1


def test_pose_frame_is_invariant_to_world_translation_and_quaternion_sign():
    settings, observation, native = fixture()
    observation["jar_pos"] = [3, 5, 7]
    observation["eef_pos"] = [3.1, 5, 7]
    observation["eef_quat"] = [0, 0, 0, -1]
    controller = GraspController(settings, True)
    controller.apply(0, native, observation)
    assert controller.apply(1, native, observation)[0][7] == -1
