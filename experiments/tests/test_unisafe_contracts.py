"""Regression checks for action coordinates and simulator buffer ownership."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_controller import backup_arm_action
from helpers.unisafe_environment import clone_observation, normalized_to_environment


class Contracts(unittest.TestCase):
    def test_terminal_observations_survive_buffer_reuse(self):
        live = {
            "rgb_camera": {"wrist_cam": torch.ones(1, 2, 2, 3)},
            "policy": {"eef_pos": torch.tensor([[0.5, 0.0, 0.2]])},
        }
        snapshot = clone_observation(live)
        live["rgb_camera"]["wrist_cam"].zero_()
        live["policy"]["eef_pos"].fill_(9)
        self.assertEqual(snapshot["rgb_camera"]["wrist_cam"].sum(), 12)
        torch.testing.assert_close(snapshot["policy"]["eef_pos"], torch.tensor([[0.5, 0.0, 0.2]]))

    def test_released_sixth_coordinate_quirk_is_explicit(self):
        raw = torch.tensor([[0.1, -0.1, 0.2, 0.3, 0.7, -0.02]])
        original = raw.clone()
        released = backup_arm_action(raw, "released")
        corrected = backup_arm_action(raw, "arm_clip")
        self.assertEqual(released[0, 5].item(), -1.0)
        self.assertAlmostEqual(corrected[0, 5].item(), -0.02)
        torch.testing.assert_close(raw, original)
        torch.testing.assert_close(released[:, :5], corrected[:, :5])

    def test_normalized_to_environment_preserves_coordinates_and_clips(self):
        action = torch.tensor([[2.0, -2.0, 0.5, -0.5, 0.0, 0.1, -1.0]])
        torch.testing.assert_close(
            normalized_to_environment(action),
            torch.tensor([[0.15, -0.15, 0.075, -0.075, 0.0, 0.015, -0.15]]),
        )


if __name__ == "__main__":
    unittest.main()
