"""Check that the release adapter preserves approach and has bounded memory."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_adaptation import GraspReleaseDelay, ReleaseDelay


class ReleaseDelayTests(unittest.TestCase):
    def apply(self, adapter, grip):
        proposal = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, grip]])
        before = proposal.clone()
        changed, log = adapter(proposal, None, None)
        torch.testing.assert_close(proposal, before)
        torch.testing.assert_close(changed[:, :6], before[:, :6])
        return changed[0, -1].item(), log

    def test_delay_preserves_approach_and_expires_without_retrigger(self):
        adapter = ReleaseDelay(hold_steps=3, min_close_steps=2)
        for grip in (1.0, -1.0, 1.0, -1.0, -1.0):
            value, log = self.apply(adapter, grip)
            self.assertEqual(value, grip)
            self.assertFalse(log['adapter_gripper_override'])
        for _ in range(3):
            value, log = self.apply(adapter, 1.0)
            self.assertEqual(value, -1.0)
            self.assertTrue(log['adapter_gripper_override'])
        for _ in range(5):
            self.assertEqual(self.apply(adapter, 1.0)[0], 1.0)
        self.assertEqual(adapter.events, 1)
        adapter.reset()
        self.assertEqual(self.apply(adapter, 1.0)[0], 1.0)
        self.assertEqual(adapter.events, 0)

    def test_grasp_rule_rejects_empty_and_transient_apertures(self):
        adapter = GraspReleaseDelay(hold_steps=3, min_close_steps=3)
        def step(grip, aperture):
            proposal = torch.tensor([[0., 0., 0., 0., 0., 0., grip]])
            obs = {"gripper_pos": torch.tensor([[aperture, -aperture]])}
            return adapter(proposal, obs, None)
        for _ in range(20):
            step(-1., 0.)
        self.assertEqual(step(1., 0.)[0][0, -1].item(), 1.)
        for aperture in [.03, .02, .012, .005, .0]:
            step(-1., aperture)
        self.assertEqual(step(1., .0)[0][0, -1].item(), 1.)
        for _ in range(5):
            step(-1., .012)
        changed, log = step(1., .012)
        self.assertEqual(changed[0, -1].item(), -1.)
        self.assertTrue(log["adapter_trigger"])
        adapter.reset()
        self.assertEqual(step(1., .012)[0][0, -1].item(), 1.)

    def test_zero_delay_is_identity(self):
        adapter = ReleaseDelay(hold_steps=0)
        for grip in [-1.0] * 10 + [0.4] * 10:
            value, log = self.apply(adapter, grip)
            self.assertAlmostEqual(value, grip)
            self.assertFalse(log['adapter_gripper_override'])


if __name__ == '__main__':
    unittest.main()
