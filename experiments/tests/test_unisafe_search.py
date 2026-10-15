"""Checks of optimizer controls and a known categorical policy-gradient direction."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_search import AdapterSearch


class SearchTests(unittest.TestCase):
    def test_common_initial_population_and_elite(self):
        domain = {"hold_steps": [0, 10], "min_close_steps": [1, 3]}
        searches = [AdapterSearch(m, domain, 4, 7) for m in (
            "evolution", "random", "policy_gradient"
        )]
        populations = [s.ask() for s in searches]
        self.assertEqual(populations[0], populations[1])
        self.assertEqual(populations[0], populations[2])
        searches[0].tell(populations[0], [-1, 0, 1, 0])
        self.assertEqual(searches[0].ask()[0], populations[0][2])
        self.assertEqual(len(searches[0].archive), 4)

    def test_policy_gradient_increases_rewarded_choice(self):
        search = AdapterSearch("policy_gradient", {"hold_steps": [0, 10]}, 4, 7)
        before = search.probabilities()[0].copy()
        search.tell([[0], [0], [1], [1]], [0, 0, 1, 1])
        after = search.probabilities()[0]
        self.assertGreater(after[1], before[1])
        self.assertAlmostEqual(after.sum(), 1.0)
        self.assertEqual(search.finalists(1), [{"hold_steps": 10}])
        control = AdapterSearch("random", {"hold_steps": [0, 10]}, 4, 7)
        control.tell([[0], [0], [1], [1]], [0, 0, 1, 1])
        np.testing.assert_array_equal(control.probabilities()[0], before)


if __name__ == '__main__':
    unittest.main()
