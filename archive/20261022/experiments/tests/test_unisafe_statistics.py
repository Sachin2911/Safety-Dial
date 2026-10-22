"""Known paired binary cases and nondegenerate uncertainty at perfect performance."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_statistics import holm, paired_binary


class PairedStatistics(unittest.TestCase):
    def test_four_discordant_pairs_have_known_exact_p(self):
        result = paired_binary([1, 1, 1, 1], [0, 0, 0, 0])
        self.assertEqual(result['mcnemar_exact_two_sided_p'], 0.125)
        self.assertEqual(result['rate_difference'], 1.)
        self.assertLess(result['conservative_exact_difference95'][0], 1.)

    def test_equal_pairs_do_not_imply_zero_uncertainty(self):
        result = paired_binary([1, 0] * 32, [1, 0] * 32)
        self.assertEqual(result['mcnemar_exact_two_sided_p'], 1.)
        low, high = result['conservative_exact_difference95']
        self.assertLess(low, 0.)
        self.assertGreater(high, 0.)
        self.assertEqual(holm({'a': .01, 'b': .04, 'c': .05}), {'a': .03, 'b': .08, 'c': .08})


if __name__ == '__main__':
    unittest.main()
