"""Prevent an unexplained policy change from being labelled an adapter effect."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from unisafe_adaptation_evaluate import matched_prefix


def row():
    return {
        'before': {'position': 0}, 'after': {'position': 1}, 'camera_sha256': 'same-input',
        'task_proposed_action': [0.1, 1.], 'proposed_action': [0.1, 1.],
        'executed_normalized_action': [0.1, 1.], 'backup_candidate_action': [0.8, 1.],
        'adapter_gripper_override': False, 'would_intervene': False, 'intervened': False,
    }


class PrefixAttribution(unittest.TestCase):
    def compare(self, left, right, modes=('filtered', 'filtered')):
        # Synthetic trace fixtures only; no simulator interactions or model queries.
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, (value, mode) in enumerate(zip((left, right), modes)):
                path = Path(directory) / str(i)
                path.mkdir()
                np.savez(path / 'initial_state.npz', initial=np.zeros(1))
                (path / 'steps.jsonl').write_text(json.dumps(value) + '\n')
                (path / 'summary.json').write_text(json.dumps({'mode': mode}))
                paths.append(path)
            return matched_prefix(*paths)

    def test_unexplained_task_difference_is_not_an_adapter_effect(self):
        a = row()
        b = copy.deepcopy(a)
        b['task_proposed_action'][0] = b['proposed_action'][0] = b['executed_normalized_action'][0] = .2
        result = self.compare(a, b)
        self.assertTrue(result['observed_prefix_equal'])
        self.assertFalse(result['pass'])
        self.assertFalse(result['first_difference_has_declared_controller_cause'])

    def test_declared_adapter_change_has_matching_original_proposal(self):
        a = row()
        b = copy.deepcopy(a)
        b['proposed_action'][-1] = b['executed_normalized_action'][-1] = -1.
        b['adapter_gripper_override'] = True
        result = self.compare(a, b)
        self.assertTrue(result['pass'])
        self.assertEqual(result['first_difference_cause'], 'declared task-adapter proposal')

    def test_filter_mode_change_requires_matching_shadow_assessment(self):
        a = row()
        b = copy.deepcopy(a)
        a['would_intervene'] = b['would_intervene'] = True
        a['intervened'] = True
        a['executed_normalized_action'] = list(a['backup_candidate_action'])
        result = self.compare(a, b, ('filtered', 'unfiltered'))
        self.assertTrue(result['pass'])
        self.assertEqual(result['first_difference_cause'], 'declared filter-mode difference')
        b['backup_candidate_action'][0] = .7
        self.assertFalse(self.compare(a, b, ('filtered', 'unfiltered'))['pass'])


if __name__ == '__main__':
    unittest.main()
