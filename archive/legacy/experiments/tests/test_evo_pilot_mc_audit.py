"""Common random numbers must be retained in the offline audit uncertainty."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evo_pilot_mc_audit import paired_mc_precision  # noqa: E402


def test_identical_policies_cancel_shared_noise_exactly():
    flags = np.array([[[0, 1, 0, 1], [1, 1, 0, 0]]])
    result = paired_mc_precision(flags, flags)
    assert result["conditional_monte_carlo_se"] == 0
    assert result["imagined_high_minus_low"] == 0
    assert result["worst_case_monte_carlo_se"] > 0


def test_known_paired_variance_and_independent_cell_scaling():
    high = np.array([[[0, 1, 0, 1]]])
    low = 1 - high
    single = paired_mc_precision(high, low)
    assert single["conditional_monte_carlo_se"] == pytest.approx(np.sqrt(1 / 3))
    repeated = paired_mc_precision(np.tile(high, (2, 3, 1)), np.tile(low, (2, 3, 1)))
    assert repeated["conditional_monte_carlo_se"] == pytest.approx(np.sqrt(1 / 18))


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.full((2, 3, 4), .5),
                                  np.zeros((2, 3, 1)), np.full((2, 3, 4), np.nan)])
def test_invalid_sample_archives_fail(bad):
    with pytest.raises(ValueError):
        paired_mc_precision(bad, bad)


def test_archive_audit_checks_saved_means_and_source_episode_roles(tmp_path):
    import json
    import yaml
    from evo_pilot_mc_audit import audit

    cfg = dict(search_seeds=[31, 32], samples={'audit_k1': 4}, noise_levels=[0., 1.],
               analysis={'low_pressure': {'population': 4, 'generation': 1},
                         'high_pressure': {'population': 8, 'generation': 2}})
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'power_pilot': cfg}))
    roles = dict(indices={'assessment': [7, 11]},
                 source_episodes={'assessment': [100, 200]})
    (tmp_path / 'episode_roles.json').write_text(json.dumps(roles))
    picks, means = [], []
    for seed in cfg['search_seeds']:
        for k in cfg['noise_levels']:
            for endpoint in cfg['analysis'].values():
                slot = len(picks)
                picks.append(dict(seed=seed, noise_k=k, slot=slot, **endpoint))
                readout = np.zeros((2, 4, 10, 3))
                readout[..., 0] = 1.2
                readout[0, 0, 3, 0] = .5
                np.savez(tmp_path / f'audit_{slot:03d}_k1.npz', readout=readout)
                means.append([.25, 0])
    (tmp_path / 'frozen_selections.json').write_text(json.dumps({'picks': picks}))
    paired = dict(assessment_indices=[7, 11], imagined_k1=means)
    np.savez(tmp_path / 'paired_assessment.npz', **paired)
    result = audit(tmp_path)
    assert result['selected_policy_archives_verified'] == 8
    assert result['incremental_costs']['real_steps'] == 0
    assert all(c['conditional_monte_carlo_se'] == 0 for c in result['contrasts'].values())
    paired['imagined_k1'][0] = [0, 0]
    np.savez(tmp_path / 'paired_assessment.npz', **paired)
    with pytest.raises(ValueError, match='sample means disagree'):
        audit(tmp_path)
