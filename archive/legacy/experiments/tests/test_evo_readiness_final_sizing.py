"""The pre-data sizing review cannot silently exceed the approved wall cap."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/scripts'))
from evo_readiness_final_sizing import review  # noqa: E402
from helpers.evoReadinessProtocol import protocol_plan  # noqa: E402
from helpers.evoReadinessQueries import digest_json  # noqa: E402


@pytest.fixture
def pilot(tmp_path):
    for name in ['configs/evo/stage5_main_candidate.yaml',
                 'docs/evoPlan/results/stage5-independent-audit/sizing_review.json']:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    p = tmp_path / 'runs/development'
    p.mkdir(parents=True)
    cfg = yaml.safe_load((ROOT / 'configs/evo/stage5_variance_pilot.yaml').read_text())
    (p / 'config.yaml').write_text(yaml.safe_dump(cfg))
    counts = protocol_plan(cfg)['counts']
    costs = dict(predictor_rows=counts['total_predictor_rows'], real_steps=counts['final_real_steps'], wall_s=1900)
    values = dict(check=dict(passed=True, main_final_bank_touched=False, development_only=True,
        all_selections_frozen_before_assessment=True, costs=costs),
        manifest=dict(kind='evo-readiness-variance-pilot-v1', costs=costs, source_sha256={}),
        launch=dict(started_at=100),
        variance=dict(first=dict(variance_components=dict(seed=.01, episode=.1, interaction=.3))))
    for name, value in values.items():
        (p / f'{name}.json').write_text(json.dumps(value))
    for search in range(32):
        folder = p / f'study/searches/search{search}/queries'
        folder.mkdir(parents=True)
        for query in range(33):
            r = dict(completed_at=1900)
            r['receipt_sha256'] = digest_json(r)
            np.savez(folder / f'{query}.npz', __receipt__=json.dumps(r))
    return tmp_path, p


def test_recommends_largest_feasible_grid_without_claiming_guaranteed_power(pilot):
    root, p = pilot
    r = review(p, root)
    assert r['recommended_search_seeds'] == 20
    assert r['evaluation_episodes'] == 320
    scenarios = r['scenarios'][1]['precision']['first']['sensitivity']
    assert scenarios[0]['approximate_two_sided_power'] > scenarios[-1]['approximate_two_sided_power']
    assert r['new_real_steps'] == r['new_predictor_rows'] == 0
    check = json.loads((p / 'check.json').read_text())
    manifest = json.loads((p / 'manifest.json').read_text())
    check['costs']['wall_s'] = manifest['costs']['wall_s'] = 3600
    (p / 'check.json').write_text(json.dumps(check))
    (p / 'manifest.json').write_text(json.dumps(manifest))
    assert review(p, root)['recommended_search_seeds'] is None


def test_rejects_final_data_or_tampered_timing(pilot):
    root, p = pilot
    path = next((p / 'study/searches').glob('*/queries/*.npz'))
    np.savez(path, __receipt__=json.dumps(dict(completed_at=0, receipt_sha256='invalid')))
    with pytest.raises(ValueError, match='receipt checksum'):
        review(p, root)
    check = json.loads((p / 'check.json').read_text())
    check['main_final_bank_touched'] = True
    (p / 'check.json').write_text(json.dumps(check))
    with pytest.raises(ValueError, match='development-only'):
        review(p, root)
