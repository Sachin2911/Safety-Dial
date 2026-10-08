"""User-authorized development work cannot masquerade as supervisor approval or final data."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evo_readiness_variance_pilot import pilot_plan
from helpers.evoReadinessProtocol import scientific_decision_blockers

ROOT = Path(__file__).resolve().parents[2]


def test_declared_pilot_has_exact_costs_and_no_collection():
    config = yaml.safe_load((ROOT / 'configs/evo/stage5_variance_pilot.yaml').read_text())
    plan = pilot_plan(config)
    assert plan['counts']['search_runs'] == 32
    assert plan['counts']['selected_policy_slots'] == 64
    assert plan['counts']['final_real_steps'] == 633600
    assert plan['counts']['total_predictor_rows'] == 55705600 + 2959360 + 2154240
    assert plan['counts']['new_source_roots'] == 0
    assert plan['counts']['maximum_new_real_steps'] == 633600
    assert config['approval']['supervisor_scientific_decision_recorded'] is False
    for change in ['final', 'decision', 'cap', 'grid']:
        bad = copy.deepcopy(config)
        if change == 'final':
            bad['variance_pilot']['new_source_episodes'] = 1
        elif change == 'decision':
            bad['approval']['user_decision_evidence']['sha256'] = '0' * 64
        elif change == 'cap':
            bad['budget']['maximum_GPU_hours_proposed'] = 3
        else:
            bad['search']['population_sizes'] = [16, 64]
        with pytest.raises(ValueError):
            pilot_plan(bad)


def test_user_decision_requires_hashed_evidence_and_full_scope(tmp_path):
    p = tmp_path / 'decision.json'
    record = dict(kind='explicit_user_authorization', readiness_extension_accepted=True,
                  matched_termination_rosarl_adaptation_accepted=True,
                  supervisor_approval_claim=False, user_messages=['synthetic test fixture only'])
    p.write_text(json.dumps(record))
    approval = dict(scientific_decision_authority='user', user_scientific_decision_recorded=True,
                    supervisor_scientific_decision_recorded=False,
                    user_decision_evidence={'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    assert scientific_decision_blockers(approval, tmp_path) == []
    record['matched_termination_rosarl_adaptation_accepted'] = False
    p.write_text(json.dumps(record))
    assert scientific_decision_blockers(approval, tmp_path)
    approval['user_decision_evidence']['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert scientific_decision_blockers(approval, tmp_path)


def test_unknown_authority_does_not_disable_review_gate():
    assert scientific_decision_blockers({'scientific_decision_authority':'automatic'}, ROOT)
    assert scientific_decision_blockers({}, ROOT)
