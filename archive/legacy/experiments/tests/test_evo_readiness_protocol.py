"""The review draft is query-free and exact budgets follow the declared factorial grid."""
import copy
import hashlib
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
from helpers.evoReadinessProtocol import launch_blockers,protocol_plan

ROOT=Path(__file__).resolve().parents[2]


def draft():
    return yaml.safe_load((ROOT/'configs/evo/stage5_readiness_draft.yaml').read_text())


def test_draft_counts_match_independent_declared_accounting_and_remain_disabled():
    cfg=draft()
    plan=protocol_plan(cfg)
    assert len(plan['search_specs'])==120
    assert plan['counts']['selected_policy_slots']==360
    assert plan['counts']['total_predictor_rows']==222201600
    assert plan['counts']['maximum_new_real_steps']==13504000
    assert plan['counts']['final_real_steps']==11584000
    assert plan['batch_shapes']=={'fitness':128,'selection':1024,'evaluation':10240}
    blockers=launch_blockers(cfg,ROOT)
    assert any('disabled review draft' in s for s in blockers)
    assert any('supervisor decision' in s for s in blockers)
    assert any('provisional' in s for s in blockers)
    assert any('no pinned private archive' in s for s in blockers)


def test_wrong_grid_batches_or_overlapping_episode_ranges_fail_before_execution():
    cfg=draft()
    missing=copy.deepcopy(cfg)
    missing['search']['penalty_modes']=['rosarl_style']
    with pytest.raises(ValueError,match='both penalty modes'):
        protocol_plan(missing)
    huge=copy.deepcopy(cfg)
    huge['bank']['evaluation_episodes']=1000
    with pytest.raises(ValueError,match='batch shapes'):
        protocol_plan(huge)
    overlap=copy.deepcopy(cfg)
    overlap['bank']['environment_seed_bases']['selection']=overlap['bank']['environment_seed_bases']['fitness']
    with pytest.raises(ValueError,match='disjoint'):
        protocol_plan(overlap)


def test_completed_check_flags_need_unchanged_evidence(tmp_path):
    cfg=draft()
    cfg['status']='locked_for_execution'
    cfg['execution_enabled']=True
    cfg['approval']['supervisor_scientific_decision_recorded']=True
    cfg['bank']['evaluation_count_is_provisional_pending_power_review']=False
    cfg['inputs']['noise_archive_revision']='a'*40
    # This synthetic fixture is only a parser test, not a scientific approval record.
    proof=tmp_path/'synthetic-proof.txt'
    proof.write_text('test fixture only')
    digest=hashlib.sha256(proof.read_bytes()).hexdigest()
    cfg['inputs']['frozen_model_probe_data_config']=proof.name
    cfg['inputs']['frozen_model_probe_data_config_sha256']=digest
    checks=cfg['approval']['required_before_execution']
    cfg['approval']['completed_checks']={name:True for name in checks}
    cfg['approval']['evidence']={name:{'path':proof.name,'sha256':digest} for name in checks}
    assert launch_blockers(cfg,tmp_path)==[]
    proof.write_text('changed fixture')
    assert len(launch_blockers(cfg,tmp_path))==len(checks)+1
