"""Whole-episode independence, charged rejections, exact resume and failure retention."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import helpers.evoReadinessBanks as banks_module
from helpers.evoReadinessBanks import CollectionRole, SourceEpisodeFailure, collect_source_banks, load_collected_roots
from helpers.evoReadinessQueries import IncompleteQueryError


def episode(n,cap):
    qp=np.zeros((n+1,9))
    qp[:,0]=np.arange(n+1)*.02
    qp[:,1]=1.2
    if n<cap:
        qp[-1,1]=.5
    return dict(qpos=qp,qvel=np.zeros_like(qp),action=np.zeros((n,6),np.float32),
                x_velocity=np.full(n,2.5))


def specs():
    return [CollectionRole(name,2,(i+1)*100,(i+1)*1000,attempts_per_root=2,max_steps=130)
            for i,name in enumerate(['fitness','selection','evaluation'])]


def arguments(counter,callback):
    return dict(actor_names=['teacher_a','teacher_b'],action_noise=[0.,.05,.1],
        provenance={'source':'fixed-for-test'},episode_callback=callback,counter=counter)


def test_rejected_episodes_charged_and_role_banks_resume_exactly(tmp_path):
    outcomes={}
    for mode in ['full','resumed']:
        counts=dict(real_steps=0,teacher_queries=0,predictor_rows=0)
        calls=[]
        def collect(request,counter):
            calls.append(request)
            n=120 if request['role']=='fitness' and request['attempt']==0 else 130
            counter['real_steps']+=n
            counter['teacher_queries']+=n
            return episode(n,130)
        path=tmp_path/mode
        args=arguments(counts,collect)
        if mode=='resumed':
            pause=collect_source_banks(path,specs(),**args,max_new_attempts=2)
            assert not pause['complete'] and pause['current_accepted']==1
            assert counts['real_steps']==250
        result=collect_source_banks(path,specs(),**args)
        assert result['complete'] and len(calls)==7
        assert counts['real_steps']==900
        assert result['accounting']['completed_costs']['real_steps']==900
        assert sum(not r['eligible'] for r in result['attempts'])==1
        bank_info,roots=load_collected_roots(path)
        keys=[r.episode for rows in roots.values() for r in rows]
        assert len(set(keys))==6
        assert all(20<=r.step<30 for rows in roots.values() for r in rows)
        assert all(len(r.policy_tape)==100 for rows in roots.values() for r in rows)
        assert all(r.meta['role']==role for role,rows in roots.items() for r in rows)
        outcomes[mode]=bank_info['roles']
        collect_source_banks(path,specs(),**args)
        assert len(calls)==7 and counts['real_steps']==900
    assert outcomes['full']==outcomes['resumed']


def test_source_attempt_cap_does_not_expand_or_repeat_completed_physics(tmp_path):
    counts=dict(real_steps=0,teacher_queries=0,predictor_rows=0)
    def short(request,counter):
        counter['real_steps']+=120
        counter['teacher_queries']+=120
        return episode(120,130)
    for _ in range(2):
        with pytest.raises(RuntimeError,match='attempt cap exhausted'):
            collect_source_banks(tmp_path,specs(),**arguments(counts,short))
    assert counts['real_steps']==480
    failure=json.loads((tmp_path/'collection_failed.json').read_text())
    assert failure['accepted']==0 and len(failure['attempts'])==4
    assert failure['accounting']['completed_costs']['real_steps']==480


def test_invalid_snapshot_retains_trajectory_and_cost_and_refuses_replay(tmp_path):
    counts=dict(real_steps=0,teacher_queries=0,predictor_rows=0)
    def invalid(request,counter):
        counter['real_steps']+=130
        counter['teacher_queries']+=130
        out=episode(130,130)
        out['qpos'][20,1]=np.nan
        return out
    with pytest.raises(ValueError,match='nonfinite'):
        collect_source_banks(tmp_path,specs(),**arguments(counts,invalid))
    assert (tmp_path/'failed-fitness-attempt00000.npz').exists()
    journal=json.loads((tmp_path/'episodes/fitness-attempt00000.pending.json').read_text())
    assert journal['measured_costs']['real_steps']==130
    with pytest.raises(IncompleteQueryError):
        collect_source_banks(tmp_path,specs(),**arguments(counts,invalid))
    assert counts['real_steps']==130


def test_source_generator_retains_only_completed_prefix_after_step_failure(monkeypatch):
    initial=np.zeros(9)
    initial[1]=1.2
    u=SimpleNamespace(data=SimpleNamespace(qpos=initial,qvel=np.zeros(9)),_get_obs=lambda:np.zeros(17))
    env=SimpleNamespace(unwrapped=u,reset=lambda seed:None)
    actor=SimpleNamespace(reset=lambda:None,act=lambda obs:np.zeros(6))
    def step(u,action,qp,qv,xv,t):
        if t==2:
            raise FloatingPointError('injected physics failure')
        qp[t+1],qv[t+1],xv[t]=initial,np.zeros(9),0.
    monkeypatch.setattr(banks_module,'_step',step)
    counts=dict(real_steps=0,teacher_queries=0,predictor_rows=0)
    with pytest.raises(SourceEpisodeFailure) as error:
        banks_module.generate_source_episode(env,actor,seed=4,max_steps=130,counter=counts)
    assert counts['real_steps']==3 and counts['teacher_queries']==3
    assert error.value.episode_prefix['action'].shape==(2,6)
    assert error.value.episode_prefix['qpos'].shape==(3,9)


def test_seed_overlap_rejected_before_source_queries(tmp_path):
    roles=specs()
    roles[1]=CollectionRole('selection',2,100,2000,max_steps=130)
    def forbidden(*args):
        raise AssertionError('collection must not run')
    with pytest.raises(ValueError,match='disjoint'):
        collect_source_banks(tmp_path,roles,**arguments(dict(real_steps=0,teacher_queries=0,predictor_rows=0),forbidden))


def test_initial_history_encoding_resumes_without_controller_outcomes(tmp_path):
    from helpers.evoReadinessBanks import encode_collected_banks,load_encoded_banks
    counter=dict(real_steps=0,teacher_queries=0,predictor_rows=0,renders=0,encodes=0)
    def collect(request,c):
        c['real_steps']+=130
        c['teacher_queries']+=130
        return episode(130,130)
    collect_source_banks(tmp_path,specs(),**arguments(counter,collect))
    calls=[]
    def encode(root):
        calls.append(root.root_id)
        counter['renders']+=3
        counter['encodes']+=3
        return np.full((3,192),float(root.episode),np.float32)
    kw=dict(encoder_provenance={'model':'frozen'},encode_callback=encode,read_costs=lambda:counter)
    part=encode_collected_banks(tmp_path,**kw,max_new_roots=2)
    assert not part['complete'] and len(calls)==2
    full=encode_collected_banks(tmp_path,**kw)
    assert full['complete'] and len(calls)==6
    assert counter['renders']==counter['encodes']==18
    assert counter['real_steps']==780 and counter['predictor_rows']==0
    encode_collected_banks(tmp_path,**kw)
    assert len(calls)==6
    encoded,roots,banks=load_encoded_banks(tmp_path)
    assert set(banks)=={'fitness','selection','evaluation'}
    for role,bank in banks.items():
        assert np.array_equal(bank.z_hist[:,0,0],[r.episode for r in roots[role]])
    with pytest.raises(ValueError,match='encoder declaration changed'):
        encode_collected_banks(tmp_path,**{**kw,'encoder_provenance':{'model':'changed'}})
    assert len(calls)==6
