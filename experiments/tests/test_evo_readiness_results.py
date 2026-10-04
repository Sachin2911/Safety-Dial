"""Final paired analysis consumes complete, unchanged queries in declared root order."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from helpers.evoReadinessAnalysis import analyze_readiness
from helpers.evoReadinessQueries import ImaginedSearchScorer,QueryStore
from helpers.evoReadinessResults import load_study_outcomes,save_paired_outcomes
from helpers.evoReadinessStudy import SearchSpec,evaluate_frozen,run_searches
from test_evo_readiness_study import FakeImaginer,real_output,study_setup


@pytest.fixture
def completed(tmp_path):
    banks,old_specs,evaluation=study_setup()
    specs=[SearchSpec(s.noise_k,replace(s.config,population=p,penalty_mode=m))
        for s in old_specs for p in [4,6] for m in ['zero','rosarl_style']]
    counts=dict(predictor_rows=0,real_steps=0)
    im=FakeImaginer(counts)
    def factory(spec,folder,identity):
        return ImaginedSearchScorer(im,None,banks['fitness'],banks['selection'],
            config=spec.config,noise_k=spec.noise_k,
            store=QueryStore(folder/'queries',study_identity=identity),read_costs=lambda:counts)
    done=run_searches(tmp_path,specs,banks,np.zeros(2),provenance={'synthetic_only':True},
        evaluation_plan=evaluation,scorer_factory=factory)
    def output(progress,reference=False):
        out=real_output(progress,reference)
        out['qpos'][:,:,0]+=np.arange(4)[:,None]*10
        out['progress']=out['qpos'][:,-1,0]-out['qpos'][:,0,0]
        # One failure between images, another at a block end.
        out['qpos'][0,5,1]=.5
        out['qpos'][1,10,1]=.5
        out['dense_violated'][:2]=True
        if not reference:
            out['readout'][...,0]=1.2
            out['readout'][1,0,0]=.5
            out['pairs']=np.column_stack([np.zeros(4,int),np.arange(4)])
        return out
    def real(theta):
        counts['real_steps']+=400
        return output(1+float(theta[0]))
    def reference():
        counts['real_steps']+=400
        return output(1,True)
    def audit(theta,k,n,seed):
        counts['predictor_rows']+=4*n*10
        ro=np.zeros((4,n,10,3))
        ro[...,0]=1.2
        ro[1,0,0,0]=.5
        return dict(readout=ro)
    evaluate_frozen(tmp_path,identity=done['identity'],evaluation_bank=banks['evaluation'],
        real_callback=real,reference_callback=reference,audit_callback=audit,
        read_costs=lambda:counts,**evaluation)
    baseline=output(1)
    roots=[SimpleNamespace(qpos=baseline['qpos'][i,0],qvel=baseline['qvel'][i,0]) for i in range(4)]
    return tmp_path,roots


def test_complete_grid_loads_into_declared_analysis_without_new_queries(completed):
    path,roots=completed
    outcomes=load_study_outcomes(path,expected_roots=roots)
    assert outcomes['search_seeds']==[31,32]
    assert len(outcomes['cells'])==16
    for key,cell in outcomes['cells'].items():
        assert np.array_equal(cell['real'],[[1,1,0,0]]*2)
        assert np.array_equal(cell['imagined'][0.],[[0,1,0,0]]*2)
        assert np.array_equal(cell['imagined'][1.],[[0,.5,0,0]]*2)
        assert np.array_equal(outcomes['diagnostics'][key]['endpoint_truth'],[[0,1,0,0]]*2)
        assert np.array_equal(outcomes['diagnostics'][key]['real_readout'],[[0,1,0,0]]*2)
    analysis=analyze_readiness(outcomes['cells'],outcomes['baseline'],search_seeds=outcomes['search_seeds'],
        episode_keys=outcomes['episode_keys'],population_sizes=[4,6],checkpoints=[1,2],
        low_pressure=(4,1),high_pressure=(6,2),replicates=50)
    assert analysis['sampling']['independent_episode_count']==4
    save_paired_outcomes(path/'export',outcomes)
    with np.load(path/'export/paired_outcomes.npz') as data:
        assert data['real'].shape==(16,2,4)
        assert data['baseline_progress'].shape==(2,4)
    axes=json.loads((path/'export/paired_axes.json').read_text())
    assert axes['episode_keys']==list(outcomes['episode_keys'])


@pytest.mark.parametrize('mutation',['numeric','request','extra','pending','incomplete','summary_cost'])
def test_incomplete_changed_or_undeclared_evaluation_is_rejected(completed,mutation):
    path,roots=completed
    query=path/'evaluation/baseline-real.npz'
    if mutation in ['numeric','request']:
        with np.load(query) as f:
            arrays={k:f[k].copy() for k in f.files}
        if mutation=='numeric':
            arrays['qpos'][0,1,0]+=1
        else:
            receipt=json.loads(str(arrays['__receipt__']))
            receipt['request']['kind']='different'
            arrays['__receipt__']=json.dumps(receipt)
        np.savez_compressed(query,**arrays)
    elif mutation=='extra':
        np.savez(path/'evaluation/undeclared.npz',x=[1])
    elif mutation=='pending':
        (path/'evaluation/partial.pending.json').write_text('{}')
    else:
        p=path/'evaluation_progress.json'
        data=json.loads(p.read_text())
        if mutation=='incomplete':
            data['complete']=False
        else:
            data['accounting']['completed_costs']['real_steps']+=1
        p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_study_outcomes(path,expected_roots=roots)


def test_initial_physics_must_match_the_declared_root_order(completed):
    path,roots=completed
    with pytest.raises(ValueError,match='initial physics states'):
        load_study_outcomes(path,expected_roots=roots[::-1])
