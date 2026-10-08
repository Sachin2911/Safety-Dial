"""Independent selection, fitness-only bounds and exact completed-generation resume."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
from helpers.evoReadinessCMA import PendingGenerationError, ReadinessCMAConfig, run_readiness_cma, selection_pick


def config(mode='zero'):
    return ReadinessCMAConfig(population=6,generations=4,sigma0=.2,seed=43,
        n_fitness_roots=12,fitness_roots_per_generation=4,fitness_samples=2,
        selection_roots=3,selection_samples=2,penalty_mode=mode,checkpoints=(0,1,2,4))


def fitness(X,indices,g):
    full=2-np.square(X-.3).sum(1)[:,None]+np.repeat(indices,2)[None]*.01
    flags=np.broadcast_to(X[:,:1]>.2,full.shape)
    return dict(violated=flags,ret=full,ret_pre=np.where(flags,full*.5,full),rows=X.shape[0]*len(indices)*2*10)


def selection(theta,g):
    # A large offset makes accidental selection leakage into fitness bounds obvious.
    full=np.full((1,6),1000-np.square(theta).sum())
    return dict(violated=np.zeros_like(full,bool),ret=full,ret_pre=full.copy(),rows=60)


@pytest.mark.parametrize('mode',['zero','rosarl_style'])
def test_completed_generation_resume_preserves_selection_and_rng(tmp_path,mode):
    cfg=config(mode)
    kwargs=dict(provenance={'root_hash':'toy-fixed-bank','model_hash':'toy'})
    full=run_readiness_cma(cfg,np.zeros(4),fitness,selection,tmp_path/'full',**kwargs)
    first=run_readiness_cma(cfg,np.zeros(4),fitness,selection,tmp_path/'resume',max_generations_this_call=2,**kwargs)
    assert first['generation']==2
    resumed=run_readiness_cma(cfg,np.zeros(4),fitness,selection,tmp_path/'resume',**kwargs)
    assert full['costs']==resumed['costs']==dict(fitness_rows=1920,selection_rows=300,candidates=24,generations=4)
    assert full['penalty']==resumed['penalty'] and full['penalty']['v_max']<10
    assert [r['asked_sha256'] for r in full['history']]==[r['asked_sha256'] for r in resumed['history']]
    for generation in cfg.checkpoints:
        a,b=full['snapshots'][generation],resumed['snapshots'][generation]
        assert a['nominee_generation']==b['nominee_generation']==0  # selection prefers baseline
        assert np.array_equal(a['theta'],b['theta']) and a['scores']==b['scores']
    with pytest.raises(ValueError):
        run_readiness_cma(replace(cfg,sigma0=.3),np.zeros(4),fitness,selection,tmp_path/'resume',**kwargs)
    with pytest.raises(ValueError):
        run_readiness_cma(cfg,np.zeros(4),fitness,selection,tmp_path/'resume',provenance={'different':'inputs'})


def test_recalculated_penalty_can_restore_a_previously_rejected_nominee():
    def nominee(g,unsafe,reward):
        return dict(generation=g,theta=np.array([g]),metrics=dict(violated=np.array([[unsafe]]),ret=np.array([[reward]]),ret_pre=np.array([[reward]])))
    pool=[nominee(0,False,1.),nominee(1,True,3.)]
    assert selection_pick(pool,0.)['nominee_generation']==1
    assert selection_pick(pool,-4.)['nominee_generation']==0


def test_partial_generation_preserved_and_not_silently_replayed(tmp_path):
    calls=[]
    def crashing(theta,g):
        calls.append(g)
        if g==1:
            raise RuntimeError('selection process interrupted')
        return selection(theta,g)
    cfg=config()
    with pytest.raises(RuntimeError,match='interrupted'):
        run_readiness_cma(cfg,np.zeros(4),fitness,crashing,tmp_path,provenance={'inputs':'fixed'})
    event=json.loads((tmp_path/'pending_generation.json').read_text())
    assert event['phase']=='nominee_selection' and event['completed_rows']==480 and event['requested_rows']==60
    with pytest.raises(PendingGenerationError):
        run_readiness_cma(cfg,np.zeros(4),fitness,crashing,tmp_path,provenance={'inputs':'fixed'})
    assert calls==[0,1]
