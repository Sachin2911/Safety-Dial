"""Guarded CLI and the actual main orchestration, using explicitly synthetic backends."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/scripts'))
import evo_readiness_main as main
import helpers.evoInputs as inputs_module
from helpers.evoReadinessProtocol import protocol_plan
from helpers.evoReadinessSearch import FixedGainResidualPolicy
from helpers.runManifest import file_sha256
from test_evo_readiness_banks import episode
from test_evo_transfer_pilot import policy_fixture

ROOT=Path(__file__).resolve().parents[2]


def test_disabled_main_cannot_create_a_run_or_load_assets(monkeypatch,capsys):
    def forbidden(*args,**kwargs):
        raise AssertionError('disabled main must stop before assets, runs or queries')
    monkeypatch.setattr(main,'load_frozen_runtime',forbidden)
    monkeypatch.setattr(main.EvoRun,'create',forbidden)
    monkeypatch.setattr(sys,'argv',['main','--execute','--run-id','forbidden'])
    with pytest.raises(SystemExit,match='before loading assets'):
        main.main()
    assert json.loads(capsys.readouterr().out)['experiment_run'] is False


def test_actual_main_orchestration_resumes_every_phase_with_synthetic_backends(tmp_path,monkeypatch):
    cfg=yaml.safe_load((ROOT/'configs/evo/stage5_main_candidate.yaml').read_text())
    cfg['bank'].update(fitness_episodes=2,selection_episodes=2,evaluation_episodes=2,max_steps_per_source_episode=130)
    cfg['search'].update(search_seeds=[31,32],population_sizes=[4,6],generations=2,checkpoints=[0,1,2],
        fitness_roots_per_generation=2,fitness_samples={'zero_noise':1,'positive_noise':2},
        selection_samples={'zero_noise':1,'positive_noise':2})
    cfg['evaluation'].update(selected_policy_count_per_run=2,audit_samples={'zero_noise':1,'positive_noise':2})
    cfg['analysis'].update(low_pressure={'population':4,'generation':1},
        high_pressure={'population':6,'generation':2},bootstrap_replicates=30)
    protocol=tmp_path/'synthetic.yaml'
    protocol.write_text(yaml.safe_dump(cfg))
    source=tmp_path/'fixture.txt'
    source.write_text('Synthetic test doubles only. Not scientific approval or experiment data.\n')
    plan=protocol_plan(cfg)
    # The real candidate stays disabled. Only this isolated test replaces its gate
    # together with every model, renderer, actor and physics backend.
    monkeypatch.setattr(main,'check_launch',lambda *_:(plan,[]))
    monkeypatch.setattr(main,'ROOT',tmp_path)
    monkeypatch.setattr(main,'source_inventory',lambda _:{source.name:file_sha256(source)})
    monkeypatch.setattr(inputs_module,'RUNS_ROOT',tmp_path/'runs')
    monkeypatch.setattr(inputs_module,'RESULTS_ROOT',tmp_path/'reports')
    monkeypatch.setattr(main,'input_revisions',lambda _: {'synthetic':True})
    monkeypatch.setattr(main,'build_manifest',lambda **kwargs:kwargs)
    external=dict(source_steps=0,real_steps=0,predictor_rows=0,fingerprints=0,encodes=0)
    def predict(z,a):
        external['predictor_rows']+=len(z)
    model=SimpleNamespace(predict=predict)
    policy=FixedGainResidualPolicy(policy_fixture(),np.ones(17))
    runtime=dict(model=model,scaler=None,probe=None,policy=policy,sigma=np.zeros(192),zero=np.zeros(1020),
        calibration_features=torch.zeros((64,444)),paths={'policies':tmp_path,'data':tmp_path},
        actor_names=['synthetic'],actor_config=SimpleNamespace(action_noise=[0.]),
        provenance={'synthetic_only':True,'calibration_features_sha256':'synthetic'})
    monkeypatch.setattr(main,'load_frozen_runtime',lambda *a,**kw:runtime)
    class Resource:
        action_space=None
        def close(self):
            pass
        def render_many(self,qp,qv):
            return np.zeros((len(qp),2,2,3),np.uint8)
    monkeypatch.setattr(main,'_physics_env',Resource)
    monkeypatch.setattr(main,'RenderContext',Resource)
    monkeypatch.setattr(main,'load_exported_actor',lambda *a,**kw:SimpleNamespace())
    def fingerprint(*args):
        external['fingerprints']+=64
    monkeypatch.setattr(main,'verify_render_fingerprint',fingerprint)
    def source_episode(env,actor,*,seed,max_steps,counter):
        counter['real_steps']+=130
        counter['teacher_queries']+=130
        external['source_steps']+=130
        out=episode(130,max_steps)
        out['qpos'][:,0]+=seed/1e5
        return out
    monkeypatch.setattr(main,'generate_source_episode',source_episode)
    def step(*args,**kwargs):
        external['real_steps']+=1
    monkeypatch.setattr(main.history_real_module,'_step',step)
    monkeypatch.setattr(main.real_module,'_step',step)
    class Imaginer:
        def __init__(self,*args,**kwargs):
            pass
        def encode(self,frames):
            external['encodes']+=len(frames)
            return torch.zeros((len(frames),192))
        def rollout_policies(self,policy,theta,zh,hist,*,n_samples,k,seed,max_batch,n_blocks=10):
            r=len(zh)
            for _ in range(n_blocks):
                model.predict(np.zeros((r*n_samples,3,192)),None)
            readout=np.zeros((1,r,n_samples,n_blocks,3))
            readout[...,0]=1.2
            readout[...,2]=2+float(theta[0,0])
            if theta[0,1]>.005:
                readout[:,0,:,5:,0]=.5
            return dict(readout=readout,root_readout=np.tile([1.2,0.,2.],(r,1)))
    monkeypatch.setattr(main,'HistoryImaginer',Imaginer)
    class Executor:
        def __init__(self,*args,**kwargs):
            self.counters=SimpleNamespace(renders=0,encodes=0)
        def arrays(self,roots,progress,reference=False):
            qp=np.stack([np.tile(r.qpos,(101,1)) for r in roots])
            qv=np.stack([np.tile(r.qvel,(101,1)) for r in roots])
            qp[:,:,0]+=np.linspace(0,progress,101)
            for _ in range(len(roots)*100):
                (main.real_module if reference else main.history_real_module)._step()
            return dict(qpos=qp,qvel=qv,actions=np.zeros((len(roots),100,6) if reference else (len(roots),10,60)),
                progress=qp[:,-1,0]-qp[:,0,0],dense_violated=np.zeros(len(roots),bool))
        def run(self,theta,roots,**kwargs):
            out=self.arrays(roots,1+float(theta[0,0]))
            self.counters.renders+=len(roots)*9
            self.counters.encodes+=len(roots)*9
            out.update(readout=np.tile([1.2,0.,2.],(len(roots),10,1)),
                dense_clearance=np.full((len(roots),100),.4),endpoint_targets=np.tile([1.2,0.,2.],(len(roots),10,1)))
            return out
        def run_tapes(self,roots,tapes):
            out=self.arrays(roots,1.,True)
            return [SimpleNamespace(qpos=q,qvel=v,actions=a) for q,v,a in zip(out['qpos'],out['qvel'],out['actions'])]
        def close(self):
            pass
    monkeypatch.setattr(main,'HistoryRealExecutor',Executor)
    run_id='synthetic-main-test'
    for index,phase in enumerate(['collection','encoding','search','evaluation',None]):
        args=['main','--protocol',str(protocol),'--execute','--run-id',run_id]
        if index:
            args.append('--resume')
        if phase:
            args+=['--stop-after',phase]
        monkeypatch.setattr(sys,'argv',args)
        main.main()
    directory=tmp_path/'runs'/run_id
    costs=json.loads((directory/'costs.json').read_text())
    assert external['source_steps']==780
    assert external['real_steps']==plan['counts']['final_real_steps']
    assert external['predictor_rows']==plan['counts']['total_predictor_rows']
    assert costs['predictor_rows']==external['predictor_rows']
    assert costs['real_steps']==external['real_steps']+780
    assert external['encodes']==18
    assert costs['diagnostic_policy_calls']==sum(s.config.population+1 for s in plan['search_specs'])
    analysis=json.loads((directory/'analysis.json').read_text())
    assert analysis['sampling']['seed_count']==2
    assert analysis['sampling']['independent_episode_count']==2
    before=external.copy()
    main.main()  # Completed queries are reused, with one disclosed startup fingerprint.
    assert external==dict(before,fingerprints=before['fingerprints']+64)
    assert (directory/'analysis/paired_outcomes.npz').exists()
    assert not (directory/'failure.json').exists()
