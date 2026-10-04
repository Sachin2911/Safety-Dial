"""Frozen local assets and action diagnostics shared by main execution and benchmarking."""
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from helpers.evoInputs import fetch_inputs,input_revisions,load_stage_config
from helpers.evoReadinessSearch import FixedGainResidualPolicy
from helpers.evoResidualPolicy import ResidualHistoryPolicy
from helpers.poseProbes import load_probe
from helpers.runManifest import array_sha256,file_sha256
from helpers.walkerLewm import load_walker_model


def load_frozen_runtime(config,repository,*,require_archived=True):
    root=Path(repository)
    inputs,impl=config['inputs'],config['implementation']
    source_path=root/impl['source_hashes']
    if file_sha256(source_path)!=impl['source_hashes_sha256']:
        raise ValueError('cached source inventory changed')
    sources=json.loads(source_path.read_text())
    for run_id,files in sources.items():
        for name,digest in files.items():
            if file_sha256(root/'runs'/run_id/name)!=digest:
                raise ValueError(f'frozen source changed: {run_id}/{name}')
    controller=root/'runs'/inputs['controller_run']/'selected_policy.npz'
    if file_sha256(controller)!=inputs['controller_sha256']:
        raise ValueError('frozen ensemble weights changed')
    revisions={inputs['controller_run']:inputs['controller_revision'],
        inputs['projection_run']:inputs['projection_revision'],
        impl['baseline_history_run']:impl['baseline_history_revision'],
        inputs['noise_run']:inputs['noise_archive_revision']}
    if require_archived:
        for run_id,revision in revisions.items():
            receipt=json.loads((root/'runs'/run_id/'hf_upload.json').read_text())
            if not revision or receipt['revision']!=revision:
                raise ValueError(f'private archive revision mismatch: {run_id}')
    actor_cfg_path=root/'configs/evo'/f'{impl["actor_pool_config"]}.yaml'
    if file_sha256(actor_cfg_path)!=impl['actor_pool_config_sha256']:
        raise ValueError('source actor pool configuration changed')
    actor_cfg=load_stage_config(impl['actor_pool_config']).fresh_bank
    source_cfg=root/inputs['frozen_model_probe_data_config']
    if file_sha256(source_cfg)!=inputs['frozen_model_probe_data_config_sha256']:
        raise ValueError('model/probe input configuration changed')
    paths=fetch_inputs(OmegaConf.load(source_cfg).inputs,names=['model','probes','data','policies'])
    manifest=paths['data']/'manifest.json'
    if file_sha256(manifest)!=actor_cfg.actor_pool_manifest_sha256:
        raise ValueError('historical actor-pool manifest changed')
    old=json.loads(manifest.read_text())['data']['policies']
    names=[row['name'] for row in old if row['len_mean']>=400 and row['return_mean']>500]
    if names!=list(actor_cfg.actor_names):
        raise ValueError('source actor mixture differs from readiness')
    actor_hashes={name:file_sha256(paths['policies']/f'{name}.pt') for name in names}
    torch.set_num_threads(int(impl['torch_threads']))
    residual_path=root/'runs'/inputs['projection_run']/'residual_policies.npz'
    if file_sha256(residual_path)!=inputs['projection_bundle_file_sha256']:
        raise ValueError('residual projection archive changed')
    with np.load(residual_path,allow_pickle=False) as f:
        residual=ResidualHistoryPolicy.from_state({k:f[k] for k in f.files},device='cuda')
    with np.load(controller,allow_pickle=False) as f:
        if not np.array_equal(residual.base_theta.cpu().numpy(),f['theta']):
            raise ValueError('embedded residual controller differs from archived ensemble')
        for key,value in residual.base.state().items():
            if key not in f or not np.array_equal(np.asarray(value),f[key]):
                raise ValueError(f'embedded ensemble interface differs: {key}')
    baseline=root/'runs'/impl['baseline_history_run']
    with np.load(baseline/'history_latents.npz',allow_pickle=False) as f:
        zh,hist=f['z_hist'],f['hist_blocks']
    calibration_path=root/'runs'/inputs['noise_run']/inputs['noise_file']
    if file_sha256(calibration_path)!=inputs['noise_file_sha256']:
        raise ValueError('frozen noise calibration changed')
    with np.load(calibration_path,allow_pickle=False) as f:
        sigma,indices=f['sigma'],f['fit_indices']
    with torch.inference_mode():
        features=residual.features(torch.as_tensor(zh[indices,-1],device='cuda'),
            torch.as_tensor(zh[indices,-2],device='cuda'),hist[indices,-1].reshape(-1,60))
        rms=residual.residual_features(features).double().square().mean(0).sqrt().clamp_min(
            float(config['policy']['coordinate_RMS_floor'])).float()
    policy=FixedGainResidualPolicy(residual,rms)
    if policy.n_params!=1020:
        raise ValueError('wrong residual dimension')
    model,scaler=load_walker_model(paths['model'],'cuda')
    probe,_=load_probe(paths['probes']/'walker_mlp.pt','cuda')
    provenance=dict(cached_sources=sources,controller_sha256=inputs['controller_sha256'],
        archive_revisions=revisions,upstream=input_revisions(paths),actor_sha256=actor_hashes,
        preconditioner_sha256=array_sha256(rms.cpu().numpy()),
        calibration_features_sha256=array_sha256(features.cpu().numpy()),
        torch_version=str(torch.__version__),torch_cuda=torch.version.cuda,
        cuda_device=torch.cuda.get_device_name(0))
    return dict(model=model,scaler=scaler,probe=probe,policy=policy,sigma=sigma,
        zero=np.zeros(1020,np.float64),calibration_features=features,feature_rms=rms,
        paths=paths,actor_config=actor_cfg,actor_names=names,provenance=provenance)


def initial_residual_diagnostics(policy,features,theta,*,on_policy_call=lambda:None):
    """Initial action deviations on the frozen 64-root calibration features only."""
    raw_rms,clipped_rms,clipped_fraction=[],[],[]
    with torch.inference_mode():
        on_policy_call()
        baseline=policy.act(np.zeros((1,policy.n_params)),features[None])
        for candidate in np.asarray(theta):
            raw=policy.residual_policy.residual(policy.raw_theta(candidate[None]),features[None])
            on_policy_call()
            action=policy.act(candidate[None],features[None])
            raw_rms.append(float(raw.double().square().mean().sqrt()))
            clipped_rms.append(float((action-baseline).double().square().mean().sqrt()))
            clipped_fraction.append(float(((baseline+raw).abs()>1).double().mean()))
    return dict(raw_residual_rms=np.asarray(raw_rms),clipped_action_delta_rms=np.asarray(clipped_rms),
                clipped_action_fraction=np.asarray(clipped_fraction))
