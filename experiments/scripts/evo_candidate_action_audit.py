#!/usr/bin/env python3
"""Recompute executed corrections from saved visual histories, without simulator queries."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
import torch
import yaml

from helpers.evoCandidateQuality import ActionLimitedPolicy
from helpers.evoFollowup import load_freeze, split_banks
from helpers.evoReadinessBanks import load_encoded_banks
from helpers.evoReadinessQueries import digest_array
from helpers.evoReadinessRuntime import load_frozen_runtime
from helpers.runManifest import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args = parser.parse_args()
    started = time.time()
    config = yaml.safe_load((args.run/'config.yaml').read_text())
    c = config['candidate_quality']
    study = args.run/'study'
    identity = json.loads((study/'study.json').read_text())['identity']
    pool,theta = load_freeze(study,'candidate_pool',identity=identity)
    _,physical,parents = load_encoded_banks(ROOT/c['reused_bank_path'])
    banks,_ = split_banks(parents,physical,c['banks'])
    runtime = load_frozen_runtime(config,ROOT,require_archived=True)
    calls = branches = blocks = 0
    inputs = {}
    with torch.inference_mode():
        for role in ['real_selection','evaluation']:
            for i,row in enumerate(pool['entries']):
                path = study/role/(row['id']+'.npz')
                inputs[str(path.relative_to(args.run))] = file_sha256(path)
                with np.load(path,allow_pickle=False) as f:
                    z = torch.as_tensor(f['z'],device='cuda')
                    acts = f['actions']
                    recorded = {k:f[k].copy() for k in ['action_delta_rms','action_delta_max','action_elements','policy_calls']}
                policy = ActionLimitedPolicy(runtime['policy'],row['cap'])
                th = torch.as_tensor(theta[i:i+1],dtype=torch.float32,device='cuda')
                zh = torch.as_tensor(banks[role].z_hist,device='cuda')
                # Match the executor's group, block, root ordering exactly for statistics.
                for start in range(0,len(banks[role]),64):
                    for b in range(10):
                        for r in range(start,min(start+64,len(banks[role]))):
                            zt = zh[r:r+1,-1] if b==0 else z[r:r+1,b-1]
                            zp = zh[r:r+1,-2] if b==0 else (zh[r:r+1,-1] if b==1 else z[r:r+1,b-2])
                            past = banks[role].hist_blocks[r:r+1,-1].reshape(1,60) if b==0 else acts[r:r+1,b-1]
                            feats = policy.features(zt,zp,torch.as_tensor(past,dtype=torch.float32,device='cuda'))
                            actual = policy.act(th,feats[:,None])[:,0].double().cpu().numpy()
                            if not np.array_equal(actual,acts[r:r+1,b]):
                                raise ValueError(f'executed policy mismatch: {role}/{row["id"]}/{r}/{b}')
                            blocks += 1
                stats = policy.statistics()
                for key,value in recorded.items():
                    if not np.array_equal(stats[key],value):
                        raise ValueError(f'action diagnostic mismatch: {role}/{row["id"]}/{key}')
                calls += policy.calls
                branches += len(banks[role])
            print(f'[action audit] {role} reproduced',flush=True)
    result = dict(passed=True,branches_verified=branches,blocks_verified=blocks,
        all_executed_actions_bitwise_reproduced=True,all_action_statistics_reproduced=True,
        real_steps=0,predictor_rows=0,renders=0,encodes=0,policy_calls=calls,
        wall_s=time.time()-started,pool_parameter_sha256=digest_array(theta),
        input_archive_sha256=inputs,script_sha256=file_sha256(Path(__file__)))
    dest = ROOT/'docs/evoPlan/results/stage6-candidate-quality'/args.run.name
    (dest/'action_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_archive_sha256'},indent=2))


if __name__=='__main__':
    main()
