#!/usr/bin/env python3
"""Post-run latent-space check; archived inference only, no model calls or new fitting."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
import yaml

from evo_repair import load_data
from helpers.evoDivergence import verified_archive
from helpers.walkerRules import health_clearance
from helpers.runManifest import file_sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    run=ap.parse_args().run
    c=yaml.safe_load((run/'config.yaml').read_text())['repair']
    declaration=json.loads((run/'declaration.json').read_text())
    roles=declaration['declaration']['roles']
    bundle=load_data(c)
    assert bundle['input_hashes']==declaration['declaration']['input_sha256']
    ev=np.array(roles['development']['indices']+roles['audit']['indices'])
    data=bundle['data']
    first=data['first'][:,ev]
    truth=data['truth'][:,ev]
    mask=((np.arange(10)+1)*10<(first+1)[...,None])&(health_clearance(truth[...,0],truth[...,1])>0)
    actual=data['z'][:,ev]
    old=bundle['base_z'][:2,:,ev]
    mse={'baseline':np.mean((old.astype(float)-actual[None].astype(float))**2,-1)[None]}
    cells=[]
    for seed in c['seeds']:
        new=[]
        for row in bundle['pool']['entries']:
            out,_=verified_archive(run/'queries'/f'seed{seed}-{row["id"]}.npz',declaration['identity'])
            new.append(out['z'])
        cells.append(np.stack(new,axis=1))
    mse['dynamics']=np.mean((np.stack(cells).astype(float)-actual[None,None].astype(float))**2,-1)
    out=dict(status='Post-run descriptive supplement; not used for the frozen gate or nominations',roles={})
    for role,ix in [('development',np.arange(24)),('audit',np.arange(24,48))]:
        summary={}
        for arm,error in mse.items():
            summary[arm]={}
            for m,mode in enumerate(c['inference_modes']):
                x=error[:,m,:,ix]
                # Explicit take avoids mixed scalar/advanced-index axis reordering.
                x=np.take(error[:,m],ix,axis=2)
                valid=mask[:,ix]
                numerator=(x*valid[None]).sum((0,1,3))
                denominator=valid.sum((0,2))*len(x)
                draws=np.random.default_rng(c['bootstrap_seed']).integers(len(ix),size=(c['bootstrap_replicates'],len(ix)))
                ratios=np.sqrt(numerator[draws].sum(1)/denominator[draws].sum(1))
                summary[arm][mode]=dict(healthy_latent_rmse=float(np.sqrt(numerator.sum()/denominator.sum())),
                    lo=float(np.quantile(ratios,.025)),hi=float(np.quantile(ratios,.975)))
        out['roles'][role]=summary
    out['script_sha256']=file_sha256(Path(__file__))
    dest=ROOT/'docs/evoPlan/results/stage6-repair'/run.name/'latent_supplement.json'
    dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
