#!/usr/bin/env python3
"""Independently replay saved repair evaluation and render the audit report."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
import yaml

from evo_repair import summarize_arms, controller_audit
from helpers.evoDivergence import verified_archive
from helpers.evoReadinessQueries import digest_json
from helpers.evoRepair import ARMS, select_repair, nominate, repair_gate, metric_arrays
from helpers.runManifest import file_sha256


def paired_changes(base, arm, truth, first, seed, replicates):
    a,b=metric_arrays(base,truth,first),metric_arrays(arm,truth,first)
    e=truth.shape[1]
    draws=np.random.default_rng(seed).integers(e,size=(replicates,e))
    out={}
    for name,num,den in [('detection','event_num','event_den'),('false_alarm','false_num','false_den')]:
        ad,bd=a[den][draws].sum(1),b[den][draws].sum(1)
        valid=(ad>0)&(bd>0)
        if not valid.any():
            out[name]=None
            continue
        values=b[num][draws].sum(1)[valid]/bd[valid]-a[num][draws].sum(1)[valid]/ad[valid]
        out[name]=dict(point=float(b[num].sum()/b[den].sum()-a[num].sum()/a[den].sum()),
            lo=float(np.quantile(values,.025)),hi=float(np.quantile(values,.975)))
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    args=ap.parse_args()
    run=args.run
    c=yaml.safe_load((run/'config.yaml').read_text())['repair']
    declaration=json.loads((run/'declaration.json').read_text())
    completion=json.loads((run/'completion.json').read_text())
    assert completion['complete'] and digest_json(declaration['declaration'])==completion['identity']==declaration['identity']
    d=declaration['declaration']
    for name,sha in d['source_sha256'].items():
        assert file_sha256(run/'source'/name)==sha
    for name,sha in d['input_sha256'].items():
        assert file_sha256(ROOT/name)==sha
    for name,sha in completion['checkpoints'].items():
        assert file_sha256(run/name)==sha
    assert file_sha256(run/'evaluation.npz')==completion['evaluation_sha256']
    for path in (run/'queries').glob('*.npz'):
        verified_archive(path,completion['identity'])
    assert not list((run/'queries').glob('*.pending.json'))
    roles=d['roles']
    keys=[k for role in roles.values() for k in role['episode_keys']]
    assert len(keys)==len(set(keys))==96
    analysis=json.loads((run/'analysis.json').read_text())
    nomination=json.loads((run/'controller_nominations.json').read_text())
    assert nomination['audit_not_yet_summarized']
    with np.load(run/'evaluation.npz',allow_pickle=False) as f:
        data={k:f[k] for k in f.files}
    n=c['episodes_per_role']
    assert data['episode_indices'].tolist()==roles['development']['indices']+roles['audit']['indices']
    summary={}
    changes={}
    for role,ix in [('development',np.arange(n)),('audit',np.arange(n,2*n))]:
        pred={a:data[a][:,:,:,ix] for a in ARMS}
        summary[role]=summarize_arms(pred,data,ix,c)
        assert summary[role]==analysis[role]
        changes[role]={a:paired_changes(pred['baseline'][:,0],pred[a][:,0],data['truth'][:,ix],data['first'][:,ix],c['bootstrap_seed'],c['bootstrap_replicates']) for a in ARMS[1:]}
        # Direct boolean count audit, independent of the metric helper's event indexing.
        first=data['first'][:,ix]
        truth=data['truth'][:,ix]
        mask=(first<100)[...,None]&(np.arange(10)[None,None,:]==np.minimum(first//10,9)[...,None])
        unsafe=(truth[...,0]<=.8)|(truth[...,0]>=2)|(np.abs(truth[...,1])>=1)
        mask &= unsafe
        assert int(mask.sum())==summary[role]['baseline']['teacher_forced']['endpoint_pairs']
        for arm in ARMS:
            ro=pred[arm][:,0]
            alarm=(ro[...,0]<=.8)|(ro[...,0]>=2)|(np.abs(ro[...,1])>=1)
            assert int((alarm&mask[None]).sum())==summary[role][arm]['teacher_forced']['event_detection']['numerator']
    choice=select_repair({a:summary['development'][a]['teacher_forced'] for a in ARMS},c['gate'])
    assert choice==analysis['choice']
    assert nominate(data['original_closed'],c['minimum_progress_ratio'])==nomination['nominations']['original']
    if choice['chosen']:
        with np.load(run/'repaired_development_closed.npz') as f:
            assert nominate(f['readout'],c['minimum_progress_ratio'])==nomination['nominations']['repaired']
    assert controller_audit(nomination['nominations'],data,np.arange(n,2*n),c)==analysis['controller_audit']
    gates={a:repair_gate(summary['audit']['baseline']['teacher_forced'],summary['audit'][a]['teacher_forced'],c['gate']) for a in ARMS[1:]}
    assert gates==analysis['audit_gates']
    costs=json.loads((run/'costs.json').read_text())
    assert costs['real_steps']==costs['encodes']==costs['renders']==0
    assert costs['predictor_training_rows']==len(c['seeds'])*c['predictor_steps']*c['batch']
    assert costs['predictor_gradient_updates']==len(c['seeds'])*c['predictor_steps']
    assert costs['probe_gradient_updates']==len(c['seeds'])*c['probe_steps']
    assert costs['predictor_inference_rows']==completion['accounting']['completed_costs']['predictor_rows']
    dest=ROOT/'docs/evoPlan/results/stage6-repair'/run.name
    audit=dict(passed=True,source_hashes=len(d['source_sha256']),input_hashes=len(d['input_sha256']),
        query_archives=len(list((run/'queries').glob('*.npz'))),checkpoint_hashes=len(completion['checkpoints']),
        numerical_summaries_reproduced=True,independent_event_counts_verified=True,
        development_choice_reproduced=True,controller_nominations_reproduced=True,
        whole_episode_splits_disjoint=True,accounting_reconciled=True,
        report_script_sha256=file_sha256(Path(__file__)))
    (dest/'integrity_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (dest/'paired_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
    lines=['# Walker predictor and probe repair','',
        'Exploratory fitting and audit on previously exposed episodes. Four disjoint groups of 24 '
        'whole source episodes separate probe fitting, predictor fitting, development, and audit. '
        'All 89 candidate branches stay with their episode. Three fitting seeds are averaged; '
        'uncertainty resamples source episodes, not candidate branches or seeds.','',
        f'Development-selected repair: **{choice["chosen"] or "none"}**. '
        f'Audit prediction gate passed for that choice: **{analysis["prediction_repair_audit_passed"]}**.','']
    def pct(v):
        return 'undefined' if v['point'] is None else f'{v["point"]*100:.1f}% [{v["lo"]*100:.1f}, {v["hi"]*100:.1f}]'
    for role in ['development','audit']:
        lines += [f'## {role}', '',
            '| Arm | One-step event detection [95%] | Healthy-endpoint false alarms [95%] | Height MAE (m) | Pitch MAE (rad) | Tape event detection |',
            '|---|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            s=summary[role][arm]['teacher_forced']
            lines.append(f'| {arm} | {pct(s["event_detection"])} | {pct(s["false_alarm"])} | '
                f'{s["healthy_height_mae"]["point"]:.4f} | {s["healthy_pitch_mae"]["point"]:.4f} | '
                f'{pct(summary[role][arm]["real_action_tape"]["event_detection"])} |')
        s=summary[role]['baseline']['teacher_forced']
        lines += ['',f'{s["endpoint_pairs"]} first-failure-containing endpoints were unsafe; '
            f'{s["failed_pairs"]} candidate/episode pairs failed densely. The endpoint can follow '
            'the event by up to 72 ms. Healthy false alarms and MAE use only endpoints strictly '
            'before the first dense event, plus all endpoints on non-failing trajectories.','']
    lines += ['## Controller-selection check','']
    if choice['chosen'] is None:
        lines += ['No repair passed the development gate. New repaired closed-loop nominations '
            'were therefore not run. This is the predeclared stopping rule, not evidence that a '
            'different repair cannot work.','']
    for name,m in analysis['controller_audit'].items():
        lines.append(f'- {name}: candidate {m["index"]:03d}, {m["failures"]}/{m["episodes"]} real failures; '
                     f'mean real progress {m["mean_progress"]:.4f}.')
    lines += ['', 'Full paired differences, progress ratios, per-seed results, real-image probe '
        'controls, and gate reasons are in analysis.json. Candidate nomination used development '
        'imagination only and was frozen before the audit summary. No controller parameters changed.','',
        '## Verification and cost','',
        f'{audit["query_archives"]} prediction archives and {audit["checkpoint_hashes"]} checkpoints verified. '
        'Primary summaries, event counts, gate decisions and nominations reproduced offline. '
        f'{costs["predictor_gradient_updates"]:,} predictor optimizer updates '
        f'({costs["predictor_training_rows"]:,} training rows), '
        f'{costs["probe_gradient_updates"]:,} probe updates '
        f'({costs["probe_training_rows"]:,} training rows), and '
        f'{costs["predictor_inference_rows"]:,} new inference predictor rows. '
        f'Elapsed compute workflow: {costs["wall_s"]/60:.2f} minutes, excluding final upload. '
        'Zero new simulator steps, renders, image encodes, or encoder updates. '
        'Historical data collection and model training remain additional costs.','',
        '![Repair comparison](repair_comparison.png)','']
    (dest/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    colors=['#858585','#3385a3','#db9c32','#34885b']
    for ax,key,title in zip(axes,['event_detection','false_alarm','healthy_height_mae'],
            ['One-step failure-endpoint detection','False alarms at healthy endpoints','Height error at healthy endpoints']):
        vals=[summary['audit'][a]['teacher_forced'][key] for a in ARMS]
        factor=1 if key.endswith('mae') else 100
        y=[v['point']*factor for v in vals]
        err=np.array([[max(0,v['point']-v['lo'])*factor for v in vals],
                      [max(0,v['hi']-v['point'])*factor for v in vals]])
        ax.bar(np.arange(4),y,color=colors,yerr=err,capsize=3)
        ax.set_xticks(np.arange(4),ARMS,rotation=20)
        ax.set_title(title,fontsize=10)
        ax.set_ylabel('m' if key.endswith('mae') else '%')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Reused audit: 24 source episodes, fixed candidate pool, three fitting seeds')
    fig.tight_layout()
    fig.savefig(dest/'repair_comparison.png',dpi=180)
    fig.savefig(dest/'repair_comparison.pdf')
    plt.close(fig)
    print(json.dumps(dict(audit=audit,choice=choice,report=str(dest)),indent=2))


if __name__=='__main__':
    main()
