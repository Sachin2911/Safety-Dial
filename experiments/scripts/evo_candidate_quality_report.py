#!/usr/bin/env python3
"""Reconcile all diagnostic queries and reproduce selections and analysis offline."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
import yaml

from helpers.evoCandidateQuality import choose_controllers, diagnostic_analysis, diagnostic_budget
from helpers.evoFollowup import load_freeze, split_banks
from helpers.evoRanking import segment_metrics
from helpers.evoReadinessBanks import load_encoded_banks
from helpers.evoReadinessQueries import QueryStore, digest_array, digest_json
from helpers.evoReadinessSearch import task_returns
from helpers.evoReadinessStudy import checked_real_output, checked_audit_output
from helpers.runManifest import file_sha256


def audit(run):
    config = yaml.safe_load((run/'config.yaml').read_text())
    c = config['candidate_quality']
    study = run/'study'
    saved = json.loads((study/'completion.json').read_text())
    declaration = json.loads((study/'study.json').read_text())
    identity = declaration['identity']
    assert saved['complete'] and digest_json(declaration['declaration'])==identity
    assert declaration['declaration']['config']==config
    pool,theta = load_freeze(study,'candidate_pool',identity=identity)
    gate,chosen = load_freeze(study,'diagnostic_choices',identity=identity)
    rows = pool['entries']
    launch = json.loads((run/'launch.json').read_text())
    for name,sha in launch['identity']['source_sha256'].items():
        assert file_sha256(run/'source'/name)==sha
    for name,sha in launch['identity']['parent_files_sha256'].items():
        assert file_sha256(ROOT/name)==sha
    encoded,physical,parents = load_encoded_banks(ROOT/c['reused_bank_path'])
    assert encoded['encoded_banks_sha256']==declaration['declaration']['provenance']['parent_bank_sha256']
    banks,roots = split_banks(parents,physical,c['banks'])
    del banks['fitness'],roots['fitness']
    assert {r:b.identity() for r,b in banks.items()}==declaration['declaration']['banks']
    stores = {r:QueryStore(study/r,study_identity=identity) for r in banks}
    accounting = {r:s.accounting() for r,s in stores.items()}
    assert accounting==saved['accounting']
    assert not any(v['pending_queries'] for v in accounting.values())
    receipts = []
    def load(role,i,k=None):
        key = rows[i]['id'] + (f'-k{k}' if k is not None else '')
        with np.load(study/role/(key+'.npz'),allow_pickle=False) as f:
            receipt = json.loads(str(f['__receipt__']))
            out = {name:f[name].copy() for name in f.files if name!='__receipt__'}
        stores[role]._validate_receipt(receipt,key)
        assert receipt['data_sha256']=={name:digest_array(v) for name,v in out.items()}
        request = receipt['request']
        assert request['candidate']==rows[i] and request['theta']==digest_array(theta[i])
        assert request['bank']==banks[role].identity()
        assert request['barrier']==(gate if role=='evaluation' else pool)['freeze_sha256']
        receipts.append((role,receipt))
        if k is None:
            checked_real_output(out,len(banks[role]),10)
            assert np.allclose(out['qpos'][:,0],np.stack([r.qpos for r in roots[role]]),rtol=0,atol=1e-12)
            if rows[i]['cap'] is not None:
                assert float(out['action_delta_max'])<=rows[i]['cap']+2e-7
        else:
            n = 1 if k==0 else c['noise_samples']
            checked_audit_output(out,len(banks[role]),n,10)
            assert request['samples']==n and request['noise']==k and request['seed']==c['audit_seed']
        return out
    selection = dict(failure=[],progress=[])
    scores = {'imagined_k0_zero':[],'imagined_k1_fixed':[]}
    check = dict(failure=[],progress=[],action_rms=[],readout_failure=[],endpoint_failure=[])
    imagined = {'0':[],'1':[]}
    for i in range(len(rows)):
        out = load('real_selection',i)
        selection['failure'].append(out['dense_violated'])
        selection['progress'].append(out['progress'])
        for k,name in [(0,'imagined_k0_zero'),(1,'imagined_k1_fixed')]:
            metrics = segment_metrics(load('imagined_selection',i,k)['readout'])
            penalty = 0 if k==0 else c['diagnostic_fixed_penalty']
            scores[name].append(float((task_returns(metrics)-penalty*metrics['violated']).mean()))
        out = load('evaluation',i)
        check['failure'].append(out['dense_violated'])
        check['progress'].append(out['progress'])
        check['action_rms'].append(float(out['action_delta_rms']))
        check['readout_failure'].append(segment_metrics(out['readout'])['violated'])
        check['endpoint_failure'].append((out['endpoint_clearance']<=0).any(-1))
        for k in [0,1]:
            imagined[str(k)].append(segment_metrics(load('evaluation',i,k)['readout'])['violated'].mean(-1))
    selection = {k:np.stack(v) for k,v in selection.items()}
    check = {k:np.stack(v) for k,v in check.items()}
    check['imagined_failure'] = {k:np.stack(v) for k,v in imagined.items()}
    choices = choose_controllers(rows,selection['failure'],selection['progress'],scores,c['minimum_progress_ratio'])
    assert gate['entries']==[dict(name=name,**v) for name,v in choices.items()]
    assert np.array_equal(chosen,np.stack([theta[v['index']] for v in choices.values()]))
    result = diagnostic_analysis(rows,selection,check,choices,progress_floor=c['minimum_progress_ratio'],
        seed=c['bootstrap_seed'],replicates=c['bootstrap_replicates'])
    assert result==saved['analysis']==json.loads((run/'analysis.json').read_text())
    totals = {}
    for record in accounting.values():
        for key,value in record['completed_costs'].items():
            totals[key] = totals.get(key,0)+value
    assert totals==saved['costs']
    plan = diagnostic_budget(config,len(rows))
    assert all(totals[key]==plan[key] for key in ['real_steps','predictor_rows'])
    costs = json.loads((run/'costs.json').read_text())
    startups = json.loads((run/'startup_receipts.json').read_text())
    assert all(costs[k]==v+(sum(s['fingerprint_renders'] for s in startups) if k=='renders' else 0)
               for k,v in totals.items())
    assert max(r['completed_at'] for role,r in receipts if role!='evaluation') < min(
        r['completed_at'] for role,r in receipts if role=='evaluation')
    # The original baseline must reproduce on the same two banks, despite instrumentation.
    baseline_matches = {}
    for role in ['real_selection','evaluation']:
        current = load(role,0)
        old_path = ROOT/c['parent_run']/'study'/role/(digest_array(theta[0])+'.npz')
        with np.load(old_path,allow_pickle=False) as old:
            for key in ['qpos','progress','dense_violated','readout','actions']:
                assert np.array_equal(current[key],old[key]), (role,key)
        baseline_matches[role] = True
    verification = dict(passed=True,queries_verified=sum(len(list((study/r).glob('*.npz'))) for r in banks),
        source_files_verified=len(launch['identity']['source_sha256']),
        parent_files_verified=len(launch['identity']['parent_files_sha256']),
        frozen_selections_reproduced=True,analysis_reproduced=True,phase_order_verified=True,
        timing_evidence='completion timestamps plus frozen barrier identities; start times are not retained',
        action_caps_verified=True,baseline_bitwise_matches=baseline_matches,
        actual_query_costs=totals,completion_sha256=file_sha256(study/'completion.json'),
        report_script_sha256=file_sha256(Path(__file__)))
    return result,verification,costs


def report(run,result,verification,costs):
    dest = ROOT/'docs/evoPlan/results/stage6-candidate-quality'/run.name
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'integrity_audit.json').write_text(json.dumps(verification,indent=2)+'\n')
    lines = ['# Walker candidate-quality development diagnostic','',
        'All results reuse development episodes. The candidate pool and selectors were frozen '
        'before their respective simulator queries. Candidate rows are correlated; '
        'intervals are exploratory and unadjusted. These 0.8-second branches do not establish '
        'full-episode controller improvement.','',
        f'The fixed pool contains {result["candidate_count"]} variants. The starting controller '
        f'failed on {result["baseline_failure"]:.2%} of {result["check_episodes"]} check episodes. '
        f'{len(result["selection_eligible_indices"])} candidates met the empirical improvement '
        f'criterion on selection episodes, {len(result["check_eligible_indices"])} on check '
        f'episodes, and {len(result["eligible_on_both"])} on both. The latter counts use '
        'retrospective check outcomes and are not validated selections.','',
        '| Frozen selector | Candidate | Check failure difference, pp [95%] | Progress ratio [95%] |',
        '|---|---|---:|---:|']
    for name,row in result['frozen_choices'].items():
        d,p = row['failure_difference'],row['progress_ratio']
        ratio = 'undefined' if p is None else f'{p["point"]:.3f} [{p["lo"]:.3f}, {p["hi"]:.3f}]'
        lines.append(f'| {name} | {row["candidate"]} | {100*d["point"]:+.2f} '
                     f'[{100*d["lo"]:+.2f}, {100*d["hi"]:+.2f}] | {ratio} |')
    lines += ['',f'Incremental cost: {costs["real_steps"]:,} real steps, '
        f'{costs["predictor_rows"]:,} predictor rows, {costs["wall_s"]/60:.2f} minutes. '
        'Zero gradient updates and zero new source-collection steps; historical collection, '
        'training and parent-search costs remain additional.','',
        f'All {verification["queries_verified"]} query archives verified. The complete analysis '
        'and all frozen selectors reproduced exactly. Both baseline replays matched the '
        'earlier development run bitwise. See analysis.json for every candidate and the '
        'dense/endpoint/readout decomposition.','',
        '![Candidate quality](candidate_quality.png)','']
    (dest/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    kinds = sorted({r['origins'][0]['kind'] for r in result['candidates']})
    for kind in kinds:
        rows = [r for r in result['candidates'] if r['origins'][0]['kind']==kind]
        axes[0].scatter([r['action_delta_rms'] for r in rows],
                        [100*r['check_failure'] for r in rows],label=kind,alpha=.75)
        axes[1].scatter([100*r['selection_failure'] for r in rows],
                        [100*r['check_failure'] for r in rows],label=kind,alpha=.75)
    for ax in axes:
        ax.axhline(100*result['baseline_failure'],color='black',ls='--',lw=1)
        ax.set_ylabel('Development-check failure (%)')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_xlabel('Same-input action correction RMS on real trajectories')
    axes[1].set_xlabel('Simulator-selection failure (%)')
    axes[1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.02,1))
    fig.suptitle('Exploratory fixed-pool diagnostic; 64 reused check episodes')
    fig.tight_layout()
    fig.savefig(dest/'candidate_quality.png',dpi=180)
    fig.savefig(dest/'candidate_quality.pdf')
    plt.close(fig)
    return dest


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args = parser.parse_args()
    results,verification,costs = audit(args.run)
    dest = report(args.run,results,verification,costs)
    print(json.dumps(dict(audit=verification,report=str(dest)),indent=2))
