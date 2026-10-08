#!/usr/bin/env python3
"""Independently reconcile a completed Stage 6 run and publish a local report."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
import yaml

from helpers.evoFollowup import analyze, load_freeze, real_selection, study_plan
from helpers.evoReadinessQueries import QueryStore, digest_array, digest_json
from helpers.evoReadinessStudy import checked_real_output
from helpers.runManifest import file_sha256


def build_report(run):
    config=yaml.safe_load((run/'config.yaml').read_text())
    c=config['followup']
    study=run/'study'
    saved=json.loads((study/'completion.json').read_text())
    declaration=json.loads((study/'study.json').read_text())
    identity=declaration['identity']
    assert saved['complete'] and digest_json(declaration['declaration'])==identity
    plan=study_plan(config)
    first,pool=load_freeze(study,'shortlists',identity=identity)
    last,chosen=load_freeze(study,'selected',identity=identity)
    launch=json.loads((run/'launch.json').read_text())
    for name,sha in launch['identity']['source_sha256'].items():
        assert file_sha256(run/'source'/name)==sha
    paths=[study/r for r in ['real_selection','evaluation','final_audits']]
    paths+=list((study/'searches').glob('*/queries/k*'))
    totals={}
    queries=0
    phase_times={'search':[], 'real_selection':[], 'evaluation':[], 'final_audits':[]}
    episode_roles=declaration['declaration']['banks']
    seen=set()
    for role,bank in episode_roles.items():
        assert not seen.intersection(bank['episodes'])
        seen.update(bank['episodes'])
    for path in paths:
        record=QueryStore(path,study_identity=identity).accounting()
        assert not record['pending_queries']
        queries+=len(list(path.glob('*.npz')))
        role=path.name if path.name in phase_times else 'search'
        for archive in path.glob('*.npz'):
            with np.load(archive,allow_pickle=False) as f:
                receipt=json.loads(f['__receipt__'].tobytes())
            phase_times[role].append(receipt['completed_at'])
            if role in ['real_selection','evaluation']:
                assert receipt['request']['bank']==episode_roles[role]
        for key,value in record['completed_costs'].items():
            totals[key]=totals.get(key,0)+value
    assert max(phase_times['search']) < min(phase_times['real_selection'])
    assert max(phase_times['real_selection']) < min(phase_times['evaluation'])
    assert max(phase_times['real_selection']) < min(phase_times['final_audits'])
    assert totals==saved['actual_costs']
    assert totals['predictor_rows']==plan['counts']['total_predictor_rows']
    per_search=[]
    for folder in sorted((study/'searches').iterdir()):
        rows=sum(QueryStore(folder/'queries'/f'k{k}',study_identity=identity).accounting()
            ['completed_costs'].get('predictor_rows',0) for k in [0,1])
        assert rows==plan['counts']['model_rows_per_search']
        per_search.append(rows)
    assert len(per_search)==plan['counts']['searches']
    def observed(role,theta):
        with np.load(study/role/(digest_array(theta)+'.npz'),allow_pickle=False) as f:
            out={k:f[k] for k in f.files if k!='__receipt__'}
        checked_real_output(out,c['banks'][role],10)
        return out
    selected_by_name={(x['shortlist_name'],x['selector']):(x,t) for x,t in zip(last['entries'],chosen)}
    for item in first['entries']:
        candidates=pool[item['offset']:item['offset']+item['count']]
        obs=[observed('real_selection',t) for t in candidates]
        decision=real_selection(np.stack([r['dense_violated'] for r in obs]),
            np.stack([r['progress'] for r in obs]),minimum_progress_ratio=c['minimum_progress_ratio'])
        for selector,slot in [('imagined',item['imagined_slot']),('verified',decision['slot'])]:
            entry,theta=selected_by_name[item['name'],selector]
            assert np.array_equal(theta,candidates[slot]) and entry['shortlist_slot']==slot
            if selector=='verified':
                assert entry['selection_decision']==decision
    outcomes=json.loads((study/'outcomes.json').read_text())
    cells={}
    baseline=observed('evaluation',pool[0])
    for entry,theta in zip(last['entries'],chosen):
        out=observed('evaluation',theta)
        key=entry['arm']+'/'+entry['selector']
        row=cells.setdefault(key,dict(failure=[],progress=[]))
        row['failure'].append(out['dense_violated'])
        row['progress'].append(out['progress'])
        recorded=next(x for x in outcomes if x['arm']==entry['arm'] and
            x['seed']==entry['seed'] and x['selector']==entry['selector'])
        assert recorded['failure']==out['dense_violated'].tolist()
        assert recorded['progress']==out['progress'].tolist()
    baseline_cell={key:np.broadcast_to(baseline[value],(len(c['seeds']),c['banks']['evaluation']))
        for key,value in [('failure','dense_violated'),('progress','progress')]}
    reproduced=analyze(cells,baseline_cell,seeds=c['seeds'],
        episode_keys=declaration['declaration']['banks']['evaluation']['episodes'],
        replicates=c['bootstrap_replicates'],bootstrap_seed=c['bootstrap_seed'],
        minimum_progress_ratio=c['minimum_progress_ratio'],phase=c['phase'])
    assert reproduced==saved['analysis']
    a=reproduced
    table=[]
    for arm in plan['arms']:
        name=arm['name']
        im,verified=[a['cells'][name+'/'+s] for s in ['imagined','verified']]
        decisions=[x for x in last['entries'] if x['arm']==name and x['selector']=='verified']
        table.append(dict(arm=name,imagined_failure=im['failure']['point'],
            verified_failure=verified['failure']['point'],
            verified_progress_ratio=verified['progress_ratio_to_baseline']['point'],
            baseline_fallbacks=sum(x['selection_decision']['baseline_fallback'] for x in decisions)))
    eligible=[]
    for penalty in c['fixed_penalties']:
        row=next(r for r in table if r['arm']==f'k1-fixed-{penalty:g}')
        if row['verified_progress_ratio'] is not None and row['verified_progress_ratio']>=c['minimum_progress_ratio']:
            eligible.append((row['verified_failure'],-row['verified_progress_ratio'],penalty))
    choice=min(eligible)[2] if eligible else 2.
    decision=dict(fixed_penalty=choice,eligible_candidates=len(eligible),
        fallback_used=not bool(eligible),rule=c['fixed_penalty_choice'],
        exploratory=True,engineering_only=c['phase']=='engineering',confirmation_sizing_locked=False,
        source_completion_sha256=file_sha256(study/'completion.json'))
    costs=json.loads((run/'costs.json').read_text())
    audit=dict(passed=True,report_script_sha256=file_sha256(Path(__file__)),
        completion_sha256=file_sha256(study/'completion.json'),
        phase_order_and_role_disjointness_verified=True,query_archives_verified=queries,source_files_verified=len(launch['identity']['source_sha256']),
        per_search_predictor_rows=sorted(set(per_search)),actual_query_costs=totals,
        real_selection_decisions_reproduced=len(first['entries']),analysis_reproduced_exactly=True,
        frozen_shortlists_sha256=first['freeze_sha256'],frozen_selections_sha256=last['freeze_sha256'])
    dest=ROOT/'docs/evoPlan/results'/('stage6-'+c['phase'])/run.name
    dest.mkdir(parents=True,exist_ok=True)
    for name,obj in [('integrity_audit.json',audit),('development_choice.json',decision),('summary_table.json',table)]:
        (dest/name).write_text(json.dumps(obj,indent=2)+'\n')
    lines=[f'# Stage 6 {c["phase"]} report', '',
        'All results below use reused Stage 5 episodes and are exploratory. The model, probe, '
        'and starting controller remain frozen. Two paired search seeds do not support a generalization claim.', '',
        f'Starting-controller final failure rate: {a["baseline"]["failure"]["point"]:.2%}.', '',
        '| Arm | Imagined-selected real failure | Simulator-selected real failure | Progress / baseline | Baseline fallbacks |',
        '|---|---:|---:|---:|---:|']
    for r in table:
        ratio='undefined' if r['verified_progress_ratio'] is None else f'{r["verified_progress_ratio"]:.3f}'
        lines.append(f'| {r["arm"]} | {r["imagined_failure"]:.2%} | {r["verified_failure"]:.2%} | {ratio} | {r["baseline_fallbacks"]}/{len(c["seeds"])} |')
    choice_text=(f'The predeclared development rule selects fixed penalty **{choice:g}**. '
        'This choice can inform a later locked confirmation protocol; confirmation sizes and fresh data have not been locked or collected.'
        if c['phase']=='development' else
        'This engineering smoke run validates execution only. It does not choose the confirmation penalty.')
    lines+=['',choice_text, '',
        f'Actual costs: {totals["predictor_rows"]:,} predictor rows; {totals["real_steps"]:,} simulator steps; '
        f'{costs["wall_s"]/60:.2f} minutes; zero gradient updates. '
        'Source-collection costs are zero for this reused-bank run; historical source collection remains charged to Stage 5.', '',
        f'All {queries} query archives and {len(launch["identity"]["source_sha256"])} source snapshots verified. '
        'Every simulator selection decision and the complete statistical analysis were independently reproduced from saved numeric outcomes.', '',
        'The planned next steps are a fresh-bank Walker2d confirmation with a locked fixed penalty, '
        'then Hopper and independent world-model replications. No fresh or cross-environment result is claimed here.', '']
    (dest/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(audit=audit,choice=decision,table=table,costs=costs),indent=2))
    return dest,table,a['baseline']['failure']['point']


def plot(dest,table,baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config=yaml.safe_load((dest/'config.yaml').read_text())['followup']
    y=np.arange(len(table))
    fig,ax=plt.subplots(figsize=(10,6))
    ax.barh(y-.18,[100*r['imagined_failure'] for r in table],height=.34,label='Imagined-selected',color='#4378a8')
    ax.barh(y+.18,[100*r['verified_failure'] for r in table],height=.34,label='Simulator-selected',color='#d38a30')
    ax.axvline(100*baseline,color='#343434',linestyle='--',label='Starting controller')
    ax.set_yticks(y,[r['arm'] for r in table])
    ax.invert_yaxis()
    ax.set_xlabel('Final simulator failure rate (%)')
    ax.set_title(f'Exploratory {config["phase"]}: {len(config["seeds"])} seeds, '
                 f'{config["banks"]["evaluation"]} reused evaluation episodes')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(dest/'development_failures.png',dpi=180)
    fig.savefig(dest/'development_failures.pdf')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--plot',action='store_true')
    args=p.parse_args()
    dest,table,baseline=build_report(args.run)
    if args.plot:
        plot(dest,table,baseline)
