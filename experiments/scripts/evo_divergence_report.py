#!/usr/bin/env python3
"""Audit archived divergence predictions and render descriptive development results."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
import yaml

from helpers.evoDivergence import MODES, summarize_cells, verified_archive, validate_cell
from helpers.evoFollowup import load_freeze, split_banks
from helpers.evoReadinessBanks import load_encoded_banks
from helpers.evoReadinessQueries import QueryStore, digest_json
from helpers.runManifest import file_sha256
from helpers.walkerRules import health_clearance

LABELS = dict(teacher_forced='Real history every block',refresh_2='Refresh every 2 blocks',
    refresh_5='Refresh every 5 blocks',real_action_tape='Real actions, no refresh',
    closed_loop='Closed-loop imagination')


def audit(run):
    config = yaml.safe_load((run/'config.yaml').read_text())
    c = config['divergence']
    launch = json.loads((run/'launch.json').read_text())
    completion = json.loads((run/'completion.json').read_text())
    assert completion['complete']
    identity = completion['study_identity']
    assert digest_json(launch['identity'])==identity
    for name,sha in launch['identity']['source_sha256'].items():
        assert file_sha256(run/'source'/name)==sha
    for name,sha in launch['identity']['parent_files_sha256'].items():
        assert file_sha256(ROOT/name)==sha
    parent = ROOT/c['parent_run']
    pc = yaml.safe_load((parent/'config.yaml').read_text())
    pool,theta = load_freeze(parent/'study','candidate_pool',identity=completion['parent_identity'])
    assert pool['freeze_sha256']==completion['pool_sha256']
    encoded,physical,parents = load_encoded_banks(ROOT/pc['candidate_quality']['reused_bank_path'])
    assert encoded['encoded_banks_sha256']==launch['identity']['parent_bank_sha256']
    banks,roots = split_banks(parents,physical,pc['candidate_quality']['banks'])
    accounting,all_cells,summary = {},{},{}
    replayed = 0
    max_action_delta = 0.
    for role in c['roles']:
        store = QueryStore(run/'queries'/role,study_identity=identity)
        accounting[role] = store.accounting()
        assert not accounting[role]['pending_queries']
        cells = []
        for i,row in enumerate(pool['entries']):
            out,receipt = verified_archive(run/'queries'/role/(row['id']+'.npz'),identity)
            validate_cell(out)
            request = receipt['request']
            assert request['role']==role and request['candidate']==row
            assert request['bank']==banks[role].identity() and request['modes']==list(MODES)
            real_path = parent/'study'/role/(row['id']+'.npz')
            assert request['parent_archive_sha256']==file_sha256(real_path)
            real,_ = verified_archive(real_path,completion['parent_identity'])
            assert np.array_equal(out['truth'],real['endpoint_targets'])
            assert np.array_equal(out['real_readout'],real['readout'])
            assert np.array_equal(out['dense_first_step'],real['dense_first_step'])
            assert np.array_equal(out['root_height_pitch'],real['qpos'][:,0,1:3])
            dense = health_clearance(real['qpos'][:,1:,1],real['qpos'][:,1:,2])<=0
            assert np.array_equal(out['dense_violated'],dense.any(1))
            assert np.array_equal(out['dense_first_step'],np.where(dense.any(1),dense.argmax(1),100))
            np.testing.assert_allclose(out['latent_rms'],
                np.sqrt(np.mean((out['predicted_z'].astype(float)-real['z'][None])**2,-1)),rtol=1e-12,atol=1e-12)
            assert all(np.array_equal(out['predicted_z'][j,:,0],out['predicted_z'][0,:,0]) for j in [1,2,3])
            max_action_delta = max(max_action_delta,float(out['initial_action_max_difference']))
            assert float(out['initial_action_max_difference'])<=c['initial_action_tolerance']
            if role=='evaluation':
                old,_ = verified_archive(parent/'study'/role/(row['id']+'-k0.npz'),completion['parent_identity'])
                assert np.array_equal(out['predicted_readout'][4],old['readout'][:,0])
                replayed += 1
            # Do not retain the large latent arrays after their independent numeric audit.
            del out['predicted_z']
            cells.append(out)
        assert len(cells)==len(pool['entries'])
        summary[role] = summarize_cells(cells,episode_keys=banks[role].episode_keys,
            seed=c['bootstrap_seed'],replicates=c['bootstrap_replicates'])
        all_cells[role] = cells
    assert accounting==completion['accounting']
    costs = json.loads((run/'costs.json').read_text())
    for key in ['predictor_rows','real_steps','renders','encodes']:
        assert costs[key]==sum(v['completed_costs'][key] for v in accounting.values())
    assert costs['predictor_rows']==427200 and costs['real_steps']==costs['renders']==costs['encodes']==0
    evidence = dict(passed=True,query_archives_verified=sum(len(v['completed_queries']) for v in accounting.values()),
        parent_files_verified=len(launch['identity']['parent_files_sha256']),
        source_files_verified=len(launch['identity']['source_sha256']),
        original_closed_loop_readouts_bitwise_reproduced=replayed,
        real_readout_replays_bitwise_verified_in_runner=len(pool['entries'])*len(c['roles']),
        dense_event_times_recomputed=True,latent_errors_recomputed=True,
        initial_real_vs_imagined_action_max_difference=max_action_delta,
        accounting_reconciled=True,report_script_sha256=file_sha256(Path(__file__)),
        completion_sha256=file_sha256(run/'completion.json'))
    return config,pool,summary,all_cells,evidence,costs



def timing_supplement(run, cells, summary, costs):
    """Post-run descriptive checks, counted independently of summary helpers."""
    result = dict(status='Post-run descriptive supplement; no new model or simulator queries', roles={})
    def unsafe(x):
        return (x[...,0]<=.8)|(x[...,0]>=2.)|(x[...,1]<=-1.)|(x[...,1]>=1.)
    for role, rows in cells.items():
        failed = np.stack([c['dense_violated'] for c in rows])
        first = np.stack([c['dense_first_step'] for c in rows])
        truth = np.stack([c['truth'] for c in rows])
        real = np.stack([c['real_readout'] for c in rows])
        pred = np.stack([c['predicted_readout'] for c in rows],axis=1)
        block = np.minimum(first//10,9)
        ci, ri = np.indices(first.shape)
        endpoint = unsafe(truth)[ci,ri,block]&failed
        probe_event = unsafe(real)[ci,ri,block]&endpoint
        probe_ever = unsafe(real).any(-1)
        mode_event = unsafe(pred)[:,ci,ri,block]&endpoint[None]
        error = np.abs(pred[...,:2]-real[None,...,:2])
        exceeded = (error[...,0]>.05)|(error[...,1]>.1)
        before = (exceeded&(((np.arange(10)+1)*10 < (first+1)[...,None])&failed[...,None])[None]).any(-1)
        s = summary[role]
        assert int(endpoint.sum())==s['event_endpoint_visible']
        assert int(probe_event.sum())==s['real_readout_event_detection']['numerator']
        data = dict(failed_pairs=int(failed.sum()),healthy_pairs=int((~failed).sum()),
            probe_segment_true_positives=int((probe_ever&failed).sum()),
            probe_segment_false_negatives=int((~probe_ever&failed).sum()),
            probe_segment_false_positives=int((probe_ever&~failed).sum()),modes={})
        for m,name in enumerate(MODES):
            assert int(mode_event[m].sum())==s['modes'][name]['event_endpoint_detection']['numerator']
            assert int(before[m].sum())==s['modes'][name]['divergence_before_dense_failure']['numerator']
            data['modes'][name] = dict(event_detected=int(mode_event[m].sum()),
                divergence_before_failure=int(before[m].sum()),
                healthy_with_any_exceedance=int((exceeded[m].any(-1)&~failed).sum()),
                healthy_with_first_endpoint_exceedance=int((exceeded[m,...,0]&~failed).sum()),
                failed_with_first_endpoint_exceedance=int((exceeded[m,...,0]&failed).sum()))
        data['selected_candidates'] = {str(i):dict(probe_event_detected=int(probe_event[i].sum()),
            unsafe_event_endpoints=int(endpoint[i].sum()),probe_segment_true_positives=int((probe_ever[i]&failed[i]).sum()),
            failed_pairs=int(failed[i].sum())) for i in [0,1,31,33]}
        result['roles'][role] = data
    launch = json.loads((run/'launch.json').read_text())
    result['timing'] = dict(budget_clock_seconds=costs['wall_s'],
        corrected_process_seconds=launch['started_at']+costs['wall_s']-launch['replacement_process_started_at'],
        note='Budget clock includes the retained zero-query failure and repair interval; process time includes setup and output writes.')
    result['independent_primary_point_counts_verified'] = True
    return result


def render(run,config,pool,summary,cells,evidence,costs):
    dest = ROOT/'docs/evoPlan/results/stage6-divergence'/run.name
    dest.mkdir(parents=True,exist_ok=True)
    analysis_path = dest/'analysis.json'
    if analysis_path.exists():
        assert json.loads(analysis_path.read_text())==summary,'analysis changed on offline replay'
    analysis_path.write_text(json.dumps(summary,indent=2)+'\n')
    (dest/'integrity_audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
    supplement = timing_supplement(run,cells,summary,costs)
    (dest/'timing_supplement.json').write_text(json.dumps(supplement,indent=2)+'\n')
    lines = ['# Where Walker imagination diverges before real failure','',
        'This is an exploratory retrospective diagnostic on the complete fixed 89-candidate '
        'pool and previously exposed source episodes. No new simulator steps or controller '
        'updates were performed. Refreshed histories and real future action tapes are '
        'privileged diagnostics, not deployable fixes or root-time risk predictions.','']
    def fmt(v):
        return 'undefined' if v['point'] is None else f'{100*v["point"]:.1f}% [{100*v["lo"]:.1f}, {100*v["hi"]:.1f}]'
    for role,s in summary.items():
        lines += [f'## {role}', '',
            f'{s["failed_pairs"]} failed candidate/episode pairs across '
            f'{s["failing_source_episodes"]} of {s["independent_source_episodes"]} source episodes. '
            f'There are {s["candidate_episode_pairs"]} correlated pairs, not that many independent episodes. '
            f'{s["event_endpoint_visible"]} first-failure-containing block ends were unsafe. '
            f'Real-image readout detected {fmt(s["real_readout_event_detection"])} of those endpoints.','',
            '| Prediction path | Failure-endpoint detection [95%] | Divergence strictly before failure [95%] | Median lead when present |',
            '|---|---:|---:|---:|']
        for name,m in s['modes'].items():
            lead = 'undefined' if m['median_lead_ms'] is None else f'{m["median_lead_ms"]:.0f} ms'
            lines.append(f'| {LABELS[name]} | {fmt(m["event_endpoint_detection"])} | '
                         f'{fmt(m["divergence_before_dense_failure"])} | {lead} |')
        lines += ['', 'Intervals resample whole source episodes and are conditional on this fixed candidate pool. '
            'Divergence means height error >0.05 m or pitch error >0.1 rad against the real-image probe. '
            'Only endpoints strictly preceding the dense event count. Detection uses the endpoint '
            'of the block containing the first dense event, which can follow that event by up to 72 ms. '
            'See analysis.json for threshold sensitivity, physical versus readout errors, and denominators.', '']
    lines += ['## Verification and cost','',
        f'All {evidence["query_archives_verified"]} new query archives verified. '
        f'All {evidence["original_closed_loop_readouts_bitwise_reproduced"]} original k0 check-bank '
        'readouts reproduced bitwise. Dense event times and latent errors were recomputed. '
        f'New cost: {costs["predictor_rows"]:,} predictor rows; '
        f'{supplement["timing"]["corrected_process_seconds"]:.2f} seconds in the corrected process, '
        f'{costs["wall_s"]/60:.2f} minutes on the inherited budget clock '
        '(including the retained zero-query failure and repair interval); '
        'zero simulator steps, renders, image encodes and gradient updates. Parent simulator '
        'and training costs remain additional.','',
        'See [interpretation](INTERPRETATION.md) and timing_supplement.json for healthy-trajectory '
        'exceedances and eventual versus immediate probe detection. These post-run checks are descriptive.','',
        '![Detection and error growth](divergence_summary.png)','',
        '![Deterministic examples](failure_traces.png)','']
    (dest/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(MODES,['#257b57','#75a863','#d4a32e','#d26c37','#6f5ba7']))
    s = summary['evaluation']
    fig,axes = plt.subplots(1,3,figsize=(14,4.6))
    ts = np.arange(1,11)*.08
    for name,m in s['modes'].items():
        errors = np.array(m['healthy_prefix_dynamics_mae'])
        axes[0].plot(ts,errors[:,0],label=LABELS[name],color=colors[name])
        axes[1].plot(ts,errors[:,1],color=colors[name])
    vals = [s['modes'][name]['event_endpoint_detection'] for name in MODES]
    axes[2].bar(np.arange(5),[100*v['point'] for v in vals],color=[colors[m] for m in MODES])
    axes[2].errorbar(np.arange(5),[100*v['point'] for v in vals],
        yerr=np.array([[100*(v['point']-v['lo']) for v in vals],
                       [100*(v['hi']-v['point']) for v in vals]]),fmt='none',ecolor='black',capsize=3)
    axes[2].axhline(100*s['real_readout_event_detection']['point'],color='black',ls='--',label='Real-image probe')
    axes[2].set_xticks(np.arange(5),['Every 1','Every 2','Every 5','Tape','Closed'],rotation=25)
    axes[2].set_ylabel('Failure-endpoint detection (%)')
    axes[2].legend(fontsize=8)
    for ax,title,unit in zip(axes[:2],['Height error before failure','Pitch error before failure'],['m','rad']):
        ax.set_xlabel('Time from root (s)')
        ax.set_ylabel(f'MAE against real-image probe ({unit})')
        ax.set_title(title)
    axes[2].set_title('Real-history refresh frequency')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Reused development check: fixed pool, 64 independent source episodes')
    fig.tight_layout(rect=(0,.1,1,.94))
    fig.savefig(dest/'divergence_summary.png',dpi=180)
    fig.savefig(dest/'divergence_summary.pdf')
    plt.close(fig)
    examples = config['divergence']['example_candidates']
    fig,axes = plt.subplots(2,len(examples),figsize=(14,6),sharex=True)
    chosen = []
    parent = ROOT/config['divergence']['parent_run']
    completion = json.loads((run/'completion.json').read_text())
    for col,i in enumerate(examples):
        out = cells['evaluation'][i]
        failed = np.flatnonzero(out['dense_violated'])
        if not len(failed):
            axes[0,col].set_title(f'Candidate {i}: no failures')
            continue
        r = int(failed[0])
        chosen.append(dict(candidate=i,episode_index=r,first_dense_step=int(out['dense_first_step'][r])))
        real,_ = verified_archive(parent/'study/evaluation'/f'candidate-{i:03}.npz',completion['parent_identity'])
        for j in [0,1]:
            ax = axes[j,col]
            ax.plot(np.arange(101)*.008,real['qpos'][r,:,j+1],color='black',label='Simulator truth',lw=2)
            ax.plot(ts,out['real_readout'][r,:,j],color='gray',ls=':',label='Real-image probe')
            for m in [0,3,4]:
                ax.plot(ts,out['predicted_readout'][m,r,:,j],color=colors[MODES[m]],label=LABELS[MODES[m]])
            for threshold in ([.8,2.] if j==0 else [-1.,1.]):
                ax.axhline(threshold,color='red',lw=.7,ls='--')
            ax.axvline((out['dense_first_step'][r]+1)*.008,color='black',lw=.7,ls='--')
            ax.spines[['top','right']].set_visible(False)
            if j==1:
                ax.set_xlabel('Time (s)')
        axes[0,col].set_title(f'Candidate {i:03}, episode index {r}')
    axes[0,0].set_ylabel('Torso height (m)')
    axes[1,0].set_ylabel('Torso pitch (rad)')
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=8)
    fig.suptitle('First failed stored check episode for each predeclared candidate')
    fig.tight_layout(rect=(0,.06,1,.95))
    fig.savefig(dest/'failure_traces.png',dpi=180)
    fig.savefig(dest/'failure_traces.pdf')
    plt.close(fig)
    (dest/'example_selection.json').write_text(json.dumps(chosen,indent=2)+'\n')
    return dest


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args = parser.parse_args()
    config,pool,summary,cells,evidence,costs = audit(args.run)
    dest = render(args.run,config,pool,summary,cells,evidence,costs)
    print(json.dumps(dict(audit=evidence,costs=costs,report=str(dest)),indent=2))
