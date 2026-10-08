"""Outcome-blind candidate sampling and bounded-action development diagnostics."""
import json
from pathlib import Path

import numpy as np
import torch

from helpers.evoFollowup import freeze, load_freeze, real_selection
from helpers.evoReadinessQueries import QueryStore, digest_array, digest_json
from helpers.runManifest import file_sha256


class ActionLimitedPolicy:
    """Bound each motor correction relative to the baseline at the SAME visited input.

    This is an action constraint, not a bound on trajectory divergence or risk.
    Uncapped actions are returned unchanged. Statistics include all executed blocks.
    """
    def __init__(self, policy, cap=None):
        if cap is not None and (not np.isfinite(cap) or cap <= 0):
            raise ValueError('cap must be finite and positive')
        self.policy, self.cap = policy, cap
        for name in ('device', 'n_params', 'action_mean', 'action_std'):
            setattr(self, name, getattr(policy, name))
        self.reset_statistics()

    def reset_statistics(self):
        self.squared = torch.zeros((), dtype=torch.float64, device=self.device)
        self.maximum = torch.zeros((), dtype=torch.float64, device=self.device)
        self.elements = self.calls = 0

    def features(self, *args):
        return self.policy.features(*args)

    def model_block(self, actions):
        return self.policy.model_block(actions)

    def act(self, theta, features):
        proposed = self.policy.act(theta, features)
        baseline = self.policy.act(torch.zeros_like(torch.as_tensor(theta)), features)
        action = proposed if self.cap is None else (
            baseline + (proposed-baseline).clamp(-self.cap, self.cap)).clamp(-1, 1)
        delta = (action-baseline).double()
        self.squared += delta.square().sum()
        self.maximum = torch.maximum(self.maximum, delta.abs().max())
        self.elements += delta.numel()
        self.calls += 2  # proposed and same-input baseline policy evaluations
        return action

    def statistics(self):
        return dict(action_delta_rms=np.asarray(float((self.squared/max(1,self.elements)).sqrt())),
                    action_delta_max=np.asarray(float(self.maximum)),
                    action_elements=np.asarray(self.elements), policy_calls=np.asarray(self.calls))


def build_pool(config, repository):
    """Read saved parameters and imagined selections, never simulator outcomes."""
    root = Path(repository)
    c = config['candidate_quality']
    parent = root/c['parent_run']/'study'
    declaration = json.loads((parent/'study.json').read_text())
    identity = declaration['identity']
    if digest_json(declaration['declaration']) != identity:
        raise ValueError('parent declaration changed')
    short, saved_theta = load_freeze(parent, 'shortlists', identity=identity)
    rows, parameters, known, inputs = [], [], {}, {}
    for name in ['study.json', 'shortlists.json', 'shortlists.npz']:
        inputs[str((parent/name).relative_to(root))] = file_sha256(parent/name)

    def add(theta, origin, cap=None):
        theta = np.asarray(theta, np.float64)
        if theta.shape != (1020,) or not np.isfinite(theta).all():
            raise ValueError('expected 1020 finite residual coefficients')
        key = digest_json(dict(theta=digest_array(theta), cap=cap))
        if key in known:
            rows[known[key]]['origins'].append(origin)
            return
        known[key] = len(rows)
        rows.append(dict(id=f'candidate-{len(rows):03d}', digest=key, cap=cap, origins=[origin]))
        parameters.append(theta.copy())

    add(np.zeros(1020), dict(kind='baseline'))
    rng = np.random.default_rng(c['sampling_seed'])
    for name in c['parent_searches']:
        item = next(e for e in short['entries'] if e['name'] == name)
        winner = saved_theta[item['offset']+item['imagined_slot']]
        add(winner, dict(kind='imagined_winner', search=name))
        for scale in c['winner_contractions']:
            add(winner*scale, dict(kind='contracted_winner', search=name, scale=scale))
        for cap in c['action_caps']:
            add(winner, dict(kind='capped_winner', search=name, cap=cap), cap=cap)
        store_path = parent/'searches'/name/'queries'/f'k{item["noise_k"]}'
        store = QueryStore(store_path, study_identity=identity)
        for generation in c['sample_generations']:
            key = f'fitness-g{generation:04d}'
            path = store_path/f'{key}.npz'
            inputs[str(path.relative_to(root))] = file_sha256(path)
            with np.load(path, allow_pickle=False) as archive:
                receipt = json.loads(str(archive['__receipt__']))
                theta = archive['theta'].copy()
            store._validate_receipt(receipt, key)
            if digest_array(theta) != receipt['data_sha256']['theta']:
                raise ValueError('archived candidate parameters changed')
            indices = rng.choice(len(theta), c['samples_per_generation'], replace=False)
            for index in indices:
                add(theta[index], dict(kind='population_sample', search=name,
                                      generation=generation, index=int(index)))
    for direction in range(c['random_directions']):
        theta = rng.standard_normal(1020)*config['policy']['initial_CMA_sigma']
        for scale in c['random_scales']:
            add(theta*scale, dict(kind='random_direction', direction=direction, scale=scale))
    return rows, np.stack(parameters), inputs


def freeze_pool(directory, rows, theta, identity):
    return freeze(directory, 'candidate_pool', rows, theta, identity=identity)


def choose_controllers(rows, failure, progress, imagined_scores, minimum_progress_ratio):
    """All choices use selection data only; baseline is available in every group."""
    kinds = sorted({o['kind'] for row in rows for o in row['origins']} - {'baseline'})
    choices = {}
    groups = {'all': list(range(len(rows)))}
    groups.update({kind: [0]+[i for i,r in enumerate(rows) if i and any(
        o['kind']==kind for o in r['origins'])] for kind in kinds})
    for kind, indices in groups.items():
        decision = real_selection(np.asarray(failure)[indices], np.asarray(progress)[indices],
                                  minimum_progress_ratio=minimum_progress_ratio)
        choices[f'simulator/{kind}'] = dict(index=indices[decision['slot']], decision=decision)
        for name, scores in imagined_scores.items():
            # Chronological index resolves all score ties, including the baseline.
            index = indices[int(np.argmax(np.asarray(scores)[indices]))]
            choices[f'{name}/{kind}'] = dict(index=index)
    return choices


def diagnostic_analysis(rows, selection, check, choices, *, progress_floor, seed, replicates):
    """Paired episode bootstrap for fixed selections; all intervals exploratory.

    Candidate-wise checks and retrospective envelopes are descriptive and subject
    to multiplicity. Candidate rows are not independent experimental replicates.
    """
    from scipy.stats import spearmanr
    f, p = check['failure'], check['progress']
    if f.shape != p.shape or f.shape[0] != len(rows) or not np.isfinite(p).all():
        raise ValueError('invalid complete candidate outcomes')
    rng = np.random.default_rng(seed)
    indices = rng.integers(f.shape[1], size=(replicates, f.shape[1]))
    def interval(x):
        draws = x[indices].mean(1)
        return dict(point=float(x.mean()), lo=float(np.quantile(draws,.025)),
                    hi=float(np.quantile(draws,.975)))
    def eligible(fail, prog):
        if prog[0].mean() <= 0:
            return []
        return np.flatnonzero((fail.mean(1)<fail[0].mean()) &
                              (prog.mean(1)>=progress_floor*prog[0].mean())).tolist()
    selection_good = eligible(selection['failure'], selection['progress'])
    check_good = eligible(f,p)
    selected = {}
    for name, record in choices.items():
        i = record['index']
        denominators = p[0,indices].mean(1)
        ratio = None if p[0].mean() <= 0 or (denominators<=0).any() else dict(
            point=float(p[i].mean()/p[0].mean()),
            lo=float(np.quantile(p[i,indices].mean(1)/denominators,.025)),
            hi=float(np.quantile(p[i,indices].mean(1)/denominators,.975)))
        selected[name] = dict(candidate=rows[i]['id'], index=i, baseline_fallback=i==0,
            failure_difference=interval(f[i].astype(float)-f[0]), progress_ratio=ratio)
    correlations = {}
    for k, values in check['imagined_failure'].items():
        x,y = values.mean(1), f.mean(1)
        correlations[k] = None if np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic)
    table = []
    for i,row in enumerate(rows):
        table.append(dict(**row, selection_failure=float(selection['failure'][i].mean()),
            check_failure=float(f[i].mean()), check_failures=int(f[i].sum()),
            check_progress=float(p[i].mean()),
            check_progress_ratio=None if p[0].mean()<=0 else float(p[i].mean()/p[0].mean()),
            action_delta_rms=float(check['action_rms'][i]),
            imagined_failure={k:float(v[i].mean()) for k,v in check['imagined_failure'].items()},
            real_readout_failure=float(check['readout_failure'][i].mean()),
            endpoint_failure=float(check['endpoint_failure'][i].mean())))
    return dict(exploratory=True, candidate_count=len(rows), check_episodes=f.shape[1],
        baseline_failure=float(f[0].mean()), baseline_progress=float(p[0].mean()),
        selection_eligible_indices=selection_good, check_eligible_indices=check_good,
        eligible_on_both=sorted(set(selection_good)&set(check_good)),
        frozen_choices=selected, descriptive_rank_correlations=correlations, candidates=table,
        retrospective_check_envelope_is_not_an_independent_selected_result=True,
        interpretation='Reused development episodes; selected contrasts have unadjusted 95% '
        'paired episode intervals conditional on this fixed pool. Candidates are correlated, '
        'not independent seeds. No confirmation, full-episode, or population-safety claim.')


def execute_diagnostic(directory, config, rows, theta, banks, *, provenance,
                       real_callback, imagined_callback, read_costs, stop_after=None):
    """Freeze pool, query selection, freeze choices, then query development checks."""
    from helpers.evoReadinessQueries import atomic_json
    from helpers.evoRanking import segment_metrics
    from helpers.evoReadinessSearch import task_returns
    from helpers.evoReadinessStudy import checked_real_output, checked_audit_output
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    c = config['candidate_quality']
    expected_roles = {'imagined_selection','real_selection','evaluation'}
    if set(banks) != expected_roles or stop_after not in (None,'selection'):
        raise ValueError('three disjoint diagnostic roles required')
    episodes = [key for b in banks.values() for key in b.episode_keys]
    if len(episodes) != len(set(episodes)):
        raise ValueError('episode leakage')
    if len(rows)!=len(theta) or rows[0]['origins'] != [dict(kind='baseline')] or np.any(theta[0]):
        raise ValueError('baseline must be first')
    declaration = dict(config=config, provenance=provenance,
                       banks={r:b.identity() for r,b in banks.items()})
    identity = digest_json(declaration)
    manifest = directory/'study.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != dict(identity=identity,declaration=declaration):
            raise ValueError('study changed')
    else:
        if any(directory.iterdir()):
            raise ValueError('unknown nonempty study directory')
        atomic_json(manifest, dict(identity=identity,declaration=declaration))
    pool = freeze_pool(directory, rows, theta, identity)
    stores = {role:QueryStore(directory/role, study_identity=identity) for role in expected_roles}
    def imagined(role, i, k, barrier):
        n = 1 if k==0 else c['noise_samples']
        key = f'{rows[i]["id"]}-k{k}'
        request = dict(candidate=rows[i], theta=digest_array(theta[i]),
            bank=banks[role].identity(), noise=k, samples=n, seed=c['audit_seed'], barrier=barrier)
        out,_ = stores[role].execute(key, request=request,
            expected_costs=dict(predictor_rows=len(banks[role])*n*10, real_steps=0),
            callback=lambda:checked_audit_output(imagined_callback(role,rows[i],theta[i],k,n),
                                                 len(banks[role]),n,10), read_costs=read_costs)
        return segment_metrics(out['readout'])
    def real(role, i, barrier):
        if role=='evaluation':
            gate,_ = load_freeze(directory,'diagnostic_choices',identity=identity)
            if gate['freeze_sha256'] != barrier:
                raise ValueError('development checks require frozen choices')
        request = dict(candidate=rows[i], theta=digest_array(theta[i]),
                       bank=banks[role].identity(), barrier=barrier)
        out,_ = stores[role].execute(rows[i]['id'], request=request,
            expected_costs=dict(real_steps=len(banks[role])*100,predictor_rows=0),
            callback=lambda:checked_real_output(real_callback(role,rows[i],theta[i]),
                                                len(banks[role]),10),read_costs=read_costs)
        return out
    scores = {'imagined_k0_zero':[], 'imagined_k1_fixed':[]}
    selection = dict(failure=[],progress=[])
    for i in range(len(rows)):
        for k,name in [(0,'imagined_k0_zero'),(1,'imagined_k1_fixed')]:
            m = imagined('imagined_selection',i,k,pool['freeze_sha256'])
            penalty = 0 if k==0 else c['diagnostic_fixed_penalty']
            scores[name].append(float((task_returns(m)-penalty*m['violated']).mean()))
        out = real('real_selection',i,pool['freeze_sha256'])
        selection['failure'].append(out['dense_violated'])
        selection['progress'].append(out['progress'])
        print(f'[quality] selection {i+1}/{len(rows)}',flush=True)
    selection = {k:np.stack(v) for k,v in selection.items()}
    choices = choose_controllers(rows,selection['failure'],selection['progress'],scores,
                                 c['minimum_progress_ratio'])
    entries = [dict(name=name,**value) for name,value in choices.items()]
    chosen_theta = np.stack([theta[x['index']] for x in entries])
    gate = freeze(directory,'diagnostic_choices',entries,chosen_theta,identity=identity)
    if stop_after=='selection':
        return dict(complete=False,phase='choices_frozen',identity=identity)
    check = dict(failure=[],progress=[],action_rms=[],readout_failure=[],endpoint_failure=[])
    imagined_failures = {'0':[],'1':[]}
    for i in range(len(rows)):
        out = real('evaluation',i,gate['freeze_sha256'])
        check['failure'].append(out['dense_violated'])
        check['progress'].append(out['progress'])
        check['action_rms'].append(float(out['action_delta_rms']))
        check['readout_failure'].append(segment_metrics(out['readout'])['violated'])
        check['endpoint_failure'].append((out['endpoint_clearance']<=0).any(-1))
        for k in [0,1]:
            m = imagined('evaluation',i,k,gate['freeze_sha256'])
            imagined_failures[str(k)].append(m['violated'].mean(-1))
        print(f'[quality] development check {i+1}/{len(rows)}',flush=True)
    check = {k:np.stack(v) for k,v in check.items()}
    check['imagined_failure'] = {k:np.stack(v) for k,v in imagined_failures.items()}
    analysis = diagnostic_analysis(rows,selection,check,choices,
        progress_floor=c['minimum_progress_ratio'],seed=c['bootstrap_seed'],
        replicates=c['bootstrap_replicates'])
    accounting = {role:store.accounting() for role,store in stores.items()}
    if any(a['pending_queries'] for a in accounting.values()):
        raise ValueError('unresolved query')
    costs = {}
    for record in accounting.values():
        for key,value in record['completed_costs'].items():
            costs[key] = costs.get(key,0)+value
    expected = diagnostic_budget(config,len(rows))
    for key in ['real_steps','predictor_rows']:
        if costs[key] != expected[key]:
            raise ValueError(f'{key} accounting mismatch')
    result = dict(complete=True,identity=identity,analysis=analysis,costs=costs,
                  accounting=accounting,pool_sha256=pool['freeze_sha256'],
                  choice_sha256=gate['freeze_sha256'])
    atomic_json(directory/'completion.json',result)
    return result


def diagnostic_budget(config, count):
    c = config['candidate_quality']
    b = c['banks']
    return dict(candidates=count,
        predictor_rows=count*(b['imagined_selection']+b['evaluation'])*(1+c['noise_samples'])*10,
        real_steps=count*(b['real_selection']+b['evaluation'])*100,
        max_hours=c['max_hours'])
