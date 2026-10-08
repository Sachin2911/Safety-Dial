"""Two freeze barriers separate search, simulator selection, and final outcomes."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from helpers.evoFollowup import analyze, freeze, load_freeze, real_selection, search, shortlist
from helpers.evoFollowup import study_plan, validate_banks
from helpers.evoReadinessCMA import ReadinessCMAConfig
from helpers.evoReadinessQueries import ImaginedSearchScorer, QueryStore, atomic_json
from helpers.evoReadinessQueries import digest_array, digest_json
from helpers.evoReadinessStudy import checked_audit_output, checked_real_output
from helpers.evoRun import noise_seed


class DualAuditScorer:
    def __init__(self, imaginer, policy, banks, cfg, arm, folder, identity, counts,
                 noisy_samples, max_batch):
        self.own_k = arm['noise_k']
        self.scorers = {}
        for k in [0,1]:
            ac = replace(cfg, fitness_samples=1 if k==0 else cfg.fitness_samples,
                         selection_samples=1 if k==0 else noisy_samples)
            self.scorers[k] = ImaginedSearchScorer(imaginer, policy, banks['fitness'],
                banks['imagined_selection'], config=ac, noise_k=k,
                store=QueryStore(Path(folder)/f'k{k}', study_identity=identity),
                read_costs=counts, max_batch=max_batch)

    def fitness(self, *args):
        return self.scorers[self.own_k].fitness(*args)

    def selection(self, *args):
        results = {k: scorer.selection(*args) for k,scorer in self.scorers.items()}
        return results[self.own_k]


def execute_study(directory, config, banks, theta0, *, provenance, scorer_factory,
                  real_callback, audit_callback, read_costs, stop_after=None):
    """Callbacks are constructed outside; no final outcomes are requested until freeze.

    real_callback(role, theta); audit_callback(theta, k, n, stream).
    All simulator selection uses real_selection, never evaluation. The latter
    bank is named only in final-stage requests after all decisions are immutable.
    """
    plan = study_plan(config)
    if plan['blockers']:
        raise ValueError('; '.join(plan['blockers']))
    c,s = config['followup'], config['followup']['search']
    if stop_after not in (None, 'search', 'selection'):
        raise ValueError('unknown phase boundary')
    validate_banks(banks)
    if any(len(banks[role]) != n for role,n in c['banks'].items()):
        raise ValueError('bank sizes differ from protocol')
    if np.asarray(theta0).ndim != 1 or not np.isfinite(theta0).all():
        raise ValueError('finite initial parameter vector required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    declaration = dict(config=c, banks=validate_banks(banks), provenance=provenance,
                       theta0_sha256=digest_array(theta0))
    identity = digest_json(declaration)
    path = directory/'study.json'
    if path.exists():
        if json.loads(path.read_text()) != dict(identity=identity,declaration=declaration):
            raise ValueError('study declaration changed')
    else:
        if any(directory.iterdir()):
            raise ValueError('unknown nonempty study directory')
        atomic_json(path, dict(identity=identity,declaration=declaration))
    entries, parameters = [], []
    for arm in plan['arms']:
        for seed in c['seeds']:
            name = f'{arm["name"]}-seed{seed}'
            folder = directory/'searches'/name
            cfg = ReadinessCMAConfig(population=arm['population'], generations=s['generations'],
                sigma0=s['sigma0'], seed=seed, n_fitness_roots=len(banks['fitness']),
                fitness_roots_per_generation=s['fitness_roots_per_generation'],
                fitness_samples=arm['samples'], selection_roots=len(banks['imagined_selection']),
                selection_samples=1 if arm['noise_k']==0 else s['noisy_selection_samples'],
                checkpoints=(0,s['generations']))
            scorer = scorer_factory(cfg, arm, folder/'queries', identity)
            print(f'[followup] search {name}', flush=True)
            state = search(cfg, theta0, scorer, folder/'cma', mode=arm['penalty_mode'],
                           fixed_penalty=arm['fixed_penalty'], provenance=declaration)
            chosen = shortlist(state, c['shortlist_size'])
            start = len(parameters)
            parameters.extend(chosen.pop('theta'))
            entries.append(dict(name=name, arm=arm['name'], noise_k=arm['noise_k'], seed=seed,
                                offset=start, count=len(parameters)-start, **chosen))
            atomic_json(directory/'search_progress.json', dict(identity=identity,
                completed_searches=len(entries), expected_searches=plan['counts']['searches']))
    first = freeze(directory, 'shortlists', entries, np.stack(parameters), identity=identity)
    if stop_after=='search':
        return dict(complete=False, phase='shortlists_frozen', identity=identity)
    first, pool = load_freeze(directory, 'shortlists', identity=identity)
    stores = {role:QueryStore(directory/role, study_identity=identity)
              for role in ['real_selection','evaluation','final_audits']}

    def real(role, theta, barrier):
        # The appropriate global freeze is required even for the shared baseline.
        gate = 'shortlists' if role=='real_selection' else 'selected'
        record,_ = load_freeze(directory,gate,identity=identity)
        if record['freeze_sha256'] != barrier:
            raise ValueError('wrong selection barrier')
        n = len(banks[role])
        request = dict(role=role, bank=banks[role].identity(), theta=digest_array(theta),
                       freeze_sha256=barrier)
        result,_ = stores[role].execute(digest_array(theta), request=request,
            expected_costs=dict(real_steps=n*100,predictor_rows=0),
            callback=lambda:checked_real_output(real_callback(role,theta),n,10),
            read_costs=read_costs)
        return result

    selected, chosen_parameters = [], []
    for entry in first['entries']:
        candidates = pool[entry['offset']:entry['offset']+entry['count']]
        observed = [real('real_selection', theta, first['freeze_sha256']) for theta in candidates]
        decision = real_selection(np.stack([r['dense_violated'] for r in observed]),
            np.stack([r['progress'] for r in observed]), minimum_progress_ratio=c['minimum_progress_ratio'])
        for selector,slot in [('imagined',entry['imagined_slot']),('verified',decision['slot'])]:
            selected.append(dict(arm=entry['arm'],seed=entry['seed'],selector=selector,
                noise_k=entry['noise_k'],shortlist_name=entry['name'],shortlist_slot=slot,
                selection_decision=decision if selector=='verified' else None))
            chosen_parameters.append(candidates[slot])
    last = freeze(directory,'selected',selected,np.stack(chosen_parameters),identity=identity)
    if stop_after=='selection':
        return dict(complete=False,phase='all_final_selections_frozen',identity=identity)
    last, chosen_parameters = load_freeze(directory,'selected',identity=identity)
    print('[followup] all simulator selections frozen; final outcomes begin',flush=True)
    baseline = real('evaluation',theta0,last['freeze_sha256'])
    outcomes, cells = [], {}
    seeds = c['seeds']
    for entry, theta in [(dict(arm='baseline',selector='baseline',seed=seeds[0]),theta0),
                         *zip(last['entries'],chosen_parameters)]:
        r = baseline if entry['arm']=='baseline' else real('evaluation',theta,last['freeze_sha256'])
        probabilities = {}
        for k in [0,1]:
            n = 1 if k==0 else s['final_audit_samples']
            stream = noise_seed(entry['seed'],300000)
            request = dict(theta=digest_array(theta), k=k, samples=n, stream=stream,
                bank=banks['evaluation'].identity(), freeze_sha256=last['freeze_sha256'])
            # Each declared slot is audited at both noise settings; duplicates stay explicit.
            key = f'{entry["arm"]}-{entry["selector"]}-{entry["seed"]}-k{k}'
            audit,_ = stores['final_audits'].execute(key,request=request,
                expected_costs=dict(predictor_rows=len(banks['evaluation'])*n*10,real_steps=0),
                callback=lambda:checked_audit_output(audit_callback(theta,k,n,stream),
                    len(banks['evaluation']),n,10),read_costs=read_costs)
            from helpers.evoRanking import segment_metrics
            probabilities[str(k)] = segment_metrics(audit['readout'])['violated'].mean(-1).tolist()
        outcomes.append(dict(**entry, failure=r['dense_violated'].tolist(),
                             progress=r['progress'].tolist(), imagined_failure=probabilities))
        if entry['arm']!='baseline':
            key = entry['arm']+'/'+entry['selector']
            cells.setdefault(key,dict(failure=[],progress=[]))
            cells[key]['failure'].append(r['dense_violated'])
            cells[key]['progress'].append(r['progress'])
    baseline_cell = {key:np.broadcast_to(baseline[value],(len(seeds),len(banks['evaluation'])))
                     for key,value in [('failure','dense_violated'),('progress','progress')]}
    analysis = analyze(cells,baseline_cell,seeds=seeds,
        episode_keys=banks['evaluation'].episode_keys, replicates=c['bootstrap_replicates'],
        bootstrap_seed=c['bootstrap_seed'], minimum_progress_ratio=c['minimum_progress_ratio'],
        phase=c['phase'])
    atomic_json(directory/'outcomes.json',outcomes)
    atomic_json(directory/'analysis.json',analysis)
    account = {role:store.accounting() for role,store in stores.items()}
    for entry in first['entries']:
        total = 0
        for k in [0,1]:
            record = QueryStore(directory/'searches'/entry['name']/'queries'/f'k{k}',
                                study_identity=identity).accounting()
            account[entry['name']+f'/k{k}'] = record
            total += record['completed_costs'].get('predictor_rows',0)
        if total != plan['counts']['model_rows_per_search']:
            raise ValueError('actual model-query budgets are not matched')
    if any(record['pending_queries'] for record in account.values()):
        raise ValueError('unresolved queries cannot count as complete')
    totals = {}
    for record in account.values():
        for k,v in record['completed_costs'].items():
            totals[k] = totals.get(k,0)+v
    if totals['predictor_rows']!=plan['counts']['total_predictor_rows']:
        raise ValueError('total predictor budget mismatch')
    if totals['real_steps']>plan['counts']['maximum_real_selection_steps']+plan['counts']['maximum_final_real_steps']:
        raise ValueError('real-query cap exceeded')
    summary = dict(complete=True,identity=identity,phase=c['phase'],analysis=analysis,
                   actual_costs=totals,planned=plan['counts'],accounting=account)
    atomic_json(directory/'completion.json',summary)
    return summary
