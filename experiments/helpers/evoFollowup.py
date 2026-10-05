"""Matched-query Walker2d follow-up: fixed penalties and independent real selection.

The Stage 5 engine is reused unchanged. Both audit noise levels are charged for
all nominees; only the arm's own audit determines its imagined ranking. Search
fitness rows match by trading population size against samples per candidate.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from helpers.evoReadinessAnalysis import CrossedBootstrap
from helpers.evoReadinessCMA import run_readiness_cma
from helpers.evoReadinessQueries import LatentBank, atomic_json, digest_array, digest_json
from helpers.evoRun import _atomic

ROLES = ('fitness', 'imagined_selection', 'real_selection', 'evaluation')


def split_banks(banks, roots, sizes):
    """Split the parent selection bank prospectively by episode order, never outcome."""
    ni, nr = sizes['imagined_selection'], sizes['real_selection']
    selections = {
        'fitness': ('fitness', 0, sizes['fitness']),
        'imagined_selection': ('selection', 0, ni),
        'real_selection': ('selection', ni, ni+nr),
        'evaluation': ('evaluation', 0, sizes['evaluation']),
    }
    out, physical = {}, {}
    for role, (parent, start, stop) in selections.items():
        b = banks[parent]
        if len(b) < stop:
            raise ValueError('parent bank too small for declared disjoint split')
        out[role] = LatentBank(b.episode_keys[start:stop], b.z_hist[start:stop].copy(),
                               b.hist_blocks[start:stop].copy())
        physical[role] = roots[parent][start:stop]
    validate_banks(out)
    return out, physical


def validate_banks(banks):
    if set(banks) != set(ROLES):
        raise ValueError('four explicitly separate episode roles required')
    seen = set()
    for role in ROLES:
        bank = banks[role]
        if seen.intersection(bank.episode_keys):
            raise ValueError(f'episode leakage at {role}')
        seen.update(bank.episode_keys)
    return {role: banks[role].identity() for role in ROLES}


def study_plan(config):
    c = config['followup']
    if c['phase'] not in ('engineering', 'development', 'confirmation'):
        raise ValueError('explicit engineering/development/confirmation phase required')
    b = c['banks']
    if set(b) != set(ROLES) or any(type(v) is not int or v < 2 for v in b.values()):
        raise ValueError('positive independent role sizes required')
    s = c['search']
    ints = ['generations', 'noisy_population', 'noisy_fitness_samples',
            'fitness_roots_per_generation', 'noisy_selection_samples', 'final_audit_samples']
    if any(type(s[k]) is not int or s[k] < 1 for k in ints):
        raise ValueError('positive integer search dimensions required')
    if s['noisy_population'] < 2 or s['fitness_roots_per_generation'] > b['fitness']:
        raise ValueError('invalid population or fitness subset')
    seeds = c['seeds']
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('at least two unique paired seeds required')
    if any(type(seed) is not int or not 0 <= seed < 2**32-1 for seed in seeds):
        raise ValueError('invalid search seed')
    penalties = c['fixed_penalties']
    if not penalties or len(set(penalties)) != len(penalties):
        raise ValueError('declare distinct fixed-penalty candidates')
    if any(not np.isfinite(x) or x <= 0 for x in penalties):
        raise ValueError('fixed penalties must be finite positive magnitudes')
    if type(c['shortlist_size']) is not int or c['shortlist_size'] < 2 or not 0 < c['minimum_progress_ratio'] <= 1:
        raise ValueError('invalid shortlist or progress floor')
    if not np.isfinite(c['max_hours']) or c['max_hours'] <= 0:
        raise ValueError('finite positive wall cap required')
    if c['bank_source'] not in ('fresh', 'reused_stage5'):
        raise ValueError('declare fresh or reused_stage5 bank source')
    if type(c['bootstrap_replicates']) is not int or c['bootstrap_replicates'] < 100:
        raise ValueError('at least 100 bootstrap replicates required')
    modes = [('zero', 0.), ('rosarl_style', 0.)]
    modes += [(f'fixed-{value:g}', float(value)) for value in penalties]
    arms = []
    for k in [0, 1]:
        samples = 1 if k == 0 else s['noisy_fitness_samples']
        population = s['noisy_population'] * s['noisy_fitness_samples'] // samples
        for mode, penalty in modes:
            arms.append(dict(name=f'k{k}-{mode}', noise_k=k, penalty_mode=mode,
                             fixed_penalty=penalty, population=population, samples=samples))
    fit = s['generations']*s['noisy_population']*s['noisy_fitness_samples']
    fit *= s['fitness_roots_per_generation']*10
    selection = (s['generations']+1)*b['imagined_selection']*(1+s['noisy_selection_samples'])*10
    jobs = len(arms)*len(seeds)
    slots = min(c['shortlist_size'], s['generations']+1)
    counts = dict(searches=jobs, fitness_rows_per_search=fit,
                  selection_audit_rows_per_search=selection,
                  model_rows_per_search=fit+selection,
                  search_predictor_rows=jobs*(fit+selection),
                  maximum_real_selection_steps=(1+jobs*(slots-1))*b['real_selection']*100,
                  maximum_final_real_steps=(1+2*jobs)*b['evaluation']*100,
                  final_audit_predictor_rows=(1+2*jobs)*b['evaluation']
                  *(1+s['final_audit_samples'])*10,
                  maximum_source_steps=sum(b.values())*4*1000
                  if c['bank_source']=='fresh' else 0)
    counts['maximum_new_real_steps'] = sum(counts[k] for k in
        ['maximum_real_selection_steps', 'maximum_final_real_steps', 'maximum_source_steps'])
    counts['total_predictor_rows'] = counts['search_predictor_rows']+counts['final_audit_predictor_rows']
    blockers = []
    if not c['execution_enabled']:
        blockers.append('execution disabled in this configuration')
    if c['phase']=='confirmation':
        if c['bank_source']!='fresh' or len(penalties)!=1:
            blockers.append('confirmation requires fresh banks and one development-locked penalty')
        if not c.get('development_lock'):
            blockers.append('development choice and sizing/throughput review must be locked first')
    elif c['bank_source']=='fresh':
        blockers.append('development uses declared reused banks; fresh confirmation is separate')
    if not np.isclose(s['sigma0'], config['policy']['initial_CMA_sigma']):
        blockers.append('search sigma must match the frozen residual preconditioner')
    return dict(arms=arms, counts=counts, blockers=blockers)


def fixed_metrics(result, magnitude):
    """Exactly U-lambda*v, keeping matched termination; old scorer stays unchanged."""
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError('positive finite fixed penalty required')
    out = dict(result)
    v = np.asarray(result['violated'], bool)
    for name in ['ret', 'ret_pre']:
        out[name] = np.asarray(result[name], float)-magnitude*v
    return out


def search(cfg, theta0, scorer, folder, *, mode, fixed_penalty, provenance,
           max_generations_this_call=None):
    if mode.startswith('fixed-'):
        cfg = replace(cfg, penalty_mode='zero')
        def fitness(*args):
            return fixed_metrics(scorer.fitness(*args), fixed_penalty)
        def selection(*args):
            return fixed_metrics(scorer.selection(*args), fixed_penalty)
    else:
        if mode not in ('zero', 'rosarl_style'):
            raise ValueError('unknown follow-up scoring rule')
        cfg = replace(cfg, penalty_mode=mode)
        fitness, selection = scorer.fitness, scorer.selection
    return run_readiness_cma(cfg, theta0, fitness, selection, folder,
        provenance=dict(**provenance, followup_mode=mode, fixed_penalty=fixed_penalty),
        max_generations_this_call=max_generations_this_call)


def shortlist(state, size):
    """Baseline plus best unique nonbaseline nominees; no real data is accepted here."""
    if type(size) is not int or size < 2:
        raise ValueError('shortlist must allow baseline and a nominee')
    g = state['generation']
    pick = state['snapshots'][g]
    scores = np.asarray(pick['scores'])
    if len(scores) != len(state['nominees']) or not np.isfinite(scores).all():
        raise ValueError('incomplete nominee audit scores')
    order = np.lexsort((np.arange(len(scores)), -scores))
    ids, seen = [0], {digest_array(state['nominees'][0]['theta'])}
    for j in order:
        digest = digest_array(state['nominees'][j]['theta'])
        if digest not in seen:
            ids.append(int(j))
            seen.add(digest)
        if len(ids) == size:
            break
    imagined = int(pick['nominee_index'])
    digest = digest_array(state['nominees'][imagined]['theta'])
    imagined_slot = next(i for i,j in enumerate(ids)
                         if digest_array(state['nominees'][j]['theta'])==digest)
    return dict(nominee_indices=ids, imagined_slot=imagined_slot,
                theta=np.stack([state['nominees'][j]['theta'] for j in ids]),
                scores=[float(scores[j]) for j in ids])


def real_selection(failures, progress, *, minimum_progress_ratio):
    """Empirical risk-first choice; baseline fallback is not a safety certificate."""
    f, p = np.asarray(failures), np.asarray(progress, float)
    if f.ndim != 2 or min(f.shape, default=0) < 1 or f.shape != p.shape or not np.isin(f, [0,1]).all():
        raise ValueError('candidate-by-independent-episode outcomes required')
    if not np.isfinite(p).all() or not 0 < minimum_progress_ratio <= 1:
        raise ValueError('finite progress and valid retention fraction required')
    risk, mean = f.mean(1), p.mean(1)
    # A nonpositive baseline denominator makes relative retention undefined.
    eligible = [] if mean[0] <= 0 else [i for i in range(1, len(f))
        if risk[i] < risk[0] and mean[i] >= minimum_progress_ratio*mean[0]]
    winner = min(eligible, key=lambda i: (risk[i], -mean[i], i)) if eligible else 0
    return dict(slot=winner, baseline_fallback=winner==0,
                empirical_failure_rates=risk.tolist(), mean_progress=mean.tolist(),
                eligible_slots=eligible, guarantee=False)


def freeze(directory, name, entries, theta, *, identity):
    """Write once, or verify the exact pre-existing freeze on resume."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path, weights = directory/f'{name}.json', directory/f'{name}.npz'
    theta = np.asarray(theta, float)
    if theta.ndim != 2 or not len(theta) or not np.isfinite(theta).all():
        raise ValueError('finite parameter matrix required')
    payload = dict(identity=identity, entries=entries, theta_sha256=digest_array(theta))
    payload['freeze_sha256'] = digest_json(payload)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError('frozen selection changed')
        load_freeze(directory, name, identity=identity)
    else:
        if weights.exists():
            with np.load(weights, allow_pickle=False) as old:
                if not np.array_equal(old['theta'], theta):
                    raise ValueError('interrupted freeze parameters differ')
        else:
            _atomic(weights, lambda f: np.savez_compressed(f, theta=theta))
        atomic_json(path, payload)
    return payload


def load_freeze(directory, name, *, identity):
    p = Path(directory)
    record = json.loads((p/f'{name}.json').read_text())
    content = {k:v for k,v in record.items() if k!='freeze_sha256'}
    if record['freeze_sha256'] != digest_json(content) or record['identity'] != identity:
        raise ValueError('freeze identity/content changed')
    with np.load(p/f'{name}.npz', allow_pickle=False) as f:
        theta = f['theta'].copy()
    if digest_array(theta) != record['theta_sha256']:
        raise ValueError('frozen parameters changed')
    return record, theta


def analyze(cells, baseline, *, seeds, episode_keys, replicates, bootstrap_seed,
            minimum_progress_ratio, phase):
    if len(set(seeds))!=len(seeds) or len(set(episode_keys))!=len(episode_keys):
        raise ValueError('independent axes require unique keys')
    shape = (len(seeds), len(episode_keys))
    boot = CrossedBootstrap(*shape, replicates=replicates, seed=bootstrap_seed)
    def checked(cell):
        f, p = np.asarray(cell['failure']), np.asarray(cell['progress'], float)
        if f.shape!=shape or p.shape!=shape or not np.isin(f,[0,1]).all():
            raise ValueError('complete binary seed-by-episode outcomes required')
        if not np.isfinite(p).all():
            raise ValueError('nonfinite progress')
        return f.astype(float), p
    bf,bp = checked(baseline)
    if not np.array_equal(bf, np.broadcast_to(bf[0],shape)) or not np.array_equal(bp,np.broadcast_to(bp[0],shape)):
        raise ValueError('one shared baseline required')
    pairs = {key: checked(cell) for key,cell in cells.items()}
    primary_key = 'k1-rosarl_style'
    vf,vp = pairs[primary_key+'/verified']
    imf,imp = pairs[primary_key+'/imagined']
    contrasts = {}
    for name, cf, cp in [('verified_minus_imagined',imf,imp),('verified_minus_baseline',bf,bp)]:
        risk = boot.interval(vf-cf, coverage=.975)
        retention = boot.ratio(vp, cp, coverage=.95)
        contrasts[name] = dict(failure_difference=risk, progress_ratio=retention,
            safety_reduction_supported=risk['hi']<0,
            progress_retained=bool(retention['defined'] and retention['lo']>=minimum_progress_ratio))
    rows = {}
    for name,(f,p) in pairs.items():
        rows[name] = dict(failure=boot.interval(f), progress=boot.interval(p),
            failure_minus_baseline=boot.interval(f-bf), progress_ratio_to_baseline=boot.ratio(p,bp))
    return dict(phase=phase, confirmatory=phase=='confirmation', primary=contrasts,
        family_coverage=.95, individual_primary_coverage=.975, secondary_coverage=.95,
        minimum_progress_ratio=minimum_progress_ratio, cells=rows,
        baseline=dict(failure=boot.interval(bf),progress=boot.interval(bp)),
        independent_episodes=len(episode_keys), paired_search_seeds=len(seeds),
        interpretation='Development is exploratory. Retaining the baseline is not improvement; '
        'short-branch and frozen-model limitations remain. Selection outcomes are not final outcomes.')
