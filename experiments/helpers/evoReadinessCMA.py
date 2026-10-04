"""Checkpointed readiness search with separate selection and fitness-only penalties.

This is implementation preparation, not approval to execute the main experiment.
A completed-generation checkpoint owns all optimiser/RNG, nominee and penalty state.
An incomplete generation is retained and refuses automatic replay because its partial
query costs need reconciliation. Existing historical experiment modules are untouched.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import pickle
from pathlib import Path

import cma
import numpy as np

from helpers.evoRanking import RosarlPenalty, rosarl_scores
from helpers.evoRun import CMAConfig, _atomic, _global_rng_kept, _pin_sampler, cma_options, generation_roots


@dataclass(frozen=True)
class ReadinessCMAConfig:
    population: int
    generations: int
    sigma0: float
    seed: int
    n_fitness_roots: int
    fitness_roots_per_generation: int
    fitness_samples: int
    selection_roots: int
    selection_samples: int
    penalty_mode: str = 'zero'
    horizon_blocks: int = 10
    checkpoints: tuple = (0, 1, 4, 16)


class PendingGenerationError(RuntimeError):
    """A partial generation must be audited before any possibly duplicated queries."""


def checked_metrics(result, candidates, segments, expected_rows):
    arrays = {k:np.asarray(result[k]) for k in ['violated', 'ret', 'ret_pre']}
    if any(a.shape != (candidates, segments) for a in arrays.values()):
        raise ValueError('score callback must return candidate-by-segment metrics')
    if not np.isin(arrays['violated'], [0, 1]).all():
        raise ValueError('violation flags must be binary')
    arrays['violated'] = arrays['violated'].astype(bool)
    arrays['ret'], arrays['ret_pre'] = [arrays[k].astype(np.float64) for k in ['ret', 'ret_pre']]
    if not all(np.isfinite(arrays[k]).all() for k in ['ret','ret_pre']):
        raise FloatingPointError('nonfinite outcomes cannot be treated as safe')
    if not np.array_equal(arrays['ret'][~arrays['violated']], arrays['ret_pre'][~arrays['violated']]):
        raise ValueError('safe outcomes must retain their full return')
    if result['rows'] != expected_rows:
        raise ValueError('callback predictor rows differ from declared query shape')
    return arrays


def selection_pick(nominees, penalty):
    """Re-score every stored nominee at one current penalty; earliest nominee wins ties."""
    scores = [float(rosarl_scores(n['metrics']['violated'], n['metrics']['ret'],
                                 n['metrics']['ret_pre'], penalty)[0]) for n in nominees]
    j = int(np.argmax(scores))
    return dict(nominee_index=j, nominee_generation=nominees[j]['generation'],
                theta=nominees[j]['theta'].copy(), scores=scores, penalty=float(penalty))


def run_readiness_cma(cfg, theta0, fitness_score, selection_score, out_dir, *, provenance,
                      max_generations_this_call=None):
    """fitness_score(X, root_ids, g) and selection_score(theta, g) return metrics+rows.

    g is 0-based for fitness and the nominee's generation for selection (baseline=0).
    The selection callback must use a fixed bank and noise stream independent of g.
    `provenance` pins all model/policy/root/scaler inputs outside this generic core.
    `max_generations_this_call` stops only after a complete checkpoint; it never alters
    the planned experiment or chooses a stopping point from results.
    """
    if cfg.penalty_mode not in ['zero', 'rosarl_style']:
        raise ValueError('unknown penalty mode')
    if min(cfg.population, cfg.n_fitness_roots, cfg.fitness_roots_per_generation,
           cfg.fitness_samples, cfg.selection_roots, cfg.selection_samples, cfg.horizon_blocks) < 1:
        raise ValueError('positive query sizes required')
    if cfg.population < 2 or cfg.generations < 0 or cfg.fitness_roots_per_generation > cfg.n_fitness_roots:
        raise ValueError('invalid population, generations or fitness bank')
    if not np.isfinite(cfg.sigma0) or cfg.sigma0 <= 0 or not 0 <= cfg.seed < 2**32-1:
        raise ValueError('finite positive sigma and a valid seed required')
    if any(g < 0 or g > cfg.generations for g in cfg.checkpoints):
        raise ValueError('checkpoint lies outside declared generation budget')
    if max_generations_this_call is not None and max_generations_this_call < 0:
        raise ValueError('nonnegative scheduling limit required')
    if not provenance:
        raise ValueError('nonempty external-input provenance required')
    theta0 = np.asarray(theta0, np.float64)
    if theta0.ndim != 1 or len(theta0) < 2 or not np.isfinite(theta0).all():
        raise ValueError('finite initial parameter vector required')
    identity_text = json.dumps(dict(config=asdict(cfg), provenance=provenance,
        theta0_sha256=hashlib.sha256(theta0.tobytes()).hexdigest(), cma_version=cma.__version__), sort_keys=True)
    identity = hashlib.sha256(identity_text.encode()).hexdigest()
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint, pending = path/'checkpoint.pkl', path/'pending_generation.json'
    def journal(**entry):
        payload = dict(identity=identity, **entry)
        _atomic(pending, lambda f:f.write((json.dumps(payload,indent=2)+'\n').encode()))
    def commit(state):
        _atomic(checkpoint, lambda f:pickle.dump(state,f,protocol=pickle.HIGHEST_PROTOCOL))
        pending.unlink(missing_ok=True)
    if checkpoint.exists():
        with checkpoint.open('rb') as f:
            state = pickle.load(f)
        if state.get('schema') != 1 or state.get('identity') != identity:
            raise ValueError('checkpoint belongs to a different declared experiment')
        if pending.exists():
            entry = json.loads(pending.read_text())
            if entry.get('identity') == identity and entry.get('target_generation') == state['generation']:
                # Checkpoint replacement completed, but deletion of the transaction marker did not.
                pending.unlink()
            else:
                raise PendingGenerationError('Incomplete generation retained; inspect its phase and charged receipts before any replay')
    else:
        if any(path.iterdir()):
            raise PendingGenerationError('No complete checkpoint; existing partial work will not be overwritten')
        base_cfg = CMAConfig(popsize=cfg.population, generations=cfg.generations, sigma0=cfg.sigma0,
            seed=cfg.seed, k_roots=cfg.fitness_roots_per_generation, cma_options={'CMA_diagonal':True})
        with _global_rng_kept():
            es = cma.CMAEvolutionStrategy(theta0, cfg.sigma0, cma_options(base_cfg))
        rows = cfg.selection_roots*cfg.selection_samples*cfg.horizon_blocks
        journal(target_generation=0, phase='baseline_selection', requested_rows=rows, completed_rows=0)
        baseline = checked_metrics(selection_score(theta0.copy(),0),1,cfg.selection_roots*cfg.selection_samples,rows)
        nominees = [dict(generation=0,theta=theta0.copy(),metrics=baseline)]
        state = dict(schema=1,identity=identity,identity_text=identity_text,generation=0,es=es,
            penalty=RosarlPenalty().state(),nominees=nominees,history=[],snapshots={},
            costs=dict(fitness_rows=0,selection_rows=rows,candidates=0,generations=0))
        if 0 in cfg.checkpoints:
            state['snapshots'][0] = selection_pick(nominees,0.)
        commit(state)
    stop = cfg.generations if max_generations_this_call is None else min(cfg.generations,state['generation']+max_generations_this_call)
    while state['generation'] < stop:
        g = state['generation']
        es = state['es']
        penalty = RosarlPenalty.from_state(state['penalty'])
        journal(target_generation=g+1,phase='sampling',requested_rows=0,completed_rows=0)
        with _global_rng_kept():
            _pin_sampler(es)
            asked = es.ask()
        X = np.asarray(asked,np.float64)
        indices = generation_roots(cfg.seed,g,cfg.n_fitness_roots,cfg.fitness_roots_per_generation)
        nf = cfg.fitness_roots_per_generation*cfg.fitness_samples
        fit_rows = cfg.population*nf*cfg.horizon_blocks
        journal(target_generation=g+1,phase='fitness',requested_rows=fit_rows,completed_rows=0)
        fit = checked_metrics(fitness_score(X.copy(),indices.copy(),g),cfg.population,nf,fit_rows)
        # Both arms observe exactly the same termination definition; only the extra penalty differs.
        penalty.update(np.where(fit['violated'],fit['ret_pre'],fit['ret']))
        applied = penalty.penalty if cfg.penalty_mode=='rosarl_style' else 0.
        scores = rosarl_scores(fit['violated'],fit['ret'],fit['ret_pre'],applied)
        order = np.lexsort((np.arange(len(scores)),-scores))
        ranks = np.empty(len(order))
        ranks[order] = np.arange(len(order))
        winner = int(order[0])
        select_rows = cfg.selection_roots*cfg.selection_samples*cfg.horizon_blocks
        journal(target_generation=g+1,phase='nominee_selection',requested_rows=select_rows,
                completed_rows=fit_rows,fitness_nominee=winner)
        selected = checked_metrics(selection_score(X[winner].copy(),g+1),1,
            cfg.selection_roots*cfg.selection_samples,select_rows)
        journal(target_generation=g+1,phase='checkpoint_commit',requested_rows=0,
                completed_rows=fit_rows+select_rows)
        with _global_rng_kept():
            es.tell(asked,ranks.tolist())
        state['nominees'].append(dict(generation=g+1,theta=X[winner].copy(),metrics=selected))
        state['generation'] = g+1
        state['penalty'] = penalty.state()
        state['history'].append(dict(generation=g+1,fitness_roots=indices.copy(),nominee=winner,
            fitness_scores=scores.copy(),applied_penalty=applied,bounds=penalty.state(),asked_sha256=hashlib.sha256(X.tobytes()).hexdigest()))
        if g+1 in cfg.checkpoints:
            state['snapshots'][g+1] = selection_pick(state['nominees'],applied)
        for key,value in [('fitness_rows',fit_rows),('selection_rows',select_rows),('candidates',cfg.population),('generations',1)]:
            state['costs'][key] += value
        commit(state)
    return state
