"""Recorded model/simulator calls for the readiness study.

A completed query is one atomic numeric archive with input identity and measured
costs. Interrupted queries refuse implicit replay, including when the exact cost
of an abrupt process death is unknown. This module does not authorize a study.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from helpers.evoRanking import segment_metrics
from helpers.evoRun import _atomic, noise_seed


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def digest_array(value):
    a = np.asarray(value)
    return digest_json(dict(dtype=a.dtype.str, shape=list(a.shape),
        bytes_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest()))


def atomic_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()
    _atomic(Path(path), lambda f:f.write(data))


@dataclass(frozen=True)
class LatentBank:
    episode_keys: tuple
    z_hist: np.ndarray
    hist_blocks: np.ndarray

    def __post_init__(self):
        n = len(self.episode_keys)
        if n < 1 or any(not isinstance(k, str) or not k for k in self.episode_keys):
            raise ValueError('nonempty source-episode keys required')
        if len(set(self.episode_keys)) != n:
            raise ValueError('one root per independent source episode required')
        z, h = np.asarray(self.z_hist), np.asarray(self.hist_blocks)
        if z.shape != (n,3,192) or h.shape != (n,2,10,6):
            raise ValueError('Walker history shapes must be (R,3,192) and (R,2,10,6)')
        if not np.isfinite(z).all() or not np.isfinite(h).all():
            raise ValueError('finite histories required')

    def identity(self):
        return dict(episodes=list(self.episode_keys),z_hist=digest_array(self.z_hist),
                    hist_blocks=digest_array(self.hist_blocks))

    def __len__(self):
        return len(self.episode_keys)


def validate_roles(banks):
    if set(banks) != {'fitness','selection','evaluation'}:
        raise ValueError('exactly fitness, selection and evaluation banks required')
    seen = set()
    for name, bank in banks.items():
        if seen.intersection(bank.episode_keys):
            raise ValueError(f'source episodes overlap at role {name}')
        seen.update(bank.episode_keys)
    return {name:bank.identity() for name,bank in banks.items()}


class IncompleteQueryError(RuntimeError):
    """A query may have consumed resources; reconcile its journal before retrying."""


class QueryStore:
    def __init__(self, directory, *, study_identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        if not study_identity:
            raise ValueError('study identity required')
        self.study_identity = study_identity

    @staticmethod
    def _cost_delta(before, after):
        if before.keys() != after.keys():
            raise ValueError('cost counter keys changed during query')
        costs = {key:after[key]-before[key] for key in before}
        if any(not isinstance(v,(int,np.integer)) or v < 0 for v in costs.values()):
            raise ValueError('cost counters must be nondecreasing integer counts')
        return {k:int(v) for k,v in costs.items()}

    def execute(self, key, *, request, expected_costs, callback, read_costs):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',key):
            raise ValueError('query key must be a simple filename component')
        identity = digest_json(dict(study=self.study_identity,key=key,request=request,
                                    expected_costs=expected_costs))
        path = self.directory/f'{key}.npz'
        pending = self.directory/f'{key}.pending.json'
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                receipt = json.loads(str(saved['__receipt__']))
                arrays = {k:saved[k].copy() for k in saved.files if k!='__receipt__'}
            if receipt['identity'] != identity or receipt['study_identity'] != self.study_identity:
                raise ValueError('saved query has different inputs or requested costs')
            if receipt['data_sha256'] != {k:digest_array(v) for k,v in arrays.items()}:
                raise ValueError('saved query content hash mismatch')
            if pending.exists():
                if json.loads(pending.read_text())['identity'] != identity:
                    raise IncompleteQueryError('pending query belongs to different inputs')
                # Atomic archive committed, process stopped before journal cleanup.
                pending.unlink()
            return arrays,receipt
        if pending.exists():
            raise IncompleteQueryError('incomplete query retained; no automatic replay')
        before = dict(read_costs())
        entry = dict(identity=identity,study_identity=self.study_identity,key=key,
            request=request,requested_costs=expected_costs,phase='querying',
            started_at=time.time(),measured_costs=None,
            abrupt_death_costs='unknown within requested bounds; not assumed zero')
        atomic_json(pending,entry)
        try:
            result = callback()
            arrays = {k:np.asarray(v) for k,v in result.items()}
            if not arrays or '__receipt__' in arrays:
                raise ValueError('nonempty numeric output with no reserved receipt key required')
            if any(v.dtype.kind not in 'biufc' or not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('query outputs must be finite numeric arrays')
            costs = self._cost_delta(before,dict(read_costs()))
            if any(costs.get(k) != v for k,v in expected_costs.items()):
                raise ValueError(f'measured costs differ from requested shape: {costs}')
            receipt = dict(identity=identity,study_identity=self.study_identity,key=key,
                request=request,costs=costs,completed_at=time.time(),
                data_sha256={k:digest_array(v) for k,v in arrays.items()})
            entry.update(phase='committing',measured_costs=costs)
            atomic_json(pending,entry)
            _atomic(path,lambda f:np.savez_compressed(f,**arrays,__receipt__=json.dumps(receipt,sort_keys=True)))
            pending.unlink()
            return arrays,receipt
        except BaseException as exc:
            entry.update(phase='failed',measured_costs=self._cost_delta(before,dict(read_costs())),
                         error=type(exc).__name__,message=str(exc))
            atomic_json(pending,entry)
            raise

    def accounting(self):
        totals,receipts = {},[]
        for path in sorted(self.directory.glob('*.npz')):
            with np.load(path,allow_pickle=False) as saved:
                receipt=json.loads(str(saved['__receipt__']))
            if receipt['study_identity'] != self.study_identity:
                raise ValueError('mixed studies in query store')
            receipts.append(dict(key=receipt['key'],costs=receipt['costs']))
            for key,value in receipt['costs'].items():
                totals[key]=totals.get(key,0)+value
        pending=[json.loads(p.read_text()) for p in sorted(self.directory.glob('*.pending.json'))]
        committed={r['key'] for r in receipts}
        for entry in pending:
            entry['archive_committed']=entry['key'] in committed
        return dict(completed_costs=totals,completed_queries=receipts,pending_queries=pending)


class ImaginedSearchScorer:
    """Candidate-local fixed-shape calls; the final evaluation bank is never supplied."""
    def __init__(self,imaginer,policy,fitness,selection,*,config,noise_k,store,read_costs,
                 max_batch=16384):
        if set(fitness.episode_keys).intersection(selection.episode_keys):
            raise ValueError('fitness and selection episodes overlap')
        if len(fitness)!=config.n_fitness_roots or len(selection)!=config.selection_roots:
            raise ValueError('bank sizes differ from search declaration')
        if noise_k==0 and (config.fitness_samples!=1 or config.selection_samples!=1):
            raise ValueError('zero noise must use one sample')
        if max(config.fitness_roots_per_generation*config.fitness_samples,len(selection)*config.selection_samples)>max_batch:
            raise ValueError('declared candidate shape exceeds imagination batch limit')
        self.imaginer,self.policy,self.fitness_bank,self.selection_bank=imaginer,policy,fitness,selection
        self.config,self.noise_k,self.store,self.read_costs=config,float(noise_k),store,read_costs
        self.max_batch=max_batch

    def _query(self,X,indices,generation,role):
        cfg=self.config
        bank=self.fitness_bank if role=='fitness' else self.selection_bank
        n=cfg.fitness_samples if role=='fitness' else cfg.selection_samples
        stream=noise_seed(cfg.seed,generation if role=='fitness' else 100000)
        X,indices=np.asarray(X,np.float64),np.asarray(indices,np.int64)
        requested=len(X)*len(indices)*n*cfg.horizon_blocks
        request=dict(theta=digest_array(X),bank=bank.identity(),indices=indices.tolist(),
            k=self.noise_k,samples=n,noise_seed=stream,horizon=cfg.horizon_blocks)
        def calculate():
            metrics={name:[] for name in ['violated','ret','ret_pre']}
            for theta in X:
                out=self.imaginer.rollout_policies(self.policy,theta[None],bank.z_hist[indices],
                    bank.hist_blocks[indices],n_samples=n,k=self.noise_k,seed=stream,
                    n_blocks=cfg.horizon_blocks,max_batch=self.max_batch)
                m=segment_metrics(out['readout'][0])
                for name in metrics:
                    metrics[name].append(m[name].ravel())
            return dict(**{k:np.stack(v) for k,v in metrics.items()},rows=requested,
                        theta=X,source_indices=indices)
        arrays,_=self.store.execute(f'{role}-g{generation:04d}',request=request,
            expected_costs=dict(predictor_rows=requested,real_steps=0),
            callback=calculate,read_costs=self.read_costs)
        return arrays

    def fitness(self,X,indices,generation):
        return self._query(X,indices,generation,'fitness')

    def selection(self,theta,generation):
        return self._query(np.asarray(theta)[None],np.arange(len(self.selection_bank)),generation,'selection')
