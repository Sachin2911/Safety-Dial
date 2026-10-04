"""Prospective whole-episode banks with fixed role ranges and charged resumable attempts.

Collection choices are deterministic functions of predeclared seeds, never of
controller outcomes. This implementation does not authorize new final episodes.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from helpers.evoReadiness import representative_root
from helpers.evoReadinessQueries import LatentBank, QueryStore, atomic_json, digest_json, validate_roles
from helpers.evoRoots import RootSet, load_roots_json, rootset_digest
from helpers.evoRun import _atomic
from helpers.evoReal import _step
from helpers.walkerRules import health_clearance, rule_unsafe


@dataclass(frozen=True)
class CollectionRole:
    name: str
    roots: int
    environment_seed_base: int
    actor_noise_seed_base: int
    attempts_per_root: int = 4
    max_steps: int = 1000


class SourceEpisodeFailure(RuntimeError):
    def __init__(self,message,episode_prefix):
        super().__init__(message)
        self.episode_prefix=episode_prefix


def generate_source_episode(env,actor,*,seed,max_steps,counter):
    """Readiness-audit collection semantics, retaining the completed prefix on failure."""
    env.reset(seed=int(seed))
    actor.reset()
    u=env.unwrapped
    qp,qv=np.empty((max_steps+1,9)),np.empty((max_steps+1,9))
    actions=np.empty((max_steps,6),np.float32)
    xv=np.empty(max_steps)
    qp[0],qv[0]=u.data.qpos.copy(),u.data.qvel.copy()
    n=0
    def prefix():
        return dict(qpos=qp[:n+1].copy(),qvel=qv[:n+1].copy(),action=actions[:n].copy(),x_velocity=xv[:n].copy())
    try:
        for t in range(max_steps):
            counter['teacher_queries']+=1
            action=np.asarray(actor.act(u._get_obs()),np.float32)
            if action.shape!=(6,) or not np.isfinite(action).all():
                raise FloatingPointError('invalid source actor action')
            actions[t]=np.clip(action,-1,1)
            counter['real_steps']+=1
            _step(u,actions[t],qp,qv,xv,t)
            n=t+1
            if bool(rule_unsafe('health',health_clearance(qp[n,1],qp[n,2]))):
                break
    except BaseException as exc:
        raise SourceEpisodeFailure(f'{type(exc).__name__}: {exc}',prefix()) from exc
    return prefix()


def validate_source_episode(episode,max_steps):
    n=len(episode['action'])
    shapes=dict(qpos=(n+1,9),qvel=(n+1,9),action=(n,6),x_velocity=(n,))
    if not 1<=n<=max_steps or any(np.asarray(episode[k]).shape!=v for k,v in shapes.items()):
        raise ValueError('invalid source trajectory shapes or length')
    if any(not np.isfinite(episode[k]).all() for k in shapes):
        raise ValueError('nonfinite source trajectory')
    if (np.abs(episode['action'])>1).any():
        raise ValueError('recorded source actions were not clipped')
    qp=np.asarray(episode['qpos'])
    failures=np.flatnonzero(rule_unsafe('health',health_clearance(qp[:,1],qp[:,2])))
    if (len(failures) and (len(failures)!=1 or failures[0]!=n)) or (n<max_steps and not len(failures)):
        raise ValueError('source episode did not stop at first dense violation or declared cap')
    return n


def validate_collection_roles(specs):
    if {s.name for s in specs}!={'fitness','selection','evaluation'} or len(specs)!=3:
        raise ValueError('exactly three independent collection roles required')
    environments,noises=set(),set()
    for spec in specs:
        if min(spec.roots,spec.attempts_per_root)<1 or spec.max_steps<=120:
            raise ValueError('positive bank size/cap and more than 120 source steps required')
        count=spec.roots*spec.attempts_per_root
        er=set(range(spec.environment_seed_base,spec.environment_seed_base+count))
        nr=set(range(spec.actor_noise_seed_base,spec.actor_noise_seed_base+count))
        if min(er|nr)<0 or max(er|nr)>=2**32 or environments&er or noises&nr:
            raise ValueError('valid disjoint environment and actor-noise seed ranges required')
        environments.update(er)
        noises.update(nr)


def collect_source_banks(directory,specs,*,actor_names,action_noise,provenance,
                         episode_callback,counter,max_new_attempts=None):
    """episode_callback(request,counter) returns a full numeric source episode.

    The counter charges physics and teacher queries at their call sites. Completed
    episodes are reused; failed or abruptly interrupted attempts cannot auto-replay.
    Pausing is a scheduling limit on fresh attempts and does not change any role cap.
    """
    validate_collection_roles(specs)
    if not actor_names or len(set(actor_names))!=len(actor_names) or not action_noise:
        raise ValueError('fixed nonempty source actor/noise pools required')
    if any(not np.isfinite(v) or v<0 for v in action_noise) or not provenance:
        raise ValueError('finite noise scales and frozen input provenance required')
    if max_new_attempts is not None and max_new_attempts<1:
        raise ValueError('positive scheduling allowance required')
    declaration=dict(schema=1,roles=[asdict(s) for s in specs],actor_names=list(actor_names),
        action_noise=list(action_noise),provenance=provenance,
        actor_choice='default_rng(SeedSequence([actor_noise_seed, 0x4d495854]))',
        root_choice='default_rng(SeedSequence([environment_seed, 0x524f4f54]))',
        root_eligibility='length > 120; root uniform [20, length-100)',
        collection_outcomes='source-teacher trajectories only, no visual controller queries')
    identity=digest_json(declaration)
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'collection_declaration.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=identity or digest_json(saved['declaration'])!=identity:
            raise ValueError('collection declaration changed; cannot resume')
    else:
        if any(directory.iterdir()):
            raise ValueError('unrecognized nonempty collection directory')
        atomic_json(path,dict(identity=identity,declaration=declaration))
    store=QueryStore(directory/'episodes',study_identity=identity)
    if list(store.directory.glob('*.pending.json')):
        from helpers.evoReadinessQueries import IncompleteQueryError
        raise IncompleteQueryError('source collection has an unresolved attempted episode')
    fresh=0
    roles,attempts={},[]
    for spec in specs:
        roots=[]
        for i in range(spec.roots*spec.attempts_per_root):
            key=f'{spec.name}-attempt{i:05d}'
            existing=(store.directory/f'{key}.npz').exists()
            if not existing and max_new_attempts is not None and fresh>=max_new_attempts:
                summary=dict(identity=identity,complete=False,accepted={k:len(v) for k,v in roles.items()},
                    current_role=spec.name,current_accepted=len(roots),attempts=attempts,accounting=store.accounting())
                atomic_json(directory/'collection_progress.json',summary)
                return summary
            env_seed=spec.environment_seed_base+i
            actor_seed=spec.actor_noise_seed_base+i
            mix=np.random.default_rng(np.random.SeedSequence([actor_seed,0x4d495854]))
            actor=actor_names[int(mix.integers(len(actor_names)))]
            sigma=float(action_noise[int(mix.integers(len(action_noise)))])
            request=dict(role=spec.name,attempt=i,actor=actor,action_noise=sigma,
                environment_seed=env_seed,actor_noise_seed=actor_seed,max_steps=spec.max_steps)
            def collect():
                before={k:counter[k] for k in ['real_steps','teacher_queries']}
                episode=None
                try:
                    episode=episode_callback(request,counter)
                    n=validate_source_episode(episode,spec.max_steps)
                    if any(counter[k]-before[k]!=n for k in before):
                        raise ValueError('source physics/teacher counts differ from trajectory length')
                    return episode
                except BaseException as exc:
                    partial=getattr(exc,'episode_prefix',episode)
                    if partial is not None:
                        _atomic(directory/f'failed-{key}.npz',lambda f:np.savez_compressed(f,**partial))
                    raise
            episode,receipt=store.execute(key,request=request,expected_costs={'predictor_rows':0},
                callback=collect,read_costs=lambda:dict(counter))
            fresh+=not existing
            n=validate_source_episode(episode,spec.max_steps)
            root=representative_root(episode,episode_id=env_seed,
                rng=np.random.default_rng(np.random.SeedSequence([env_seed,0x524f4f54])))
            row=dict(**request,steps=n,eligible=root is not None,query_costs=receipt['costs'])
            if root is not None:
                root.meta.update(role=spec.name,actor=actor,sigma=sigma,seed=env_seed,
                    actor_noise_seed=actor_seed,source_file=f'episodes/{key}.npz',collection_identity=identity)
                roots.append(root)
                row.update(root_id=root.root_id,root_step=root.step)
            attempts.append(row)
            if len(roots)==spec.roots:
                break
        if len(roots)!=spec.roots:
            failure=dict(identity=identity,complete=False,error='declared source-attempt cap exhausted',
                role=spec.name,accepted=len(roots),required=spec.roots,attempts=attempts,accounting=store.accounting())
            atomic_json(directory/'collection_failed.json',failure)
            raise RuntimeError('source-attempt cap exhausted; cannot fill with old or sibling roots')
        roles[spec.name]=roots
        atomic_json(directory/f'{spec.name}_roots.json',dict(roots=[r.to_dict() for r in roots]))
    bank_roles={}
    for name,roots in roles.items():
        path=directory/f'{name}_roots.json'
        bank_roles[name]=dict(roots=len(roots),rootset_digest=rootset_digest(RootSet.from_roots(name,roots)),
            path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            episode_keys=[f'environment-seed:{r.episode}' for r in roots])
    complete=dict(identity=identity,complete=True,roles=bank_roles,attempts=attempts,
        accounting=store.accounting(),provenance=provenance)
    complete['banks_sha256']=digest_json(complete)
    atomic_json(directory/'banks.json',complete)
    atomic_json(directory/'collection_progress.json',complete)
    return complete


def load_collected_roots(directory):
    directory=Path(directory)
    banks=json.loads((directory/'banks.json').read_text())
    declared=json.loads((directory/'collection_declaration.json').read_text())
    if banks.get('banks_sha256')!=digest_json({k:v for k,v in banks.items() if k!='banks_sha256'}):
        raise ValueError('collected bank metadata changed')
    if banks['identity']!=declared['identity'] or digest_json(declared['declaration'])!=declared['identity']:
        raise ValueError('collection identity changed')
    if not banks['complete'] or banks['accounting']['pending_queries']:
        raise ValueError('only complete source banks can be encoded or evaluated')
    roles={}
    for name,spec in banks['roles'].items():
        path=directory/spec['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
            raise ValueError('collected root file hash mismatch')
        roles[name]=load_roots_json(path)
        if rootset_digest(RootSet.from_roots(name,roles[name]))!=spec['rootset_digest']:
            raise ValueError('collected root content differs from declared identity')
    return banks,roles


def encode_collected_banks(directory,*,encoder_provenance,encode_callback,read_costs,max_new_roots=None):
    """Cache three initial history frames per root, without controller outcome queries."""
    directory=Path(directory)
    collection,roots=load_collected_roots(directory)
    if not encoder_provenance or (max_new_roots is not None and max_new_roots<1):
        raise ValueError('encoder identity and a positive scheduling allowance required')
    declaration=dict(collection_sha256=collection['banks_sha256'],encoder=encoder_provenance,
                     shape=[3,192],encoding_batch='one root, three frames per call')
    identity=digest_json(declaration)
    path=directory/'encoding_declaration.json'
    if path.exists():
        if json.loads(path.read_text())!={'identity':identity,'declaration':declaration}:
            raise ValueError('encoder declaration changed; cannot reuse bank features')
    else:
        atomic_json(path,dict(identity=identity,declaration=declaration))
    store=QueryStore(directory/'encoded_roots',study_identity=identity)
    fresh,completed=0,0
    role_files={}
    for name,rows in roots.items():
        zs=[]
        for root in rows:
            key=f'{name}-episode{root.episode}'
            existing=(store.directory/f'{key}.npz').exists()
            if not existing and max_new_roots is not None and fresh>=max_new_roots:
                progress=dict(identity=identity,complete=False,encoded_roots=completed,accounting=store.accounting())
                atomic_json(directory/'encoding_progress.json',progress)
                return progress
            def encode():
                z=np.asarray(encode_callback(root))
                if z.shape!=(3,192):
                    raise ValueError('encoder must return three 192-dimensional history latents')
                return {'z_hist':z}
            result,_=store.execute(key,request=dict(root_id=root.root_id,collection_sha256=collection['banks_sha256']),
                expected_costs=dict(real_steps=0,predictor_rows=0,renders=3,encodes=3),
                callback=encode,read_costs=read_costs)
            zs.append(result['z_hist'])
            fresh+=not existing
            completed+=1
        features=directory/f'{name}_history_latents.npz'
        payload=dict(z_hist=np.stack(zs),hist_blocks=np.stack([r.history_actions for r in rows]))
        if features.exists():
            with np.load(features,allow_pickle=False) as old:
                if set(old.files)!=set(payload) or any(not np.array_equal(old[k],v) for k,v in payload.items()):
                    raise ValueError('existing role cache differs from verified root features')
        else:
            _atomic(features,lambda f:np.savez_compressed(f,**payload))
        role_files[name]=dict(path=features.name,sha256=hashlib.sha256(features.read_bytes()).hexdigest(),
            episode_keys=collection['roles'][name]['episode_keys'])
    result=dict(identity=identity,complete=True,collection_sha256=collection['banks_sha256'],
                roles=role_files,accounting=store.accounting())
    result['encoded_banks_sha256']=digest_json(result)
    atomic_json(directory/'encoded_banks.json',result)
    atomic_json(directory/'encoding_progress.json',result)
    return result


def load_encoded_banks(directory):
    directory=Path(directory)
    collection,roots=load_collected_roots(directory)
    encoded=json.loads((directory/'encoded_banks.json').read_text())
    if encoded.get('encoded_banks_sha256')!=digest_json({k:v for k,v in encoded.items() if k!='encoded_banks_sha256'}):
        raise ValueError('encoded bank metadata changed')
    if not encoded['complete'] or encoded['collection_sha256']!=collection['banks_sha256'] or encoded['accounting']['pending_queries']:
        raise ValueError('incomplete or mismatched encoded banks')
    banks={}
    for role,spec in encoded['roles'].items():
        path=directory/spec['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
            raise ValueError('history feature file hash mismatch')
        if spec['episode_keys']!=collection['roles'][role]['episode_keys']:
            raise ValueError('encoded source-episode order differs from collection')
        with np.load(path,allow_pickle=False) as features:
            banks[role]=LatentBank(tuple(spec['episode_keys']),features['z_hist'].copy(),features['hist_blocks'].copy())
        if not np.array_equal(banks[role].hist_blocks,np.stack([r.history_actions for r in roots[role]])):
            raise ValueError('action histories differ from recorded source episodes')
    validate_roles(banks)
    return encoded,roots,banks
