"""Freeze diverse episode-excluded PushT cases against the validated full memory."""
import json
from pathlib import Path
import socket
import sys
import time
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'experiments'),str(ROOT/'experiments/scripts')]
from helpers import manip as M  # noqa: E402
from helpers.pusht_authority import distinct_start  # noqa: E402
import manip_pipeline as P  # noqa: E402
BASE = ROOT/'outputs/pusht-validated-20261102'
REFERENCE = ROOT/'outputs/pusht-ap-reproduction-20261102'


def main():
    assert socket.gethostname() == 'c45db82630e6'
    started = time.time()
    BASE.mkdir(exist_ok=True, parents=True)
    assert not (BASE/'data-receipt.json').exists()
    source = json.loads((REFERENCE/'freeze.json').read_text())
    cache = json.loads((REFERENCE/'cache-complete.json').read_text())
    assert M.digest(REFERENCE/'latents.npy') == cache['sha256']
    assert M.digest(ROOT/'assets/pusht/weights.pt') == source['checkpoint_sha256']
    assert M.digest(ROOT/'data/pusht_expert_train.h5') == source['data_sha256']
    M.save(BASE/'data-receipt.json',dict(started=started, dataset={'sha256':source['data_sha256']},
           checkpoint_sha256=source['checkpoint_sha256'], cache_sha256=cache['sha256'],
           baseline_freeze_sha256=M.digest(REFERENCE/'freeze.json')))
    rng = np.random.default_rng(18100)
    memory = source['memory_episodes']
    old = json.loads((ROOT/'configs/manip-screen/pusht-panel.json').read_text())
    excluded = {c['episode'] for k in ['stage_a','stage_b','development'] for c in old[k]}
    excluded.update(q['episode'] for q in source['queries'])
    previous = json.loads((ROOT/'outputs/pusht-authority-20261102/pusht/panel.json').read_text())
    excluded.update(c['episode'] for c in previous['cases'])
    allowed = set(range(18685)) - set(memory) - excluded
    candidates, rejected = [], []
    with h5py.File(ROOT/'data/pusht_expert_train.h5','r') as data:
        lengths, offsets = P.episode_index(data)
        starts = np.asarray(data['state'][offsets],np.float64)
        assert len(lengths)==185*101
        for group in range(185):
            assert np.allclose(starts[group*101:(group+1)*101], starts[group*101], atol=1e-4, rtol=0)
        for group in rng.permutation(185):
            chosen = None
            episodes = [e for e in range(int(group)*101,(int(group)+1)*101)
                        if e in allowed and lengths[e]>=141]
            for episode in rng.permutation(episodes):
                for local in rng.permutation(int(lengths[episode])-140)[:64]:
                    start = int(offsets[episode]+local)
                    if P.initially_successful('pusht',data,start,start+140):
                        continue
                    state = np.asarray(data['state'][start],np.float64)
                    if any(not distinct_start(state,c['start_state']) for c in candidates):
                        continue
                    states = np.asarray(data['state'][start:start+141],np.float64)
                    layout = M.pusht_layout(states,'on_path',M.TASKS['pusht'])
                    if layout is None:
                        continue
                    chosen = dict(id=f"validated-scene-{len(candidates):02d}",group=int(group),
                                  episode=int(episode), start_row=start, goal_row=start+140,
                                  horizon=140, start_state=state.tolist(), **layout)
                    break
                if chosen is not None:
                    break
            if chosen is None:
                rejected.append(dict(group=int(group),reason='no_eligible_geometry_window'))
                continue
            candidates.append(chosen)
            if len(candidates)==20:
                break
    panel = dict(status='ready' if len(candidates)==20 else 'insufficient_distinct_cases',
                 cases=candidates[:8], reserved_final_cases=candidates[8:],
                 memory_episodes=memory, excluded_episodes=sorted(set(range(18685))-set(memory)),
                 excluded_previous_episodes=sorted(excluded), rejected=rejected,
                 seed=18100, created=time.time(), grouping='101 episodes per source group',
                 memory_scope='episode excluded; sibling demonstrations allowed')
    M.save(BASE/'pusht/panel.json',panel)
    M.save(BASE/'pusht/memory-v1/panel.json',dict(memory_episodes=memory))
    M.save(BASE/'pusht/memory-v1/complete.json',dict(frames=0,reused_frames=2336736,
           reused_path=str(REFERENCE/'latents.npy'),sha256=cache['sha256']))
    print(json.dumps(dict(status=panel['status'],development=len(panel['cases']),
                         reserved=len(panel['reserved_final_cases']),groups=[c['group'] for c in candidates])),flush=True)
    assert panel['status']=='ready'


if __name__=='__main__':
    main()
