"""Freeze and inspect saved training panels only; no new environment episodes."""
import hashlib
import json
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/safety-dial-20261102'


def main():
    assert socket.gethostname() == '075cb160c9b3'
    OUT.mkdir(exist_ok=False, parents=True)
    paths = [ROOT / 'outputs/ppo-20261101/run-v1/summary.json',
             ROOT / 'outputs/ppo-20261101/l4-summary-for-analysis.json',
             ROOT / 'SafetyDialDevelopment.md', Path(__file__)]
    hashes = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        target = OUT / 'sources' / rel
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes(path.read_bytes())
        hashes[str(rel)] = hashlib.sha256(target.read_bytes()).hexdigest()
    freeze = dict(created=time.time(), sources=hashes, risk_targets=[0, .25, .5, .75, 1],
                  simulator_action_cap=0, max_seconds=1800, final_outcomes_used=False)
    (OUT / 'freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
    summaries = [json.loads(path.read_text()) for path in paths[:2]]
    groups = {}
    for summary in summaries:
        assert summary['status'] == 'complete'
        for key, search in summary['searches'].items():
            pool = [c for c in summary['candidates'] if c['search'] == key and c['eligible']]
            assert pool and all(c['n'] == c['episodes'] == 8 for c in pool)
            selected = []
            for epsilon in freeze['risk_targets']:
                eligible = [c for c in pool if c['violations'] / c['n'] <= epsilon]
                best = max(eligible, key=lambda c: (c['complete'], c['safe'], -c['violations'])) if eligible else None
                selected.append(dict(epsilon=epsilon, candidate=None if best is None else {
                    k: best[k] for k in ['id', 'complete', 'safe', 'violations', 'n']
                }))
            useful = [s['candidate'] for s in selected if s['candidate'] is not None
                      and s['candidate']['complete'] >= 4]
            groups[key] = dict(host=summary['host'], optimizer=search['optimizer'],
                               candidates=len(pool), selections=selected,
                               distinct_selected=len({s['candidate']['id'] for s in selected
                                                      if s['candidate'] is not None}),
                               useful_risk_variation=len({c['violations'] for c in useful}) >= 2)
    result = dict(groups=groups, zero_event_iid_upper95=1-.05**(1/8),
                  warning='Illustration assumes IID; reused constructed panels do not certify risk',
                  completed=time.time(), simulator_actions=0, final_outcomes_used=False)
    assert result['completed'] - freeze['created'] < 1800
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: dict(candidates=v['candidates'], distinct=v['distinct_selected'],
                              useful_risk_variation=v['useful_risk_variation'])
                      for k, v in groups.items()}), flush=True)


if __name__ == '__main__':
    main()
