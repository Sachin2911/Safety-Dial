"""Generate a dedicated AAC benchmark copy; never edit the upstream evaluator.

The generated module requires an explicitly installed AACRuntime. A future authorized
main child owns that bootstrap, service/lease checks and execution. This module only
builds source and provenance; it never runs the evaluator or imports a simulator.
"""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def instrument_source(original, upstream):
    source = original

    def replace(before, after):
        nonlocal source
        if source.count(before) != 1:
            raise ValueError('AAC source anchor is not unique: ' + before[:90])
        source = source.replace(before, after, 1)

    replace('REPO_ROOT = Path(__file__).resolve().parents[2]',
            f'REPO_ROOT = Path({str(Path(upstream).resolve())!r})\n'
            'from helpers.maniguard_aac_runtime import get_runtime\nAAC = get_runtime()')
    replace('    resolve_source_config(cfg)\n',
            '    resolve_source_config(cfg)\n    AAC.validate_config(cfg)\n')
    replace('    import omnigibson as og\n',
            '    import omnigibson as og\n    AAC.install_accounting(og)\n')
    replace('    policy, client_type = connect_policy(cfg)\n',
            '    policy, client_type = AAC.connect_policy(cfg)\n')
    replace('            step_idx = 0\n',
            '            AAC.start(scene_info, obs, env, robot, monitor, episode_seed)\n'
            '            step_idx = 0\n')
    replace('                chunk = query_policy(policy, obs, client_type, cfg)\n',
            '                chunk = AAC.query_policy(policy, obs, client_type, cfg,\n'
            '                                         step_idx, robot, _remap_obs_for_openpi)\n')
    replace('                chunk_len = min(cfg.execute_horizon, len(chunk), cfg.max_steps - step_idx)',
            '                chunk_len = min(len(chunk), cfg.max_steps - step_idx)')
    replace('                    rollout_diagnostics["phase"] = "env_step"\n',
            '                    AAC.before_action(step_idx, ci, chunk[ci], ctrl_action, action_clipped,\n'
            '                                      action_space.low, action_space.high)\n'
            '                    rollout_diagnostics["phase"] = "env_step"\n')
    replace('                    rollout_diagnostics["env_steps_returned"] += 1\n',
            '                    AAC.applied(step_idx, action_clipped)\n'
            '                    rollout_diagnostics["env_steps_returned"] += 1\n')
    replace('                    if goal_checker is not None:\n',
            '                    AAC.observed(step_idx, obs, monitor)\n'
            '                    if goal_checker is not None:\n')
    replace('                        if inst_success:\n',
            '                        AAC.write("goal", step=step_idx, success=inst_success, detail=goal_detail)\n'
            '                        if inst_success:\n')
    replace('            all_results.append(result)\n',
            '            AAC.finish(result)\n            all_results.append(result)\n')
    replace('        all_results.append(result)\n        if recorder is not None:',
            '        AAC.finish(result)\n        all_results.append(result)\n        if recorder is not None:')
    compile(source, 'benchmark_aac.py', 'exec')
    return source


def prepare_runtime(output, *, root=ROOT):
    """Archive a checked generated source; caller must subsequently bootstrap execution."""
    root, output = Path(root), Path(output)
    upstream = root / 'third_party/ManiGuard'
    pin = json.loads((root / 'configs/maniguard/phase2/sources.json').read_text())['maniguard']['revision']
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    original = subprocess.check_output(['git', '-C', str(upstream), 'show',
                                        pin + ':maniguard/eval/benchmark.py'], text=True)
    if revision != pin or (upstream / 'maniguard/eval/benchmark.py').read_text() != original:
        raise ValueError('AAC source generation requires the unchanged pinned benchmark')
    source = instrument_source(original, upstream)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'benchmark_aac.py'
    with path.open('x') as stream:
        stream.write(source)
    provenance = {'revision': revision, 'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
                  'runtime_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'source_hook_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'ordinary_query_replaced': True, 'runtime_bootstrap_required': True,
                  'live_authorization_integrated': False}
    with (output / 'aac-source-provenance.json').open('x') as stream:
        json.dump(provenance, stream, indent=2)
        stream.write('\n')
    return path
