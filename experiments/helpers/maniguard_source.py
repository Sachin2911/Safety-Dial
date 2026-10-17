"""Build a checked, additive runtime copy of the pinned upstream evaluator."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def prepare_runtime(output):
    upstream = ROOT / "third_party/ManiGuard"
    spec = json.loads((ROOT / "configs/maniguard/sources.json").read_text())
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != spec["maniguard"]["revision"]:
        raise ValueError("ManiGuard source revision differs from the study pin")
    original = subprocess.check_output([
        "git", "-C", str(upstream), "show", "HEAD:maniguard/eval/benchmark.py",
    ], text=True)
    if (upstream / "maniguard/eval/benchmark.py").read_text() != original:
        raise ValueError("Upstream evaluator has unrecorded local edits")
    source = original

    def replace(before, after):
        nonlocal source
        if source.count(before) != 1:
            raise ValueError(f"Runtime patch anchor is not unique: {before[:100]}")
        source = source.replace(before, after, 1)

    replace('REPO_ROOT = Path(__file__).resolve().parents[2]',
            f'REPO_ROOT = Path({str(upstream)!r})\nfrom helpers.maniguard_runtime import STUDY')
    replace('    return out\n\n\ndef query_policy',
            '    out["study_episode_id"] = STUDY.job["episode_id"]\n'
            '    return out\n\n\ndef query_policy')
    replace('    import omnigibson as og\n',
            '    import omnigibson as og\n    STUDY.install_accounting(og)\n')
    replace('    policy, client_type = connect_policy(cfg)\n',
            '    policy, client_type = connect_policy(cfg)\n'
            '    STUDY.validate_policy(policy, client_type)\n')
    replace('            step_idx = 0\n',
            '            STUDY.start(scene_info, obs)\n            step_idx = 0\n')
    replace('                chunk = query_policy(policy, obs, client_type, cfg)\n',
            '                STUDY.before_query(step_idx, obs, STUDY.frame(\n'
            '                    env, robot, monitor, step_idx, _eef_jacobian_arm))\n'
            '                chunk = query_policy(policy, obs, client_type, cfg)\n'
            '                STUDY.after_query(step_idx, chunk)\n')
    replace('                chunk_len = min(cfg.execute_horizon, len(chunk), cfg.max_steps - step_idx)',
            '                chunk_len = min(STUDY.config.horizon, len(chunk), cfg.max_steps - step_idx)')
    replace('                    if np.shape(ctrl_action) != action_space.shape:',
            '                    ctrl_action = STUDY.action(step_idx, ci, ctrl_action, STUDY.frame(\n'
            '                        env, robot, monitor, step_idx, _eef_jacobian_arm))\n'
            '                    if np.shape(ctrl_action) != action_space.shape:')
    replace('                    rollout_diagnostics["env_steps_returned"] += 1',
            '                    STUDY.applied(step_idx, action_clipped)\n'
            '                    rollout_diagnostics["env_steps_returned"] += 1')
    replace('                    if goal_checker is not None:\n',
            '                    STUDY.observed(step_idx, obs, monitor)\n'
            '                    if goal_checker is not None:\n')
    replace('                if step_idx % 50 == 0 or step_idx == 1:',
            '                    if STUDY.should_replan(step_idx, STUDY.frame(\n'
            '                        env, robot, monitor, step_idx, _eef_jacobian_arm)):\n'
            '                        break\n\n'
            '                if step_idx % 50 == 0 or step_idx == 1:')
    replace('            all_results.append(result)\n',
            '            STUDY.finish(result)\n            all_results.append(result)\n')
    replace('        all_results.append(result)\n        if recorder is not None:',
            '        STUDY.finish(result)\n        all_results.append(result)\n'
            '        if recorder is not None:')
    compile(source, 'benchmark_instrumented.py', 'exec')
    destination = Path(output) / 'benchmark_instrumented.py'
    destination.write_text(source)
    provenance = {"revision": revision,
                  "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                  "runtime_sha256": hashlib.sha256(source.encode()).hexdigest()}
    (Path(output) / 'runtime-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return destination
