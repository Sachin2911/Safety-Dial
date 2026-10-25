"""Apply passive DSRL06 query/scorer/warm-up hooks to pinned ManiGuard 6bf4c61."""

import argparse
import difflib
import hashlib
from pathlib import Path

ORIGINAL_SHA256 = "0991d5fdf5f65b4d70f113fa57bbff33190f4d65d5eb32683e87b44968ccae29"
GOAL_CHECKER_SHA256 = "ace47ce46cc0a540c759b0f5d0c89f050bdebbe83877e6052e770bf09acc74b0"
EDITS = (
    (
        "    return out\n\n\ndef query_policy",
        "    if '_dsrl06' in obs:\n        out['_dsrl06'] = obs['_dsrl06']\n"
        "    return out\n\n\ndef query_policy",
    ),
    (
        "        result = policy.infer(_remap_obs_for_openpi(obs, cfg))\n",
        "        result = policy.infer(_remap_obs_for_openpi(obs, cfg))\n"
        "        if '_dsrl06' in result:\n            obs['_dsrl06_response'] = result['_dsrl06']\n",
    ),
    (
        "                env.step(torch.from_numpy(wa).unsqueeze(0))\n",
        "                if recorder is not None and hasattr(recorder, 'warmup_before'):\n"
        "                    recorder.warmup_before(wa)\n"
        "                env.step(torch.from_numpy(wa).unsqueeze(0))\n"
        "                if recorder is not None and hasattr(recorder, 'warmup_after'):\n"
        "                    recorder.warmup_after()\n",
    ),
    (
        "                recorder.start(env, robot, obs, monitor, episode_seed, scene_info)\n",
        "                recorder.start(env, robot, obs, monitor, episode_seed, scene_info)\n"
        "                if hasattr(recorder, 'initial_goal'):\n"
        "                    recorder.initial_goal(goal_checker, env)\n",
    ),
    (
        "                chunk = query_policy(policy, obs, client_type, cfg)\n",
        "                if recorder is not None and hasattr(recorder, 'before_query'):\n"
        "                    recorder.before_query(step_idx, obs)\n"
        "                chunk = query_policy(policy, obs, client_type, cfg)\n"
        "                if recorder is not None and hasattr(recorder, 'after_query'):\n"
        "                    recorder.after_query(step_idx, obs, chunk)\n",
    ),
    (
        "                        inst_success, goal_detail = goal_checker.check(env)\n",
        "                        inst_success, goal_detail = goal_checker.check(env)\n"
        "                        if recorder is not None and hasattr(recorder, 'goal'):\n"
        "                            recorder.goal(step_idx, inst_success, goal_detail,\n"
        "                                          first_contact_step=first_contact_step)\n",
    ),
)


def patched(source):
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Evaluator is not the frozen ManiGuard 6bf4c61 source")
    for old, new in EDITS:
        if source.count(old) != 1:
            raise ValueError("DSRL06 observer anchor is not unique")
        source = source.replace(old, new)
    compile(source, "benchmark.py", "exec")
    return source


def restore(source):
    for old, new in reversed(EDITS):
        if source.count(new) != 1:
            raise ValueError("The installed hook is not the frozen DSRL06 patch")
        source = source.replace(new, old)
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Restored source differs from the frozen evaluator")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--diff", type=Path, required=True)
    args = parser.parse_args()
    goal = args.checkout / "maniguard/eval/goal_checker.py"
    if hashlib.sha256(goal.read_bytes()).hexdigest() != GOAL_CHECKER_SHA256:
        raise ValueError("Initial goal read was not audited for this GoalChecker source")
    path = args.checkout / "maniguard/eval/benchmark.py"
    source = path.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA256:
        source = restore(source)
    result = patched(source)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.diff.write_text(
        "".join(
            difflib.unified_diff(
                source.splitlines(True),
                result.splitlines(True),
                fromfile="a/maniguard/eval/benchmark.py",
                tofile="b/maniguard/eval/benchmark.py",
            )
        )
    )
    path.write_text(result)
    print("DSRL06 evaluator hook SHA256", hashlib.sha256(result.encode()).hexdigest())


if __name__ == "__main__":
    main()
