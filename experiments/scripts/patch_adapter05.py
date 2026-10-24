"""Apply the minimal, reversible action hook to exactly ManiGuard 6bf4c61."""

import argparse
import difflib
import hashlib
from pathlib import Path

ORIGINAL_SHA256 = "0991d5fdf5f65b4d70f113fa57bbff33190f4d65d5eb32683e87b44968ccae29"
EDITS = (
    (
        "        recorder = make_observer(cfg, scene_info)\n",
        "        recorder = make_observer(cfg, scene_info)\n        action_adapter = None\n",
    ),
    (
        "            robot = env.robots[0]\n",
        "            robot = env.robots[0]\n"
        "            if os.environ.get('SAFETY_DIAL_ADAPTER_CONFIG'):\n"
        "                from adapter05_runtime import create_adapter\n"
        "                action_adapter = create_adapter(cfg, robot)\n",
    ),
    (
        "                env.step(torch.from_numpy(wa).unsqueeze(0))\n",
        "                if action_adapter is not None:\n"
        "                    action_adapter.warmup_before(wa)\n"
        "                env.step(torch.from_numpy(wa).unsqueeze(0))\n"
        "                if action_adapter is not None:\n"
        "                    action_adapter.warmup_after()\n",
    ),
    (
        '                    _record_action_finiteness(rollout_diagnostics, ctrl_action, "controller")\n',
        "                    if action_adapter is not None:\n"
        "                        ctrl_action = action_adapter.apply(step_idx, ctrl_action)\n"
        '                    _record_action_finiteness(rollout_diagnostics, ctrl_action, "controller")\n',
    ),
    (
        '                    rollout_diagnostics["env_steps_returned"] += 1\n',
        '                    rollout_diagnostics["env_steps_returned"] += 1\n'
        "                    if action_adapter is not None:\n"
        "                        action_adapter.after_step(step_idx)\n",
    ),
    (
        "            all_results.append(result)\n            if recorder is not None:\n",
        "            all_results.append(result)\n            if action_adapter is not None:\n"
        "                action_adapter.finish(result)\n            if recorder is not None:\n",
    ),
    (
        "        all_results.append(result)\n        if recorder is not None:\n",
        "        all_results.append(result)\n        if action_adapter is not None:\n"
        "            action_adapter.finish(result)\n        if recorder is not None:\n",
    ),
)


def patched(source):
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Evaluator source differs from the pinned, audited revision")
    for old, new in EDITS:
        if source.count(old) != 1:
            raise ValueError("Action hook anchor is not unique")
        source = source.replace(old, new)
    compile(source, "benchmark.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--diff", type=Path, required=True)
    args = parser.parse_args()
    path = args.checkout / "maniguard/eval/benchmark.py"
    original = path.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_SHA256:
        restored = original
        for old, new in reversed(EDITS):
            restored = restored.replace(new, old)
        if patched(restored) != original:
            raise ValueError("Existing patch differs from the audited action hook")
        original = restored
    result = patched(original)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.diff.write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                result.splitlines(True),
                fromfile="a/maniguard/eval/benchmark.py",
                tofile="b/maniguard/eval/benchmark.py",
            )
        )
    )
    path.write_text(result)
    print("patched evaluator sha256", hashlib.sha256(result.encode()).hexdigest())


if __name__ == "__main__":
    main()
