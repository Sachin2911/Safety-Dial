"""Fresh explicit action/accounting hooks at the pinned public evaluator revision."""

import argparse
import difflib
import hashlib
from pathlib import Path

ORIGINAL_SHA256 = "0991d5fdf5f65b4d70f113fa57bbff33190f4d65d5eb32683e87b44968ccae29"
EDITS = [
    (
        "    import omnigibson as og\n\n    # When evaluating safety",
        "    import omnigibson as og\n"
        "    if os.environ.get('JAR_GRASP_PROBE_PLAN'):\n"
        "        from jar_grasp_probe import install_accounting\n"
        "        install_accounting(og)\n\n    # When evaluating safety",
    ),
    (
        "                env.step(torch.from_numpy(wa).unsqueeze(0))\n",
        "                if os.environ.get('JAR_GRASP_PROBE_PLAN'):\n"
        "                    from jar_grasp_probe import mark\n"
        "                    mark('warmup_attempt')\n"
        "                env.step(torch.from_numpy(wa).unsqueeze(0))\n"
        "                if os.environ.get('JAR_GRASP_PROBE_PLAN'):\n"
        "                    mark('warmup_return')\n",
    ),
    (
        "                chunk = query_policy(policy, obs, client_type, cfg)\n",
        "                if os.environ.get('JAR_GRASP_PROBE_PLAN'):\n"
        "                    from jar_grasp_probe import mark\n"
        "                    mark('query_attempt', step=step_idx)\n"
        "                chunk = query_policy(policy, obs, client_type, cfg)\n"
        "                if os.environ.get('JAR_GRASP_PROBE_PLAN'):\n"
        "                    mark('query_return', step=step_idx)\n",
    ),
    (
        '                    _record_action_finiteness(rollout_diagnostics, ctrl_action, "controller")\n',
        "                    if os.environ.get('JAR_GRASP_PROBE_PLAN'):\n"
        "                        from jar_grasp_probe import modify_action\n"
        "                        ctrl_action = modify_action(recorder, step_idx, ctrl_action)\n"
        '                    _record_action_finiteness(rollout_diagnostics, ctrl_action, "controller")\n',
    ),
]


def patch(source):
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Unexpected evaluator source; refusing to patch")
    for old, new in EDITS:
        if source.count(old) != 1:
            raise ValueError("Non-unique explicit hook anchor")
        source = source.replace(old, new)
    compile(source, "benchmark.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("diff", type=Path)
    args = parser.parse_args()
    path = args.checkout / "maniguard/eval/benchmark.py"
    original = path.read_text()
    modified = patch(original)
    args.diff.write_text("".join(difflib.unified_diff(
        original.splitlines(True), modified.splitlines(True),
        fromfile="a/maniguard/eval/benchmark.py", tofile="b/maniguard/eval/benchmark.py",
    )))
    path.write_text(modified)
    print(hashlib.sha256(modified.encode()).hexdigest())


if __name__ == "__main__":
    main()
