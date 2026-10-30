"""Generate a separate instrumented evaluator without editing the public checkout."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path


UPSTREAM_SHA256 = "0991d5fdf5f65b4d70f113fa57bbff33190f4d65d5eb32683e87b44968ccae29"


def instrument(source, checkout):
    if hashlib.sha256(source.encode()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Expected the pinned unmodified ManiGuard 6bf4c61 evaluator")
    edits = [
        ("REPO_ROOT = Path(__file__).resolve().parents[2]",
         f"REPO_ROOT = Path({str(checkout)!r})"),
        ("    import omnigibson as og\n\n    # When evaluating safety",
         "    import omnigibson as og\n"
         "    from maniguard_adapter_runtime import install_accounting\n"
         "    install_accounting(og)\n\n    # When evaluating safety"),
        ("                env.step(torch.from_numpy(wa).unsqueeze(0))\n",
         "                from maniguard_adapter_runtime import mark\n"
         "                mark('warmup_attempt')\n"
         "                env.step(torch.from_numpy(wa).unsqueeze(0))\n"
         "                mark('warmup_return')\n"),
        ("                chunk = query_policy(policy, obs, client_type, cfg)\n",
         "                from maniguard_adapter_runtime import mark\n"
         "                mark('query_attempt', step=step_idx)\n"
         "                chunk = query_policy(policy, obs, client_type, cfg)\n"
         "                mark('query_return', step=step_idx)\n"),
        ('                    _record_action_finiteness('
         'rollout_diagnostics, ctrl_action, "controller")\n',
         "                    from maniguard_adapter_runtime import modify_action\n"
         "                    ctrl_action = modify_action(recorder, step_idx, ctrl_action)\n"
         '                    _record_action_finiteness('
         'rollout_diagnostics, ctrl_action, "controller")\n'),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError("Ambiguous evaluator instrumentation anchor")
        source = source.replace(old, new)
    compile(source, "benchmark_adapter.py", "exec")
    return source


def prepare(checkout, output):
    checkout = Path(checkout).resolve()
    source = (checkout / "maniguard/eval/benchmark.py").read_text()
    modified = instrument(source, checkout)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "benchmark_adapter.py").write_text(modified)
    (output / "evaluator.patch").write_text("".join(difflib.unified_diff(
        source.splitlines(True), modified.splitlines(True),
        fromfile="upstream/benchmark.py", tofile="generated/benchmark_adapter.py")))
    report = {"upstream_sha256": UPSTREAM_SHA256,
              "generated_sha256": hashlib.sha256(modified.encode()).hexdigest(),
              "checkout": str(checkout), "original_checkout_modified": False}
    (output / "evaluator-source.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.checkout, args.out)))
