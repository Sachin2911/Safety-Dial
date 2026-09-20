"""Test side of the environment equivalence check. Runs in the project venv.

Compares the vendored `experiments/helpers/locoEnv.py` against reference trajectories emitted by
`emit_reference_traj.py` from a real safety-gymnasium install, and writes
`notes/envEquivalence.md`.

This is the artefact that makes vendoring defensible. Without it, "we ran the Safety-Gymnasium
benchmark" is an assertion; with it, it is a measurement.

    uv run python experiments/scripts/verify_env_equivalence.py

Exits non-zero if any gate fails. Gates G3 (per-trajectory cost sum) and G4 (first termination
index) are the two that can void the paper: cost is the Mode A channel and termination is the
Mode B channel, so a disagreement in either means the vendored env is not the benchmark. G5 and
G6 are diagnostics that say *why* if those fail, distinguishing float reassociation (smooth
exponential drift from ~1e-16, benign) from a real solver difference (a step change).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers import equivCheck as ec  # noqa: E402
from helpers import locoEnv as le  # noqa: E402


class VendoredAdapter:
    """The vendored env, in the same interface as `UpstreamAdapter`.

    Built with `time_limit=False` and the raw six-tuple, so nothing truncates the tape.
    """

    def __init__(self, robot: str, version: str):
        self.env = le.make_loco_env(robot, version, six_tuple=True, time_limit=False)
        self.u = self.env
        self.reset_qpos = None
        self.reset_qvel = None

    def reset(self, seed=0):
        self.env.reset(seed=int(seed))
        self.reset_qpos = self.u.data.qpos.copy()
        self.reset_qvel = self.u.data.qvel.copy()

    def set_state(self, qpos, qvel):
        self.u.set_state(np.asarray(qpos, float), np.asarray(qvel, float))

    def step(self, action):
        obs, reward, cost, terminated, _, info = self.env.step(action)
        return (obs, float(reward), float(cost), bool(terminated),
                float(info["x_velocity"]), self.u.data.qpos.copy(), self.u.data.qvel.copy())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="Walker2d")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--report", default=str(REPO_ROOT / "notes" / "envEquivalence.md"))
    args = ap.parse_args()

    ref_path = Path(args.ref) if args.ref else (
        REPO_ROOT / "data" / "equiv" / f"ref_{args.robot}_{args.version}.npz"
    )
    if not ref_path.is_file():
        print(f"missing reference trajectories at {ref_path}\n"
              f"Generate them first with:\n"
              f"  uv run --isolated --no-project --python 3.10 \\\n"
              f'    --with "safety-gymnasium==1.0.0" --with "numpy<2" \\\n'
              f"    python experiments/scripts/emit_reference_traj.py "
              f"--robot {args.robot} --version {args.version}")
        return 2

    blob = np.load(ref_path, allow_pickle=False)
    n_trials, horizon = int(blob["n_trials"]), int(blob["horizon"])

    adapter = VendoredAdapter(args.robot, args.version)
    adapter.reset(seed=0)
    nq, nv = adapter.u.model.nq, adapter.u.model.nv
    a_dim = int(adapter.env.action_space.shape[0])
    specs = ec.build_specs(nq, nv, a_dim, horizon)
    assert len(specs) == n_trials, f"trial corpus mismatch: {len(specs)} vs {n_trials}"

    print(f"=== equivalence: {args.robot}-{args.version}, {n_trials} trials x {horizon} steps ===")

    comparisons = []
    g1_fail = 0
    for i, spec in enumerate(specs):
        test = ec.run_trial(adapter, spec, horizon)
        ref = {
            k: blob[f"t{i}_{k}"]
            for k in ("qpos", "qvel", "obs", "reward", "cost", "terminated", "x_velocity")
        }
        ref["cost_sum"] = float(ref["cost"].sum())
        ref["first_term"] = int(np.argmax(ref["terminated"])) if ref["terminated"].any() else -1

        if spec["kind"] == "reset":
            # G1: if the reset distributions differ, nothing downstream is comparable.
            if not (np.array_equal(blob[f"t{i}_reset_qpos"], test["reset_qpos"])
                    and np.array_equal(blob[f"t{i}_reset_qvel"], test["reset_qvel"])):
                g1_fail += 1

        comparisons.append(ec.compare(ref, test))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_trials}")

    import json
    ref_manifest_path = ref_path.with_name(ref_path.stem + "_manifest.json")
    ref_manifest = (json.loads(ref_manifest_path.read_text())
                    if ref_manifest_path.is_file() else {})
    test_manifest = le.env_manifest(adapter.env)

    summary = ec.summarise(comparisons, ref_manifest, test_manifest)
    summary["gates"]["G1 reset state exact"] = g1_fail == 0

    print()
    for k, v in summary["gates"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  max drift @200  {summary['g5_drift_200']:.3e}")
    print(f"  max drift @1000 {summary['g6_drift_1000']:.3e}")
    print(f"  cost-sum exact  {summary['g3_pass']}/{summary['n_trials']}")
    print(f"  term-index      {summary['g4_pass']}/{summary['n_trials']}")

    summary["verdict"] = "PASS" if all(summary["gates"].values()) else "FAIL"
    Path(args.report).write_text(ec.format_report(summary, ref_manifest, test_manifest))
    print(f"\nVERDICT: {summary['verdict']}   (report written to {args.report})")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
