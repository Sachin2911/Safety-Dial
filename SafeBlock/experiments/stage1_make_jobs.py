"""Generate the frozen Stage 1 job list (SafeBlock/configs/stage1-jobs.json)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "SafeBlock/configs/stage1-freeze.json"
PROBES = ROOT / "SafeBlock/configs/probes-20261101.json"
OUT = ROOT / "SafeBlock/configs/stage1-jobs.json"


def main():
    freeze = json.loads(FREEZE.read_text())
    probes = json.loads(PROBES.read_text())
    base = ["--freeze", str(FREEZE.relative_to(ROOT))]
    S = freeze["seeds"]
    rng = lambda a: list(range(a[0], a[1] + 1))  # noqa: E731
    jobs = {}
    # Smoke: wiring check on one start.
    jobs["SMK"] = {"jobs": [
        {"id": "SMK-residual-g5000000", "task": "hard", "condition": "ap_rank", "seed": 4310000, "args": base + ["--adapter", "residual", "--genotype-seed", "5000000", "--bound", "0.5", "--gripper-bound", "2.0"]},
        {"id": "SMK-selection-g5100000", "task": "hard", "condition": "ap_rank", "seed": 4310000, "args": base + ["--adapter", "selection", "--genotype-seed", "5100000"]},
        {"id": "SMK-residual-zero", "task": "hard", "condition": "ap_rank", "seed": 4300000, "args": base + ["--adapter", "residual", "--zero-genotype", "--bound", "0.5", "--gripper-bound", "2.0"]},
    ]}
    # Parity: 6 starts x {bare, residual zero, selection zero} + one bare repeat for determinism.
    parity = []
    for seed in rng(S["parity"]):
        parity.append({"id": f"S1P-{seed}-bare", "task": "hard", "condition": "ap_rank", "seed": seed, "args": base})
        parity.append({"id": f"S1P-{seed}-residual-zero", "task": "hard", "condition": "ap_rank", "seed": seed, "args": base + ["--adapter", "residual", "--zero-genotype", "--bound", "0.5", "--gripper-bound", "2.0"]})
        parity.append({"id": f"S1P-{seed}-selection-zero", "task": "hard", "condition": "ap_rank", "seed": seed, "args": base + ["--adapter", "selection", "--zero-genotype"]})
    parity.append({"id": f"S1P-{S['parity'][0]}-bare-repeat", "task": "hard", "condition": "ap_rank", "seed": S["parity"][0], "args": base})
    jobs["S1P"] = {"jobs": parity}
    # Sweep: 4 starts x (bare + 24 residual genotypes in a 2x2 design + 24 selection genotypes).
    sweep = []
    design = []
    for i in range(24):
        gripper = 2.0 if i < 12 else 0.0
        bound = 0.25 if (i % 12) < 6 else 0.5
        design.append({"genotype_seed": 5000000 + i, "bound": bound, "gripper_bound": gripper})
    for seed in rng(S["sweep_starts"]):
        sweep.append({"id": f"S1S-{seed}-bare", "task": "hard", "condition": "ap_rank", "seed": seed, "args": base})
        for d in design:
            sweep.append({"id": f"S1S-{seed}-residual-g{d['genotype_seed']}-B{d['bound']}-G{d['gripper_bound']}", "task": "hard", "condition": "ap_rank", "seed": seed, "genotype": d, "args": base + ["--adapter", "residual", "--genotype-seed", str(d["genotype_seed"]), "--bound", str(d["bound"]), "--gripper-bound", str(d["gripper_bound"])]})
        for i in range(24):
            g = 5100000 + i
            sweep.append({"id": f"S1S-{seed}-selection-g{g}", "task": "hard", "condition": "ap_rank", "seed": seed, "genotype": {"genotype_seed": g}, "args": base + ["--adapter", "selection", "--genotype-seed", str(g)]})
    jobs["S1S"] = {"design": design, "jobs": sweep}
    # Calibration: 48 bare starts.
    jobs["S1C"] = {"jobs": [{"id": f"S1C-{seed}-bare", "task": "hard", "condition": "ap_rank", "seed": seed, "args": base} for seed in rng(S["calibration"])]}
    # Reset test: 6 fresh + one in-process job (warm-up on the first seed, then the six).
    reset_seeds = rng(S["reset_test"])
    reset = [{"id": f"S1R-{seed}-fresh", "task": "hard", "condition": "ap_rank", "seed": seed, "args": base} for seed in reset_seeds]
    reset.append({"id": "S1R-in-process", "task": "hard", "condition": "ap_rank", "seed": reset_seeds[0], "args": base + ["--in-process-seeds", ",".join(map(str, reset_seeds))]})
    jobs["S1R"] = {"jobs": reset}
    out = {"written_utc": freeze["written_utc"], "freeze": str(FREEZE.relative_to(ROOT)), "library": probes["library"], "planner": probes["planner"], "probes": jobs}
    OUT.write_text(json.dumps(out, indent=2))
    print({k: len(v["jobs"]) for k, v in jobs.items()}, "total", sum(len(v["jobs"]) for v in jobs.values()))


if __name__ == "__main__":
    main()
