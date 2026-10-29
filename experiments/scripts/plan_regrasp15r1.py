"""Write the frozen task 15 recovery design (configs/regrasp15r1/recovery.json).

The recovery replicates the declared stage C and stage D schedule as one complete cohort
on a replacement host: four stage A engineering episodes, then stage C (native against
the frozen stage B choice, 48 episodes each, the original interleaved order and job
numbers), then stage D (project-0.10, 48 episodes, by seed then task). Nothing is chosen
here: the arms, tasks, seeds and order come from the live design, the original stage C
decision (copied byte for byte and bound to its stage B evidence) and the stage D
declaration. Cumulative ceilings import the original host's conservative counters, and
no original episode counts as recovery work.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "configs/regrasp15/design.json"
DECLARATION = ROOT / "progress/receipts/15-regrasp-stage-d.json"
EVIDENCE = ROOT / "progress/receipts/15-regrasp-recovery-evidence-20261026.json"
HISTORICAL = ROOT / "outputs/regrasp15/remote"
OUT_DIR = ROOT / "configs/regrasp15r1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_c_jobs(design, arm):
    """The rule in run_regrasp15.Queue.stage_jobs, unchanged."""
    arms = ["native", arm]
    jobs, number = [], 300
    for seed in design["stage_c_seeds"]:
        for task in design["stage_c_tasks"]:
            for name in (arms if (seed + task) % 2 == 0 else arms[::-1]):
                jobs.append({"job": number, "stage": "stage_c", "arm": name,
                             "task": task, "seed": seed})
                number += 1
    return jobs


def stage_d_jobs(declaration):
    """The rule in stage_d_regrasp15.StageD.stage_jobs, unchanged."""
    jobs, number = [], 500
    for seed in declaration["seeds"]:
        for task in declaration["tasks"]:
            jobs.append({"job": number, "stage": "stage_d", "arm": declaration["arm"],
                         "task": task, "seed": seed})
            number += 1
    return jobs


def build(decision_path):
    design = json.loads(DESIGN.read_text())
    declaration = json.loads(DECLARATION.read_text())
    decision = json.loads(Path(decision_path).read_text())
    evidence = json.loads(EVIDENCE.read_text())
    if decision["design_sha256"] != sha256(DESIGN):
        raise ValueError("The stage C decision is not bound to the live design")
    for name, expected in decision["evidence_sha256"].items():
        if sha256(HISTORICAL / name) != expected:
            raise ValueError(f"Original stage B evidence changed: {name}")
    if decision["arm"] not in design["arms"] or declaration["arm"] not in design["arms"]:
        raise ValueError("Recovery arms must be frozen design arms")
    counters = evidence["historical_counters"]
    worst_steps = 2000  # max_steps in configs/regrasp15/jar_eval.yaml
    return {
        "schema": "regrasp15-recovery-v1",
        "cohort": "r1",
        "namespace": "outputs/regrasp15r1",
        "rental": "progress/rental-regrasp15r1-20261026.json",
        "design_sha256": sha256(DESIGN),
        "stage_c_decision": {"path": "configs/regrasp15r1/historical-stage-c-decision.json",
                             "sha256": sha256(decision_path), "arm": decision["arm"],
                             "rule": decision["rule"]},
        "stage_d_declaration_sha256": sha256(DECLARATION),
        "engineering_jobs": design["stage_a_jobs"],
        "stage_c_jobs": stage_c_jobs(design, decision["arm"]),
        "stage_d_jobs": stage_d_jobs(declaration),
        "imports": {
            "source": "progress/receipts/15-regrasp-recovery-evidence-20261026.json",
            "simulator_starts": counters["simulator_starts"]["conservative"],
            "replacements": counters["replacements"],
            "policy_controls": counters["policy_controls"]["conservative"],
            "native_queries": counters["native_queries"]["conservative"],
            "candidate_decodes": counters["candidate_decodes"]["conservative"],
        },
        "ceilings": {**design["ceilings"], **declaration["ceilings"]},
        "worst_case_per_start": {"policy_controls": worst_steps,
                                 "native_queries": worst_steps // 8 + 1,
                                 "candidate_decodes": design["k"] * worst_steps // 8},
        "probe_native_queries_reserved": 16,
        "readmission": "Stage C starts only after outputs/regrasp15r1/admission-c.json records "
                       "admit true for this freeze and rental, written from measured "
                       "engineering timings.",
        "completion": "A stage is complete only when every expected job validated, no start "
                      "is unresolved, the queue finished and no job is missing.",
        "analysis": {
            "stage_c_primary": {"stage": "stage_c", "arms": ["native", decision["arm"]]},
            "stage_d": {"treated": ["stage_d", declaration["arm"]],
                        "comparator": ["stage_c", "native"]},
            "historical": "The original host's episodes are a separate, descriptive cohort "
                          "and are never pooled with the recovery cohort.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, default=HISTORICAL / "stage-c-decision.json")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copy = OUT_DIR / "historical-stage-c-decision.json"
    if copy.exists() and sha256(copy) != sha256(args.decision):
        raise SystemExit("A different historical decision copy already exists")
    shutil.copyfile(args.decision, copy)
    recovery = build(copy)
    path = OUT_DIR / "recovery.json"
    path.write_text(json.dumps(recovery, indent=2) + "\n")
    print("wrote", path.relative_to(ROOT), sha256(path), "jobs",
          len(recovery["engineering_jobs"]), len(recovery["stage_c_jobs"]),
          len(recovery["stage_d_jobs"]))


if __name__ == "__main__":
    main()
