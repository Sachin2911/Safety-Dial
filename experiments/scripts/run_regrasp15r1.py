"""Run the task 15 recovery cohort: stage A engineering, then stage C, then stage D.

Every episode runs through the frozen queue (run_regrasp15.Queue.run and drain): the
same evaluator command, adapter hook, validation and decision replay, one replacement
per incomplete job, and a halt on two consecutive incomplete attempts or a validation
failure. This module only rebinds that queue to the recovery namespace and rental,
takes the job lists from configs/regrasp15r1/recovery.json, enforces the five
cumulative ceilings including the original host's imported counters, admits stage C
only after the measured re-admission and stage D only after a semantically complete
stage C, and writes semantic completion records. Run with the behavior environment's
Python on the recovery rental.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
sys.path.insert(0, str(ROOT / "experiments/helpers"))
import run_regrasp15 as base  # noqa: E402

OUT = ROOT / "outputs/regrasp15r1"
RECOVERY = ROOT / "configs/regrasp15r1/recovery.json"
RENTAL = ROOT / "progress/rental-regrasp15r1-20261026.json"
DESIGN = ROOT / "configs/regrasp15/design.json"
DECLARATION = ROOT / "progress/receipts/15-regrasp-stage-d.json"
base.OUT = OUT
base.SERVER_LEDGER = OUT / "server/candidates.jsonl"
STAGES = ("stage_a", "stage_c", "stage_d")
INCOMPLETE_EXIT = 3


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_jobs(recovery, stage):
    key = {"stage_a": "engineering_jobs", "stage_c": "stage_c_jobs",
           "stage_d": "stage_d_jobs"}[stage]
    return recovery[key]


def completion(history, jobs, finished):
    """Complete means every expected job validated, no start unresolved, queue finished."""
    expected = [j["job"] for j in jobs]
    wanted = set(expected)
    validated = {e["job"] for e in history if e["event"] == "validated" and e["job"] in wanted}
    exits = {(e["job"], e["attempt"]) for e in history if e["event"] == "simulator_exit"}
    unresolved = [[e["job"], e["attempt"]] for e in history
                  if e["event"] == "start_reserved" and e["job"] in wanted
                  and (e["job"], e["attempt"]) not in exits]
    missing = [job for job in expected if job not in validated]
    return {"utc": utc(), "expected_jobs": len(expected), "validated_jobs": sorted(validated),
            "missing_jobs": missing, "unresolved_starts": unresolved,
            "abandoned_jobs": sorted({e["job"] for e in history
                                      if e["event"] == "job_abandoned" and e["job"] in wanted}),
            "finished": bool(finished),
            "complete": bool(finished) and not missing and not unresolved}


def recorded_complete(path, history, jobs):
    """A stored completion record counts only if it agrees with the ledger itself."""
    if not path.exists():
        return False
    stored = json.loads(path.read_text())
    again = completion(history, jobs, stored.get("finished"))
    return bool(stored.get("complete")) and again["complete"] and (
        stored.get("missing_jobs") == [] and stored.get("unresolved_starts") == [])


def cumulative_usage(history, recovery, decodes):
    """Original-host imports plus recovery use; an unvalidated start costs its worst case."""
    imports, worst = recovery["imports"], recovery["worst_case_per_start"]
    starts = [e for e in history if e["event"] == "start_reserved"]
    validated = {(e["job"], e["attempt"]): e for e in history if e["event"] == "validated"}
    controls = queries = 0
    for start in starts:
        done = validated.get((start["job"], start["attempt"]))
        if done is None:
            controls += worst["policy_controls"]
            queries += worst["native_queries"]
        else:
            steps = done["outcome"]["steps"]
            controls += steps
            queries += math.ceil(steps / 8) + 1  # chunk queries plus the reseed probe
    return {
        "simulator_starts": imports["simulator_starts"] + len(starts),
        "replacements": imports["replacements"] + sum(s["attempt"] > 1 for s in starts),
        "policy_controls": imports["policy_controls"] + controls,
        "native_queries": imports["native_queries"] + queries
        + recovery["probe_native_queries_reserved"],
        "candidate_decodes": imports["candidate_decodes"] + decodes,
    }


def refusal_reasons(used, recovery, attempt, *, now, admission, stop):
    ceilings, worst = recovery["ceilings"], recovery["worst_case_per_start"]
    reasons = []
    if now >= admission:
        reasons.append("less_than_45_minutes_before_deadline")
    if stop:
        reasons.append("stop_file")
    if used["simulator_starts"] + 1 > ceilings["simulator_starts"]:
        reasons.append("simulator_starts")
    if attempt > 1 and used["replacements"] + 1 > ceilings["replacements"]:
        reasons.append("replacements")
    for key in ("policy_controls", "native_queries", "candidate_decodes"):
        if used[key] + worst[key] > ceilings[key]:
            reasons.append(key)
    return reasons


def gate(stage, history, recovery, *, out, plan_sha, instance):
    """Raise unless the stage's predecessors are complete and admitted."""
    if stage == "stage_a":
        return
    engineering = stage_jobs(recovery, "stage_a")
    if not recorded_complete(out / "stage_a-complete.json", history, engineering):
        raise ValueError("Stage A must be complete before confirmation episodes")
    passed = json.loads((out / "stage-a-pass.json").read_text())
    if not passed.get("pass"):
        raise ValueError("Stage A did not pass its engineering rules")
    admission_path = out / "admission-c.json"
    if not admission_path.exists():
        raise ValueError("Stage C needs the measured re-admission")
    admission = json.loads(admission_path.read_text())
    if not (admission.get("admit_stage_c") is True and admission.get("freeze_sha256") == plan_sha
            and admission.get("instance_id") == instance):
        raise ValueError("The measured re-admission does not admit stage C for this freeze")
    if stage == "stage_d" and not recorded_complete(
            out / "stage_c-complete.json", history, stage_jobs(recovery, "stage_c")):
        raise ValueError("Stage D needs a complete stage C native comparator")


class RecoveryQueue(base.Queue):
    def __init__(self, plan_path, stage):
        self.recovery = json.loads(RECOVERY.read_text())
        plan = json.loads(plan_path.read_text())
        if plan.get("stage") != "recovery_live":
            raise ValueError("The recovery queue runs only under the recovery live freeze")
        frozen = plan["source_sha256"]
        if frozen.get(str(RECOVERY.relative_to(ROOT))) != sha256(RECOVERY):
            raise ValueError("The recovery design is not part of this freeze")
        decision = self.recovery["stage_c_decision"]
        if (sha256(ROOT / decision["path"]) != decision["sha256"]
                or self.recovery["design_sha256"] != sha256(DESIGN)
                or self.recovery["stage_d_declaration_sha256"] != sha256(DECLARATION)):
            raise ValueError("Recovery design is not bound to the frozen decision and design")
        self.plan_sha = sha256(plan_path)
        super().__init__(plan_path, DESIGN, RENTAL, stage)

    def stage_jobs(self):
        gate(self.stage, self.history, self.recovery, out=OUT, plan_sha=self.plan_sha,
             instance=self.rental["instance_id"])
        return stage_jobs(self.recovery, self.stage)

    def admit(self, job, attempt):
        used = cumulative_usage(self.history, self.recovery, self.decodes())
        reasons = refusal_reasons(used, self.recovery, attempt, now=time.time(),
                                  admission=self.admission, stop=(OUT / "STOP").exists())
        if reasons:
            self.record("admission_refused", job=job["job"], attempt=attempt,
                        reasons=reasons, usage=used)
            return False
        if attempt > 1:
            self.replacements += 1
        self.starts += 1
        self.record("start_reserved", **job, attempt=attempt, start_number=self.starts,
                    cumulative_start=used["simulator_starts"] + 1, usage_before=used)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    queue = RecoveryQueue(args.plan.resolve(), args.stage)
    try:
        finished = queue.drain()
    except Exception as error:
        (OUT / "STOP").write_text(f"{args.stage}: {type(error).__name__}\n")
        queue.record("queue_error", kind=type(error).__name__, detail=str(error))
        raise
    queue.record("queue_finished", stage=args.stage, starts=queue.starts,
                 replacements=queue.replacements, finished=finished)
    result = completion(queue.history, queue.jobs, finished)
    result["usage"] = cumulative_usage(queue.history, queue.recovery, queue.decodes())
    (OUT / f"{args.stage}-complete.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.stage == "stage_a":
        rules = base.stage_a_rules(queue)
        (OUT / "stage-a-pass.json").write_text(json.dumps(rules, indent=2) + "\n")
        if not rules["pass"]:
            raise SystemExit("Stage A failed its engineering rules")
    if not result["complete"]:
        print(json.dumps({"incomplete": args.stage, "missing": len(result["missing_jobs"])}))
        raise SystemExit(INCOMPLETE_EXIT)


if __name__ == "__main__":
    main()
