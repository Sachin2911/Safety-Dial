"""Freeze source and full ordered schedules; never overwrite an earlier freeze."""

import argparse
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import utc
from helpers.unisafe09_protocol import design, digest, schedule


def sources():
    paths = set()
    for directory, pattern in (
        ("experiments/helpers", "unisafe09*.py"),
        ("experiments/scripts", "*unisafe09*"),
        ("experiments/tests", "test_unisafe09*.py"),
        ("configs/unisafe09", "*"),
    ):
        paths.update(p for p in (ROOT / directory).glob(pattern) if p.is_file())
    paths.add(ROOT / "experiments/helpers/dsrl06_infra.py")
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["preparation", "A", "B", "C"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--engineering", type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Freeze already exists; choose a fresh version path")
    spec = design()
    maximum_hours = spec["proposed_rental"]["maximum_rental_hours"]
    authorization = {"user_launch": True, "persistent_engineering_repairs": True}
    access = json.loads((ROOT / "progress/receipts/09-unisafe-access.json").read_text())
    selected = None
    selection = None
    if args.stage == "C":
        if not args.selection:
            raise ValueError("Confirmation needs the frozen deterministic selection")
        selection = json.loads(args.selection.read_text())
        if not selection["admit_signal"]:
            raise ValueError("No signal admits confirmation")
        selected = selection["conditions"]
    if maximum_hours == 15:
        approval_path = ROOT / "progress/receipts/09-unisafe-time-extension-authorization.json"
        trigger_path = ROOT / "progress/receipts/09-unisafe-C-original-window-projection.json"
        approval = json.loads(approval_path.read_text())
        trigger = json.loads(trigger_path.read_text())
        if (
            args.stage != "C"
            or approval["user_answer"] != "Approve conditional time extension"
            or approval["total_cap_usd"] != 10
            or not trigger["qualifies"]
            or trigger["original_window_fits"]
            or trigger["selection_sha256"] != digest(args.selection)
        ):
            raise ValueError("The conditional time extension is not authorized for this stage")
        authorization["conditional_time_extension"] = {
            "approval_sha256": digest(approval_path),
            "trigger_sha256": digest(trigger_path),
            "allocation_deadline_utc": approval["allocation_deadline_utc"],
        }
    elif maximum_hours != 12:
        raise ValueError("Unapproved rental-hour ceiling")
    stage = "A" if args.stage == "preparation" else args.stage
    jobs = schedule(stage, selected)
    acceptance = None
    if stage in ("B", "C"):
        if not args.engineering:
            raise ValueError("Scored stages need successful engineering acceptance")
        acceptance = json.loads(args.engineering.read_text())
        if not acceptance["passed"]:
            raise ValueError("Engineering did not pass")
    # Check retained historical JSON manifests for exact integer seed collisions.
    new_seeds = {j["seed"] for j in schedule("A")} | {j["seed"] for j in schedule("B")}
    new_seeds |= set(range(9300000, 9300064))
    checked = 0

    def scan(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"seed", "policy_seed"} and isinstance(item, int) and item in new_seeds:
                    raise ValueError(f"Historical seed collision: {item}")
                if key in {"seeds", "initial_seeds", "test_seeds"} and isinstance(item, list):
                    if new_seeds.intersection(x for x in item if isinstance(x, int)):
                        raise ValueError("Historical seed panel collision")
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)

    for base in ("archive/20261022/configs/unisafe", "archive/20261022/docs/research"):
        for path in (ROOT / base).rglob("*.json"):
            if "unisafe" in str(path).lower():
                scan(json.loads(path.read_text()))
                checked += 1
    record = {
        "schema": "unisafe09-freeze-v1",
        "utc": utc(),
        "stage": args.stage,
        "authorization": authorization,
        "money": {
            "total_usd": 10,
            "active_usd": 8,
            "reserve_usd": 2,
            "hourly_ceiling_usd": 0.65,
            "maximum_hours": maximum_hours,
        },
        "source_sha256": sources(),
        "design": spec,
        "access": access,
        "jobs": jobs,
        "historical_seed_manifests_checked": checked,
        "selection": selection,
        "engineering_acceptance": acceptance,
        "engineering_acceptance_sha256": digest(args.engineering) if args.engineering else None,
    }
    if args.stage == "preparation":
        record["planned_response_jobs"] = schedule("B")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    if args.bundle:
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.bundle, "w:gz") as archive:
            for name in record["source_sha256"]:
                archive.add(ROOT / name, arcname=name)
            for path in (
                args.output,
                ROOT / "progress/receipts/09-unisafe-access.json",
                ROOT / "progress/09-unisafe-dial-plan.md",
            ):
                archive.add(path, arcname=str(path.resolve().relative_to(ROOT)))
            rental = ROOT / "progress/rental-unisafe09-20261025.json"
            if rental.exists():
                archive.add(rental, arcname=str(rental.relative_to(ROOT)))
    print(
        json.dumps(
            {
                "freeze": str(args.output),
                "sha256": digest(args.output),
                "source_files": len(record["source_sha256"]),
                "jobs": len(jobs),
            }
        )
    )


if __name__ == "__main__":
    main()
