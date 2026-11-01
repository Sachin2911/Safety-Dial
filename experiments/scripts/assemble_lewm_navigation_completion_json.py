"""Assemble35 complete owner-bound stages after exact four-source accounting.

Original full trees remain immutable outside the canonical replacement project.
Only JSON is copied by default. Optional media adds exactly the48 predeclared Nav
repeat0 trace/history pairs, read as raw bytes only. Nothing enters canonical results.
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
OLD_PODS = {"navigation": "vukovc3t4o1dui", "pusht": "wq4ar2qx2ff6r4", "recovery": "nj2fqkp2prgk8x"}
LEGACY = "experiments/scripts/assemble_lewm_navigation_analysis_inputs.py"
LEGACY_SHA = "fbab5d294ea597647112260a3c09c5364594ff5213216ad26edfe0804fa98b82"
COLLECTOR = "experiments/scripts/collect_lewm_campaign_20261031_v4.py"
COLLECTOR_SHA = "d9d5da4893b4eb483bd072766fa6e428865bb43ea988e98d760f2d4845a7a7fb"
NAV_REPORT = "312d4ba2e8f68ca10fa1898b0c127f0a4975d5f7f8c0235b7376feb9990eda2d"
PUSHT_REPORT = "207b7fec8a1f9c9d5c69a9ba29e4000402fae831e7881030a43cd11bcbe05b53"
RECOVERY_REPORT = "ba2028222ad233eab0f5ae107eed9b88a346c9ce397dda50027827bde5b8dcfb"
ACCOUNTING = "lewm-campaign-accounting-20261031-v1"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def relative(value):
    value = str(value)
    if "/Safety-Dial/" in value:
        value = value.split("/Safety-Dial/", 1)[1]
    path = Path(value)
    require(
        not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in value
        and str(path) == value,
        "Unsafe evidence path",
    )
    return value


def module(name, expected):
    path = ROOT / name
    require(sha(path) == expected, "Immutable reused source changed: " + name)
    spec = importlib.util.spec_from_file_location("completion_" + path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def ref(path):
    return {"path": str(Path(path)), "sha256": sha(path)}


def accounting_reports(paths, hashes, actual):
    require(len(paths) == len(hashes) == 4, "Exactly four SHA-pinned owning-host reports required")
    reports = {}
    for path, expected in zip(paths, hashes, strict=True):
        require(sha(path) == expected, "Accounting report bytes differ")
        value = load(path)
        pod = value["owning_pod_id"]
        require(
            value["schema"] == ACCOUNTING
            and value["complete_final_accounting"] is True
            and value["unresolved"] == []
            and pod not in reports,
            "One complete authoritative report per source pod is required",
        )
        reports[pod] = {"path": Path(path), "sha256": expected, "value": value}
    require(set(reports) == {*OLD_PODS.values(), actual}, "Four physical source pods differ")
    require(
        reports[OLD_PODS["navigation"]]["sha256"] == NAV_REPORT,
        "Retain exact original Nav accounting",
    )
    require(
        reports[OLD_PODS["pusht"]]["sha256"] == PUSHT_REPORT,
        "Retain exact qualified PushT accounting",
    )
    recovery = reports[OLD_PODS["recovery"]]["value"]
    require(
        reports[OLD_PODS["recovery"]]["sha256"] == RECOVERY_REPORT
        and recovery.get("collection_execution_pod_id") == actual
        and recovery.get("recorded_source_owner_pod_id") == OLD_PODS["recovery"]
        and recovery.get("numeric_collection_executed_on_explicit_authorized_replacement") is True,
        "Recovery needs the explicitly qualified preserved-ledger reconstruction",
    )
    return reports


def merge_accounting(args):
    require(
        os.environ.get("RUNPOD_POD_ID") == args.pod_id, "Merge only on the declared rented host"
    )
    reports = accounting_reports(args.reports, args.report_sha256, args.pod_id)
    collector = module(COLLECTOR, COLLECTOR_SHA)
    result = collector.merge(argparse.Namespace(reports=args.reports, pod_id=args.pod_id))
    require(
        result["complete_final_accounting"] is True and not result["unresolved"],
        "Global accounting incomplete",
    )
    result["four_source_pods_explicitly_verified"] = sorted(reports)
    result["assembly_merge_source_sha256"] = sha(__file__)
    write_new(args.out, result)
    return {"output": str(args.out), "sha256": sha(args.out), "complete_final_accounting": True}


def origins(values, receipts):
    roots, proofs = {}, {}
    for value in values:
        label, path = value.split("=", 1)
        require(label in OLD_PODS and label not in roots, "One original tree per source owner")
        roots[label] = Path(path).resolve()
    for value in receipts:
        label, path = value.split("=", 1)
        require(label in OLD_PODS and label not in proofs, "One extraction proof per source owner")
        proofs[label] = Path(path)
    require(set(roots) == set(proofs) == set(OLD_PODS), "All original source trees/proofs required")
    output = {}
    for label, pod in OLD_PODS.items():
        root = roots[label]
        require(
            not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),
            "Originals remain outside canonical project",
        )
        proof = load(proofs[label])
        manifest_path = root / "MANIFEST-20261031.json"
        require(
            proof["schema"] == "lewm-preserved-origin-extraction-v1"
            and proof["verified"] is True
            and proof["original_owner_pod_id"] == pod
            and Path(proof["destination"]).resolve() == root
            and proof["every_extracted_member_hashed_and_readback_verified"] is True
            and proof["embedded_manifest_sha256"] == sha(manifest_path),
            "Exact read-only all-member original extraction required",
        )
        manifest = load(manifest_path)
        require(
            manifest["pod_id"] == pod and proof["origin_files_read_only"] is True,
            "Original owner/immutability differs",
        )
        output[pod] = {
            "root": root,
            "files": manifest["files"],
            "proof": ref(proofs[label]),
            "archive": proof["archive"],
        }
    return output


def copy_checked(source, destination, expected):
    require(source.is_file() and not source.is_symlink(), "Only regular evidence files")
    require(sha(source) == expected, "Source evidence SHA differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), "Never overwrite assembled evidence")
    partial = destination.with_name(destination.name + ".assembly-part")
    with source.open("rb") as incoming, partial.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    require(sha(partial) == expected, "Copied evidence readback differs")
    os.link(partial, destination)
    partial.unlink()


def assemble(args):
    sys.path.insert(0, str(ROOT / "experiments/scripts"))
    import analyze_lewm_navigation_campaign_20261031_v4 as analysis

    plan = analysis.verify_plan(args.plan)
    actual = os.environ.get("RUNPOD_POD_ID")
    value = analysis.continuation_authority(ROOT, plan)
    require(
        actual == value["actual_final_host"]["pod_id"] == args.pod_id,
        "Assemble only on the actual replacement host",
    )
    reports = accounting_reports(args.reports, args.report_sha256, actual)
    require(
        sha(args.accounting_merge) == args.accounting_merge_sha256, "Global accounting SHA differs"
    )
    aggregate = load(args.accounting_merge)
    require(
        aggregate["schema"] == ACCOUNTING + "-aggregate"
        and aggregate["complete_final_accounting"] is True
        and not aggregate["unresolved"]
        and {row["sha256"] for row in aggregate["source_reports"]}
        == {row["sha256"] for row in reports.values()},
        "Assembly must follow the exact complete four-source accounting merge",
    )
    destination = args.destination.resolve()
    require(
        not destination.exists()
        and not destination.is_relative_to(ROOT)
        and not ROOT.is_relative_to(destination),
        "Use a fresh sibling analysis-only tree, never canonical project/parent",
    )
    inputs = load(args.inputs)
    require(
        inputs["schema"] == "lewm-navigation-analysis-inputs-20261031-v4"
        and inputs["plan"]["sha256"] == sha(args.plan)
        and inputs["final_execution_relocation"] == plan["final_execution_relocation"]
        and inputs["original_inputs"] == plan["original_analysis_inputs"]
        and inputs["all_relocated_namespaces_absent_at_continuation_binding"] is True,
        "Exact prospective V4 continuation binding required",
    )
    analysis.relocation_helpers().validate_binding(ROOT, inputs, value)
    source_origins = origins(args.origin, args.origin_receipt)
    legacy = module(LEGACY, LEGACY_SHA)
    stages = []
    for entry in inputs["training_runs"]:
        owner = plan["profiles"][entry["family"]]["host_assignment"][str(entry["repeat_id"])]
        stages.append({**entry, "kind": "training", "owning_pod_id": owner})
    for entry in inputs["profiles"]:
        _, protocol = analysis.load_reference(ROOT, entry["protocol"])
        owner = analysis.relocation_helpers().actual_host(value, entry)["pod_id"]
        stages.append(
            {
                **entry,
                "output_namespace": protocol["output_namespace"],
                "kind": "final",
                "owning_pod_id": owner,
            }
        )
    require(
        len(stages) == 35 and len({s["output_namespace"] for s in stages}) == 35,
        "All24 training and11 final namespaces required",
    )
    files = {}

    def add(name, source, expected, owner, role):
        name = relative(name)
        require(
            source.resolve().is_relative_to(
                ROOT if owner is None or owner == actual else source_origins[owner]["root"]
            ),
            "Evidence source escaped owner tree",
        )
        record = {
            "sha256": expected,
            "bytes": source.stat().st_size,
            "owning_pod_id": owner,
            "role": role,
            "source": str(source),
        }
        if name in files:
            require(files[name]["sha256"] == expected, "Conflicting assembled metadata bytes")
        else:
            files[name] = record

    for stage in stages:
        owner = stage["owning_pod_id"]
        source_root = ROOT if owner == actual else source_origins[owner]["root"]
        selected = {}
        legacy.select_namespace(source_root, stage, selected)
        physical = {
            relative(row["source"]["path"]): row["source"]["sha256"]
            for row in reports[owner]["value"]["components"]
            if row["kind"] == "physical_episode"
        }
        for name, record in selected.items():
            require(name.endswith(".json"), "Only JSON stage evidence is assembled")
            if owner != actual:
                original = source_origins[owner]["files"].get(name)
                require(
                    original and original["sha256"] == record["sha256"],
                    "Old stage evidence differs from verified original archive",
                )
            if name.endswith("/episode/result.json"):
                require(
                    physical.get(name) == record["sha256"],
                    "Physical inner episode is absent from its complete owner accounting",
                )
            add(name, source_root / name, record["sha256"], owner, "complete_owned_stage")
    # Source-freeze/inner/state reconciliation remains in the frozen V4 analyzer;
    # select_namespace already enforces complete retirement, exact job/inner hashes.
    metadata = set()
    metadata.update(
        str(p.relative_to(ROOT)) for p in (ROOT / "configs/lewm_campaign_20261031").rglob("*.json")
    )
    metadata.update(
        str(p.relative_to(ROOT)) for p in (ROOT / "results/evidence-audits").rglob("*.json")
    )
    metadata.update(
        {
            relative(args.plan),
            relative(args.inputs),
            plan["original_analysis_inputs"]["path"],
            value["capacity_failure_evidence"]["path"],
            value["software_qualification"]["path"],
        }
    )
    for entry in inputs["profiles"]:
        if analysis.relocation_helpers().relocated_row(value, entry):
            source = load(ROOT / Path(entry["state_path"]).parent / "source-freeze.json")
            sidecar = source["continuation"]
            require(
                sha(ROOT / sidecar["path"]) == sidecar["sha256"], "Continuation sidecar changed"
            )
            metadata.add(sidecar["path"])
    for name in sorted(metadata):
        name = relative(name)
        add(name, ROOT / name, sha(ROOT / name), None, "shared_frozen_metadata")
    # Copy already-SHA-verified account JSONs as metadata only, allowing downstream
    # descriptive reporting to keep every ref inside the analysis-only sibling.
    for pod, account in reports.items():
        files[".accounting/" + pod + ".json"] = {
            "sha256": account["sha256"],
            "bytes": account["path"].stat().st_size,
            "owning_pod_id": pod,
            "role": "authoritative_accounting_metadata",
            "source": str(account["path"]),
        }
    files[".accounting/aggregate.json"] = {
        "sha256": args.accounting_merge_sha256,
        "bytes": args.accounting_merge.stat().st_size,
        "owning_pod_id": None,
        "role": "complete_four_source_accounting_metadata",
        "source": str(args.accounting_merge),
    }
    if args.with_media:
        nav = source_origins[OLD_PODS["navigation"]]
        first_family_cases = None
        for family in ("primary50", "secondary30"):
            entries = [
                e for e in inputs["profiles"] if e["family"] == family and e["repeat_id"] == 0
            ]
            require(
                len(entries) == 1 and entries[0]["host"]["pod_id"] == OLD_PODS["navigation"],
                "Only originalNavR0 showcase",
            )
            _, protocol = analysis.load_reference(ROOT, entries[0]["protocol"])
            cases = {j["case"]["case_id"]: j["case"] for j in protocol["jobs"]}
            chosen = {
                min(
                    (c for c in cases.values() if c["condition"] == cond),
                    key=lambda c: (c["source_episode"], c["source_step"], c["case_id"]),
                )["case_id"]
                for cond in analysis.CONDITIONS
            }
            require(
                first_family_cases is None or chosen == first_family_cases,
                "Both endpoints must show the same frozen first-admitted cases",
            )
            first_family_cases = chosen
            jobs = [j for j in protocol["jobs"] if j["case"]["case_id"] in chosen]
            require(len(jobs) == 24, "Four frozen cases times six showcase arms")
            for job in jobs:
                prefix = protocol["output_namespace"] + "/jobs/" + job["job_id"] + "/episode/"
                for suffix in ("trace.npz", "selector-history.json"):
                    name = prefix + suffix
                    original = nav["files"].get(name)
                    require(
                        original is not None, "Fixed showcase media absent from original archive"
                    )
                    add(
                        name,
                        nav["root"] / name,
                        original["sha256"],
                        OLD_PODS["navigation"],
                        "fixed_NavR0_media_raw_bytes",
                    )
        require(
            sum(r["role"] == "fixed_NavR0_media_raw_bytes" for r in files.values()) == 96,
            "Exactly48 trace/history pairs required",
        )
    destination.mkdir(parents=True)
    write_new(
        destination / ".analysis-only-input-root.json",
        {
            "never_collect_from_this_tree": True,
            "four_source_accounting": ref(args.accounting_merge),
            "new_model_simulator_calls": 0,
        },
    )
    for name, record in sorted(files.items()):
        copy_checked(Path(record["source"]), destination / name, record["sha256"])
    require(
        all(sha(destination / name) == record["sha256"] for name, record in files.items()),
        "Final assembly readback differs",
    )
    result = {
        "schema": "lewm-completion-analysis-assembly-v1",
        "verified": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "execution_pod_id": actual,
        "source_sha256": sha(__file__),
        "immutable_legacy_selector_sha256": LEGACY_SHA,
        "plan": ref(args.plan),
        "inputs": ref(args.inputs),
        "accounting_merge": ref(args.accounting_merge),
        "accounting_reports": [
            {"pod_id": pod, "sha256": r["sha256"]} for pod, r in reports.items()
        ],
        "destination": str(destination),
        "namespaces": stages,
        "namespace_count": 35,
        "files": files,
        "media_pairs": 48 if args.with_media else 0,
        "canonical_foreign_physical_imports": 0,
        "new_model_queries": 0,
        "new_simulator_actions": 0,
    }
    write_new(destination / ".assembly-manifest.json", result)
    write_new(args.out, result)
    return {
        "verified": True,
        "namespace_count": 35,
        "files": len(files),
        "media_pairs": result["media_pairs"],
        "receipt": str(args.out),
        "sha256": sha(args.out),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    merger = commands.add_parser("merge-accounting")
    assembler = commands.add_parser("assemble")
    for command in (merger, assembler):
        command.add_argument("--pod-id", required=True)
        command.add_argument("--reports", type=Path, nargs=4, required=True)
        command.add_argument("--report-sha256", nargs=4, required=True)
        command.add_argument("--out", type=Path, required=True)
    assembler.add_argument("--plan", type=Path, required=True)
    assembler.add_argument("--inputs", type=Path, required=True)
    assembler.add_argument("--accounting-merge", type=Path, required=True)
    assembler.add_argument("--accounting-merge-sha256", required=True)
    assembler.add_argument("--origin", action="append", required=True)
    assembler.add_argument("--origin-receipt", action="append", required=True)
    assembler.add_argument("--destination", type=Path, required=True)
    assembler.add_argument("--with-media", action="store_true")
    args = parser.parse_args()
    require(not args.out.exists(), "Never overwrite assembly/merge evidence")
    print(
        json.dumps(merge_accounting(args) if args.command == "merge-accounting" else assemble(args))
    )


if __name__ == "__main__":
    main()
