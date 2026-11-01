"""Build a reviewable private release only after all experiment evidence is verified."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from scripts.collect_lewm_campaign import collect
from helpers.safety_data import write_json
from helpers.safety_model import file_digest


SOURCES = (
    "outputs/pusht-preset-20261029-v1",
    "outputs/pusht-preset-motion-h5-20261029",
    "outputs/pusht-budget150-20261030-v1",
    "outputs/lewm-preset-expansion-20261030",
    "configs/pusht_preset_20261029",
    "configs/lewm_portable_20261030",
)


def forbidden(path):
    return (
        path.name in {".env", "config.toml", "known_hosts"}
        or path.name.startswith(".env.")
        or path.suffix in {".pem", ".key"}
        or path.name == "core"
        or path.name.startswith("core.")
    )


def local_dependencies(files):
    """Retain imported repository helpers without importing or running their code."""
    pending = [p for p in files if p.suffix == ".py"]
    visited = set()
    extra = set()
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        for node in ast.walk(ast.parse(source.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
                if node.module in {"helpers", "scripts"}:
                    modules += [node.module + "." + alias.name for alias in node.names]
            for module in modules:
                if not module.startswith(("helpers.", "scripts.")):
                    continue
                path = ROOT / "experiments" / (module.replace(".", "/") + ".py")
                if path.exists() and path not in visited:
                    extra.add(path)
                    pending.append(path)
    return extra


def package(dest):
    accounting = collect(require_complete=True)
    base = ROOT / "outputs/lewm-preset-expansion-20261030"
    studies = {}
    for name in ("tworoom", "reacher", "cube-v4"):
        out = base / name
        post = out / "artifact-postprocessing.json"
        if name == "cube-v4" and (
            not post.exists() or json.loads(post.read_text())["status"] != "complete"
        ):
            raise RuntimeError("Cube artifact postprocessing is incomplete")
        if len(list((out / "media").glob("*.mp4"))) != 12:
            raise RuntimeError("Missing reserved-case paired videos: " + name)
        if (
            not (out / "media/hazard-comparison.png").exists()
            or not (out / "media/control-comparison.png").exists()
        ):
            raise RuntimeError("Missing comparison gallery")
        report = json.loads((out / "report.json").read_text())
        studies[report["environment"]] = {
            "selected": report["selection"]["selected"],
            "evaluation": report["evaluation"],
            "motion_calibration": report["motion_report"]["calibration"],
            "audit_sha256": file_digest(out / "independent-audit.json"),
        }
    primary = json.loads((ROOT / SOURCES[0] / "report.json").read_text())
    sensitivity = json.loads((ROOT / SOURCES[2] / "report.json").read_text())
    studies["pusht"] = {
        "selected": primary["selected"],
        "evaluation_at50": primary["evaluation"],
        "budget_sensitivity": sensitivity["summaries"],
        "exact_original_prefixes": sensitivity["all_original_prefixes_match"],
    }
    dest.mkdir(parents=True, exist_ok=False)
    files = []
    for source in SOURCES:
        for p in sorted((ROOT / source).rglob("*")):
            if p.is_symlink():
                raise RuntimeError("Review symlink before packaging: " + str(p))
            if not p.is_file():
                continue
            if forbidden(p):
                raise RuntimeError("Unexpected sensitive or core-dump filename: " + str(p))
            if p.suffix == ".part":
                raise RuntimeError("Incomplete file: " + str(p))
            if p.parent == base and p.name in {
                "release-preparation.json",
                "release-preparation.log",
            }:
                continue
            if p.is_relative_to(base / "checkpoints"):
                continue
            files.append(p)
    # Include final review/analysis programs, in addition to immutable study snapshots.
    for pattern in (
        "*portable*.py",
        "*cube_bounded.py",
        "*cube_goal_repair.py",
        "*cube_calibration.py",
        "collect_lewm_campaign.py",
        "package_lewm_campaign.py",
        "verify_lewm_release.py",
    ):
        files.extend((ROOT / "experiments/scripts").glob(pattern))
    files.extend((ROOT / "experiments/tests").glob("test_*portable*.py"))
    files.append(ROOT / "experiments/tests/test_cube_action_contract.py")
    for name, sha256 in json.loads((ROOT / SOURCES[1] / "freeze.json").read_text())[
        "source_sha256"
    ].items():
        path = ROOT / name
        if file_digest(path) != sha256:
            raise RuntimeError("PushT training source changed before preservation: " + name)
        files.append(path)
    for name in (
        "report_pusht_preset.py",
        "media_pusht_preset.py",
        "audit_pusht_learned_motion.py",
    ):
        files.append(ROOT / "experiments/scripts" / name)
    files.extend(local_dependencies(files))
    files = sorted(set(files))
    members = {
        str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": file_digest(p)}
        for p in files
    }
    write_json(
        dest / "evidence-manifest.json",
        {
            "members": members,
            "file_count": len(members),
            "note": "Original paths are relative to the repository. Exact public world-model objects are supplied separately under checkpoints. No account files or environment dumps are included.",
        },
    )
    archive = dest / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    models = dest / "checkpoints"
    models.mkdir()
    for p in sorted((base / "checkpoints").glob("*.ckpt")):
        shutil.copy2(p, models / p.name)
    for name, path in {
        "pusht": ROOT / SOURCES[1] / "model.pt",
        "tworoom": base / "tworoom/motion/model.pt",
        "reacher": base / "reacher/motion/model.pt",
        "cube": base / "cube-v4/motion/model.pt",
    }.items():
        shutil.copy2(path, models / (name + "-safety.pt"))
    for name, path in [
        ("pusht", ROOT / SOURCES[0]),
        ("tworoom", base / "tworoom"),
        ("reacher", base / "reacher"),
        ("cube", base / "cube-v4"),
    ]:
        target = dest / "media" / name
        target.mkdir(parents=True)
        for p in (path / "media").glob("*"):
            if p.is_file():
                shutil.copy2(p, target / p.name)
        if (path / "safety-adaptation.png").exists():
            shutil.copy2(path / "safety-adaptation.png", target / "safety-adaptation.png")
    summary = {
        "studies": studies,
        "accounting": accounting,
        "source_root": "evidence.tar.gz",
        "limits": "Small one-seed pilots, no evolutionary or optimizer-superiority claim. Native observed proprioception, not image-only safety. Protected body only, not complete arm/object protection. Empirical calibration is not a universal guarantee. All failed and interrupted engineering work is retained.",
    }
    write_json(dest / "results.json", summary)
    lines = [
        "# SafeLeWM learned-safety preset comparison",
        "",
        "Frozen LeWM task cost, learned native-proprioception body motion, shared hazard geometry and bounded development-only planner search.",
        "",
        "The full evidence archive retains training/calibration data, source/config freezes, all scored and excluded engineering runs, actual commands, dense monitors and source checksums. Public world-model objects and final safety heads are exposed under checkpoints.",
        "",
        "Reserved results use six hazard cases and six paired no-added-hazard controls per arm. PushT primary limit50; its separate fixed-setting sensitivity reports50/100/150. Other environments use150 with all three checkpoints.",
        "",
    ]
    for name, data in studies.items():
        result = data["evaluation_at50"] if name == "pusht" else data["evaluation"]
        selected = data["selected"]["id"]
        a = result["authors"]
        b = result[selected]
        lines.append(
            f"- {name}: hazard safe completions authors {a['hazard']['safe_completions']}/6, selected {b['hazard']['safe_completions']}/6; controls {a['control']['safe_completions']}/6 and {b['control']['safe_completions']}/6. [Paired replays](media/{name})."
        )
    lines += [
        "",
        "These are descriptive pilots. TwoRoom is a within-room panel. Cube uses object-position success; earlier gripper-reaching studies are different tasks. Native monitors label real executions, with no simulator previews during candidate planning.",
        "",
        "Both arms use the complete SafeCEM and learned-safety stack. Authors refers to the authors' planning hyperparameters in this modified stack, not a replication of their original benchmark evaluation. Search includes one manually chosen anchor and six random settings; selecting the anchor is not evidence that random search discovered it.",
        "",
        "Safety weights are fitted separately per environment using native observed proprioception. Geometry remains analytic and shared. Protection covers the declared navigation disk, fingertip disk or pinch-site disk, not the complete arm or manipulated object. No universal safety or image-only transfer claim is made.",
        "",
        "Endpoint clearance was screened before comparison; whole-task connectivity and action-budget feasibility were not certified. Reacher's strong gate has a closed direct opening after adding its fixed prediction-error allowance. Routes around the ends were not ruled out. See each environment's task-progress-diagnostics.json inside the archive.",
        "",
        "Cube's valid comparison is cube-v4. Its safety forecasts and applied commands share the native action bounds. The latent task cost retains the same upstream unbounded candidate imaginations used elsewhere in the study. Earlier Cube engineering passes, including goal-image and actuator-contract defects, are preserved and excluded from performance summaries. Recalibration used fresh groups and unchanged learned weights after the initial release check failed.",
        "",
        "Interaction totals retain failed and interrupted work. Unjournaled work is reported with explicit conservative bounds, not invented exact counts. Planner-time counters omit a final exception-raising planning call; full episode and stage wall times include it. Concurrent jobs limit latency comparisons.",
        "",
        "For paired trajectory plots, red is the hazard and amber includes the protected-body radius plus the requested margin. Grey is the source reference, and green is actual motion. Inactive-control rescoring distinguishes deliberate adaptation from incidental clearance.",
        "",
        "The unchanged PushT LeWM object is already stored in the earlier private release at revision34256873f32a27e90c16ee081d30660aecc6acb4, SHA256d27504690172f6efa31277107a4d54e68f18760520664bc8a33d3b9d1b558952.",
        "",
    ]
    (dest / "README.md").write_text("\n".join(lines))
    manifest = {
        str(p.relative_to(dest)): {"bytes": p.stat().st_size, "sha256": file_digest(p)}
        for p in sorted(dest.rglob("*"))
        if p.is_file()
    }
    write_json(
        dest / "release-manifest.json",
        {
            "files": manifest,
            "file_count": len(manifest),
            "total_bytes": sum(v["bytes"] for v in manifest.values()),
        },
    )
    # Read every archived member back, not just the archive header.
    with tarfile.open(archive, "r:gz") as tar:
        seen = set()
        for entry in tar:
            if not entry.isfile() or entry.name not in members:
                raise RuntimeError("Unexpected archive member")
            blob = tar.extractfile(entry)
            digest = hashlib.file_digest(blob, "sha256").hexdigest()
            if digest != members[entry.name]["sha256"]:
                raise RuntimeError("Archive member hash mismatch")
            seen.add(entry.name)
        if seen != set(members):
            raise RuntimeError("Incomplete archive")
    write_json(
        dest / "verification.json",
        {
            "archive_members_verified": len(members),
            "archive_sha256": file_digest(archive),
            "payload_files": len(manifest) + 2,
            "upload_status": "Not uploaded; exact payload requires approval.",
        },
    )
    print(
        json.dumps(
            {
                "release": str(dest),
                "files": len(manifest) + 2,
                "bytes_before_verification_metadata": sum(v["bytes"] for v in manifest.values()),
                "archive_sha256": file_digest(archive),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    package(parser.parse_args().out)


if __name__ == "__main__":
    main()
