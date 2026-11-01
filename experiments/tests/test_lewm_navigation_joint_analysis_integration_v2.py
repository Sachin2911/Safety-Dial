"""Synthetic rented-only joint-builder -> analysis-v3 contract integration.

All scientific inputs are fabricated JSON or dummy bytes. No actual final data,
arrays, checkpoint, renderer, simulator or numerical analysis is used.
"""

import collections
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

REAL_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REAL_ROOT / "experiments/scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REAL_ROOT / "experiments"))

pytestmark = pytest.mark.skipif(
    os.environ.get("RUNPOD_POD_ID") not in ("vukovc3t4o1dui", "wq4ar2qx2ff6r4"),
    reason="Synthetic integration belongs only on the declared rented hosts",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return path


def ref(path, root):
    return {"path": str(path.relative_to(root)), "sha256": sha(path)}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def byte_asset(root, name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SYNTHETIC DUMMY BYTES, NOT A MODEL OR ARRAY\n")
    return name, sha(path)


def fixture(root, joint):
    actual_sources = [
        *joint.NEW_SOURCES,
        "experiments/scripts/freeze_lewm_navigation_final_comparison.py",
    ]
    for name in actual_sources:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REAL_ROOT / name).read_bytes())
    shared = dict(
        byte_asset(root, name)
        for name in (
            "assets/world.ckpt",
            "assets/safety.pt",
            "assets/scalers.json",
            "assets/memory.npz",
            "assets/memory-metadata.json",
            "assets/readout.npz",
        )
    )
    common_sources = {
        name: sha(root / name)
        for name in (
            "experiments/helpers/lewm_navigation_joint_final_v2.py",
            "experiments/scripts/freeze_lewm_navigation_final_comparison.py",
        )
    }
    design = {
        "world_model": "assets/world.ckpt",
        "motion_model": "assets/safety.pt",
        "scalers": "assets/scalers.json",
        "memory_path": "assets/memory.npz",
        "memory_metadata_path": "assets/memory-metadata.json",
        "memory_sha256": shared["assets/memory.npz"],
        "goal_readout": {"path": "assets/readout.npz", "sha256": shared["assets/readout.npz"]},
        "asset_sha256": shared,
        "planner": {"checked": True, "horizon": 2, "iterations": 8},
        "controller_backend": "observed-target-v2",
        "controller_config": {"fixture": True},
        "progress_backend": "navigation-append-fsync-v1",
        "body_radius": 7,
        "margin": 3,
        "motion_allowance": 12.405431747436523,
        "precision": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
    }
    original = save(root / "configs/original-secondary-manifest.json", {"synthetic_original": True})
    proof = save(
        root / "configs/unstarted-proof.json",
        {
            "schema": "lewm-secondary-r2-unstarted-proof-v1",
            "pod_id": "wq4ar2qx2ff6r4",
            "old_secondary_r2_output_exists": False,
            "old_secondary_r2_output_children": [],
            "original_secondary_manifest": ref(original, root),
        },
    )
    relocation = save(
        root / "configs/relocation.json",
        {
            "schema": "lewm-navigation-secondary-relocation-v1",
            "study_family": "secondary30",
            "repeat_id": 2,
            "old_host": {"host": "pusht", "pod_id": "wq4ar2qx2ff6r4"},
            "new_host": {"host": "recovery", "pod_id": "nj2fqkp2prgk8x"},
            "whole_unstarted_repeat_block": True,
            "scientific_settings_unchanged": True,
            "all_cases_seeds_budgets_preserved": True,
            "all_local_methods_and_controls_moved": True,
            "original_secondary_manifest": ref(original, root),
            "unstarted_proof": ref(proof, root),
            "final_pixels_read": 0,
        },
    )
    openings, locks = {}, {}
    for family, cap in (("primary", 50), ("secondary", 30)):
        entries = []
        for repeat in range(3):
            nav = repeat == 0 or (family == "primary" and repeat == 2)
            host = {
                "repeat_id": repeat,
                "host": "navigation" if nav else "pusht",
                "pod_id": "vukovc3t4o1dui" if nav else "wq4ar2qx2ff6r4",
            }
            if family == "secondary" and repeat == 2:
                host.update(host="recovery", pod_id="nj2fqkp2prgk8x")
            chosen = {
                arm: {
                    "parameters": [0.25] * 10,
                    "selected_is_trained": False,
                    "selected_source_arm": "native" if arm == "native" else "strong_anchor",
                    "provenance": {"kind": "synthetic_anchor", "is_trained": False},
                }
                for arm in joint.primary.ARMS
            }
            current = copy.deepcopy(design)
            current.update(execution_pod_id=host["pod_id"], maximum_episode_actions=cap)
            lock = {
                "schema": "lewm-navigation-selection-lock-v1",
                "repeat_id": repeat,
                "host": host,
                "final_pixels_read": 0,
                "no_best_repeat_selection": True,
                "denominator_each": 8,
                "evaluation": current,
                "selected": chosen,
                "source_sha256": common_sources,
            }
            path = save(root / f"configs/{family}-lock-r{repeat}.json", lock)
            entries.append({**ref(path, root), "repeat_id": repeat, "host": host})
            locks[(family, repeat)] = lock
        openings[family] = save(
            root / f"configs/{family}-opening.json",
            {
                "schema": "lewm-navigation-final-opening-lock-v1"
                if family == "primary"
                else "lewm-navigation-final-opening-lock-v2",
                "all_three_repeat_selections_fixed": True,
                "best_repeat_selection_forbidden": True,
                "final_pixels_read": 0,
                "selection_locks": entries,
                **(
                    {"secondary_relocation": ref(relocation, root)} if family == "secondary" else {}
                ),
            },
        )
        if family == "secondary":
            # Replace only this fabricated opening with the NEW real sealer's
            # output, exercising its three-host metadata contract end to end.
            openings[family].rename(
                openings[family].with_name("secondary-opening-manual-preserved.json")
            )
            sealer = module(
                "secondary_opening_integration_v2",
                SCRIPTS / "freeze_lewm_navigation_selection_opening_v2.py",
            )
            sealer.ROOT = root
            sealer.seal(
                SimpleNamespace(
                    locks=[root / item["path"] for item in entries],
                    relocation_record=relocation,
                    out=openings[family],
                )
            )
    activation = save(
        root / "configs/activation.json",
        {
            "schema": "lewm-navigation-secondary-study-choice-v1",
            "activate_secondary": True,
            "all_six_selection_locks_required_before_final": True,
            "final_pixels_read": 0,
        },
    )
    joint_path = root / "configs/joint-opening.json"
    joint.seal(
        SimpleNamespace(
            primary_opening=openings["primary"],
            secondary_opening=openings["secondary"],
            activation_record=activation,
            relocation_record=relocation,
            namespace="results/synthetic-joint-final",
            out=joint_path,
        )
    )
    prep_path = save(
        root / "configs/primary-preparation.json",
        {
            "schema": "lewm-navigation-final-case-preparation-v1",
            "execution_enabled": True,
            "final_opening": ref(openings["primary"], root),
            "final_episode_actions": 50,
            "source_sha256": common_sources,
        },
    )
    joint_prep_path = root / "configs/joint-preparation.json"
    joint.preparation(
        SimpleNamespace(
            joint_opening=joint_path, primary_preparation_protocol=prep_path, out=joint_prep_path
        )
    )
    output = root / "results/navigation-final-cases-v1"
    cases = []
    for episode in range(5000, 5016):
        condition = ("none", "circle3", "circle5", "circle7")[episode % 4]
        name, fingerprint = byte_asset(
            root, f"results/navigation-final-cases-v1/inputs/synthetic-{episode}.npz"
        )
        cases.append(
            {
                "case_id": f"synthetic-{episode}-{condition}",
                "split": "heldout_final",
                "source_episode": episode,
                "source_step": 0,
                "goal_step": 25,
                "condition": condition,
                "shape": {"kind": "none"}
                if condition == "none"
                else {
                    "kind": "island",
                    "radius": int(condition[6:]),
                    "centre": [60, 60],
                    "axis": [1, 0],
                },
                "seed": 713000 + episode,
                "input_path": name,
                "input_sha256": fingerprint,
                "image_style_flag": False,
                "initial_xy": [40, 40],
                "goal_xy": [70, 120],
            }
        )
    panel = save(
        output / "panel.json",
        {
            "phase": "awaiting_visual_review",
            "cases": cases,
            "final_opening": {"opening_sha256": sha(openings["primary"])},
            "final_pixels_read": 32,
            "synthetic_fabricated_metadata": True,
        },
    )
    save(
        output / "source-freeze.json",
        {"protocol_sha256": sha(prep_path), "sources": common_sources},
    )
    save(
        output / "status.json",
        {
            "phase": "awaiting_visual_review",
            "error": None,
            "controller_episodes": 0,
            "new_model_queries": 0,
            "final_pixels_read": 32,
            "synthetic_fabricated_metadata": True,
        },
    )
    save(
        output / "joint-preparation.json",
        {
            "protocol_sha256": sha(joint_prep_path),
            "joint_opening": ref(joint_path, root),
            "all_six_repeat_selections_fixed": True,
        },
    )
    gallery_name, gallery_sha = byte_asset(
        root, "results/navigation-final-cases-v1/quality/synthetic-gallery.png"
    )
    review = save(
        output / "visual-review.json",
        {
            "approved": True,
            "images_clean": True,
            "panel_sha256": sha(panel),
            "reviewed_utc": "2026-10-31T00:00:00+00:00",
            "galleries": {gallery_name: gallery_sha},
            "reviewer": "synthetic fixture",
        },
    )
    decoded_sources = dict(
        byte_asset(root, name)
        for name in (
            "experiments/helpers/lewm_navigation_decoded_cost.py",
            "experiments/scripts/run_lewm_navigation_campaign_v3.py",
        )
    )
    qual_name, qual_sha = byte_asset(root, "environment/synthetic-qualification.json")
    aux = save(
        root / "configs/aux-source.json",
        {
            "evaluation": {
                "runtime_backend": "navigation-campaign-v3",
                "asset_sha256": {qual_name: qual_sha},
            },
            "source_sha256": decoded_sources,
            "software_qualification": {
                "path": qual_name,
                "sha256": qual_sha,
                "actual_passed_tests": 9,
            },
        },
    )
    manifests = []
    for family, cap in (("primary", 50), ("secondary", 30)):
        base_folder = root / f"configs/{family}-training"
        master_folder = (
            root / "configs/secondary-relocated" if family == "secondary" else base_folder
        )
        schedule, file_refs = [], []
        for repeat in range(3):
            folder = master_folder if family == "secondary" and repeat == 2 else base_folder
            runs = {}
            for method in ("cma", "random", "reinforce"):
                path = save(
                    folder / f"training-{method}-r{repeat}.json", {"max_episode_actions": cap}
                )
                runs[method] = {
                    "protocol": str(path),
                    "output_namespace": f"results/synthetic-{family}-r{repeat}-{method}",
                }
                file_refs.append(ref(path, root))
            fixed = save(
                folder / f"fixed-tuning-r{repeat}.json",
                {
                    "evaluation": {"maximum_episode_actions": cap},
                    "output_namespace": f"results/synthetic-{family}-fixed-r{repeat}",
                },
            )
            file_refs.append(ref(fixed, root))
            schedule.append(
                {
                    "repeat_id": repeat,
                    "runs": runs,
                    "fixed_tuning": f"results/synthetic-{family}-fixed-r{repeat}",
                }
            )
        manifests.append(
            save(
                master_folder / "manifest.json",
                {"schedule": schedule, "files": file_refs, "synthetic_only": True},
            )
        )
    return SimpleNamespace(
        joint=joint_path,
        joint_prep=joint_prep_path,
        panel=panel,
        review=review,
        aux=aux,
        activation=activation,
        relocation=relocation,
        unstarted_proof=proof,
        training=manifests,
        locks=locks,
    )


def test_joint_profiles_bind_to_analysis_v3_and_tampered_protocol_is_rejected(
    tmp_path, monkeypatch
):
    # A canonical sibling tree avoids an artificial nested Safety-Dial prefix in
    # the production archive-path normalizer. It is snapshotted in retained NFS.
    root = Path("/workspace/SafetyDial-joint-analysis-synthetic-v2")
    assert not root.exists(), "Use a fresh synthetic integration namespace"
    root.mkdir()
    joint = module("joint_final_integration", SCRIPTS / "freeze_lewm_navigation_joint_final_v2.py")
    analysis = module(
        "analysis_v3_integration", SCRIPTS / "analyze_lewm_navigation_campaign_20261031_v3.py"
    )
    monkeypatch.setattr(joint, "ROOT", root)
    monkeypatch.setattr(joint.primary, "ROOT", root)
    monkeypatch.setattr(analysis, "ROOT", root)
    monkeypatch.setattr(
        joint,
        "__file__",
        str(root / "experiments/scripts/freeze_lewm_navigation_joint_final_v2.py"),
    )
    monkeypatch.setattr(
        joint.primary,
        "__file__",
        str(root / "experiments/scripts/freeze_lewm_navigation_final_comparison.py"),
    )
    summary = {
        "synthetic_only": True,
        "actual_final_metadata_read": 0,
        "new_model_simulator_renderer_queries": 0,
        "array_or_checkpoint_opens": 0,
    }
    try:
        data = fixture(root, joint)
        plan = save(
            root / "configs/synthetic-analysis-plan.json",
            {
                "schema": "lewm-navigation-analysis-plan-20261031-v1",
                "version": 3,
                "analysis_script_sha256": sha(
                    SCRIPTS / "analyze_lewm_navigation_campaign_20261031_v3.py"
                ),
                "bootstrap": {"draws": 10000, "seed": 803101},
                "source_sha256": {
                    "experiments/helpers/lewm_navigation_joint_final_v2.py": sha(
                        root / "experiments/helpers/lewm_navigation_joint_final_v2.py"
                    )
                },
                "execution": {
                    "allowed_pod_ids": ["vukovc3t4o1dui", "wq4ar2qx2ff6r4", "nj2fqkp2prgk8x"]
                },
                "secondary_activation": ref(data.activation, root),
                "secondary_relocation": ref(data.relocation, root),
                "secondary_unstarted_proof": ref(data.unstarted_proof, root),
                "joint_final_manifest_schema": "lewm-navigation-joint-final-comparison-v2",
                "auxiliary_decoded_native": {
                    "eligible_pod_ids_by_endpoint": {
                        "50": ["vukovc3t4o1dui", "wq4ar2qx2ff6r4"],
                        "30": ["vukovc3t4o1dui", "wq4ar2qx2ff6r4", "nj2fqkp2prgk8x"],
                    }
                },
                "profiles": {
                    family: {
                        "training_manifest": ref(data.training[index], root),
                        "host_assignment": {
                            str(r): data.locks[(label, r)]["host"]["pod_id"] for r in range(3)
                        },
                    }
                    for index, (family, label) in enumerate(
                        (("primary50", "primary"), ("secondary30", "secondary"))
                    )
                },
                "synthetic_only": True,
            },
        )
        original_open = Path.open
        heavy_before = set(sys.modules)

        def deny_arrays(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if path.suffix in {".npz", ".npy", ".h5", ".pt", ".ckpt"} and "r" in mode:
                raise AssertionError("Synthetic builder/binder opened an array/checkpoint")
            return original_open(path, *args, **kwargs)

        with monkeypatch.context() as io_guard:
            io_guard.setattr(Path, "open", deny_arrays)
            generated = joint.comparison(
                SimpleNamespace(
                    joint_opening=data.joint,
                    joint_preparation_protocol=data.joint_prep,
                    panel=data.panel,
                    review=data.review,
                    out_dir=root / "configs/joint-final-profiles",
                    namespace="results/synthetic-joint-final",
                    aux_source_protocol=data.aux,
                    aux_endpoints=[30, 50],
                )
            )
            manifest = root / generated["path"]
            bound = root / "configs/analysis-inputs.json"
            result = analysis.bind_inputs(
                SimpleNamespace(
                    plan=plan, manifest=manifest, training_manifest=data.training, out=bound
                )
            )
            inputs = json.loads(bound.read_text())
            assert result["profiles"] == len(inputs["profiles"]) == 11
            assert len(inputs["training_runs"]) == 24
            assert collections.Counter(p["family"] for p in inputs["profiles"]) == {
                "primary50": 3,
                "secondary30": 3,
                "aux-decoded-native30": 3,
                "aux-decoded-native50": 2,
            }
            secondary2 = next(
                p
                for p in inputs["profiles"]
                if p["family"] == "secondary30" and p["repeat_id"] == 2
            )
            assert secondary2["host"]["host"] == "recovery"
            frozen = json.loads(manifest.read_text())
            target = root / frozen["files"][0]["path"]
            shutil.copyfile(target, root / "configs/pristine-generated-protocol.json")
            target.write_text(target.read_text() + "\n")
            with pytest.raises(ValueError, match="Final protocol bytes differ from joint manifest"):
                analysis.bind_inputs(
                    SimpleNamespace(
                        plan=plan,
                        manifest=manifest,
                        training_manifest=data.training,
                        out=root / "configs/should-not-bind.json",
                    )
                )
            assert not (root / "configs/should-not-bind.json").exists()
            summary.update(
                profile_count=11,
                training_entries=24,
                secondary_repeat2_host="recovery",
                changed_protocol_bytes_rejected=True,
                joint_builder_sha256=sha(SCRIPTS / "freeze_lewm_navigation_joint_final_v2.py"),
                analysis_v3_sha256=sha(SCRIPTS / "analyze_lewm_navigation_campaign_20261031_v3.py"),
            )
        assert not (
            {name.split(".")[0] for name in set(sys.modules) - heavy_before}
            & {"numpy", "torch", "h5py", "matplotlib", "PIL"}
        )
    finally:
        save(tmp_path / "integration-report.json", summary)
        shutil.copytree(root, tmp_path / "full-synthetic-root-snapshot")
        (tmp_path / "external-synthetic-root.txt").write_text(str(root) + "\n")
