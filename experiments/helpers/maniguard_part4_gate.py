"""Fail-closed physical stage contract and truthful unstarted-stage receipts."""

from helpers.maniguard_part4_protocol import CONFIG, atomic, clock, digest, read


def authorize_acquisition(run):
    offline = read(run / "offline-audit.json")
    if not offline["gate"]["eligible"]:
        raise ValueError("Frozen offline premise did not pass: " + offline["gate"]["status"])
    required = {
        "scene-engineering.json": ("valid_scene_builder", "reset_repeatable", "saved_input_policy_checked"),
        "capacity-decision.json": ("fits_original_allocation",),
        "shared-initial-gate.json": ("eligible",),
        "comparative-freeze.json": ("sealed_before_method_specific_outcomes",),
    }
    for name, flags in required.items():
        receipt = read(run / name)
        if not all(receipt.get(flag) is True for flag in flags):
            raise ValueError("Missing physical acquisition gate: " + name)
        if receipt.get("protocol_sha256") != digest(CONFIG / "offline-protocol.json"):
            raise ValueError("Gate uses a different target/model protocol")
    if clock()["physical_remaining_seconds"] <= 0:
        raise ValueError("Original allocation has reached its physical cutoff")
    return True


def close_failed_premise(run):
    audit = read(run / "offline-audit.json")
    if audit["gate"]["eligible"] or audit["gate"]["shared_initial_eligible"]:
        raise ValueError("Cannot claim a terminal failed-premise diagnostic while further stages remain eligible")
    stages = {
        "scene_engineering": {"status": "unstarted_after_failed_offline_gate", "physical_attempts": 0},
        "shared_initial": {"planned": 8, "executed": 0},
        "acquisition": {m: {"planned": 12, "executed": 0, "proposals": [], "lineage": [],
                            "archive": [], "status": "unstarted_by_scientific_gate"}
                        for m in ("R", "U", "E", "D")},
        "confirmation": {"planned": 8, "executed": 0, "records": []},
        "repair": {"status": "unstarted_by_scientific_gate", "exports": []},
        "fresh_audit": {"planned": 16, "executed": 0, "labels_opened": False,
                        "predictions": [], "tasks_selected": False},
    }
    for method, value in stages["acquisition"].items():
        atomic(run / "acquisition" / method / "unstarted.json", value)
    atomic(run / "scene-manifest.json", {
        "status": "unstarted_after_failed_offline_gate", "generated_scenes": [],
        "physical_validity_receipts": [], "static_metadata_inspected": True,
        "reason": "A transform-only scene edit would not establish coherent physics, object references, goal metadata, reachability or post-settling validity.",
    })
    terminal = {"schema": "maniguard-part4-terminal-v1", "status": "scientifically_gated",
                "reason": audit["gate"], "offline_audit_sha256": digest(run / "offline-audit.json"),
                "stages": stages, "new_physical_attempts": 0, "new_controls": 0,
                "new_physics_steps": 0, "new_VLA_queries": 0, "new_encoder_calls": 0,
                "historical_donors_reused": 24, "clock": clock(),
                "interpretation": "Completed bounded premise diagnostic. No evidence for or against evolutionary acquisition or predictor repair efficacy."}
    atomic(run / "terminal.json", terminal)
    return terminal
