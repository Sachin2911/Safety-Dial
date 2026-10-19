"""Timing-only selection that protects a complete matched pilot and native horizons."""

import calendar
import hashlib
import json
import subprocess
import time
from pathlib import Path

from helpers.maniguard_latent_queue import ROOT
from helpers.maniguard_latent_results import read_episode


def planning_bound(timings, horizon):
    """Combine the slowest startup and control throughput, allowing full timeouts.

    The first saved prequery frame provides a filesystem wall-time observation of
    startup. The exact values and file hashes are retained in the decision receipt.
    Scaling only the warm segment avoids scaling one-off startup as policy work.
    """
    if not timings or horizon < 1:
        raise ValueError("Need measured development timing and a native horizon")
    for row in timings:
        if not 0 <= row["startup_seconds"] < row["elapsed_seconds"] or not 0 < row["steps"] <= horizon:
            raise ValueError("Invalid measured development timing")
    startup = max(row["startup_seconds"] for row in timings)
    seconds_per_control = max((row["elapsed_seconds"] - row["startup_seconds"]) / row["steps"]
                              for row in timings)
    return max(180., 1.25 * max(row["elapsed_seconds"] for row in timings),
               1.25 * (startup + horizon * seconds_per_control))


def comparison_design(family):
    train = {"jar": ["task_0019", "task_0009"],
             "clutter": ["task_0034", "task_0002"]}[family]
    audit = {"jar": ["task_0016", "task_0012", "task_0011", "task_0003"],
             "clutter": ["task_0048", "task_0050", "task_0048", "task_0050"]}[family]
    historical = json.loads((ROOT / "configs/maniguard/phase2/splits.json").read_text())["families"][family]
    if not set(train).issubset(historical["train"]) or not set(audit).issubset(historical["validation"]):
        raise ValueError("Prospective cases breach the preserved main-study partition")
    if set(train) & set(audit) or (set(train) | set(audit)) & set(historical["test"]):
        raise ValueError("Audit and training tasks or reserved main-test tasks overlap")
    orders = [["native", "fixed", "ea", "random"], ["fixed", "random", "native", "ea"],
              ["ea", "native", "random", "fixed"], ["random", "ea", "fixed", "native"]]
    return {"training_panels": [[{"scene": task + "/base", "seed": 6101910 + 10 * g + i}
                                  for i, task in enumerate(train)] for g in range(3)],
            "audit_blocks": [{"scene": task + "/base", "seed": 6101940 + i, "order": orders[i]}
                             for i, task in enumerate(audit)],
            "schedule": "Initial candidates in birth order; descendants interleave methods at each slot, reversing method order on odd slots; cases in listed order.",
            "exposure": {"training": "Programme-exposed tasks, independent new policy seeds.",
                         "audit": "Reserved from this study's development and optimizer feedback. Historical validation allocation does not prove absence of prior programme exposure; conservatively treated as exposed.",
                         "clutter_scope": "Two tasks, each checked at two new seeds. Known Part1 comparison tasks 0029 and 0032 are excluded; a whole-task generalization claim is not made.",
                         "foundation_training": "No foundation-policy training-data exclusion claim."}}


def select(run, development_protocol, *, now=None):
    run, development_protocol = Path(run), Path(development_protocol)
    protocol = json.loads(development_protocol.read_text())
    queue = json.loads((run / "queue.json").read_text())
    if any(a["stage"] not in ["engineering", "development"] for a in queue["attempts"]):
        raise ValueError("Capacity selection must precede every comparative physical attempt")
    now = time.time() if now is None else now
    cutoff = calendar.timegm(time.strptime(protocol["physical_cutoff_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    full = [a for a in queue["attempts"] if not a["engineering"]]
    invalid = sum(a.get("status") == "invalid" for a in full)
    if invalid > 4:
        raise ValueError("The full-attempt replacement allowance was exceeded")
    families = {}
    for family, horizon in [("jar", 2000), ("clutter", 1100)]:
        timings = []
        for condition in ["native", "bias"]:
            job = queue["jobs"][f"dev-{family}-{condition}"]
            if job["status"] != "completed":
                raise ValueError("Complete all four development episodes before capacity selection")
            output = Path(job["output"])
            verified = read_episode(output, require_full=True)
            attempt = next(a for a in full if a["id"] == job["attempt_ids"][-1])
            terminal = json.loads(Path(attempt["terminal_receipt"]).read_text())
            elapsed = terminal["finished_epoch"] - terminal["started_epoch"]
            observation = output / "query_observations/000000.npz"
            first_query = observation.stat().st_mtime
            startup = first_query - terminal["started_epoch"]
            timings.append({"job": job["definition"]["id"], "elapsed_seconds": elapsed,
                            "steps": verified["physical"]["steps"], "startup_seconds": startup,
                            "first_query_saved_mtime": first_query,
                            "first_query_sha256": hashlib.sha256(observation.read_bytes()).hexdigest(),
                            "attempt": attempt["id"]})
        authority_path = run / ("authority-" + family) / "summary.json"
        authority = json.loads(authority_path.read_text())
        if authority["decoder_calls"] != 160:
            raise ValueError("Incomplete bounded authority audit")
        bound = planning_bound(timings, horizon)
        families[family] = {"development_timings": timings, "episode_seconds": bound,
                            "authority_pass": authority["authority_pass"],
                            "same_noise_identity_pass": authority["same_noise_identity_pass"],
                            "authority_summary_sha256": hashlib.sha256(
                                authority_path.read_bytes()).hexdigest()}
    authority_pass = all(f["authority_pass"] and f["same_noise_identity_pass"]
                         for f in families.values())
    valid_remaining, replacements_remaining = 56, 4 - invalid
    reserve = 900
    for family in families.values():
        family["required_seconds"] = (valid_remaining + replacements_remaining) * family["episode_seconds"] + reserve
        family["fits_time"] = family["required_seconds"] <= cutoff - now
    selected = next((name for name in ["jar", "clutter"] if families[name]["fits_time"]
                     and authority_pass), None)
    decision = {**protocol, "schema": "maniguard-latent-capacity-v1",
                "status": "capacity_selected_frozen" if selected else "capacity_insufficient_or_authority_failed",
                "selected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                          text=True).strip(),
                "development_protocol_sha256": hashlib.sha256(development_protocol.read_bytes()).hexdigest(),
                "family": selected, "capacity": {"fits": selected is not None,
                    "episode_seconds": families[selected]["episode_seconds"] if selected else None,
                    "available_seconds": cutoff - now, "valid_remaining": valid_remaining,
                    "replacements_remaining": replacements_remaining, "lifecycle_backup_reserve_seconds": reserve,
                    "families": families, "all_development_families_authority_pass": authority_pass,
                    "reason": "Complete 56 remaining valid episodes plus remaining replacements and lifecycle/backups must fit before the unchanged physical cutoff. Full native timeout throughput is included. Comparative scores are not inputs."},
                "unselected_designs": {name: comparison_design(name) for name in ["jar", "clutter"]}}
    if selected:
        decision.update(comparison_design(selected))
    return decision
