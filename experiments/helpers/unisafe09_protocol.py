"""Task-9 immutable schedules, deterministic selection and input validation."""

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def design():
    return json.loads((ROOT / "configs/unisafe09/design.json").read_text())


def condition(condition_id):
    if condition_id == "native_direct":
        return None
    return next(dict(c) for c in design()["conditions"] if c["id"] == condition_id)


def schedule(stage, selected=None):
    spec = design()
    panel = spec["stages"][stage]
    if stage == "A":
        jobs = [dict(job) for job in panel["jobs"]]
    else:
        ids = panel["condition_ids"] if stage == "B" else selected
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("A complete unique condition list is required")
        if stage == "C":
            if not set(panel["required_conditions"]) <= set(ids) or len(ids) > 6:
                raise ValueError("Invalid confirmation condition list")
        seeds = list(range(panel["seed_start"], panel["seed_stop_exclusive"]))
        rng = random.Random(spec["schedule"]["rng_seed"])
        rng.shuffle(seeds)
        initial_rotation = rng.randrange(len(ids))
        jobs = []
        for block, seed in enumerate(seeds):
            rotation = (initial_rotation + block) % len(ids)
            for cid in ids[rotation:] + ids[:rotation]:
                jobs.append(
                    {
                        "id": f"{stage}-{seed}-{cid}",
                        "seed": seed,
                        "policy_seed": seed,
                        "condition": cid,
                    }
                )
    for job in jobs:
        job.update(
            stage=stage,
            continuation=panel["continuation"],
            condition_spec=condition(job["condition"]),
        )
    return jobs


def select_confirmation(rows):
    """Require a complete B panel; thresholds never inspect confirmation outcomes."""
    spec = design()
    ids = spec["stages"]["B"]["condition_ids"]
    expected = {(j["seed"], j["condition"]) for j in schedule("B")}
    keys = [(r["seed"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Selection requires the complete, unique, valid B panel")
    stats = {}
    for cid in ids:
        group = [r for r in rows if r["condition"] == cid]
        stats[cid] = {
            "C": sum(r["safe_success"] for r in group),
            "V": sum(r["violation"] for r in group),
            "J": sum(r["J"] for r in group) / len(group),
        }
    ref_ids = list(spec["references"].values())
    r, e = stats["reference_random"], stats["d15_g100"]
    eligible = [c for c in ids if stats[c]["C"] >= 8 and stats[c]["V"] <= r["V"]]

    def dominates(a, b):
        x, y = stats[a], stats[b]
        return (
            x["C"] >= y["C"]
            and x["V"] <= y["V"]
            and x["J"] <= y["J"]
            and (x["C"] > y["C"] or x["V"] < y["V"] or x["J"] < y["J"])
        )

    frontier = [c for c in eligible if not any(dominates(a, c) for a in eligible)]
    speed = (
        min(frontier, key=lambda c: (stats[c]["J"], -stats[c]["C"], stats[c]["V"], c))
        if frontier
        else None
    )
    completion = (
        min(frontier, key=lambda c: (-stats[c]["C"], stats[c]["V"], stats[c]["J"], c))
        if frontier
        else None
    )
    tradeoff = bool(
        speed
        and completion
        and speed != completion
        and stats[completion]["C"] >= stats[speed]["C"] + 2
        and stats[completion]["J"] >= 1.1 * stats[speed]["J"]
    )
    qualifying = []
    signals = []
    for c in dict.fromkeys([speed, completion]):
        if c is None:
            continue
        x = stats[c]
        improvement = c not in ref_ids and x["C"] >= r["C"] + 2
        efficiency = c not in ref_ids and x["J"] <= 0.9 * r["J"] and x["C"] >= r["C"]
        if improvement or efficiency or tradeoff:
            qualifying.append(c)
            signals.append(
                {
                    "condition": c,
                    "improvement": improvement,
                    "efficiency": efficiency,
                    "tradeoff": tradeoff,
                }
            )
    ablations = [
        c
        for c in ("history_only", "grasp_latch")
        if abs(stats[c]["C"] - e["C"]) >= 4 or abs(stats[c]["V"] - e["V"]) >= 2
    ]
    ablations.sort(key=lambda c: (-abs(stats[c]["C"] - e["C"]), -abs(stats[c]["V"] - e["V"]), c))
    extras = [c for c in qualifying if c not in ref_ids][:2]
    extras += [c for c in ablations if c not in extras][: 2 - len(extras)]
    contrasts = []
    for c in extras:
        if c in qualifying:
            contrasts.append([c, "reference_random"])
        if c in {"history_only", "grasp_latch"}:
            contrasts.append([c, "d15_g100"])
    if speed != completion and speed in qualifying and completion in qualifying:
        pair = [completion, speed]
        if pair not in contrasts and pair[::-1] not in contrasts:
            contrasts.append(pair)
    return {
        "admit_signal": bool(signals or ablations),
        "statistics": stats,
        "frontier": frontier,
        "speed_endpoint": speed,
        "completion_endpoint": completion,
        "signals": signals,
        "qualifying_ablations": ablations,
        "conditions": ref_ids + extras,
        "primary_contrasts": contrasts,
    }


def verify_freeze(path):
    freeze = json.loads(Path(path).read_text())
    if not freeze["authorization"]["user_launch"]:
        raise ValueError("Execution is not authorized")
    for name, expected in freeze["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    return freeze
