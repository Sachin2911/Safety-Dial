"""Audit the finished adapter pilot and aggregate its recorded descriptive metrics.

This workstation-only reporting step never starts a simulator or changes scoring.
Run analyze_adapter05.py and finish the visual reviews first.
"""

import collections
import datetime as dt
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "outputs/adapter05/remote"
OUT = ROOT / "outputs/adapter05/analysis"


def read(path):
    return json.loads(path.read_text())


def weighted_stat(episodes, select):
    values = [select(e) for e in episodes]
    values = [v for v in values if v["n"]]
    return sum(v["n"] * v["mean"] for v in values) / sum(v["n"] for v in values)


def main():
    assert (RAW / "pilot-complete.txt").exists(), "Queue still active"
    assert not (RAW / "STOP").exists(), "Queue stopped early"
    plan = read(ROOT / "progress/receipts/05-adapter-plan-v4.json")
    summary = read(OUT / "summary.json")
    episodes = summary["episodes"]
    expected = {j["job"]: j for j in plan["engineering_jobs"] + plan["jobs"]}
    assert len(episodes) == len(expected) == 38
    assert {e["job"] for e in episodes} == set(expected)
    for episode in episodes:
        for key in ("stage", "task", "seed", "arm", "preferred_lane"):
            assert episode[key] == expected[episode["job"]][key], (episode["job"], key)
        assert episode["attempt"] == 1
    reviews = read(OUT / "review-notes.json")
    validations = sorted(RAW.glob("episodes/*/validation.json"))
    assert len(validations) == len(reviews) == 38
    assert set(reviews) == {p.parent.name for p in validations}
    for review in reviews.values():
        assert set(review["reviewed"]) == {
            "timeline.png",
            "joint_targets.png",
            "event_frames.png",
            "adapter_activity.png",
        }
    checks = []
    for validation in validations:
        folder = validation.parent
        steps = read(validation)["steps"]
        with np.load(next(folder.glob("recordings/*.npz")), allow_pickle=False) as data:
            numeric = [key for key in data.files if data[key].dtype.kind in "fciub"]
            assert all(np.isfinite(data[key]).all() for key in numeric), folder.name
        videos = {}
        for camera in ("main", "wrist"):
            path = next(folder.glob(f"eval/task_*/base_{camera}.mp4"))
            cap = cv2.VideoCapture(str(path))
            assert cap.isOpened(), str(path)
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "declared_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
            }
            cap.release()
            assert info["width"] == info["height"] == 256
            assert info["declared_frames"] == steps + 1
            videos[camera] = info
        checks.append(
            {
                "episode": folder.name,
                "numeric_fields": len(numeric),
                "all_numeric_finite": True,
                "videos": videos,
            }
        )
    (OUT / "additional-recording-checks.json").write_text(
        json.dumps(
            {
                "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "scope": "Final 38 recordings; all numeric NPZ fields and both video headers. "
                "Full frame decoding is separately performed by analyze_adapter05.py.",
                "recordings": len(checks),
                "videos": 2 * len(checks),
                "checks": checks,
            },
            indent=2,
        )
        + "\n"
    )
    accounting = summary["accounting"]
    attempts = accounting["attempts"]
    assert len(attempts) == 38 and all(a["validated"] for a in attempts)
    assert all(a["truncated_journal_lines"] == 0 for a in attempts)
    assert all(a["warmup_attempts"] == a["warmup_returned"] == 10 for a in attempts)
    stages = {}
    for stage in ("engineering", "pilot"):
        ledger = [
            json.loads(line) for line in (RAW / f"attempts_{stage}.jsonl").read_text().splitlines()
        ]
        counts = collections.Counter(e["event"] for e in ledger)
        n = 2 if stage == "engineering" else 36
        for key in ("start_reserved", "simulator_launch", "simulator_exit", "validated"):
            assert counts[key] == n, (stage, key, counts[key])
        exits = [e for e in ledger if e["event"] == "simulator_exit"]
        assert all(e["rc"] == 0 and e["completed_row"] for e in exits)
        start = next(e for e in ledger if e["event"] == "queue_start")
        end = next(e for e in ledger if e["event"] == "queue_finished")
        assert end["replacements"] == 0
        rows = [a for a in attempts if a["stage"] == stage]
        for a in rows:
            assert a["policy_commands"] == a["policy_returned"]
            assert a["policy_query_attempts"] == a["policy_query_returns"]
            assert a["policy_query_returns"] == (a["policy_returned"] + 7) // 8
            assert a["reseed_probe_queries"] == a["reseed_probe_query_returns"] == 1
        fields = (
            "warmup_attempts",
            "warmup_returned",
            "policy_commands",
            "policy_returned",
            "policy_query_attempts",
            "policy_query_returns",
            "reseed_probe_queries",
            "reseed_probe_query_returns",
        )
        totals = {key: sum(a[key] for a in rows) for key in fields}
        logs = list((RAW / "logs" / stage).glob("probe*.log"))
        totals["startup_query_attempts"] = sum(
            p.read_text().count("adapter05_query_attempt") for p in logs
        )
        totals["startup_query_returns"] = sum(
            p.read_text().count("adapter05_query_returned") for p in logs
        )
        totals["all_query_attempts"] = (
            totals["policy_query_attempts"]
            + totals["reseed_probe_queries"]
            + totals["startup_query_attempts"]
        )
        totals["all_query_returns"] = (
            totals["policy_query_returns"]
            + totals["reseed_probe_query_returns"]
            + totals["startup_query_returns"]
        )
        stages[stage] = {
            "starts": n,
            "replacements": 0,
            "queue_start_utc": start["utc"],
            "queue_finish_utc": end["utc"],
            "queue_elapsed_s": (
                dt.datetime.fromisoformat(end["utc"]) - dt.datetime.fromisoformat(start["utc"])
            ).total_seconds(),
            "subprocess_elapsed_sum_s": sum(e["elapsed_s"] for e in exits),
            "recorder_overhead_sum_s": sum(
                e["recorder_overhead_s"] for e in ledger if e["event"] == "validated"
            ),
            **totals,
        }
    aggregates = {}
    for arm in ("native", "fixed", "adaptive"):
        cells = [e for e in episodes if e["stage"] == "pilot" and e["arm"] == arm]
        assert len(cells) == 12
        assert collections.Counter(e["preferred_lane"] for e in cells) == {0: 6, 1: 6}
        n = sum(e["row"]["steps"] for e in cells)
        aggregates[arm] = {
            "lag_absolute_error_means_rad": [
                weighted_stat(cells, lambda e, lag=lag: e["lag_error_rad"][lag]["all"])
                for lag in range(9)
            ],
            "realized_step_rms_mean_rad": weighted_stat(
                cells, lambda e: e["windows"]["all"]["realized_move_rms_rad"]
            ),
            "post_action_rms_mean_rad": weighted_stat(
                cells, lambda e: e["windows"]["all"]["post_action_error_rms_rad"]
            ),
            "correction_absolute_mean_rad": weighted_stat(cells, lambda e: e["correction_abs_rad"]),
            "saturated_joint_fraction": sum(
                e["saturated_joint_fraction"] * e["row"]["steps"] for e in cells
            )
            / n,
            "clipped_action_fraction": sum(
                e["arm_clipped_action_fraction"] * e["row"]["steps"] for e in cells
            )
            / n,
        }
    source = read(ROOT / "progress/receipts/05-adapter-analysis-source.json")
    for name, expected_hash in source["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_hash, name
    result = {
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "full_plan_completed": True,
        "all_visual_reviews_present": True,
        "stages": stages,
        "pilot_tracking": aggregates,
        "aggregation": "Tracking statistics weight every recorded step/joint equally. "
        "The rows are descriptive of distinct factual trajectories, not paired causal effects.",
    }
    (OUT / "completion-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
