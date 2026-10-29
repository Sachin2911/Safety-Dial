"""Requirement-level completion and exact private snapshot coverage audit."""

import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm17_protocol import CONFIG, OUT, digest, write_json
from helpers.lewm17_remote import finance, rental, utc


def main():
    stats = json.loads((OUT / "analysis/statistics.json").read_text())
    audit = json.loads((ROOT / "progress/receipts/17-final-audit.json").read_text())
    assert stats["episodes"] == audit["expected"] == 7800 and audit["passed"]
    assert stats["stages"] == {
        "E": 16,
        "E2": 32,
        "C": 72,
        "F": 288,
        "A": 720,
        "T": 4320,
        "V": 720,
        "H": 1632,
    }
    assert stats["failed_attempts"] == 0
    for path in (OUT / "attempts").glob("*/result.json"):
        row = json.loads(path.read_text())
        trial = row["job"]
        assert trial["params"]["objective"] == "latent"
        assert trial["params"]["checked"] or trial["condition"] == "native"
        assert trial["margin"] in ([4.0, 6.0] if trial["env"] == "tworoom" else [0.01, 0.015])
        if "weights" in trial["params"]:
            weights = trial["params"]["weights"]
            assert len(weights) == 6 and all(len(r) == 6 for r in weights)
            assert all(-3 <= v <= 3 for r in weights for v in r)
    trial_groups = collections.defaultdict(list)
    for path in (OUT / "attempts").glob("*/result.json"):
        trial = json.loads(path.read_text())["job"]
        if trial.get("optimizer_run"):
            trial_groups[trial["optimizer_run"]].append(trial)
    assert len(trial_groups) == 30
    all_trials = [
        json.loads(p.read_text())["job"] for p in (OUT / "attempts").glob("*/result.json")
    ]
    final_seeds = {j["seed"] for j in all_trials if j["stage"] == "H"}
    other_seeds = {j["seed"] for j in all_trials if j["stage"] != "H"}
    assert final_seeds == set(
        json.loads((CONFIG / "final-planner-seeds.json").read_text())["seeds"]
    )
    assert not final_seeds & other_seeds
    amendment = json.loads(
        (ROOT / "progress/receipts/17-final-planner-seed-amendment.json").read_text()
    )
    final_access = json.loads((OUT / "final-access.json").read_text())
    assert amendment["recorded_epoch"] < final_access["utc_epoch"]
    for name, trials in trial_groups.items():
        assert collections.Counter(j["stage"] for j in trials) == {"A": 24, "T": 144, "V": 24}
        folder = OUT / "optimizers" / name
        for round_index in range(6):
            before = json.loads((folder / f"round{round_index:02d}-before.json").read_text())
            expected = sorted(
                (point["case_id"], point["margin"], point["planner_seed"])
                for point in before["points"]
            )
            for candidate in range(8):
                own = [
                    j
                    for j in trials
                    if j["stage"] == "T" and j["candidate"] == f"round{round_index}-c{candidate}"
                ]
                assert len(own) == 3
                actual = sorted((j["case"]["id"], j["margin"], j["seed"]) for j in own)
                assert actual == expected
                assert all(j["params"]["stochastic"] for j in own)
                assert all(j["params"]["weights"] == before["candidates"][candidate] for j in own)
        validation = json.loads((folder / "validation-freeze.json").read_text())
        selected_run = json.loads((folder / "selected.json").read_text())
        assert selected_run["weights"] == validation["weights"][selected_run["selected_index"]]
        signatures = []
        for candidate in range(4):
            own = [
                j for j in trials if j["stage"] == "V" and j["candidate"] == f"finalist{candidate}"
            ]
            assert len(own) == 6
            assert all(not j["params"]["stochastic"] for j in own)
            assert all(j["params"]["weights"] == validation["weights"][candidate] for j in own)
            signatures.append(sorted((j["case"]["id"], j["margin"], j["seed"]) for j in own))
        assert all(signature == signatures[0] for signature in signatures)
    ready = json.loads((OUT / "environment/prepared.json").read_text())
    model_hashes = {}
    for env in ("tworoom", "reacher"):
        path = ROOT / f"data/lewm17/stablewm/checkpoints/{env}/lewm_object.ckpt"
        model_hashes[env] = digest(path)
        assert model_hashes[env] == ready[env]["checkpoint_sha256"]
        assert digest(CONFIG / f"{env}-scalers.json") == ready[env]["scalers_sha256"]
    live = []
    for path in (OUT / "control/workers").glob("*.json"):
        worker = json.loads(path.read_text())
        proc = Path(f"/proc/{worker['pid']}")
        if not proc.exists():
            continue
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z" or fields[19] != worker["start_ticks"]:
            continue
        if worker["script"].encode() in (proc / "cmdline").read_bytes().split(b"\0"):
            live.append(worker)
    assert not live, "Experiment workers must be terminal before final preservation audit"
    receipt = json.loads((OUT / "control/preservation-latest.json").read_text())
    assert receipt["verified"] and receipt["private"]
    snapshot = OUT / "preservation" / receipt["prefix"].split("/")[-1]
    manifest = json.loads((snapshot / "files.json").read_text())
    assert digest(snapshot / "index.json") == receipt["index_sha256"]
    required = set()
    for folder in (OUT / "attempts", OUT / "optimizers", OUT / "analysis", OUT / "source", CONFIG):
        required.update(
            p for p in folder.rglob("*") if p.is_file() and not p.name.endswith(".part")
        )
    required.update(p for p in (OUT / "logs").glob("*") if p.is_file())
    for relative in (
        "progress/receipts/17-lewm-evolution-results.md",
        "progress/receipts/17-final-statistics.json",
        "progress/receipts/17-final-audit.json",
        "progress/receipts/17-completion-status.json",
        "progress/receipts/17-final-planner-seed-amendment.json",
        "progress/receipts/17-synthetic-test-work.json",
        "progress/receipts/17-reporting-revision.json",
        "progress/receipts/17-engineering-final-audit.json",
        "progress/receipts/17-implementation-audit.json",
        "progress/receipts/17-optimizer-comparison.json",
        "progress/receipts/17-final-evaluation.json",
        "progress/README.md",
        "progress/receipts/README.md",
        "progress/theOdessy.tex",
        "progress/theOdessy.pdf",
        "docs/research/research_notes/LeWM evolution task 17.md",
        "outputs/lewm17/final-selection.json",
        "outputs/lewm17/final-access.json",
        "outputs/lewm17/final-integrity-access.json",
        "outputs/lewm17/environment/campaign-tests.txt",
        "outputs/lewm17/environment/installed-freeze.txt",
    ):
        required.add(ROOT / relative)
    for path in required:
        relative = str(path.relative_to(ROOT))
        assert path.is_file() and relative in manifest, (
            f"Missing required preserved artifact: {relative}"
        )
        assert digest(path) == manifest[relative]["sha256"], f"Unpreserved change: {relative}"
    assert "17 passed" in (OUT / "environment/campaign-tests.txt").read_text()
    assert "Task 17:" in (ROOT / "progress/theOdessy.tex").read_text()
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "--",
            "configs/lewm16",
            "experiments/helpers/lewm16*",
            "experiments/scripts/*lewm16*",
            "progress/receipts/16-*",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    financial = finance(rental())
    assert financial["conservative_total_usd"] < 9 and not financial["closeout_due"]
    result = {
        "utc": utc(),
        "scientific_completion_verified": True,
        "all_7800_attempts_and_30_runs_verified": True,
        "private_static_files_verified": len(required),
        "verified_snapshot": receipt,
        "frozen_model_checkpoint_sha256": model_hashes,
        "historical_task16_diff_empty": True,
        "registered_experiment_workers_live": live,
        "financial_reserve_verified": financial,
        "remaining_authorized_operation": "Final private snapshot including this audit, then control_lewm17.py finish",
        "external_absence": "Not observed; requires an external check after provider deletion",
    }
    write_json(ROOT / "progress/receipts/17-completion-audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
