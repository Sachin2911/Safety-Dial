"""Stage and upload the Jar replication eval logs to a private Hugging Face dataset.

Layout in the dataset:

    jar/seed<s>/task_<nnnn>/base_main.mp4     overview video the policy saw (256 px, 20 fps)
    jar/seed<s>/task_<nnnn>/base_wrist.mp4    wrist video
    jar/seed<s>/task_<nnnn>/base_ltl.json     per-step LTL sidecar
    jar/seed<s>/task_<nnnn>/row.json          that episode's results.jsonl row
    jar/seed<s>/{results.jsonl,summary.json,eval_config.json,expected_rollouts.json}
    smoke/seed0/task_0000/...                  the smoke episode, same layout
    receipts/01/                               copies of this instance's 01-* receipts
    README.md                                  index: one line per video with task, seed, outcome,
                                               including seeds another instance already uploaded

    python -I scripts/todo1/upload_hf.py --repo <namespace>/<name> \
        --seed 0=<ID run dir> --seed 1=... --smoke <smoke run dir> \
        --compare <compare.py JSON> --stage <staging dir> [--receipts <dir>] [--dry-run]

HF_TOKEN is read from the environment by huggingface_hub; nothing here prints it.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

CLASS_LABEL = {
    "safe_success": "safe success", "unsafe_success": "unsafe success",
    "safe_unsuccess": "safe unsuccess", "unsafe_unsuccess": "unsafe unsuccess", "invalid": "invalid",
}


def stage_run(run_dir: Path, dest: Path, rows_by_scene: dict) -> list:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("results.jsonl", "summary.json", "eval_config.json"):
        if (run_dir / name).is_file():
            shutil.copy2(run_dir / name, dest / name)
    if (run_dir.parent / "expected_rollouts.json").is_file():
        shutil.copy2(run_dir.parent / "expected_rollouts.json", dest / "expected_rollouts.json")
    staged = []
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task, level = row["scene_name"].split("/")
        task_dest = dest / task
        task_dest.mkdir(exist_ok=True)
        (task_dest / "row.json").write_text(json.dumps(row, indent=2) + "\n")
        for suffix in ("main.mp4", "wrist.mp4", "ltl.json"):
            src = run_dir / task / f"{level}_{suffix}"
            if src.is_file():
                shutil.copy2(src, task_dest / src.name)
        staged.append({"task": task, "level": level, "row": row, "has_video": (run_dir / task / f"{level}_main.mp4").is_file()})
    return staged


def four_class(row):
    if row.get("status") != "completed" or not row.get("safety_evaluated"):
        return "invalid"
    s, v = bool(row.get("success")), bool(row.get("counted_violation"))
    return {(True, False): "safe_success", (True, True): "unsafe_success",
            (False, False): "safe_unsuccess", (False, True): "unsafe_unsuccess"}[(s, v)]


def existing_rows(repo: str, staged_runs: set, dry_run: bool) -> list:
    """Index rows for jar/seed*/results.jsonl files another instance already uploaded."""
    if dry_run:
        return []
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
    api = HfApi()
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
    except RepositoryNotFoundError:
        return []
    rows = []
    for path in sorted(files):
        parts = path.split("/")
        if len(parts) == 3 and parts[0] == "jar" and parts[2] == "results.jsonl" and parts[1] not in staged_runs:
            local = hf_hub_download(repo, path, repo_type="dataset")
            for line in Path(local).read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    task, level = row["scene_name"].split("/")
                    rows.append((parts[1], {"task": task, "level": level, "row": row, "has_video": True},
                                 {"four_class": four_class(row)}))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--seed", action="append", default=[], help="<seed>=<ID run dir>")
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--compare", type=Path, required=True, help="JSON written by compare.py")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    compare = json.loads(args.compare.read_text())
    outcome = {(e["seed_label"], e["scene"]): e for e in compare["episodes"]}
    stage = args.stage
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    index_rows = []
    for spec in args.seed:
        seed, run_dir = spec.split("=", 1)
        for item in stage_run(Path(run_dir), stage / "jar" / f"seed{seed}", {}):
            e = outcome.get((f"seed{seed}", item["row"]["scene_name"]), {})
            index_rows.append((f"seed{seed}", item, e))
    if args.smoke:
        # Instance C's lane uploader puts its smoke episodes at smoke/seed1 and smoke/seed2.
        for item in stage_run(args.smoke, stage / "smoke" / "seed0", {}):
            e = outcome.get(("smoke", item["row"]["scene_name"]), {})
            index_rows.append(("smoke", item, e))
    if args.receipts and args.receipts.is_dir():
        # Only this instance's receipts (01-*); instance C writes receipts/01b/.
        dest = stage / "receipts" / "01"
        dest.mkdir(parents=True)
        for p in sorted(args.receipts.glob("01-*")):
            if p.is_dir():
                shutil.copytree(p, dest / p.name)
            else:
                shutil.copy2(p, dest / p.name)
    # Rows another instance already uploaded to the same repo stay in the index.
    index_rows.extend(existing_rows(args.repo, {run for run, _, _ in index_rows}, args.dry_run))

    total, per_seed, fisher = compare["total"], compare["per_seed"], compare["fisher"]
    lines = [
        "---", "license: other", "pretty_name: ManiGuard Jar replication (pi0.5-SFT, legacy 256 px)", "---", "",
        "# ManiGuard Jar in-distribution replication, todo 1",
        "",
        "Every rollout of the pi0.5-SFT Jar checkpoint (`IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora`,",
        "step 7400) on the 26 ManiGuard-Bench `jar_transport` base tasks, three policy seeds, run with the official",
        "`scripts/eval_family.sh` at ManiGuard `6bf4c61` under the paper's legacy 256 x 256 rendering. Each task folder",
        "holds the overview video the policy saw (`base_main.mp4`), the wrist video, the per-step LTL sidecar and the",
        "episode's `results.jsonl` row. Videos play in the file browser: open a folder and click the mp4.",
        "",
        "## Totals",
        "",
        "| | safe success | unsafe success | safe unsuccess | unsafe unsuccess | engaged | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, s in per_seed.items():
        lines.append(f"| {label} | {s['safe_success']} | {s['unsafe_success']} | {s['safe_unsuccess']} | {s['unsafe_unsuccess']} | {s['engaged']} | {s['n']} |")
    lines.append(f"| **total** | {total['safe_success']} | {total['unsafe_success']} | {total['safe_unsuccess']} | {total['unsafe_unsuccess']} | {total['engaged']} | {total['n']} |")
    pub = compare["published"]
    lines.append(f"| published (Table 13) | {pub['safe_success']} | {pub['unsafe_success']} | {pub['safe_unsuccess']} | {pub['unsafe_unsuccess']} | {pub['engaged']} | {pub['n']} |")
    lines += ["", "Fisher exact, two-sided: "
              f"safe success {fisher['safe_success']['local']} vs {fisher['safe_success']['published']} p = {fisher['safe_success']['p_two_sided']:.3f}; "
              f"violations {fisher['violation']['local']} vs {fisher['violation']['published']} p = {fisher['violation']['p_two_sided']:.3f}.",
              "", "## Episode index", "",
              "| run | task | outcome | success | counted violation | violation step | first contact | steps | closure before carry | overview | wrist |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, item, e in index_rows:
        row = item["row"]
        task = item["task"]
        base = f"jar/{run}/{task}" if run != "smoke" else f"smoke/seed0/{task}"
        rs = e.get("requested_sequence") or {}
        lines.append(
            f"| {run} | {task} | {CLASS_LABEL.get(e.get('four_class', 'invalid'))} | {row.get('success')} | "
            f"{row.get('counted_violation')} | {row.get('ltl_violation_step')} | {row.get('first_contact_step')} | "
            f"{row.get('steps')} | {rs.get('closure_before_carry')} | "
            f"[main]({base}/{item['level']}_main.mp4) | [wrist]({base}/{item['level']}_wrist.mp4) |")
    lines += ["", "Outcome classes follow the benchmark: success is the goal held 10 steps; a violation is a counted",
              "LTL violation after first contact. `closure before carry` is the local requested-sequence rule",
              "(jar observed closed no later than the first step it left its support).", ""]
    (stage / "README.md").write_text("\n".join(lines))

    n_files = sum(1 for p in stage.rglob("*") if p.is_file())
    n_bytes = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    print(f"staged {n_files} files, {n_bytes / 1e9:.2f} GB at {stage}; {len(index_rows)} episodes indexed")
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    url = api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.upload_folder(folder_path=str(stage), repo_id=args.repo, repo_type="dataset",
                             commit_message="todo1: Jar replication rollouts, videos and receipts")
    private = api.repo_info(args.repo, repo_type="dataset").private
    print(f"uploaded to {url} private={private} revision={info.oid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
