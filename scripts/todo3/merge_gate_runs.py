"""Merge the per-job run leaves of the todo 3 gate into one run directory per arm and seed.

Each job wrote outputs/eval_logs/gate03/<arm>/seed<s>/<task>/ID/ (results.jsonl with one row,
eval_config.json, summary.json, <task>/base_main.mp4, base_wrist.mp4, base_ltl.json) because
the lane wrapper takes one output leaf per invocation. For the dataset (jar/gate03_<arm>/seed<s>/
with the <n>_<outcome> folders of scripts/todo1/upload_hf.py) the six task leaves of one
(arm, seed) are merged into outputs/eval_logs/gate03_merged/<arm>/seed<s>/ in the layout of a
family run: results.jsonl (six rows, the completed attempt of each job; a job with no completed
row contributes its last attempt's row), eval_config.json, expected_rollouts.json (rollouts
concatenated), summary.json (counts), and <task>/base_*.{mp4,json}. Nothing under gate03/ is
modified.

    python3 -I scripts/todo3/merge_gate_runs.py [--plan progress/receipts/03-plan.json] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path("/workspace/Safety-Dial")
LOGS = ROOT / "third_party/ManiGuard/outputs/eval_logs"


def rows(leaf):
    p = LOGS / leaf / "ID" / "results.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default=str(ROOT / "progress/receipts/03-plan.json"))
    ap.add_argument("--out", default=str(LOGS / "gate03_merged"))
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    report = {}
    for arm in plan["arms"]:
        for seed in plan["seeds"]:
            dest = out / arm / f"seed{seed}"
            dest.mkdir(parents=True)
            merged, expected, cfg_done = [], [], False
            for job in [j for j in plan["jobs"] if j["arm"] == arm and j["seed"] == seed]:
                chosen = None
                for leaf in (job["leaf"], job["leaf"] + "_rerun1"):
                    rs = rows(leaf)
                    done = [r for r in rs if r.get("status") == "completed"]
                    if done:
                        chosen = (leaf, done[-1]); break
                    if rs:
                        chosen = (leaf, rs[-1])
                if chosen is None:
                    continue
                leaf, row = chosen
                row = {**row, "gate03_job": job["job"], "gate03_leaf": leaf, "gate03_port": job["port"]}
                merged.append(row)
                src = LOGS / leaf
                if not cfg_done and (src / "ID" / "eval_config.json").is_file():
                    shutil.copy2(src / "ID" / "eval_config.json", dest / "eval_config.json"); cfg_done = True
                if (src / "expected_rollouts.json").is_file():
                    expected += json.loads((src / "expected_rollouts.json").read_text())["rollouts"]
                task_src = src / "ID" / job["task"]
                if task_src.is_dir():
                    (dest / job["task"]).mkdir(exist_ok=True)
                    for f in task_src.iterdir():
                        shutil.copy2(f, dest / job["task"] / f.name)
            (dest / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in merged))
            (dest / "expected_rollouts.json").write_text(json.dumps(
                {"schema_version": 1, "family": "jar_transport", "rollouts": expected}, indent=2) + "\n")
            done = [r for r in merged if r.get("status") == "completed"]
            (dest / "summary.json").write_text(json.dumps({
                "run": f"gate03/{arm}/seed{seed}", "n_attempted": len(merged), "n_scenes": len(done),
                "n_failed": len(merged) - len(done), "n_success": sum(1 for r in done if r.get("success")),
                "n_ltl_violated": sum(1 for r in done if r.get("counted_violation")),
                "n_engaged": sum(1 for r in done if r.get("first_contact_step") is not None)}, indent=2) + "\n")
            report[f"{arm}/seed{seed}"] = (len(merged), len(done))
    print({k: v for k, v in report.items()})
    print(f"merged {sum(v[0] for v in report.values())} rows ({sum(v[1] for v in report.values())} completed) into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
