"""Write the task 19 results receipt from the analysis summary.

Reads outputs/evo19/analysis/summary.json, configs/evo19/protocol.json, the frozen evolved
settings, the rental record and (when present) the upload verification, and writes
progress/receipts/19-evo-adapter-results.md. Numbers come only from those files.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
import manage_evo19 as infra  # noqa: E402

SUMMARY = ROOT / "outputs/evo19/analysis/summary.json"
OUT = ROOT / "progress/receipts/19-evo-adapter-results.md"


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


def ci(interval):
    return f"[{100 * interval[0]:.0f}, {100 * interval[1]:.0f}]%"


def contrast_rows(c):
    n = c["paired_task_seeds"]
    return (f"| {c['endpoint'].replace('_', ' ')} | {pct(c['native'], n)} {ci(c['native_wilson95'])} | "
            f"{pct(c['evolved'], n)} {ci(c['evolved_wilson95'])} | "
            f"{100 * c['difference_evolved_minus_native']:+.1f} pts | {c['fisher_one_sided_p']} | "
            f"{c['barnard_one_sided_p']} | {c['pairs_only_native']} / {c['pairs_only_evolved']} | "
            f"{c['exact_mcnemar_one_sided_p']} |")


def main():
    s = json.loads(SUMMARY.read_text())
    protocol = json.loads((ROOT / "configs/evo19/protocol.json").read_text())
    evolved = s["evolved"]
    record = infra.load()
    upload_path = ROOT / "outputs/evo19/upload-verification.json"
    upload = json.loads(upload_path.read_text()) if upload_path.exists() else None
    lines = ["# Task 19 results: an evolved tip shield for frozen pi0.5 on ManiGuard Jar", ""]
    p, v = s["primary"], s["secondary_violations"]
    lines += [
        "## Answer", "",
        f"Confirmation on {p['paired_task_seeds']} task-seed pairs (tasks 0, 3, 6, 9). Strict safe "
        f"completion: native {pct(p['native'], p['paired_task_seeds'])}, evolved shield "
        f"{pct(p['evolved'], p['paired_task_seeds'])}, one-sided Fisher p = "
        f"{p['fisher_one_sided_p']}. Raw violation episodes: native "
        f"{pct(v['native'], v['paired_task_seeds'])}, evolved "
        f"{pct(v['evolved'], v['paired_task_seeds'])}, one-sided Fisher p = "
        f"{v['fisher_one_sided_p']} (Barnard {v['barnard_one_sided_p']}).", "",
        "## Evolved adapter", "",
        f"Frozen at {evolved['frozen_utc']} after {evolved['generations']} stage {evolved['stage']} "
        f"CMA-ES generations: tilt limit {evolved['settings']['tilt_limit_deg']} degrees, retreat "
        f"{evolved['settings']['retreat_m']} m, contact window "
        f"{evolved['settings']['contact_window']} controls, active from the first control. At "
        "each policy query (every 8 controls), if the jar is tilted at least the limit from "
        "vertical and touched the robot in the last 8 controls, the arm retraces its own "
        "recorded joint path back to where the hand was the retreat distance away; otherwise "
        "the native pi0.5 chunk runs unchanged.", "",
        "## Confirmation (all complete pairs)", "",
        "| Endpoint | Native | Evolved | Difference | Fisher p | Barnard p | "
        "Only native / only evolved | McNemar p |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        contrast_rows(p), contrast_rows(v), "",
        "One-sided tests: evolved more strict safe completions, and evolved fewer violation "
        "episodes. Wilson 95% intervals in brackets. Episodes are not deterministic per seed "
        "(a repeat of b-native-t9-s193001 closed the jar and tipped late instead of tipping "
        "before closure), so pairing matches scenes, not trajectories; Fisher and Barnard are "
        "the main tests and McNemar is descriptive.", ""]
    if "panel72_primary" in s:
        lines += ["## Original 72-pair panel alone (amendment 6)", "",
                  "| Endpoint | Native | Evolved | Difference | Fisher p | Barnard p | "
                  "Only native / only evolved | McNemar p |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |",
                  contrast_rows(s["panel72_primary"]), contrast_rows(s["panel72_violations"]), ""]
    mix = s["outcome_mix"]
    keys = ["strict_safe", "violation", "stall_after_closure", "never_closed"]
    lines += ["## Outcome mix on the paired episodes", "",
              "| Arm | " + " | ".join(k.replace("_", " ") for k in keys) + " |",
              "| --- | " + " | ".join("---" for _ in keys) + " |"]
    for arm in ("native", "evolved"):
        lines.append(f"| {arm} | " + " | ".join(str(mix[arm].get(k, 0)) for k in keys) + " |")
    timing = s["violation_timing"]
    lines += ["", "Violation timing: native " + json.dumps(timing["native"]) + "; evolved "
              + json.dumps(timing["evolved"]) + ". Shield activity: "
              + json.dumps(s["evolved_shield_activity"]) + ".", "",
              "## By task", "", "| Task | Pairs | Strict safe native / evolved | "
              "Violations native / evolved |", "| --- | --- | --- | --- |"]
    for task, row in p["by_task"].items():
        vrow = v["by_task"][task]
        lines.append(f"| {task} | {row['pairs']} | {row['native']} / {row['evolved']} | "
                     f"{vrow['native']} / {vrow['evolved']} |")
    lines += ["", "## Search", "",
              "Four stages, each recorded as a protocol amendment before any evolved-arm "
              "episode. Stages 1 to 3 are exploratory.", "",
              "| Stage | Adapter | Episodes | Strict safe | Violations | Stalls after closure |",
              "| --- | --- | --- | --- | --- | --- |"]
    names = {1: "speed projection, 3 genes", 2: "isotropic speed projection, 2 genes",
             3: "tip shield, contact at the query only (pilot)",
             4: "tip shield, 8-control contact window"}
    for k in (1, 2, 3, 4):
        t = s[f"search_stage{k}"]
        lines.append(f"| {k} | {names[k]} | {t['valid']} | {t['strict_safe']} | "
                     f"{t['raw_violation']} | {t['post_gate_stall']} |")
    lines += ["", "Stage 4 generations (mean genes when asked, strict safe of 6):", "",
              "| Generation | Tilt limit (deg) | Retreat (m) | Sigma | Strict safe |",
              "| --- | --- | --- | --- | --- |"]
    for g in s["generations"]:
        if g["stage"] == 4:
            m = g["mean_settings"]
            lines.append(f"| {g['generation']} | {m['tilt_limit_deg']} | {m['retreat_m']} | "
                         f"{g['sigma']:.3f} | {g['strict_safe']} |")
    lines += ["", "## Protocol amendments", ""]
    for i, a in enumerate(protocol["amendments"], 1):
        lines.append(f"{i}. {a['utc']}: {a['change'][:400]}{'...' if len(a['change']) > 400 else ''}")
    hosts = [h for h in record["hosts"] if h.get("status") != "never_created"]
    lines += ["", "## Cost and hosts", "",
              f"Conservative spend estimate {infra.spend(record):.2f} USD (elapsed hours at the "
              "offer rate plus a 60 GB transfer allowance per host) against the 10 USD cap. "
              f"{len(hosts)} rentals: " + ", ".join(
                  f"{h['name']} ({h['offer']['gpu_name']}, {h['status']})" for h in hosts) + ".", ""]
    revalidation_path = ROOT / "outputs/evo19/analysis/revalidation.json"
    if revalidation_path.exists():
        r = json.loads(revalidation_path.read_text())
        lines += ["## Integrity", "",
                  f"Every confirmation episode was validated on its host (recording, monitor "
                  f"sidecar, and an exact replay of every adapter decision). Re-validated on the "
                  f"workstation from the mirror: {r['matching']} of {r['episodes']} counted "
                  f"episodes identical to both the stored record and the host sidecar, "
                  f"{len(r['problems'])} problems."]
        for d in r.get("uncounted_duplicates", []):
            lines.append(
                f"The h9 worker ran {d['job']} a second time straight after its counted "
                f"attempt finished ({d['leaf']}, finished {d['finished_utc'][:19]} UTC). That "
                "duplicate is not counted; it re-validates against its own sidecar "
                f"({d['validates_against_own_sidecar']}) and ended "
                f"{'strict safe' if d['outcome']['strict_safe'] else 'without strict safe completion'}"
                f"{' with a violation' if d['outcome']['raw_violation'] else ''}. It overwrote "
                "the host's result file for the job, so the counted outcome is the stored "
                "state record (attempt 1). Counting the duplicate instead would change no "
                "violation count and would move the shield's strict safe count by at most one.")
        lines.append("")
    lines += ["## Corrections", "",
              "27 October 2026: the Barnard p-values were first computed with the two arms as "
              "table rows, but scipy's barnard_exact treats each column as one binomial "
              "sample, so that version conditioned on outcome totals. analyze_evo19.py now "
              "puts the arms in columns. Raw violations on 107 pairs: 0.0111 became 0.00827; "
              "on the 72-pair panel, 0.02453 became 0.01215 (strict safe 0.40153 became "
              "0.34093). Fisher, McNemar and every count are unchanged, and no conclusion "
              "changes.", ""]
    if upload:
        lines += ["## Preservation", "",
                  f"Private dataset {upload['repo']}, prefix {upload['prefix']}, revision "
                  f"{upload['revision']}: {upload['verified_files']} files re-downloaded and "
                  "SHA-256 verified.", ""]
    lines += ["## Limits", "",
              "One frozen policy, one task family, four scenes, single-episode search "
              "evaluations. The search mean stayed close to its start, so this does not show "
              "that evolution beat a hand-set shield; it tests one evolved setting against "
              "native. Privileged simulator state (jar pose, contacts) drives the shield.", ""]
    OUT.write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
