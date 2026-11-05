#!/usr/bin/env python3
"""Build AP dataset navigation from explicit audited JSON; never publish or simulate.

Usage:
  python3 ap_catalog.py --data catalog-data.json --media-index media-index.json

``--data`` is the output of ap_reproduction/audit.py with an added presentation
object. Results, Wilson intervals, paired differences, bootstrap intervals and
measured scientific totals come only from that audit. Required presentation keys:

  report_id: a new folder name, such as ap-reproduction-20261105
  date: ISO date of the report
  setup:
    common: [{"setting": "Planner", "value": "..."}, ...]
    environments: {"tworoom": [setting/value rows], "reacher": [...],
                   "pusht": [...], "cube": [...]}
  evidence: [{"label": "Frozen protocol", "path": "reports/.../protocol.md",
              "revision": "40-character HF commit", "sha256": "optional hash",
              "description": "..."}, ...]
  runs: [{"run_id": "...", "kind": "evaluation", "environment": "tworoom",
          "revision": "40-character HF commit", "status": "verified",
          "description": "..."}, ...]

Optional presentation keys: overview_pdf (report-relative PDF path),
accounting_report (report-relative JSON path),
testing_implications (list of strings), limitations
(list of strings), and accounting (setting/value rows for measured non-scientific
work). Do not put monetary or rental details in study accounting.

The separate media index contains selection_verified=true, selection (description
of first AP-CEM success/failure in original query order), and examples. Each
example has environment, outcome, case (native compact record), source_run,
source_revision, source_trace, video (frame_steps and frames), and files. Files
use path, source_path, bytes and sha256. The MP4 path is the intended HF path;
the builder does not copy video bytes. Optional note is shown beside the player.
Represent an absent outcome with case=null and reason; this is accepted only if
the audited AP-CEM counts confirm that outcome is absent. All four environments
need success/failure entries. The index's first-case selection must be verified
from the complete original query order before it is supplied to this builder.

Default output is the adjacent content-ap/. Failed/incomplete audits are rejected
unless --allow-partial is explicit; such output is prominently marked incomplete.
All output is Markdown/JSON. This module imports only the Python standard library.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
from html import escape
import json
import math
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote


REPO = "Sachioster/LeWMExp1_AP_Reproduction"
BASE = f"https://huggingface.co/datasets/{REPO}"
TASKS = {"tworoom": "TwoRoom", "reacher": "Reacher", "pusht": "PushT", "cube": "OGBench Cube"}
ARMS = {"gaussian_final": "Final-goal CEM", "gaussian_observed": "AP-CEM"}
VIDEO_NOTE = (
    "These are original AP-CEM evaluation frames captured from the planner's existing "
    "observations, with the fixed goal alongside. Frames are normally five actions apart "
    "and played at 2 frames per second. The final frame can precede the stopping action. "
    "The success/failure label comes from the recorded evaluation outcome. "
    "No replay or reconstruction was used. Sign in with dataset access to play the clips."
)
COMPARISON_NOTE = (
    "AP-CEM and final-goal CEM start from the same recorded state, environment seed, "
    "optimizer seed and fixed goal. The 30 fps widescreen videos show AP ghosts in teal "
    "and CEM ghosts in orange, matching the trajectory lines, without action bars. "
    "Pixel-color masks tint the main view; individual previews keep their original colors. "
    "Motion interpolation smooths each "
    "original clip before blending; the in-between images are visual estimates, not new "
    "observations. Both planners follow the same clock at 20 recorded actions per playback "
    "second. Paths connect saved states. The last available image is held and labeled; "
    "it can precede the scored stopping state. Original unmodified clips remain linked "
    "below. No additional simulation or model evaluation was used."
)
DEFAULT_IMPLICATIONS = [
    "Use the paired AP-CEM and final-goal CEM results as the long-range recorded-goal "
    "reference for subsequent adapter tests; the earlier nearby-goal results use different cases.",
    "These tests do not establish evolutionary benefit or safety. Later comparisons need "
    "their own fixed cases, interaction budgets, completion measures and safety evaluation.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path_text(value):
    require(isinstance(value, str) and value, "Expected a repository-relative path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in value,
            f"Unsafe repository path: {value}")
    require(not any(ord(character) < 32 for character in value), "Control character in path")
    return value


def revision(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
            "Evidence and original runs require pinned 40-character revisions")
    return value


def url(path, *, raw=False, rev="main", tree=False):
    path_text(path)
    mode = "resolve" if raw else "tree" if tree else "blob"
    return f"{BASE}/{mode}/{quote(rev, safe='')}/{quote(path, safe='/')}"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def table(headers, rows):
    lines = ["| " + " | ".join(map(cell, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
    return "\n".join(lines)


def pct(value):
    return f"{value * 100:.1f}%"


def pp(value):
    return f"{value * 100:+.1f} pp"


def interval(value, *, difference=False):
    formatter = pp if difference else pct
    return f"{formatter(value[0])} to {formatter(value[1])}"


def validate_interval(value, label, *, difference=False):
    bound = -1 if difference else 0
    require(isinstance(value, list) and len(value) == 2
            and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
            and bound <= value[0] <= value[1] <= 1, f"Invalid interval: {label}")


def validate(data, media, allow_partial):
    complete = data.get("passed") is True and not data.get("errors") and not data.get("missing_cases")
    require(complete or allow_partial, "Audit is incomplete or failed; use --allow-partial explicitly")
    presentation = data["presentation"]
    require(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", presentation["report_id"]), "Invalid report_id")
    datetime.date.fromisoformat(presentation["date"])
    if presentation.get("overview_pdf"):
        require(path_text(presentation["overview_pdf"]).endswith(".pdf"),
                "Overview link must identify a report-relative PDF")
    if presentation.get("accounting_report"):
        require(path_text(presentation["accounting_report"]).endswith(".json"),
                "Accounting link must identify a report-relative JSON file")
    require(presentation["setup"]["common"], "Exact shared setup rows are required")
    require(presentation["evidence"] and presentation["runs"], "Evidence and run links are required")
    for rows in [presentation["setup"]["common"], *presentation["setup"]["environments"].values(),
                 presentation.get("accounting", [])]:
        for row in rows:
            require(set(row) >= {"setting", "value"}, "Setting rows need setting and value")
    for entry in presentation["evidence"]:
        path_text(entry["path"])
        revision(entry["revision"])
        require(entry.get("label") and entry.get("description"), "Evidence needs a label and description")
        if entry.get("sha256"):
            require(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), "Invalid evidence hash")
    run_ids = set()
    run_revisions = {}
    for run in presentation["runs"]:
        require(re.fullmatch(r"[A-Za-z0-9_-]+", run["run_id"]), "Invalid run identifier")
        require(run["run_id"] not in run_ids, "Duplicate run identifier")
        run_ids.add(run["run_id"])
        revision(run["revision"])
        run_revisions[run["run_id"]] = run["revision"]
        require(all(run.get(key) for key in ("kind", "status", "description")), "Incomplete run metadata")
        require(run.get("environment") in (None, "all", *TASKS), "Unknown run environment")

    summary = data["summary"]
    require(set(summary) <= set(TASKS), "Unknown result environment")
    if complete:
        require(set(summary) == set(TASKS), "A completed report requires all four environments")
        require(data["scientific_totals"]["evaluated_episodes"] == 1024,
                "A completed report requires all 1,024 arm/case combinations")
        require(data["paired_bootstrap"]["resamples"] == 10000,
                "Expected the predeclared 10,000 paired bootstrap samples")
        require(data["paired_bootstrap"]["seed"] == 20260913, "Bootstrap seed differs")
    for task in TASKS:
        require(presentation["setup"]["environments"].get(task), f"Exact {task} setup is absent")
        result = summary.get(task, {})
        for arm in ARMS:
            row = result.get(arm)
            if row is None:
                require(not complete, f"Missing completed result: {task}/{arm}")
                continue
            n, success = row["episodes"], row["successes"]
            require(type(n) is int and type(success) is int and 0 <= success <= n <= 128 and n > 0,
                    f"Invalid count: {task}/{arm}")
            require(not complete or n == 128, f"Incomplete panel: {task}/{arm}")
            require(math.isclose(row["success_rate"], success / n), "Rate differs from counts")
            require(row["unfinished"] == n - success, "Unfinished count differs")
            validate_interval(row["wilson_95"], f"{task}/{arm}")
        if "paired_difference" in result:
            require(all(arm in result for arm in ARMS), "Paired difference lacks one arm")
            require(result["gaussian_final"]["episodes"] == result["gaussian_observed"]["episodes"],
                    "Paired panels differ in size")
            difference = result["gaussian_observed"]["success_rate"] - result["gaussian_final"]["success_rate"]
            require(math.isclose(result["paired_difference"], difference, abs_tol=1e-12),
                    "Paired difference differs from outcome rates")
    intervals = data.get("paired_bootstrap", {}).get("intervals_95", {})
    for key, value in intervals.items():
        require(key in (*TASKS, "macro"), "Unknown paired interval")
        validate_interval(value, key, difference=True)
    if complete:
        require(set(intervals) == {*TASKS, "macro"}, "Missing paired confidence intervals")
        require(math.isclose(data["macro_paired_difference"],
                             sum(summary[task]["paired_difference"] for task in TASKS) / 4,
                             abs_tol=1e-12), "Macro difference differs from equal task weighting")

    require(media.get("selection_verified") is True, "Media first-case selection is not verified")
    require(media.get("selection"), "Media selection description is missing")
    examples = {}
    for example in media["examples"]:
        task, outcome = example["environment"], example["outcome"]
        require(task in TASKS and outcome in ("success", "failure"), "Invalid media identity")
        require((task, outcome) not in examples, "Duplicate media outcome")
        examples[task, outcome] = example
        row = summary.get(task, {}).get("gaussian_observed")
        case = example.get("case")
        if case is None:
            require(example.get("reason"), "Unavailable media needs a reason")
            absent = row is not None and row["successes" if outcome == "success" else "unfinished"] == 0
            require(absent or not complete, "An observed outcome requires its original selected clip")
            continue
        require(case["arm"] == "gaussian_observed" and case["condition"] == "clean",
                "Gallery clips must show clean AP-CEM cases")
        require(case["success"] is (outcome == "success") and case["completed"] is True,
                "Clip outcome differs from its case record")
        require(type(case["query_ordinal"]) is int and 0 <= case["query_ordinal"] < 128,
                "Invalid original query ordinal")
        require(0 <= case["primitive_steps"] <= case["budget"], "Invalid clip action count")
        require(case["task"] == task and case["identity"]["record_id"], "Clip task/identity differs")
        require(example["source_run"] in run_ids, "Clip source is absent from the run index")
        revision(example["source_revision"])
        require(example["source_revision"] == run_revisions[example["source_run"]],
                "Clip revision differs from its indexed original run")
        path_text(example["source_trace"])
        require(example["source_trace"].startswith("runs/" + example["source_run"] + "/"),
                "Clip trace is outside its indexed original run")
        frames = example["video"]["frame_steps"]
        require(frames and frames[0] == 0 and example["video"]["frames"] == len(frames),
                "Invalid native video frame index")
        require(all(type(step) is int and 0 <= step <= case["primitive_steps"] for step in frames)
                and all(b - a == 5 for a, b in zip(frames, frames[1:])),
                "Video does not follow the original five-action planner observation spacing")
        mp4s = []
        for file in example["files"]:
            path_text(file["path"])
            path_text(file["source_path"])
            require(file["source_path"].startswith("runs/" + example["source_run"] + "/"),
                    "Clip source file is outside its indexed original run")
            require(re.fullmatch(r"[0-9a-f]{64}", file["sha256"]), "Invalid media hash")
            require(type(file["bytes"]) is int and file["bytes"] > 0, "Invalid media size")
            if file["path"].endswith(".mp4"):
                mp4s.append(file)
        require(len(mp4s) == 1, "Each selected case requires one MP4")
        plot = example.get("trajectory_plot")
        if plot:
            path_text(plot["path"])
            require(plot["path"].endswith(".png") and type(plot["bytes"]) is int and plot["bytes"] > 0,
                    "Invalid derived trajectory image")
            require(re.fullmatch(r"[0-9a-f]{64}", plot["sha256"]), "Invalid trajectory image hash")
            require(plot.get("coordinate_note") and plot.get("representation"),
                    "Derived trajectory image needs an explicit coordinate description")
            require(plot["source_trajectory"] == example["source_trajectory"]
                    and plot["source_trajectory_sha256"] == example["source_trajectory_sha256"],
                    "Derived image source differs from the selected saved trajectory")
    require(set(examples) == {(task, outcome) for task in TASKS for outcome in ("success", "failure")},
            "Media index must account for success and failure in all four environments")
    return complete, examples


class Catalog:
    def __init__(self, data, media, allow_partial=False, comparisons=None):
        self.complete, self.examples = validate(data, media, allow_partial)
        self.data, self.media = data, media
        self.presentation = data["presentation"]
        self.report = "reports/" + self.presentation["report_id"]
        self.media_root = "media/" + self.presentation["report_id"]
        self.comparisons = {}
        if comparisons is not None:
            require(comparisons.get("repo_id") == REPO
                    and comparisons.get("derived_comparisons_verified") is True,
                    "Comparison videos require verified saved-evidence derivation")
            for entry in comparisons["comparisons"]:
                key = entry["environment"], entry["outcome"]
                require(key in self.examples and key not in self.comparisons,
                        "Invalid or duplicate comparison selection")
                original = self.examples[key].get("case")
                require(original is not None
                        and entry["query_ordinal"] == original["query_ordinal"],
                        "Comparison differs from the original first-case selection")
                require(entry.get("comparison_checks") == {
                    "same_seed": True, "same_initial_state": True, "same_goal": True},
                    "Comparison lacks matched seed/state/goal checks")
                require(entry.get("simulator_actions") == 0 and entry.get("model_queries") == 0,
                        "Comparison must use existing recordings only")
                output = entry["output"]
                require(path_text(output["path"]).endswith(".mp4")
                        and type(output["bytes"]) is int and output["bytes"] > 0
                        and re.fullmatch(r"[0-9a-f]{64}", output["sha256"]),
                        "Comparison output needs its exact size and hash")
                if entry.get("poster"):
                    poster = entry["poster"]
                    require(path_text(poster["path"]).endswith(".png")
                            and type(poster["bytes"]) is int and poster["bytes"] > 0
                            and re.fullmatch(r"[0-9a-f]{64}", poster["sha256"]),
                            "Comparison poster needs its exact size and hash")
                self.comparisons[key] = entry
            require(not self.complete or set(self.comparisons) == {
                key for key, example in self.examples.items() if example.get("case") is not None},
                "Completed gallery needs every selected paired comparison")

    def link(self, label, suffix="README.md"):
        return f"[{label}]({url(self.report + '/' + suffix)})"

    def navigation(self):
        links = [f"[Dataset home]({BASE})", self.link("Results")]
        if self.presentation.get("overview_pdf"):
            links.append(self.link("Two-page overview", self.presentation["overview_pdf"]))
        links += [self.link("Exact setup", "setups.md"), self.link("Evidence", "evidence.md"),
                  self.link("Original runs", "runs.md")]
        return " · ".join(links)

    def result_table(self, tasks=None, *, show_intervals=False):
        rows = []
        for task in tasks or TASKS:
            result = self.data["summary"].get(task, {})
            row = [self.link(TASKS[task], f"environments/{task}.md")]
            for arm in ("gaussian_observed", "gaussian_final"):
                panel = result.get(arm)
                value = "Not available" if panel is None else (
                    f"{panel['successes']}/{panel['episodes']} ({pct(panel['success_rate'])})")
                if panel is not None and show_intervals:
                    value += f"; 95% CI {interval(panel['wilson_95'])}"
                row.append(value)
            paired = result.get("paired_difference")
            bounds = self.data.get("paired_bootstrap", {}).get("intervals_95", {}).get(task)
            value = "Not available" if paired is None else pp(paired)
            if paired is not None and show_intervals:
                value += (f"; 95% CI {interval(bounds, difference=True)}"
                          if bounds else "; CI unavailable")
            row.append(value)
            rows.append(row)
        return table(["Environment", "AP-CEM", "Final-goal CEM", "Paired AP-CEM gain"], rows)

    def headline(self):
        if not self.complete:
            return "**Incomplete reproduction. The audit has not passed; missing or failed work is disclosed below.**"
        totals = {arm: sum(self.data["summary"][task][arm]["successes"] for task in TASKS) for arm in ARMS}
        return (f"**AP-CEM completed {totals['gaussian_observed']}/512 cases; "
                f"matched final-goal CEM completed {totals['gaussian_final']}/512.** "
                "Both planners used the same 128 original queries in each of four environments.")

    def statistics_note(self):
        text = ("Arm intervals are Wilson 95% intervals. The difference uses paired query bootstrap "
                "intervals with 10,000 resamples and seed 20260913. Positive values favor AP-CEM. "
                "These are fixed original queries, not independent query or training-seed replications.")
        if "macro_paired_difference" in self.data:
            bounds = self.data.get("paired_bootstrap", {}).get("intervals_95", {}).get("macro")
            text += f" Equal-weight task-average difference: **{pp(self.data['macro_paired_difference'])}**"
            text += f" (95% CI {interval(bounds, difference=True)})." if bounds else "."
        return text

    def practical_note(self):
        panels = [self.data["summary"].get(task, {}).get("gaussian_observed", {}) for task in TASKS]
        if not self.complete or not all("meets_practical_tolerance" in panel for panel in panels):
            return ""
        passed = sum(panel["meets_practical_tolerance"] is True for panel in panels)
        return (f"AP-CEM meets the predeclared practical tolerance in **{passed}/4 environments**: "
                "a success rate no more than 10 percentage points below its published reference. "
                "This descriptive check is separate from passing the evidence audit; it does not "
                "establish statistical equivalence.")

    def gallery(self, task, *, include_plots=False, heading_level=3):
        heading = "#" * heading_level
        blocks = []
        for outcome in ("success", "failure"):
            example = self.examples[task, outcome]
            title = f"{TASKS[task]}: first AP-CEM {outcome}"
            if example.get("case") is None:
                blocks.append(f"{heading} {outcome.title()}\n\n{example['reason']}")
                continue
            case = example["case"]
            file = next(file for file in example["files"] if file["path"].endswith(".mp4"))
            video_url = url(file["path"], raw=True)
            trace_url = url(example["source_trace"], rev=example["source_revision"])
            frame_steps = example["video"]["frame_steps"]
            player = (f'<video controls playsinline preload="none" width="100%" '
                      f'src="{escape(video_url, quote=True)}" title="{escape(title, quote=True)}">'
                      f'<a href="{escape(video_url, quote=True)}">Watch {escape(title)}</a></video>')
            detail = (f"Original query **{case['query_ordinal']}** (zero-based), "
                      f"record `{case['identity']['record_id']}`. "
                      f"Recorded stop after **{case['primitive_steps']} actions**; "
                      f"last displayed observation after **{frame_steps[-1]} actions**.")
            block = (f"{heading} {outcome.title()}\n\n{player}\n\n{detail}\n\n"
                     f"[Open video]({video_url}) · [Original case record]({trace_url})")
            plot = example.get("trajectory_plot")
            if plot:
                plot_url = url(plot["path"], raw=True)
                block += f" · [Trajectory plot]({plot_url})"
                if include_plots:
                    block += (f"\n\n![{TASKS[task]} {outcome}: saved physical trajectory]({plot_url})"
                              f"\n\n{plot['coordinate_note']} This plot is derived from all saved physical "
                              "states, including the stop; the original video shows planner observations.")
            if example.get("note"):
                block += f"\n\n{example['note']}"
            blocks.append(block)
        return "\n\n".join(blocks)

    def comparison_gallery(self, task, *, heading_level=3):
        heading = "#" * heading_level
        blocks = []
        for outcome in ("success", "failure"):
            entry = self.comparisons.get((task, outcome))
            if entry is None:
                continue
            case = self.examples[task, outcome]["case"]
            title = f"{TASKS[task]}: AP-CEM versus final-goal CEM, query {case['query_ordinal']}"
            video_url = (url(entry["output"]["path"], raw=True)
                         + "?v=" + entry["output"]["sha256"][:16])
            poster = ""
            if entry.get("poster"):
                poster_url = (url(entry["poster"]["path"], raw=True)
                              + "?v=" + entry["poster"]["sha256"][:16])
                poster = f' poster="{escape(poster_url, quote=True)}"'
            player = (f'<video controls playsinline preload="none" width="100%" '
                      f'src="{escape(video_url, quote=True)}"{poster} '
                      f'title="{escape(title, quote=True)}">'
                      f'<a href="{escape(video_url, quote=True)}">Watch {escape(title)}</a></video>')
            original_links = []
            for arm, label in (("gaussian_observed", "Original AP"),
                               ("gaussian_final", "Original CEM")):
                source = next(item for item in entry["sources"]
                              if item["source_path"].endswith("_" + arm + ".mp4"))
                original_links.append(
                    f"[{label}]({url(source['source_path'], raw=True, rev=source['source_revision'])})")
            blocks.append(
                f"{heading} First AP-CEM {outcome}: matched comparison\n\n{player}\n\n"
                f"Original query **{case['query_ordinal']}**. Shared environment seed "
                f"**{case['identity']['environment_seed']}**, optimizer seed **{case['optimizer_seed']}**; "
                "the starting state and goal match exactly.\n\n"
                f"[Open comparison]({video_url}) · " + " · ".join(original_links) + " · "
                f"[Pairing and derivation evidence]({url(self.media_root + '/comparisons.json')})")
        return "\n\n".join(blocks)

    def setup_rows(self, task=None):
        setup = self.presentation["setup"]
        rows = setup["common"] if task is None else setup["environments"][task]
        return table(["Setting", "Value"], [(row["setting"], row["value"]) for row in rows])

    def interpretation(self):
        limits = [
            "This tests long-range recorded-goal completion with a supplied temporal offset. "
            "It is not unrestricted canonical benchmark completion.",
            "The nearby-goal baseline uses different cases and is not the paired control here.",
            "No safety monitor or evolutionary optimizer was evaluated in this reproduction.",
        ] + self.presentation.get("limitations", [])
        return "\n\n".join(limits)

    def implications(self):
        return "\n".join("- " + item for item in self.presentation.get(
            "testing_implications", DEFAULT_IMPLICATIONS))

    def evidence_page(self):
        rows = []
        for entry in self.presentation["evidence"]:
            label = f"[{entry['label']}]({url(entry['path'], rev=entry['revision'])})"
            rows.append([label, entry["description"], f"`{entry['revision']}`"])
        text = f"# Evidence guide\n\n{self.navigation()}\n\n"
        text += ("Results come from saved native traces and independent physical-state checks. "
                 "Original run links are pinned to their verified Hugging Face revisions.\n\n")
        text += table(["Evidence", "What it contains", "Pinned revision"], rows)
        text += (f"\n\n[Audit used for this page]({url(self.report + '/audit.json')}) · "
                 f"[Catalog input]({url(self.report + '/catalog-data.json')}) · "
                 f"[Media provenance and hashes]({url(self.media_root + '/index.json')})\n\n")
        if self.presentation.get("accounting_report"):
            text += self.link("Full work accounting", self.presentation["accounting_report"]) + "\n\n"
        totals = self.data.get("scientific_totals", {})
        text += "## Recorded scientific work\n\n" + table(
            ["Measure", "Recorded value"], [(key.replace("_", " "), value) for key, value in totals.items()])
        text += ("\n\nOnly values present in the audit are shown. Initially successful cases are included "
                 "in the success denominators. Unreported engagement measures are not inferred.\n")
        if self.presentation.get("accounting"):
            text += "\n## Preparation and engineering\n\n" + table(
                ["Measure", "Recorded value"], [(row["setting"], row["value"])
                                               for row in self.presentation["accounting"]]) + "\n"
        if not self.complete:
            text += "\n## Incomplete audit\n\n" + "\n".join(
                "- " + str(error) for error in self.data.get("errors", []))
            text += f"\n\nMissing case records: {len(self.data.get('missing_cases', []))}.\n"
        return text

    def run_page(self):
        rows = []
        for run in self.presentation["runs"]:
            target = url("runs/" + run["run_id"], rev=run["revision"], tree=True)
            rows.append([f"[{run['run_id']}]({target})", run["kind"],
                         TASKS.get(run.get("environment"), run.get("environment") or "Shared"),
                         run["status"], run["description"]])
        return (f"# Original run index\n\n{self.navigation()}\n\n"
                "These links preserve original outputs, including preparation, engineering and failed "
                "attempts when present in the supplied run index. Gallery pages do not replace them.\n\n"
                + table(["Run", "Purpose", "Environment", "Status", "Contents"], rows) + "\n")

    def pages(self):
        nav = self.navigation()
        result = self.result_table()
        common = (f"{self.headline()}\n\n{result}\n\n{self.statistics_note()}\n\n"
                  "Both use frozen LeWM checkpoints and the same CEM optimizer. AP-CEM scores a "
                  "retrieved short-range observation; the matched control scores the final goal.")
        if self.practical_note():
            common += "\n\n" + self.practical_note()
        report = (f"# AP checkpoint reproduction\n\nReport date: {self.presentation['date']}.\n\n"
                  f"{common}\n\n{nav}\n\n## What this means\n\n{self.interpretation()}\n\n"
                  f"## What this means for our testing\n\n{self.implications()}\n")
        video_title = "Matched ghost comparisons" if self.comparisons else "Videos by environment"
        video_note = COMPARISON_NOTE if self.comparisons else VIDEO_NOTE
        card = ("---\npretty_name: LeWMExp1 AP Reproduction\nlanguage:\n  - en\ntags:\n"
                "  - robotics\n  - world-models\n  - planning\n  - video\n  - reproducibility\n"
                "viewer: false\n---\n# LeWM Experiment 1: AP reproduction\n\n"
                f"{common}\n\n{nav}\n\n## {video_title}\n\n{video_note}\n\n")
        for task, name in TASKS.items():
            card += f"### {name}\n\n{self.link('Results, setup and original clips', f'environments/{task}.md')}\n\n"
            gallery = self.comparison_gallery if self.comparisons else self.gallery
            card += gallery(task, heading_level=4) + "\n\n"
        card += f"## What this means for our testing\n\n{self.implications()}\n\n{self.interpretation()}\n"
        pages = {
            "README.md": card,
            self.report + "/README.md": report,
            self.report + "/evidence.md": self.evidence_page(),
            self.report + "/runs.md": self.run_page(),
            "reports/README.md": "# Experiment reports\n\n" + table(
                ["Study", "Scope", "Audit"], [[self.link("AP checkpoint reproduction"),
                 "Four environments; paired AP-CEM and final-goal CEM on original recorded goals",
                 "Passed" if self.complete else "Incomplete"]]) + "\n",
            "media/README.md": (f"# Evaluation videos\n\n{video_note}\n\n"
                                f"[Browse by environment]({BASE}#{video_title.lower().replace(' ', '-')}) · "
                                f"[Media index]({url(self.media_root + '/index.json')})\n"),
        }
        setup = f"# Exact study setup\n\n{nav}\n\n## Shared settings\n\n{self.setup_rows()}\n"
        for task, name in TASKS.items():
            setup += f"\n## {name}\n\n{self.setup_rows(task)}\n"
            panel = self.data["summary"].get(task, {})
            unfinished = "; ".join(f"{label}: {panel[arm]['unfinished']}/{panel[arm]['episodes']} unfinished"
                                   for arm, label in ARMS.items() if arm in panel)
            external = []
            for arm, label in ARMS.items():
                row = panel.get(arm, {})
                if "paper_successes" in row and "paper_episodes" in row:
                    external.append(f"{label}: {row['paper_successes']}/{row['paper_episodes']}")
            page = (f"# {name}: AP reproduction\n\n{nav}\n\n"
                    f"{self.result_table([task], show_intervals=True)}\n\n"
                    f"{unfinished}.\n\n{self.statistics_note()}\n\n")
            detail = []
            for arm, label in ARMS.items():
                row = panel.get(arm, {})
                items = []
                for key, description in (("initial_successes", "initially successful"),
                                         ("engaged", "executed at least one action")):
                    if key in row:
                        items.append(f"{row[key]}/{row['episodes']} {description}")
                if "policy_actions" in row:
                    items.append(f"{row['policy_actions']:,} recorded policy actions")
                if items:
                    detail.append(label + ": " + "; ".join(items) + ".")
            if detail:
                page += "\n\n".join(detail) + "\n\n"
            if external:
                page += ("Published external reference: " + "; ".join(external)
                         + ". Agreement with these published scores is not statistical equivalence.\n\n")
            ap_panel = panel.get("gaussian_observed", {})
            if "meets_practical_tolerance" in ap_panel:
                status = "Meets" if ap_panel["meets_practical_tolerance"] else "Does not meet"
                page += (f"**{status} the AP-CEM practical tolerance** of no more than 10 percentage "
                         "points below its published reference. This is a descriptive score check, "
                         "not the evidence-audit result.\n\n")
            if self.comparisons:
                page += (f"## Matched ghost comparisons\n\n{COMPARISON_NOTE}\n\n"
                         f"{self.comparison_gallery(task)}\n\n")
            page += (f"## Original AP-CEM clips\n\n{VIDEO_NOTE}\n\n{self.media['selection']}\n\n"
                     f"{self.gallery(task, include_plots=True)}\n\n## Environment setup\n\n{self.setup_rows(task)}\n\n"
                     f"{self.link('Shared planner and evaluation settings', 'setups.md')} · "
                     f"{self.link('Original protocol and evidence', 'evidence.md')}\n")
            pages[self.report + f"/environments/{task}.md"] = page
        pages[self.report + "/setups.md"] = setup
        return pages


def build(data_path, media_path, output, *, allow_partial=False, comparisons_path=None):
    data_bytes, media_bytes = data_path.read_bytes(), media_path.read_bytes()
    data, media = json.loads(data_bytes), json.loads(media_bytes)
    comparison_bytes = comparisons_path.read_bytes() if comparisons_path else None
    comparisons = json.loads(comparison_bytes) if comparison_bytes is not None else None
    catalog = Catalog(data, media, allow_partial, comparisons)
    payloads = {path: text.encode() for path, text in catalog.pages().items()}
    audit = {key: value for key, value in data.items() if key != "presentation"}
    payloads[catalog.report + "/audit.json"] = (json.dumps(audit, indent=2, allow_nan=False) + "\n").encode()
    payloads[catalog.report + "/catalog-data.json"] = data_bytes
    payloads[catalog.media_root + "/index.json"] = media_bytes
    if comparison_bytes is not None:
        payloads[catalog.media_root + "/comparisons.json"] = comparison_bytes
    manifest = {
        "repo_id": REPO, "report_id": catalog.presentation["report_id"],
        "audit_passed": catalog.complete, "published": False,
        "inputs": {"data_sha256": hashlib.sha256(data_bytes).hexdigest(),
                   "media_index_sha256": hashlib.sha256(media_bytes).hexdigest()},
        "files": {path: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                  for path, content in sorted(payloads.items())},
    }
    if comparison_bytes is not None:
        manifest["inputs"]["comparisons_index_sha256"] = hashlib.sha256(comparison_bytes).hexdigest()
    payloads["catalog-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output = output.resolve()
    for name, content in payloads.items():
        destination = output / name
        require(destination.resolve().is_relative_to(output), "Output path escapes destination")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--media-index", type=Path, required=True)
    parser.add_argument("--comparisons-index", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "content-ap")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    try:
        manifest = build(args.data, args.media_index, args.output, allow_partial=args.allow_partial,
                         comparisons_path=args.comparisons_index)
    except (KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"Invalid catalog input: {error}\n")
    print(json.dumps({"output": str(args.output), "files": len(manifest["files"]) + 1,
                      "audit_passed": manifest["audit_passed"], "published": False}))


if __name__ == "__main__":
    main()
