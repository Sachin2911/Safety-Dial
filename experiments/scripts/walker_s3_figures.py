#!/usr/bin/env python3
"""Present finalized Walker S3 evidence in a fresh sibling directory.

Reads JSON and hashes completed outputs only. No model, simulator, training, network,
new qualification decision or mutation of the original run/results is involved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))
from helpers.managedWorkflow import file_hash, output_hash

VARIABLES = ("height", "pitch", "speed")
UNITS = {"height": "m", "pitch": "rad", "speed": "m/s"}
R2_THRESHOLDS = {"height": 0.9, "pitch": 0.9, "speed": 0.8}
FINAL_ERROR_CEILINGS = {"height": 0.3, "pitch": 0.6, "speed": 1.0}


def finite(value):
    return value is not None and math.isfinite(float(value))


def number(value):
    return f"{float(value):.6g}" if finite(value) else "undefined"


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def check(condition, message):
    if not condition:
        raise ValueError(message)


def flag(value):
    check(type(value) is bool, "Saved gate flags must be JSON booleans")
    return "pass" if value else "does not pass"


def metric_cells(metric):
    """Preserve saved undefined values; validate count/rate contradictions."""
    counts = [metric[k] for k in ("n", "n_accepted", "n_false_safe")]
    check(all(type(v) is int and v >= 0 for v in counts), "Invalid decision counts")
    n, accepted, false_safe = counts
    check(false_safe <= accepted <= n, "Inconsistent decision counts")
    if finite(metric.get("acceptance_rate")):
        check(n > 0 and math.isclose(metric["acceptance_rate"], accepted / n),
              "Acceptance rate disagrees with saved counts")
    if finite(metric.get("fsa")):
        check(accepted > 0 and not metric.get("n_accepted_censored", 0)
              and math.isclose(metric["fsa"], false_safe / accepted),
              "Defined FSA disagrees with counts or unresolved acceptance")
    return [str(n), str(accepted), str(false_safe), number(metric.get("n_accepted_censored")),
            number(metric.get("acceptance_rate")), number(metric.get("fsa"))]


def validate_report(report, rows):
    check(report.get("role") == "development", "S3 presentation requires development evidence")
    gate = report["gate"]
    for key in ("go", "health_rule_ok", "speed_rule_ok", "height_pass", "pitch_pass", "speed_pass"):
        flag(gate[key])
    check(gate["go"] == gate["health_rule_ok"], "Recorded go and health qualification disagree")
    expected_rules = (["health"] + (["speed"] if gate["speed_rule_ok"] else [])) if gate["go"] else []
    check(gate["active_rules"] == expected_rules, "Recorded active rules disagree with saved gate")
    check(gate["status"] == ("pass" if gate["go"] else "no_go"), "Recorded gate status disagrees")
    check(rows and len({(row["root"], row["tape"]) for row in rows}) == len(rows),
          "Empty or duplicated saved gate row identities")
    for source in ("real_readout_abs_error_by_block", "imagined_abs_error_by_block"):
        for variable in VARIABLES:
            values = report[source][variable]
            check(len(values) == 10, "Expected the recorded ten-block horizon")
            check(all(not finite(v) or v >= 0 for v in values), "Negative absolute error")
    for rule in ("health", "speed"):
        decision = report["decisions_m0"][rule]
        diagnostic = report["decision_diagnostics"][rule]
        check(decision["n"] == len(rows), "Decision count differs from saved rows")
        check(diagnostic["n_unsafe"] == decision["n_unsafe"], "Unsafe counts disagree")
        check(diagnostic["n_safe"] + diagnostic["n_unsafe"] == len(rows),
              "Readout support counts disagree")
        for source in ("real_readout", "imagined"):
            metric_cells(decision[source])
            check(decision[source]["n"] == len(rows), "Source decision counts differ")
    return gate


def completed_evidence(run_dir, results_dir, workflow_state, stage_id):
    """Require a completed managed subprocess before opening its scientific results."""
    state = json.loads(workflow_state.read_text())
    stage = state.get("stages", {}).get(stage_id, {})
    check(stage.get("status") in ("completed", "gate_stopped"), "S3 stage is not finalized")
    attempts = stage.get("attempts", [])
    check(attempts and attempts[-1].get("finished_unix") is not None
          and attempts[-1].get("returncode") in (0, 2), "S3 subprocess has not finalized")
    recorded = stage.get("output_hashes", {})
    for directory in (run_dir, results_dir):
        check(str(directory) in recorded, f"Workflow did not freeze {directory}")
        check(output_hash(directory) == recorded[str(directory)], f"Finalized output changed: {directory}")
    for filename in ("gate.json", "gate_rows.json"):
        check(file_hash(run_dir / filename) == file_hash(results_dir / filename),
              f"Run/results copies disagree: {filename}")
    report = json.loads((results_dir / "gate.json").read_text())
    rows = json.loads((results_dir / "gate_rows.json").read_text())
    gate = validate_report(report, rows)
    saved_gate = stage["gate"]
    check(saved_gate["key"] == "gate.go" and saved_gate["passes"] == gate["go"]
          and Path(saved_gate["path"]).resolve() == run_dir / "gate.json"
          and saved_gate["sha256"] == file_hash(run_dir / "gate.json"),
          "Workflow gate is not this saved S3 gate")
    check(stage["status"] == ("completed" if gate["go"] else "gate_stopped"),
          "Workflow status differs from recorded gate")
    return report, rows, stage


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def markdown(report, rows):
    gate = report["gate"]
    status = "Health qualification passed" if gate["go"] else "Stopped at the health qualification gate"
    lines = [f"# Walker S3: {status}", "", f"Run: `{report['run_id']}`. Role: development.", "",
             "The recorded health result controls continuation. Speed is an optional additional rule; "
             "a speed failure alone does not stop a passing health result. This supplement preserves "
             "the saved gate without making a new qualification decision.", "",
             f"Saved active rules: {', '.join(gate['active_rules']) or 'none'}. "
             f"Evidence: {len(rows)} paired branches from {len({r['root'] for r in rows})} roots.", "",
             "## Probe readout", "",
             "Probe metrics use a trajectory-disjoint validation partition of the separate probe collection. "
             "The gate uses the MLP probe. Linear results are a reported reference. R² thresholds "
             "and final imagined-error ceilings below come from the versioned `walkerValidation.py` "
             "protocol; its SHA is in the manifest. The gate report saves the resulting flags.", ""]
    lines += [table(["Variable", "Linear R²", "MLP R²", "MLP threshold", "Recorded result"],
                    [[v.title(), number(report['probes']['linear'][v + '_r2']),
                      number(report['probes']['mlp'][v + '_r2']), f"≥ {R2_THRESHOLDS[v]}",
                      flag(gate[v + '_pass'])] for v in VARIABLES]), "", "![Probe R²](probe_r2.png)", "",
              "## Qualification components", ""]
    components = []
    for rule, variables in (("health", ("height", "pitch")), ("speed", ("speed",))):
        limits = "; ".join(f"{v} < {FINAL_ERROR_CEILINGS[v]} {UNITS[v]}" for v in variables)
        growth = "; ".join(f"{v}: {flag(gate['imagined_error_grows'][v])}" for v in variables)
        components.append([rule.title(), limits, flag(gate['imagined_error_useful'][rule]), growth,
                           flag(gate['real_readout_tracks_truth'][rule]), flag(gate[rule + '_rule_ok'])])
    lines += [table(["Rule", "Final-error ceiling", "Useful error", "Last > first error",
                     "Readout tracks truth", "Recorded qualification"], components), "",
              "## Real-readout decisions", ""]
    t = gate["thresholds"]
    lines += [f"Recorded minimum support: {t['min_unsafe']} unsafe and {t['min_unsafe']} safe branches "
              f"per rule; unsafe recall ≥ {number(t['min_recall'])}; specificity ≥ {number(t['min_specificity'])}.", "",
              table(["Rule", "Unsafe", "Safe", "Unsafe detected", "Safe accepted", "Recall", "Specificity"],
                    [[r.title()] + [number(report['decision_diagnostics'][r][k]) for k in
                     ('n_unsafe', 'n_safe', 'unsafe_detected', 'safe_accepted', 'unsafe_recall', 'specificity')]
                     for r in ('health', 'speed')]), "",
              "At margin zero, acceptance is predicted clearance ≥ 0. Health truth at zero clearance "
              "is unsafe; speed truth at zero clearance is safe. FSA is unsafe accepted / all accepted. "
              "Undefined values, including zero-acceptance FSA, remain undefined.", "",
              table(["Rule", "Prediction", "Branches", "Accepted", "Unsafe accepted", "Accepted unresolved", "Acceptance rate", "FSA"],
                    [[r.title(), label] + metric_cells(report['decisions_m0'][r][source])
                     for r in ('health', 'speed')
                     for source, label in (('real_readout', 'Real-image readout'), ('imagined', 'Imagined'))]), "",
              "## Error by prediction horizon", "",
              "Each block contains 10 environment steps (0.08 s); block 10 reaches 0.8 s. Values are "
              "the saved mean absolute errors. Separate axes preserve the physical units. Gaps mean "
              "undefined values, not zero error. These means have no saved uncertainty intervals.", "",
              "![Errors by horizon](horizon_errors.png)", ""]
    lines += [table(["Block"] + [f"{v.title()} {s} ({UNITS[v]})" for v in VARIABLES for s in ('real', 'imagined')],
                    [[str(k + 1)] + [number(report[source][v][k]) for v in VARIABLES
                     for source in ('real_readout_abs_error_by_block', 'imagined_abs_error_by_block')]
                     for k in range(10)]), "", "## Evidence limits", "",
              "This is a presentation of saved development evidence, not a final-test result or a new "
              "gate evaluation. Probe predictions and dense branch trajectories were not saved in these "
              "JSON reports, so the figures do not independently verify those underlying measurements. "
              "No model, simulator or network operation was performed. Input and presentation-source "
              "hashes are recorded in `manifest.json`.", ""]
    return "\n".join(lines)


def figures(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4), constrained_layout=True)
        for ax, variable in zip(axes, VARIABLES, strict=True):
            vals = [report['probes'][kind][variable + '_r2'] for kind in ('linear', 'mlp')]
            for x, value, color in zip((0, 1), vals, ('#777777', '#0072B2'), strict=True):
                if finite(value):
                    ax.scatter([x], [value], color=color, s=65, zorder=3)
                    ax.annotate(number(value), (x, value), xytext=(0, 8), textcoords='offset points', ha='center')
                else:
                    ax.text(x, .08, 'undefined', transform=ax.get_xaxis_transform(), ha='center')
            ax.axhline(R2_THRESHOLDS[variable], color='#D55E00', linestyle='--', label='MLP threshold')
            limits = [0., 1.] + [float(v) for v in vals if finite(v)]
            span = max(limits) - min(limits)
            ax.set_ylim(min(limits) - .12 * span, max(limits) + .2 * span)
            ax.set_xlim(-.45, 1.45)
            ax.set_xticks([0, 1], ['Linear', 'MLP'])
            ax.set_title(variable.title())
            ax.set_ylabel('Validation R²')
            ax.grid(axis='y', alpha=.2)
        fig.legend(*axes[-1].get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(.5, -.06), fontsize=9)
        fig.suptitle('Trajectory-split probe validation: saved results')
        fig.savefig(destination / 'probe_r2.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), constrained_layout=True)
        for ax, variable in zip(axes, VARIABLES, strict=True):
            missing = []
            for source, label, color, marker in (
                ('real_readout_abs_error_by_block', 'Real-image readout', '#0072B2', 'o'),
                ('imagined_abs_error_by_block', 'Imagined', '#D55E00', 's'),
            ):
                values = report[source][variable]
                y = [float(v) if finite(v) else np.nan for v in values]
                ax.plot(range(1, 11), y, marker=marker, markersize=4, label=label, color=color)
                n_missing = sum(not finite(v) for v in values)
                if n_missing:
                    missing.append(f'{label}: {n_missing} undefined')
            ax.set_title(variable.title())
            ax.set_xlabel('Prediction block (0.08 s each)')
            ax.set_ylabel(f'Mean absolute error ({UNITS[variable]})')
            ax.set_xticks([1, 2, 4, 6, 8, 10])
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
            if missing:
                ax.set_title(variable.title() + '\n' + '\n'.join(missing))
        fig.legend(*axes[-1].get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(.5, -.06), ncol=2, fontsize=9)
        fig.suptitle('Saved errors by horizon; units and scales differ across variables')
        fig.savefig(destination / 'horizon_errors.png', dpi=180, bbox_inches='tight')
        plt.close(fig)


def create_presentation(run_dir, results_dir, output_dir, workflow_state, *, stage_id='physical_probe_gate'):
    run_dir, results_dir, output_dir, workflow_state = map(
        lambda p: Path(p).resolve(), (run_dir, results_dir, output_dir, workflow_state))
    check(not output_dir.exists(), 'Refusing to overwrite existing presentation output')
    check(output_dir.parent == results_dir.parent, 'Presentation must be a sibling of the results directory')
    check(all(not output_dir.is_relative_to(source) and not source.is_relative_to(output_dir)
              for source in (run_dir, results_dir)), 'Output must not overlap either immutable source tree')
    report, rows, stage = completed_evidence(run_dir, results_dir, workflow_state, stage_id)
    input_hashes = {str(folder / name): file_hash(folder / name)
                    for folder in (run_dir, results_dir)
                    for name in ('gate.json', 'gate_rows.json', 'manifest.json')}
    sources = [Path(__file__).resolve(), REPO / 'experiments/helpers/managedWorkflow.py',
               REPO / 'experiments/helpers/walkerValidation.py', REPO / 'experiments/scripts/walker_s3_gate.py']
    source_hashes = {str(path.relative_to(REPO)): file_hash(path) for path in sources}
    output_dir.mkdir(parents=False, exist_ok=False)
    (output_dir / 'README.md').write_text(markdown(report, rows))
    figures(report, output_dir)
    presentation = {'run_id': report['run_id'], 'role': 'development', 'recorded_gate': report['gate'],
                    'probe_metrics': report['probes'], 'decisions_m0': report['decisions_m0'],
                    'decision_diagnostics': report['decision_diagnostics'],
                    'undefined_policy': 'Nonfinite/missing numeric values are null; never replaced by zero.',
                    'new_gate_decision': False}
    (output_dir / 'presentation.json').write_text(json.dumps(json_safe(presentation), indent=2, allow_nan=False) + '\n')
    _, _, after = completed_evidence(run_dir, results_dir, workflow_state, stage_id)
    check(stage == after, 'Finalized S3 stage changed during presentation')
    check(all(file_hash(Path(path)) == expected for path, expected in input_hashes.items()),
          'Input report changed during presentation')
    check(all(file_hash(REPO / name) == expected for name, expected in source_hashes.items()),
          'Presentation source changed while generating figures')
    manifest = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'run_id': report['run_id'], 'inputs_sha256': input_hashes, 'source_sha256': source_hashes,
                'workflow_state': str(workflow_state), 'workflow_stage': stage_id,
                'completed_source_output_hashes': stage['output_hashes'],
                'recorded_gate_go': report['gate']['go'], 'input_artifacts_unchanged': True,
                'gate_recomputed': False, 'model_calls': 0, 'simulator_steps': 0, 'network_calls': 0,
                'threshold_origin': 'R2/useful-error constants: versioned walkerValidation.py; diagnostic thresholds: saved gate.json.',
                'files_sha256': {p.name: file_hash(p) for p in sorted(output_dir.iterdir()) if p.is_file()}}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'results-dir', 'output-dir', 'workflow-state'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage-id', default='physical_probe_gate')
    args = parser.parse_args()
    result = create_presentation(args.run_dir, args.results_dir, args.output_dir,
                                 args.workflow_state, stage_id=args.stage_id)
    print(json.dumps({'status': result['status'], 'output_dir': str(args.output_dir),
                      'recorded_gate_go': result['recorded_gate_go']}))


if __name__ == '__main__':
    main()
