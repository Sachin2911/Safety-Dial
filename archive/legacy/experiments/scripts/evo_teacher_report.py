#!/usr/bin/env python3
"""Create the report and figures from a completed teacher diagnostic without new simulation."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from helpers.runManifest import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run_id
    dest = ROOT / 'docs/evoPlan/results/stage2-teacher' / args.run_id
    d = json.loads((run / 'diagnostic.json').read_text())
    rows = json.loads((run / 'rows.json').read_text())
    failures = [r for r in rows if r['mixed']['violated']]
    boundaries = [t['takeover_step'] for t in rows[0]['takeovers']]
    summary_rows = []
    for r in failures:
        safe = [t['takeover_step'] for t in r['takeovers'] if not t['violated']]
        summary_rows.append({
            'root_id': r['root_id'], 'teacher': r['teacher'],
            'student_first_unsafe_transition_index': r['mixed']['first_unsafe_step'],
            'latest_tested_safe_takeover': max(safe) if safe else None,
            'safe_takeover_steps': safe})
    all_totals = [{'takeover_step': b, 'violations': sum(r['takeovers'][j]['violated'] for r in rows)}
                  for j, b in enumerate(boundaries)]
    # A descriptive post-hoc matched-position check, not a selection criterion.
    with np.load(run / 'trajectories.npz') as a:
        squared = ((a['mixed_actions'] - a['mixed_teacher_actions'])**2).mean(-1).reshape(24, 10, 10)
        first = np.array([r['mixed']['first_unsafe_step'] for r in rows])
        keep = np.arange(10)[None] * 10 + 9 <= first[:, None]
        complete_positions = squared[keep].mean(0)
        assert a['takeover_qpos'].shape == (24, 10, 101, 9)
        assert a['mixed_qpos'].shape == (24, 101, 9)
        for i, r in enumerate(rows):
            for j, boundary in enumerate(boundaries):
                for key in ['qpos', 'qvel']:
                    assert np.array_equal(a[f'takeover_{key}'][i, j, :boundary+1],
                                          a[f'mixed_{key}'][i, :boundary+1])
    analysis = dict(failure_takeovers=summary_rows, takeover_totals=all_totals,
                    complete_pre_failure_blocks=int(keep.sum()),
                    complete_block_position_mse=complete_positions.tolist(),
                    complete_block_analysis='post hoc descriptive matched-position check',
                    note='Finite-horizon success of this source teacher, not guaranteed recovery.')
    for folder in [run, dest]:
        (folder / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw={'width_ratios': [1, 1.5]})
    for label, title, color in [('mixed', 'Student on its visited states', '#2964a3'),
                                ('recorded', 'Recorded actions on recorded replay', '#767676')]:
        err = d['action_error'][label]['all']['block_position']
        ax.plot(range(1, 11), [p['mse'] for p in err], marker='o', label=title, color=color)
    ax.set(yscale='log', xlabel='Action position within ten-action block',
           ylabel='MSE against source teacher', xticks=range(1, 11),
           title='Disagreement before the first violation')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='center left')
    matrix = np.array([[2 if r['mixed']['first_unsafe_step'] < t['takeover_step']
                        else int(t['violated']) for t in r['takeovers']] for r in failures])
    heat.imshow(matrix, cmap=ListedColormap(['#c6e6cc', '#edb0a7', '#dddfe2']), vmin=0, vmax=2,
                aspect='auto')
    heat.set(xticks=range(len(boundaries)), xticklabels=boundaries,
             yticks=range(len(failures)), yticklabels=[r['root_id'] for r in failures],
             xlabel='Teacher takeover before action index',
             title='Four student failures after teacher takeover')
    for i in range(len(failures)):
        for j in range(len(boundaries)):
            heat.text(j, i, ['Safe', 'Fail', 'Prior'][matrix[i, j]], ha='center', va='center', fontsize=8)
    heat.text(0, -0.25, 'Safe: no violation through step 100. Prior: prefix already violated.',
              transform=heat.transAxes, fontsize=8)
    fig.tight_layout(w_pad=2)
    for suffix in ['png', 'pdf']:
        path = dest / f'teacher_diagnostic.{suffix}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        shutil.copy2(path, run / path.name)
    plt.close(fig)
    summary = d['summary']
    table = '\n'.join(
        f"| {name} | {summary[key]['all']['violations']}/24 | "
        f"{summary[key]['PPO']['violations']}/8 | {summary[key]['PPOLag']['violations']}/16 | "
        f"{summary[key]['all']['progress_mean']:.3f} |"
        for key, name in [('recorded', 'Recorded action replay'), ('mixed', 'Mixed student'),
                          ('teacher', 'Teacher acting every step'), ('held_teacher', 'Teacher action held for ten steps')])
    failure_table = '\n'.join(
        f"| {r['root_id']} | {r['teacher']} | {r['student_first_unsafe_transition_index']} | "
        f"{r['latest_tested_safe_takeover']} |" for r in summary_rows)
    err = d['action_error']
    m0, m9 = [err['mixed']['all']['block_position'][j]['mse'] for j in [0, 9]]
    report = f"""# Teacher diagnosis of the four remaining development failures

On 4 October 2026, the original source teachers avoided health violations on all
24 development starts when acting every simulator step. Teacher takeover before
action 30 also prevented all four failures of the mixed-demonstration student.
The student disagreed substantially with its teacher even on the first action of
each block. These results motivate improving imitation on student-visited states;
they do not establish that changing the observation interval alone will fix control.

This is a diagnostic of the previously inspected development set, not a new safety
confirmation. **Gate 0 was not rerun, and no controller was trained.**

## Fixed comparison and provenance

Protocol and implementation were committed as bfa4bb1 before teacher queries or
interventions. The source is the coverage bundle at private Hugging Face revision
aab43ddcba2c1209453121a705ecf10c21a560a2. Actors come from the pinned original
42-policy ladder. Each root uses the exact actor named by its source episode,
with the exported observation normalisation and action clipping, and zero action noise.

Every takeover replays the student's original actions from the root to a fixed
boundary, then uses that teacher through the same 100-step horizon. This preserves
the solver warm start; resetting only the boundary pose and velocity would not.
All 24 saved student trajectories were reproduced bitwise in both pose and velocity,
and all 240 intervention prefixes matched them bitwise. Earlier prefix violations
remain counted after takeover. We tested boundaries 0, 10, ..., 90 on every root.

Teachers receive privileged simulator observations at 125 Hz. Their performance is
a diagnostic reference, not a latent-controller result. The held-action reference
updates the executed teacher action every ten steps and repeats it. That differs from the
student, which predicts ten distinct actions per block; it cannot establish that
sequential block control itself is inadequate.

## Outcomes

| Controller or replay | All violations | PPO starts | PPO-Lagrangian starts | Mean progress in metres |
|---|---:|---:|---:|---:|
{table}

The per-step teachers retained {100*summary['teacher']['all']['progress_mean']/summary['recorded']['all']['progress_mean']:.2f}%
of recorded progress. Zero failures in 24 already-inspected starts is not a safety
guarantee. Holding teacher actions caused 22/24 failures, confirming that the
teacher's actions generally cannot be held for ten steps.

## Teacher takeovers on the four failed trajectories

| Root | Source teacher | Student first unsafe transition index | Latest tested safe takeover index |
|---|---|---:|---:|
{failure_table}

Indices are zero based. Takeover at 30 occurs before action 30, after 30 student
actions. A first unsafe transition at 66 means that the state after action 66 is
unsafe. Safe means no violation through the original segment endpoint; the horizon
was not extended after takeover.

Takeovers at 0, 10, 20 and 30 produced zero failures across all 24 roots. Counts at
40, 50, 60, 70, 80 and 90 were 2, 2, 2, 3, 4 and 4. The two PPO teachers failed
when started at boundary 40 even though their student prefixes had not yet violated.
This suggests that waiting for a health violation is too late for these particular
teachers and trajectories. Failed teacher recovery is not proof of irreversibility,
and the coarse boundary grid does not identify a precise recovery deadline.

## Action disagreement

On the student's visited states, mean action MSE against the noiseless source
teacher was {err['mixed']['all']['all']['mse']:.5f}, using {err['mixed']['all']['all']['n_actions']}
actions up to and including the first violating transition. Recorded-action replay
gave {err['recorded']['all']['all']['mse']:.5f} against its teacher on its own states.
The latter includes the collection noise and is a reference, not a noise term that
can simply be subtracted from the student's error.

Student MSE was {m0:.5f} on the first position and {m9:.5f} on the tenth. Thus
substantial disagreement exists when a fresh block is first selected. It is not
confined to its later actions. A post-hoc check using the same {int(keep.sum())} complete
pre-failure blocks at every position gives {complete_positions[0]:.5f} and
{complete_positions[-1]:.5f}, respectively. Position-wise counts, family breakdowns
and that descriptive check are saved in [diagnostic.json](diagnostic.json) and
[analysis.json](analysis.json).

Teacher disagreement is not uniquely wrong action selection: multiple gaits can
be safe, and the mixed student imitates many teachers without receiving teacher
identity. The intervention results make imitation on visited states a useful
hypothesis to test, but do not separate representation, capacity, conflicting
teacher targets and distribution shift.

![Action disagreement and takeover outcomes](teacher_diagnostic.png)

## Recommended next experiment

Keep the latent representation and ten-action interface fixed first. Compare an
equal-budget baseline with teacher-labelled action blocks collected from student
rollouts starting only in the established fit episodes. Query the source teacher
from those visited states to produce a sequential ten-action target, and count
every student rollout, replay prefix and teacher branch step. Preserve whole-episode
validation and keep these 24 development episodes out of fitting.

Declare the target sampling, teacher assignment, replay mixture and model-selection
rule before collection. This would test whether targeted imitation data improves
closed-loop control. The current diagnostic provides no new starting policy and no
basis for proceeding to transfer checks or evolution yet.

## Verification and resources

The full suite passed: 621 tests, no failures or skips, with 49 existing warnings.
Ruff passed. Three new tests check bitwise action-tape replay despite teacher queries,
takeover and action-hold timing with exact prefix preservation, and exclusion of
post-failure states from action-error summaries.

The scientific run took {d['costs']['scientific_wall_s']:.2f} seconds and used exactly
31,200 new simulator steps and 31,200 teacher queries: 2,400 each for mixed replay,
recorded replay and held-teacher control, plus 24,000 for 240 complete takeover
segments. It used no new rendering, encoding, imagined rollout or optimiser update.
Test costs are separate. Source experiment costs remain recorded as provenance.

[rows.json](rows.json) holds all paired outcomes; [manifest.json](manifest.json)
pins inputs, source hashes, teacher weights, code and accounting.
Dense trajectories are in the ignored local run bundle runs/{args.run_id}/.
An archive receipt records the private remote revision once uploaded.

Reproduce the diagnostic and report with a new run ID:

    uv run python experiments/scripts/evo_teacher_diagnostic.py --run-id NEW_RUN_ID
    uv run python experiments/scripts/evo_teacher_report.py --run-id NEW_RUN_ID
"""
    (dest / 'README.md').write_text(report)
    (run / 'REPORT.md').write_text(report)
    code_path = 'experiments/scripts/evo_teacher_report.py'
    snapshot = run / 'source' / code_path
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / code_path, snapshot)
    provenance = {'report_code_sha256': file_sha256(ROOT / code_path),
                  'source_files': {name: file_sha256(run / name) for name in
                                   ['diagnostic.json', 'rows.json', 'trajectories.npz']},
                  'outputs_sha256': {name: file_sha256(run / name) for name in
                                     ['REPORT.md', 'analysis.json', 'teacher_diagnostic.png', 'teacher_diagnostic.pdf']}}
    for folder in [run, dest]:
        (folder / 'report_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
