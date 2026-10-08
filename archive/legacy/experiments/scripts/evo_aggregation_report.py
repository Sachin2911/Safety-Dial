#!/usr/bin/env python3
"""Audit and report one completed aggregation comparison without additional simulation."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
import numpy as np

from helpers.runManifest import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run_id
    dest = ROOT / 'docs/evoPlan/results/stage2-aggregation' / args.run_id
    d = json.loads((run / 'diagnostic.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    fit = json.loads((run / 'fit.json').read_text())
    history = json.loads((run / 'training_history.json').read_text())
    rows = json.loads((run / 'development_rows.json').read_text())
    plan = json.loads((run / 'collection_plan.json').read_text())
    assert manifest['repo']['commit'].startswith('2a15aae')
    assert len(history) == 600 and len(rows) == 24
    assert all(r['episode'] % 4 == 0 for r in rows)
    assert len({r['episode'] for r in rows}) == 24
    assert len(plan['roots']) == len({r['episode'] for r in plan['roots']}) == 128
    for name, expected in manifest['code_sha256'].items():
        assert file_sha256(ROOT / name) == expected
    for family, key in [('PPOLag', 'lag'), ('PPO', 'ppo')]:
        episodes = {r['episode'] for r in plan['roots'] if r['family'] == family}
        assert len(episodes) == 64
        assert episodes <= set(plan['source_coverage_plan'][f'fit_episodes_selected_{key}'])
        assert not episodes.intersection(plan['source_coverage_plan'][f'{key}_validation_episodes'])
    assert d['costs']['real_steps'] == 171200
    assert d['costs']['optimizer_updates'] == 27000
    collection = {}
    with np.load(run / 'labels.npz') as labels, np.load(run / 'mixture_indices.npz') as mixture:
        assert labels['history_z'].shape == (128, 3, 192)
        for key in ['lag', 'ppo']:
            old, new = mixture[f'old_{key}'], mixture[f'new_{key}']
            assert len(old) == len(set(old)) == 17245
            assert len(new) == 5748 and np.ptp(np.bincount(new, minlength=640)) == 1
        for arm in ['teacher_states', 'student_states']:
            assert labels[f'{arm}_X'].shape == (128, 10, 384)
            assert labels[f'{arm}_Y'].shape == (128, 10, 60)
            assert np.isfinite(labels[f'{arm}_X']).all()
            assert np.isfinite(labels[f'{arm}_Y']).all()
            assert np.abs(labels[f'{arm}_Y']).max() <= 1
            with np.load(run / f'collection_{arm}.npz') as a:
                qp = a['qpos']
                unsafe = (qp[:, 1:, 1] <= .8) | (qp[:, 1:, 1] >= 2) | (abs(qp[:, 1:, 2]) >= 1)
                collection[arm] = dict(violations=int(unsafe.any(1).sum()),
                                       ppo_violations=int(unsafe[64:].any(1).sum()),
                                       ppolag_violations=int(unsafe[:64].any(1).sum()))
                if arm == 'teacher_states':
                    assert np.array_equal(labels[f'{arm}_Y'],
                                          a['actions'].reshape(128, 10, 60).astype(np.float32))
            selected = fit[arm]['selected']
            expected = min([r for r in history if r['arm'] == arm],
                           key=lambda r: (r['val_mse'], r['seed'], r['epoch']))
            assert all(selected[k] == expected[k] for k in ['seed', 'epoch', 'val_mse'])
            with np.load(run / f'{arm}_policy.npz') as p, np.load(
                    run / f"{arm}_seed_{selected['seed']}.npz") as seed:
                assert np.array_equal(p['theta'], seed['theta']) and p['theta'].size == 139196
            with np.load(run / f'real_{arm}.npz') as a:
                qp = a['qpos']
                unsafe = (qp[:, 1:, 1] <= .8) | (qp[:, 1:, 1] >= 2) | (abs(qp[:, 1:, 2]) >= 1)
                assert np.array_equal(unsafe.any(1), a['dense_violated'])
                assert int(unsafe.any(1).sum()) == d['development'][arm]['violations']
                for i, r in enumerate(rows):
                    assert bool(unsafe[i].any()) == r[arm]['violated']
                    assert qp[i, -1, 0] - qp[i, 0, 0] == r[arm]['progress']
    improved = sum(r['teacher_states']['violated'] and not r['student_states']['violated'] for r in rows)
    worsened = sum(not r['teacher_states']['violated'] and r['student_states']['violated'] for r in rows)
    analysis = dict(collection=collection, newly_safe_student_vs_teacher=improved,
                    newly_unsafe_student_vs_teacher=worsened,
                    conclusion='No development improvement from this fixed single aggregation round.')
    for folder in (run, dest):
        (folder / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    recorded = d['recorded']['progress_mean']
    names = {'teacher_states': 'Teacher-state control', 'student_states': 'Student-state augmentation'}
    table = '\n'.join(f"| {names[arm]} | {d['development'][arm]['violations']}/24 | "
        f"{d['by_family']['PPO'][arm]['violations']}/8 | "
        f"{d['by_family']['PPOLag'][arm]['violations']}/16 | "
        f"{d['development'][arm]['progress_mean']:.3f} | "
        f"{100*d['development'][arm]['progress_mean']/recorded:.1f}% |"
        for arm in names)
    selections = '\n'.join(f"| {names[arm]} | {fit[arm]['selected']['seed']} | "
        f"{fit[arm]['selected']['epoch']} | {fit[arm]['selected']['val_mse']:.5f} |" for arm in names)
    delta, progress = d['paired']['violation_difference'], d['paired']['progress_difference']
    report = f"""# One teacher labelled aggregation round did not improve development safety

On 4 October 2026, adding teacher-labelled examples from student-visited training
states produced 6/24 development health violations. Its matched control, which
received teacher labels from teacher-visited states, also produced 6/24. The
previous mixed-demonstration controller had 4/24. The new student-state arm made
less mean forward progress than the matched control, although the paired interval
includes zero.

This fixed single-round comparison does not support replacing the previous
checkpoint. **Gate 0 was not rerun.** No transfer check or evolutionary search ran.

## Protocol and scope

The protocol and implementation were committed as 2a15aae before collection or
outcome inspection. Architecture, frozen encoder, feature normalisation, ten-action
output interface, optimiser, three fitting seeds, epoch count and validation
criterion stayed fixed. The source controller and caches are pinned to coverage
archive revision aab43ddcba2c1209453121a705ecf10c21a560a2.

We selected 64 PPO and 64 PPO-Lagrangian episodes from the existing selected
fit pools, with one seeded start per episode. Eligibility used episode length
alone. Teacher-state and student-state arms share these 128 roots. Teacher labels
come from each source episode's original actor with zero action noise.

Fit episodes are in setA.h5. Development episodes are in the separately collected
roots.h5; episode numbers are local to each file. Neither validation episodes nor
development trajectories supplied new training labels. Collection did not filter
states by health, failure, or teacher success.

The fixed student generated one 100-step rollout per root. The control used the
source teacher's 100-step rollout from the same root. At every ten-step boundary,
both arms replayed their exact prefix from the root and queried the teacher for
the next ten sequential actions. This preserved solver warm starts. All 2,560
label branches verified their prefix pose and velocity against the supplying
trajectory, and control targets matched their teacher rollout actions exactly.

Both arms were charged 83,200 simulator steps: 12,800 rollout steps and 70,400
prefix-plus-label steps. Replaying prefixes on the control arm also verified
identical solver-state handling. These equal budgets are for this diagnostic
implementation, not a claim that this is the cheapest way to obtain teacher labels.

## Training and selection

Each arm collected 1,280 unique action-block labels, 640 per family. An epoch had
45,986 examples: 34,490 shared original examples and 11,496 draws from the new
labels, a 25.0% augmentation fraction after integer rounding. Repetition was nearly
uniform, with the same original indices and new root/block indices in both arms.

Each arm trained three fresh initialisations for 100 epochs, batch size 1,024,
with AdamW at learning rate 0.001 and weight decay 0.0001. This was 13,500 optimiser
updates per arm. The 384-256-128-60 ReLU network has 139,196 parameters.

The unchanged whole-episode validation pool assigned half its weight to each
family. Recorded-action validation MSE alone selected checkpoints; development
outcomes played no part. Neither checkpoint was refit on validation examples.

| Arm | Selected seed | Selected epoch | Family balanced validation MSE |
|---|---:|---:|---:|
{selections}

The earlier mixed checkpoint had validation MSE 0.08781. Both new arms selected
earlier checkpoints under the common criterion. The total training searches were
matched; selected checkpoints need not have the same training age. The inherited
feature scaler remains unchanged and was originally fitted using validation
features as well as fit features, as documented in the preceding reports.

## Development outcomes

| Controller | Health violations | PPO starts | PPO-Lagrangian starts | Mean progress in metres | Recorded progress retained |
|---|---:|---:|---:|---:|---:|
| Previous mixed controller, archived reference | 4/24 | 2/8 | 2/16 | 2.093 | 99.2% |
{table}

Recorded action replay had no violations and mean progress {recorded:.3f} m.
The previous controller and recorded replay are reused archival outcomes, not
new evaluations. The matched causal comparison here is student-state augmentation
against teacher-state control.

Relative to the teacher-state control, {improved} starts became safe and {worsened}
became unsafe. The paired violation-rate difference was {100*delta['point']:.1f}
percentage points, with a source-episode bootstrap interval of
[{100*delta['lo']:.1f}, {100*delta['hi']:.1f}]. The progress difference was
{progress['point']:.3f} m, with interval [{progress['lo']:.3f}, {progress['hi']:.3f}] m.
Both intervals include zero. These are repeatedly inspected development states,
not an untouched confirmation set.

Both new controllers were imagined safe on all 24 starts despite six real
violations each. Probes on real encoded frames marked 9/24 control segments and
7/24 treatment segments unsafe. Those model and readout errors remain separate
from the imitation comparison.

## Interpretation and next decision

The new collection did reach different states: the frozen student violated on
21/128 training rollouts, including 17 PPO and 4 PPO-Lagrangian starts, whereas
the teacher rollouts had no violations. All planned labels were retained.

The result is limited to one frozen collection policy, 128 starts, a 25% replay
mixture, this network and the unchanged offline selection rule. New labels were
repeated roughly nine times per epoch, while replacing 11,496 distinct original
examples. Thus the comparison with the earlier controller also changes diversity
and supervision; it cannot identify which of those changes caused its worse count.
The paired arms control those choices while testing the visited-state distribution.

Do not adopt either new checkpoint or proceed to Gate 0 on the strength of this
run. Keep the earlier mixed controller as the current reference. Before collecting
more data or adding aggregation rounds, a useful next diagnostic would separate
latent-state information limits from conflicting targets across source teachers.
That diagnostic is a proposal, not work performed here. This result does not
establish that all teacher-guided aggregation methods fail.

## Verification and resources

All 626 tests passed with 49 existing warnings and no skips or failures. Ruff
passed. Five new tests cover whole-episode role isolation, deterministic sampling,
matched augmentation budgets, exact solver-prefix labels, protocol invariants,
and current/previous latent feature alignment. Selected weights, hashes, masks,
counts and saved trajectories were audited independently after the run.

The scientific run took {d['costs']['wall_s']:.2f} seconds. It used 171,200 new
simulator steps: 166,400 for collection and 4,800 for two development evaluations.
It also used 480 imagined predictor rows, 27,000 optimiser updates, 3,560 renders
including 64 fingerprint frames, and 3,496 encoded frames, excluding black batch
padding. Test execution costs are separate. Existing source-data and model-training
costs remain recorded as upstream provenance.

[diagnostic.json](diagnostic.json) records summaries and intervals.
[development_rows.json](development_rows.json) contains every paired result.
[collection_plan.json](collection_plan.json), [fit.json](fit.json), and
[training_history.json](training_history.json) record sampling and selection.
[manifest.json](manifest.json) pins inputs, teacher weights, source code and costs.
Weights, labels and dense trajectories are in the ignored local run bundle
runs/{args.run_id}/. An upload receipt records the private archive when complete.

Reproduce with a new run ID:

    uv run python experiments/scripts/evo_aggregation_diagnostic.py --run-id NEW_RUN_ID
    uv run python experiments/scripts/evo_aggregation_report.py --run-id NEW_RUN_ID
"""
    (dest / 'README.md').write_text(report)
    (run / 'REPORT.md').write_text(report)
    code = 'experiments/scripts/evo_aggregation_report.py'
    path = run / 'source' / code
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / code, path)
    provenance = dict(report_code_sha256=file_sha256(ROOT / code),
        source_files={name: file_sha256(run / name) for name in
                      ['diagnostic.json', 'development_rows.json', 'labels.npz',
                       'fit.json', 'training_history.json', 'collection_plan.json']},
        report_sha256=file_sha256(run / 'REPORT.md'), audit='all assertions passed')
    for folder in (run, dest):
        (folder / 'report_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
