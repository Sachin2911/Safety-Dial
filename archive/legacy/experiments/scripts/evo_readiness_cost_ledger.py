#!/usr/bin/env python3
"""Build a scoped, deduplicated cost ledger from preserved experiment manifests."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root=ROOT):
    root = Path(root)
    manifests = {}
    for path in sorted((root / 'docs/evoPlan/results').glob('*/*/manifest.json')):
        data = json.loads(path.read_text())
        run_id = data.get('run_id', '')
        if not run_id.startswith('walker2d-evo-'):
            continue
        if run_id in manifests and manifests[run_id][1]['costs'] != data['costs']:
            raise ValueError(f'inconsistent duplicate costs: {run_id}')
        manifests.setdefault(run_id, (path, data))
    rows = []
    for run_id, (path, data) in sorted(manifests.items()):
        costs = data['costs']
        if 'real_steps' in costs:
            real = costs['real_steps']
            step_fields = ['real_steps']
        elif 'new_real_steps' in costs:
            real = costs['new_real_steps']
            step_fields = ['new_real_steps']
        elif {'real_steps_bc', 'real_steps_recorded_replay'} <= costs.keys():
            step_fields = ['real_steps_bc', 'real_steps_recorded_replay']
            real = sum(costs[k] for k in step_fields)
        else:
            raise ValueError(f'unknown incremental real-step schema: {run_id}')
        if not isinstance(real, int) or real < 0:
            raise ValueError(f'invalid recorded step cost: {run_id}')
        model = costs.get('predictor_rows', costs.get('imagined_rows'))
        if model is None and isinstance(costs.get('imagined'), dict):
            model = costs['imagined'].get('imagined_rows')
        rows.append(dict(run_id=run_id, manifest=str(path.relative_to(root)),
            manifest_sha256=sha(path), real_steps=real, real_step_fields=step_fields,
            predictor_rows=model,
            gradient_updates=costs.get('gradient_updates', costs.get('optimizer_updates')),
            elapsed_seconds=costs.get('wall_s', data.get('wall_clock_s')),
            included_in_evolution_real_step_subtotal=True))
    data_path = root / 'data/hf/safetydial-walker2d-data/data/walker2d-data-20260926-3/manifest.json'
    model_path = root / 'data/hf/safetydial-walker2d/lewm-a/walker2d-lewm-a-recovery-20260927-1/manifest.json'
    data = json.loads(data_path.read_text())
    model = json.loads(model_path.read_text())
    stats = data['data']['stats']
    dataset = dict(setA=stats['A_competent']['n_steps'] + stats['A_early']['n_steps'],
                   probe=stats['probe']['n_steps'], roots=stats['roots']['n_steps'])
    return dict(scope='Preserved Walker evolution run manifests plus shared original dataset collection; not a lifetime total.',
        evolution_runs=rows, recorded_evolution_real_steps=sum(r['real_steps'] for r in rows),
        recorded_evolution_predictor_rows=sum(r['predictor_rows'] for r in rows if r['predictor_rows'] is not None),
        predictor_row_accounting_complete=all(r['predictor_rows'] is not None for r in rows),
        shared_original_collection_steps=dataset,
        shared_original_collection_total=sum(dataset.values()),
        original_LeWM_final_optimizer_index=model['metrics']['step'],
        original_LeWM_training_episodes=len(model['data']['episode_splits']['training']),
        original_LeWM_validation_episodes=len(model['data']['episode_splits']['validation']),
        shared_asset_manifest_sha256={str(p.relative_to(root)): sha(p) for p in [data_path, model_path]},
        accounting_rules=[
            'Each run ID appears once. Shared upstream, reused steps and nested executor subtotals are not added again.',
            'The failed information setup and its resumed run are separate incremental entries; the resume cumulative field is not summed.',
            'Stored source transitions count once. Training epochs and repeated clip sampling do not create more simulator steps.',
            'The final LeWM optimizer index is not a metered lifetime total including any repeated recovery work.',
            'PPO/PPO-Lagrangian training interactions and historical S0/S1 failed-run or provenance gaps remain unquantified here.',
            'Additional historical probe qualification and S4 evaluation/construction costs are outside this evolution-run subtotal.',
            'This ledger contains only finished or explicitly archived failed runs. A still-running job is absent until its manifest exists.'
        ], generator_sha256=sha(Path(__file__)), incremental_costs=dict(real_steps=0, predictor_rows=0, gradient_updates=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(runs=len(result['evolution_runs']),
        recorded_evolution_real_steps=result['recorded_evolution_real_steps'],
        shared_original_collection_total=result['shared_original_collection_total']), indent=2))


if __name__ == '__main__':
    main()
