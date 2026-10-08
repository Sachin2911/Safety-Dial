#!/usr/bin/env python3
"""Declared four-arm variance pilot on cached development episodes only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
import numpy as np
import yaml
from omegaconf import OmegaConf

from helpers.evoHistoryImagine import HistoryImaginer
from helpers.evoHistoryReal import HistoryRealExecutor
from helpers.evoInputs import EvoRun, input_revisions
from helpers.evoReadiness import reference_arrays
from helpers.evoReadinessAnalysis import analyze_readiness
from helpers.evoReadinessProtocol import protocol_plan, scientific_decision_blockers
from helpers.evoReadinessQueries import ImaginedSearchScorer, LatentBank, QueryStore, atomic_json
from helpers.evoReadinessResults import load_study_outcomes, save_paired_outcomes
from helpers.evoReadinessRuntime import load_frozen_runtime
from helpers.evoReadinessSearch import pilot_uncertainty
from helpers.evoReadinessStudy import evaluate_frozen, run_searches
from helpers.evoRoots import load_roots_json
from helpers.locoData import RenderContext
from helpers.runManifest import build_manifest, file_sha256
from helpers.walkerValidation import verify_render_fingerprint
import helpers.evoHistoryReal as history_module
import helpers.evoReal as real_module


def pilot_plan(config, repository=ROOT):
    blockers = scientific_decision_blockers(config['approval'], repository)
    if blockers:
        raise ValueError('; '.join(blockers))
    if config['status'] != 'declared_development_variance_pilot' or config['execution_enabled'] is not True:
        raise ValueError('a declared development pilot is required')
    pilot = config['variance_pilot']
    if pilot['new_source_episodes'] != 0 or pilot['main_final_bank_touched'] is not False:
        raise ValueError('this pilot cannot collect or use new final episodes')
    if config['implementation']['collection_purpose'] != 'previously_inspected_development_only':
        raise ValueError('cached development roles required')
    if (config['search']['population_sizes'] != [16, 256]
            or config['search']['checkpoints'] != [0, 1, 16]
            or config['search']['generations'] != 16
            or len(config['search']['search_seeds']) != 4
            or [config['bank'][f'{role}_episodes'] for role in ['fitness', 'selection', 'evaluation']] != [96, 64, 96]
            or config['budget']['maximum_GPU_hours_proposed'] > 2):
        raise ValueError('pilot grid, cached bank sizes and two-hour cap must match its bounded declaration')
    plan = protocol_plan(config)
    # The inherited bank schema specifies array sizes, not permission to collect.
    plan['counts'].update(new_source_roots=0, maximum_source_collection_steps=0,
        maximum_new_real_steps=plan['counts']['final_real_steps'], initial_history_renders_and_encodes=0)
    return plan


def source_inventory(config_path, config):
    files = [str(Path(__file__).relative_to(ROOT)), str(config_path.resolve().relative_to(ROOT)),
             config['approval']['user_decision_evidence']['path'], 'pyproject.toml', 'uv.lock',
             'configs/evo/inputs.yaml', 'configs/evo/power_pilot_sources.json',
             'configs/evo/stage2_readiness_baseline.yaml',
             'experiments/tests/test_evo_readiness_variance_pilot.py']
    files += sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'experiments/helpers').glob('*.py'))
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *files], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *files], cwd=ROOT, check=True)
    return {name: file_sha256(ROOT / name) for name in sorted(set(files))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/evo/stage5_variance_pilot.yaml')
    parser.add_argument('--run-id')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.protocol.read_text())
    plan = pilot_plan(config)
    if not args.execute:
        print(json.dumps(dict(counts=plan['counts'], experiment_run=False), indent=2))
        return
    if not args.run_id:
        parser.error('--run-id is required for execution')
    sources = source_inventory(args.protocol, config)
    run = EvoRun.create(OmegaConf.create(config), 's5-variance-pilot', args.run_id, resume=args.resume)
    identity = dict(config_sha256=file_sha256(args.protocol), source_sha256=sources)
    launch_path = run.run_dir / 'launch.json'
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
        if launch['identity'] != identity:
            raise ValueError('declared pilot or sources changed since launch')
    else:
        started = time.time()
        launch = dict(identity=identity, started_at=started,
                      deadline=started + config['budget']['maximum_GPU_hours_proposed'] * 3600)
        run.write_json('launch.json', launch)
        for name in sources:
            dest = run.run_dir / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
    def deadline():
        if time.time() > launch['deadline']:
            raise TimeoutError('pilot wall cap reached; no automatic extension')
    deadline()
    settings = config['variance_pilot']
    role_path = ROOT / settings['source_roles_file']
    if file_sha256(role_path) != settings['source_roles_sha256']:
        raise ValueError('cached development roles changed')
    roles = json.loads(role_path.read_text())['indices']
    roles['evaluation'] = roles.pop('assessment')
    runtime = load_frozen_runtime(config, ROOT, require_archived=True)
    base = ROOT / 'runs' / config['implementation']['baseline_history_run']
    if file_sha256(base / 'real_recorded.npz') != settings['recorded_reference_sha256']:
        raise ValueError('recorded reference archive changed')
    roots = load_roots_json(base / 'roots.json')
    with np.load(base / 'history_latents.npz', allow_pickle=False) as saved:
        zh, hist = saved['z_hist'].copy(), saved['hist_blocks'].copy()
    banks = {name: LatentBank(tuple(f'{base.name}:episode:{roots[i].episode}' for i in ids),
                             zh[ids], hist[ids]) for name, ids in roles.items()}
    provenance = dict(**runtime['provenance'], source_sha256=sources,
        role_source_sha256=settings['source_roles_sha256'], role='inspected_development_only',
        user_decision_sha256=config['approval']['user_decision_evidence']['sha256'])
    model = runtime['model']
    original_predict = model.predict
    original_steps = {history_module: history_module._step, real_module: real_module._step}
    counter = dict(predictor_rows=0, real_steps=0)
    ex = ctx = None
    startup_path = run.run_dir / 'startup_receipts.json'
    startups = json.loads(startup_path.read_text()) if startup_path.exists() else []
    def predict(z, a, *rest, **kwargs):
        deadline()
        if counter['predictor_rows'] + len(z) > plan['counts']['total_predictor_rows']:
            raise RuntimeError('declared predictor-row cap exceeded')
        counter['predictor_rows'] += len(z)
        return original_predict(z, a, *rest, **kwargs)
    def wrap(original):
        def step(*args, **kwargs):
            deadline()
            if counter['real_steps'] >= plan['counts']['final_real_steps']:
                raise RuntimeError('declared real-step cap exceeded')
            counter['real_steps'] += 1
            return original(*args, **kwargs)
        return step
    model.predict = predict
    for module, original in original_steps.items():
        module._step = wrap(original)
    def counts():
        return dict(counter, renders=0 if ex is None else ex.counters.renders,
                    encodes=0 if ex is None else ex.counters.encodes)
    try:
        im = HistoryImaginer(model, runtime['scaler'], runtime['probe'], device='cuda', sigma=runtime['sigma'])
        work = run.run_dir / 'study'
        def factory(spec, folder, study_identity):
            print(f'[variance-pilot] search {spec.name}', flush=True)
            scorer = ImaginedSearchScorer(im, runtime['policy'], banks['fitness'], banks['selection'],
                config=spec.config, noise_k=spec.noise_k,
                store=QueryStore(folder / 'queries', study_identity=study_identity), read_costs=counts)
            original_fitness = scorer.fitness
            def fitness(theta, indices, generation):
                result = original_fitness(theta, indices, generation)
                print(f'[variance-pilot] {spec.name} generation {generation + 1}/{spec.config.generations}; process rows {counter["predictor_rows"]}', flush=True)
                return result
            scorer.fitness = fitness
            return scorer
        done = run_searches(work, plan['search_specs'], banks, runtime['zero'], provenance=provenance,
                            evaluation_plan=plan['evaluation_plan'], scorer_factory=factory)
        assert done['complete']
        print('[variance-pilot] all selections frozen; starting paired assessment', flush=True)
        ctx = RenderContext()
        verify_render_fingerprint(ctx, runtime['paths']['data'] / 'setA.h5')
        startups.append(dict(time=time.time(), fingerprint_renders=64))
        atomic_json(startup_path, startups)
        ex = HistoryRealExecutor(model, runtime['scaler'], runtime['probe'], runtime['policy'],
                                 device='cuda', ctx=ctx, max_envs=64, encode_batch=64)
        ids = roles['evaluation']
        selected_roots = [roots[i] for i in ids]
        def real(theta):
            return ex.run(theta[None], selected_roots, z_hist=zh[ids], record_qpos=True)
        def reference():
            return reference_arrays(ex.run_tapes(selected_roots, np.stack([r.policy_tape for r in selected_roots])))
        def audit(theta, k, n, seed):
            out = im.rollout_policies(runtime['policy'], theta[None], zh[ids], hist[ids],
                                     n_samples=n, k=k, seed=seed, max_batch=16384)
            return dict(readout=out['readout'][0], root_readout=out['root_readout'])
        evaluated = evaluate_frozen(work, identity=done['identity'], evaluation_bank=banks['evaluation'],
            real_callback=real, reference_callback=reference, audit_callback=audit,
            read_costs=counts, **plan['evaluation_plan'])
        assert evaluated['complete']
        outcomes = load_study_outcomes(work, expected_roots=selected_roots)
        controls = {}
        for current, old_name in [('baseline-real', 'real_controller'), ('reference-real', 'real_recorded')]:
            with np.load(work / 'evaluation' / f'{current}.npz') as current_data, np.load(base / f'{old_name}.npz') as old:
                fields = ['actions', 'qpos', 'qvel', 'progress', 'dense_violated'] + (['readout'] if current == 'baseline-real' else [])
                controls[current] = {key: bool(np.array_equal(current_data[key], old[key][ids])) for key in fields}
        if not all(all(row.values()) for row in controls.values()):
            raise ValueError('archived real controls did not reproduce exactly')
        save_paired_outcomes(run.run_dir / 'analysis', outcomes)
        a = config['analysis']
        analysis = analyze_readiness(outcomes['cells'], outcomes['baseline'],
            search_seeds=outcomes['search_seeds'], episode_keys=outcomes['episode_keys'],
            population_sizes=config['search']['population_sizes'], checkpoints=[1, 16],
            replicates=a['bootstrap_replicates'], bootstrap_seed=a['bootstrap_seed'])
        cells = outcomes['cells']
        def amplification(k, mode):
            high, low = cells[(k, mode, 256, 16)], cells[(k, mode, 16, 1)]
            return (high['real'].astype(float) - high['imagined'][k]) - (low['real'].astype(float) - low['imagined'][k])
        contrasts = dict(k0_zero=amplification(0., 'zero'),
                         combined_minus_k0=amplification(1., 'rosarl_style') - amplification(0., 'zero'))
        variance = {}
        for name, matrix in contrasts.items():
            result = pilot_uncertainty(matrix, seed=a['bootstrap_seed'], n_boot=a['bootstrap_replicates'])
            result['warning'] = 'Development-only four-seed variance estimate, with truncated components; not guaranteed main-study power.'
            variance[name] = result
        accounting = [evaluated['accounting']]
        accounting += [QueryStore(work / 'searches' / s.name / 'queries', study_identity=done['identity']).accounting()
                       for s in plan['search_specs']]
        costs = {}
        for record in accounting:
            assert not record['pending_queries']
            for key, value in record['completed_costs'].items():
                costs[key] = costs.get(key, 0) + value
        assert costs['predictor_rows'] == plan['counts']['total_predictor_rows']
        assert costs['real_steps'] == plan['counts']['final_real_steps']
        costs['renders'] += sum(s['fingerprint_renders'] for s in startups)
        costs.update(new_source_episodes=0, gradient_updates=0, wall_s=time.time() - launch['started_at'])
        run.write_json('analysis.json', analysis)
        run.write_json('variance.json', variance)
        run.write_json('check.json', dict(passed=True, controls=controls, costs=costs,
            main_final_bank_touched=False, development_only=True, all_selections_frozen_before_assessment=True))
        report = '# Four-arm development variance pilot\n\nThe fixed four-arm pilot completed on previously inspected episodes. Variance estimates support planning only; no final confirmation or safe-controller claim is made. See analysis.json, variance.json and check.json.\n'
        (run.results_dir / 'README.md').write_text(report)
        run.finish(build_manifest(run_id=run.run_id, kind='evo-readiness-variance-pilot-v1',
            costs=costs, started_at=launch['started_at'], upstream_revisions=input_revisions(runtime['paths']),
            data=provenance, seeds=dict(search=config['search']['search_seeds']),
            extra=dict(source_sha256=sources, main_study=False, user_authorized=True, supervisor_approval_claim=False)),
            upload=False, readme=report)
        print(json.dumps(dict(complete=True, costs=costs, primary=analysis['primary'])), flush=True)
    except BaseException as exc:
        run.write_json('failure.json', dict(error=type(exc).__name__, message=str(exc),
            current_process_costs=counts(), note='Completed query receipts and unresolved journals retain earlier costs. No automatic retry.'))
        raise
    finally:
        model.predict = original_predict
        for module, original in original_steps.items():
            module._step = original
        if ex is not None:
            ex.close()
        if ctx is not None:
            ctx.close()


if __name__ == '__main__':
    main()
