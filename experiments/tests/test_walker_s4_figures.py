"""Synthetic report presentation tests; no production evidence or model execution."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/walker_s4_figures.py'
spec = importlib.util.spec_from_file_location('walker_s4_figures_test', path)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def metric(false_safe=20, *, unresolved=0, accepted=50):
    return {'m': 0., 'n': 100, 'n_accepted': accepted, 'n_false_safe': false_safe,
            'acceptance_rate': accepted / 100, 'n_censored': unresolved,
            'n_accepted_censored': unresolved,
            'fsa_lower': false_safe / accepted if accepted else float('nan'),
            'fsa_upper': (false_safe + unresolved) / accepted if accepted else float('nan'),
            'fsa': false_safe / accepted if accepted and not unresolved else float('nan')}


def pair(reference=None, updated=None, *, bounds=(-.15, .05)):
    reference = metric() if reference is None else reference
    updated = metric(15) if updated is None else updated
    point = updated['fsa'] - reference['fsa']
    defined = figures.finite(point)
    lo, hi = bounds if defined else (float('nan'), float('nan'))
    direction = ('decrease' if defined and hi < 0 else 'increase' if defined and lo > 0 else
                 'inconclusive' if defined else 'undefined')
    return {'point': point, 'lo': lo, 'hi': hi, 'defined': defined,
            'direction': direction, 'interval_excludes_zero': direction in ('decrease', 'increase'),
            'undefined_reason': ('zero_acceptance' if not reference['n_accepted'] or not updated['n_accepted'] else
                                 'accepted_censored_futures' if not defined else None),
            'n_boot': 900, 'n_boot_requested': 1000, 'n_boot_undefined': 100,
            'cluster_unit': 'source_episode', 'n_source_episodes': 20, 'n_roots': 20,
            'reference': reference, 'updated': updated}


def gate(stopped, rules):
    t = {'min_unsafe': 10, 'min_false_safe': 5, 'min_imagination': 3, 'min_share': .25}
    evidence = {rule: {'n': 100, 'n_unsafe': 2 if stopped else 20,
                       'n_false_safe': 0 if stopped else 5, 'n_imagination': 0 if stopped else 3,
                       'imagination_share': None if stopped else .6, 'pass': not stopped} for rule in rules}
    return {'go': not stopped, 'repair_rules': [] if stopped else rules, 'evidence': evidence,
            'role': 'development', 'thresholds': t, 'status': 'diagnostic_stop' if stopped else 'pass'}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def freeze(tmp_path, *, stopped=False, rules=('health',), undefined=True):
    """Producer-shaped JSON mirrors, with explicitly synthetic numeric examples."""
    run, results, bank = (tmp_path / n for n in ('synthetic-s4', 'results', 'bank'))
    for d in (run, results, bank):
        d.mkdir()
    g = gate(stopped, list(rules))
    rows = [{'bank': 'dev', 'root_id': f'dev-{i // 5}', 'branch': i} for i in range(100)]
    def export(name, value):
        for d in (run, results, bank):
            dump(d / name, value)
        p = run / name
        return {'file': name, 'sha256': figures.file_hash(p), 'bytes': p.stat().st_size}
    report = {'run_id': run.name, 'status': 'diagnostic_stop' if stopped else 'complete'}
    if stopped:
        report.update(gate=g, final_test_evaluated=False,
                      saved_evaluation_rows={'development': export('development_rows.json', rows)})
    else:
        exports = {name: [{**r, 'bank': name} for r in rows] for name in ('dev', 'test', 'stress')}
        report.update(development_gate=g, active_rules=list(rules), repair_rules=list(rules),
            planned_budgets=[128, 512], acquisition_seeds=[0, 1, 2], evaluation_steps=121600,
            saved_evaluation_rows={
                'baseline_decomposition': export('baseline_decomposition_rows.json', exports),
                'no_update_evaluation': export('no_update_evaluation_rows.json', exports)},
            adaptation={'no_update': {name: {r: {'margin': 0., 'at_matched': metric(20)} for r in rules}
                                      for name in ('dev', 'test', 'stress')}, 'random': {}, 'boundary': {}},
            boundary_vs_random={'contrast': 'boundary minus random', 'cluster_unit': 'source_episode',
                'confidence_level': .95, 'by_budget': {str(b): {'by_seed': {}} for b in (128, 512)}})
        for seed in (0, 1, 2):
            for arm in figures.ARMS:
                curve = []
                for budget in (128, 512):
                    m = metric() if arm == 'random' else metric(30 if seed == 1 else 15 if budget == 128 else 10,
                        unresolved=int(undefined and seed == 2))
                    evaluation = {name: {r: {'margin': 0., 'at_matched': copy.deepcopy(m)} for r in rules}
                                  for name in ('dev', 'test', 'stress')}
                    rel = f'{arm}-s{seed}-b{budget}/evaluation_rows.json'
                    dump(run / rel, exports)
                    curve.append({'budget': budget, 'root_generation_steps': 300000,
                        'common_seed_steps': 12800, 'acquired_steps': budget * 100,
                        'charged_steps': 312800 + budget * 100, 'optimizer_steps': 1500,
                        'model_candidate_queries': 0 if arm == 'random' else 640 if budget == 128 else 1152,
                        'training_time_s': 12., 'eval': evaluation,
                        'evaluation_rows': {'file': rel, 'sha256': figures.file_hash(run / rel)}})
                report['adaptation'][arm][str(seed)] = curve
            for i, budget in enumerate((128, 512)):
                points = {a: report['adaptation'][a][str(seed)][i] for a in figures.ARMS}
                entry = {'charged_steps': points['random']['charged_steps'],
                         'evaluation_rows_sha256': {a: points[a]['evaluation_rows']['sha256'] for a in figures.ARMS}}
                for name in figures.BANKS:
                    entry[name] = {r: pair(points['random']['eval'][name][r]['at_matched'],
                                           points['boundary']['eval'][name][r]['at_matched'],
                        bounds=(.1, .3) if seed == 1 else (-.12, -.05) if budget == 512 else (-.15, .05)) for r in rules}
                report['boundary_vs_random']['by_budget'][str(budget)]['by_seed'][str(seed)] = entry
    manifest = {'run_id': run.name, 'data': {'saved_evaluation_rows': report['saved_evaluation_rows']},
                'costs': {'generated_evaluation_steps' if stopped else 'evaluation_steps': 121600}}
    for d in (run, results, bank):
        dump(d / 'manifest.json', manifest)
    for d in (run, results):
        dump(d / 'decomposition_gate.json', g)
    dump(run / 'study.json', report)
    dump(bank / ('diagnostic.json' if stopped else 'study.json'), report)
    dump(results / 'study.json', dict(report, bank_hf_revision='a' * 40))
    state = {'stages': {'prospective_acquisition': {
        'status': 'gate_stopped' if stopped else 'completed',
        'attempts': [{'finished_unix': 1., 'returncode': 2 if stopped else 0}],
        'output_hashes': {str(d): figures.output_hash(d) for d in (run, results, bank)},
        'gate': {'path': str(run / 'decomposition_gate.json'), 'key': 'go', 'passes': g['go'],
                 'sha256': figures.file_hash(run / 'decomposition_gate.json')}}}}
    state_path = tmp_path / 'workflow.json'
    dump(state_path, state)
    return run, results, bank, state_path


def test_undefined_pair_stays_undefined_despite_usable_bootstrap_replicates():
    item = pair(updated=metric(0, unresolved=1))
    cells = figures.paired_cells(item)
    assert cells[:4] == ['undefined', '[undefined, undefined]', 'undefined', 'accepted_censored_futures']
    assert cells[4] == '900/1000'
    item['lo'], item['hi'] = -.1, -.01
    with pytest.raises(ValueError, match='Undefined point'):
        figures.paired_cells(item)


def test_zero_acceptance_never_becomes_zero_fsa_or_direction():
    item = pair(updated=metric(0, accepted=0))
    assert figures.paired_cells(item)[3] == 'zero_acceptance'
    item['point'], item['defined'] = 0., True
    with pytest.raises(ValueError, match='undefined constituent'):
        figures.paired_cells(item)


def test_saved_point_outside_percentile_interval_is_not_a_negative_errorbar():
    item = pair(updated=metric(10), bounds=(-.12, -.05))
    cells = figures.paired_cells(item)
    assert cells[:3] == ['-0.2', '[-0.12, -0.05]', 'decrease']


@pytest.mark.parametrize('change', [dict(direction='increase'), dict(n_boot_undefined=0), dict(lo=None)])
def test_inconsistent_saved_interval_or_support_is_rejected(change):
    item = pair()
    item.update(change)
    with pytest.raises(ValueError):
        figures.paired_cells(item)


@pytest.mark.parametrize('stopped,rules', [(True, ('health',)), (False, ('health',)), (False, ('health', 'speed'))])
def test_end_to_end_saved_schema_and_static_figures(tmp_path, stopped, rules):
    run, results, bank, state = freeze(tmp_path, stopped=stopped, rules=rules)
    before = {str(d): figures.output_hash(d) for d in (run, results, bank)}
    output = tmp_path / 'figures'
    provenance = figures.create_presentation(run, results, bank, output, state)
    assert provenance['study_status'] == ('diagnostic_stop' if stopped else 'complete')
    assert {str(d): figures.output_hash(d) for d in (run, results, bank)} == before
    assert (output / 'development_gate.png').read_bytes().startswith(b'\x89PNG')
    assert (output / 'development_gate.pdf').read_bytes().startswith(b'%PDF')
    assert (output / 'paired_fsa.png').exists() is (not stopped)
    assert (output / 'paired_fsa.pdf').exists() is (not stopped)
    text = (output / 'README.md').read_text()
    assert '121600' in text
    if stopped:
        assert 'No final-test measurements or acquisition comparison' in text
        assert '## Paired acquisition estimates' not in text
    else:
        assert '900/1000' in text and 'accepted_censored_futures' in text
        assert '325600' in text and '364000' in text
        assert 'not a lifetime simulator total' in text
    presentation = json.loads((output / 'presentation.json').read_text(), parse_constant=lambda s: pytest.fail(s))
    assert presentation['gate_recomputed'] is False
    with pytest.raises(ValueError, match='overwrite'):
        figures.create_presentation(run, results, bank, output, state)


def test_pending_stage_refuses_before_scientific_reads(tmp_path):
    state = tmp_path / 'workflow.json'
    dump(state, {'stages': {'prospective_acquisition': {'status': 'running'}}})
    with pytest.raises(ValueError, match='not finalized'):
        figures.create_presentation(tmp_path / 'absent', tmp_path / 'results', tmp_path / 'bank', tmp_path / 'figures', state)
    assert not (tmp_path / 'figures').exists()


@pytest.mark.parametrize('relative', ['results/figures', 'bank/figures', 'synthetic-s4/figures', 'results'])
def test_same_or_nested_output_refused(tmp_path, relative):
    run, results, bank, state = freeze(tmp_path, stopped=True)
    with pytest.raises(ValueError):
        figures.create_presentation(run, results, bank, tmp_path / relative, state)


def test_tampered_completed_input_rejected_before_any_output(tmp_path):
    run, results, bank, state = freeze(tmp_path)
    (results / 'study.json').write_text('{}')
    with pytest.raises(ValueError, match='output changed'):
        figures.create_presentation(run, results, bank, tmp_path / 'figures', state)
    assert not (tmp_path / 'figures').exists()


def test_mutation_during_plotting_cannot_publish_complete_manifest(tmp_path, monkeypatch):
    run, results, bank, state = freeze(tmp_path, stopped=True)
    def mutate(_gate, _output):
        (results / 'study.json').write_text('{}')
    monkeypatch.setattr(figures, 'plot_gate', mutate)
    with pytest.raises(ValueError, match='output changed'):
        figures.create_presentation(run, results, bank, tmp_path / 'figures', state)
    assert not (tmp_path / 'figures/manifest.json').exists()
