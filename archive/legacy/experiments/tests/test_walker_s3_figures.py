"""Synthetic saved-report checks only; no production artifacts or model calls."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/walker_s3_figures.py'
spec = importlib.util.spec_from_file_location('walker_s3_figures_test', path)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)

def evidence(*, health_pass=True, undefined=False):
    from helpers.dialMetrics import fsa
    from helpers.walkerValidation import decision_diagnostics, gate_decision

    probes = {'linear': {'height_r2': -.4, 'pitch_r2': .5, 'speed_r2': .4},
              'mlp': {'height_r2': .95 if health_pass else .7, 'pitch_r2': .96, 'speed_r2': .7}}
    errors = {'height': np.linspace(.02, .15, 10).tolist(),
              'pitch': np.linspace(.03, .25, 10).tolist(),
              'speed': np.linspace(.1, .5, 10).tolist()}
    if undefined:
        probes['linear']['height_r2'] = float('-inf')
        probes['mlp']['height_r2'] = float('nan')
        errors['height'] = [float('nan')] * 10
    rows = []
    for i in range(24):
        rows.append({'root': f'w-e0-t{i // 3}', 'tape': i % 3,
                     **{f'true_{r}': -1 if i < 12 else 1 for r in ('health', 'speed')},
                     **{f'real_{r}': -1 if i < 10 else 1 for r in ('health', 'speed')},
                     **{f'imag_{r}': 1 if 8 <= i < 12 or i >= 18 else -1 for r in ('health', 'speed')}})
    diagnostics, decisions = {}, {}
    for rule in ('health', 'speed'):
        u = np.array([r[f'true_{rule}'] < 0 for r in rows])
        real = np.array([r[f'real_{rule}'] for r in rows])
        imag = np.array([r[f'imag_{rule}'] for r in rows])
        diagnostics[rule] = decision_diagnostics(real, u)
        decisions[rule] = {'n': len(rows), 'n_unsafe': int(u.sum()),
                           'real_readout': fsa(real, u, 0), 'imagined': fsa(imag, u, 0)}
    gate = gate_decision(probes['mlp'], errors, diagnostics)
    return {'run_id': 'synthetic-s3', 'role': 'development', 'probes': probes,
            'gate': gate, 'decision_diagnostics': diagnostics, 'decisions_m0': decisions,
            'imagined_abs_error_by_block': errors,
            'real_readout_abs_error_by_block': {v: [.01] * 10 for v in figures.VARIABLES}}, rows


def frozen_bundle(tmp_path, **kwargs):
    report, rows = evidence(**kwargs)
    run, results = tmp_path / 'run', tmp_path / 'results'
    for directory in (run, results):
        directory.mkdir()
        (directory / 'gate.json').write_text(json.dumps(report))
        (directory / 'gate_rows.json').write_text(json.dumps(rows))
        (directory / 'manifest.json').write_text(json.dumps({'run_id': report['run_id']}))
    state_path = tmp_path / 'workflow.json'
    stage = {'status': 'completed' if report['gate']['go'] else 'gate_stopped',
             'attempts': [{'finished_unix': 1., 'returncode': 0 if report['gate']['go'] else 2}],
             'output_hashes': {str(p): figures.output_hash(p) for p in (run, results)},
             'gate': {'key': 'gate.go', 'path': str(run / 'gate.json'),
                      'sha256': figures.file_hash(run / 'gate.json'), 'passes': report['gate']['go']}}
    state_path.write_text(json.dumps({'stages': {'physical_probe_gate': stage}}))
    return run, results, state_path


def test_speed_failure_alone_does_not_change_recorded_health_go():
    report, rows = evidence()
    assert figures.validate_report(report, rows)['go'] is True
    assert report['gate']['speed_rule_ok'] is False
    text = figures.markdown(report, rows)
    assert 'Health qualification passed' in text
    assert 'Saved active rules: health.' in text
    assert 'MLP threshold' in text and '≥ 0.9' in text and '≥ 0.8' in text


def test_undefined_probe_and_horizon_values_remain_undefined():
    report, rows = evidence(undefined=True)
    figures.validate_report(report, rows)
    text = figures.markdown(report, rows)
    assert 'Stopped at the health qualification gate' in text
    assert 'undefined' in text and 'nan' not in text and 'inf' not in text
    clean = figures.json_safe(report)
    assert clean['probes']['mlp']['height_r2'] is None
    assert clean['imagined_abs_error_by_block']['height'] == [None] * 10
    json.dumps(clean, allow_nan=False)


def test_zero_acceptance_not_reported_as_zero_fsa():
    metric = {'n': 4, 'n_accepted': 0, 'n_false_safe': 0, 'acceptance_rate': 0., 'fsa': float('nan')}
    assert figures.metric_cells(metric)[-1] == 'undefined'
    with pytest.raises(ValueError, match='Defined FSA'):
        figures.metric_cells(dict(metric, fsa=0.))


def test_undefined_saved_point_is_not_filled_from_counts():
    metric = {'n': 4, 'n_accepted': 2, 'n_false_safe': 1, 'acceptance_rate': .5, 'fsa': None}
    assert figures.metric_cells(metric)[-1] == 'undefined'


def test_gate_flag_contradiction_is_rejected_not_recomputed():
    report, rows = evidence()
    report['gate']['go'] = False
    with pytest.raises(ValueError, match='go and health'):
        figures.validate_report(report, rows)


@pytest.mark.parametrize('health_pass,undefined', [(True, False), (False, True)])
def test_fresh_sibling_presentation_preserves_sources_and_saves_figures(tmp_path, health_pass, undefined):
    run, results, state = frozen_bundle(tmp_path, health_pass=health_pass, undefined=undefined)
    before = {str(p): figures.output_hash(p) for p in (run, results)}
    out = tmp_path / 'figures'
    manifest = figures.create_presentation(run, results, out, state)
    assert manifest['recorded_gate_go'] is health_pass
    assert manifest['gate_recomputed'] is False
    assert {str(p): figures.output_hash(p) for p in (run, results)} == before
    for name in ('probe_r2.png', 'horizon_errors.png'):
        assert (out / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    parsed = json.loads((out / 'presentation.json').read_text())
    assert parsed['recorded_gate']['go'] is health_pass
    assert manifest['source_sha256']['experiments/helpers/walkerValidation.py']
    assert len(manifest['files_sha256']) == 4
    with pytest.raises(ValueError, match='overwrite'):
        figures.create_presentation(run, results, out, state)


def test_unfinished_stage_rejected_before_reading_scientific_artifacts(tmp_path):
    state = tmp_path / 'workflow.json'
    state.write_text(json.dumps({'stages': {'physical_probe_gate': {'status': 'running'}}}))
    with pytest.raises(ValueError, match='not finalized'):
        figures.create_presentation(tmp_path / 'absent-run', tmp_path / 'absent-results', tmp_path / 'figures', state)
    assert not (tmp_path / 'figures').exists()


def test_changed_completed_report_rejected_before_output(tmp_path):
    run, results, state = frozen_bundle(tmp_path)
    (results / 'gate.json').write_text('{}')
    with pytest.raises(ValueError, match='output changed'):
        figures.create_presentation(run, results, tmp_path / 'figures', state)
    assert not (tmp_path / 'figures').exists()


@pytest.mark.parametrize('destination', ['results/new', 'run/new', 'results'])
def test_source_nested_or_same_output_is_rejected(tmp_path, destination):
    run, results, state = frozen_bundle(tmp_path)
    with pytest.raises(ValueError):
        figures.create_presentation(run, results, tmp_path / destination, state)


def test_accepted_unresolved_future_is_explicit_without_invented_fsa():
    metric = {'n': 4, 'n_accepted': 2, 'n_false_safe': 0, 'n_accepted_censored': 1,
              'acceptance_rate': .5, 'fsa': float('nan')}
    cells = figures.metric_cells(metric)
    assert cells[3] == '1' and cells[-1] == 'undefined'
    with pytest.raises(ValueError, match='unresolved acceptance'):
        figures.metric_cells(dict(metric, fsa=0.))


def test_displayed_thresholds_match_canonical_gate_boundaries():
    from helpers.walkerValidation import gate_decision

    report, _ = evidence()
    probes = {v + '_r2': value for v, value in figures.R2_THRESHOLDS.items()}
    errors = report['imagined_abs_error_by_block']
    diagnostics = report['decision_diagnostics']
    for variable in figures.VARIABLES:
        at_boundary = gate_decision(probes, errors, diagnostics)
        assert at_boundary[variable + '_pass'] is True
        below = dict(probes, **{variable + '_r2': figures.R2_THRESHOLDS[variable] - 1e-6})
        assert gate_decision(below, errors, diagnostics)[variable + '_pass'] is False
        at_ceiling = {k: list(v) for k, v in errors.items()}
        at_ceiling[variable][-1] = figures.FINAL_ERROR_CEILINGS[variable]
        rule = 'speed' if variable == 'speed' else 'health'
        assert gate_decision(probes, at_ceiling, diagnostics)['imagined_error_useful'][rule] is False
