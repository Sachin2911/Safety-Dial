import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/mainplan_report_pdf.py'
spec = importlib.util.spec_from_file_location('pdf_report', SOURCE)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write(path, value):
    path.write_text(json.dumps(value, indent=2))
    return path


def composition(tmp_path):
    image = Image.new('RGB', (1200, 540), '#f4f7fa')
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 60, 1110, 450), outline='#253c56', width=4)
    for i, height in enumerate((80, 230, 140, 290)):
        draw.rectangle((180 + i * 220, 450 - height, 290 + i * 220, 450), fill='#39688e')
    image.save(tmp_path / 'synthetic.png')
    write(tmp_path / 'evidence.json', {'synthetic_only': True, 'point': None})
    content = {'version': 1, 'draft': True, 'scope': 'through_s4', 'title': 'Synthetic renderer validation',
        'subtitle': 'Typography, measured pagination, tables and captions', 'author': 'Sachin Mohan',
        'date': '27 September 2026', 'evidence': ['evidence.json'], 'sections': [
        {'title': '1. Paragraph flow', 'blocks': [
            {'type': 'paragraph', 'text': ('This synthetic passage exercises paragraph flow without reporting experimental outcomes. ' * 85)},
            {'type': 'source', 'text': 'Example source: https://example.org/' + 'long-path-segment/' * 12,
             'url': 'https://example.org/'}, {'type': 'page_break'},
            {'type': 'paragraph', 'text': 'An explicit page break begins this paragraph.'}]},
        {'title': '2. Long table', 'blocks': [{'type': 'table', 'title': 'Repeated headers and continued rows',
            'headers': ['Case', 'Saved value', 'Interpretation'], 'widths': [1, 1, 3],
            'rows': [[str(i), 'undefined' if i % 4 == 0 else str(i / 100),
                      'Synthetic row only. ' * (180 if i == 4 else 5)] for i in range(32)]}]},
        {'title': '3. Figure with a long caption', 'blocks': [{'type': 'figure', 'path': 'synthetic.png',
            'caption': 'Figure 1. Synthetic illustration. ' + 'This long caption stays with its image and describes layout validation only. ' * 17,
            'source': 'Locally generated synthetic image; no research measurements.'}]}]}
    return write(tmp_path / 'spec.json', content)


def workflow(tmp_path, stop=None):
    stages, recipes = {}, []
    for i, sid in enumerate(('lewm_a_full', 'physical_probe_gate', 'prospective_acquisition')):
        output = tmp_path / sid
        output.mkdir()
        write(output / 'record.json', {'synthetic': True})
        recipe = {'id': sid, 'argv': ['synthetic'], 'timeout_seconds': 60,
                  'outputs': [str(output)], 'fresh_outputs': [str(output)]}
        stage = {'status': 'gate_stopped' if stop == sid else 'completed',
                 'attempts': [{'finished_unix': 1, 'returncode': 2 if stop == sid else 0}]}
        if i:
            gate = write(output / 'gate.json', {'gate': {'go': stop != sid}})
            recipe['gate'] = {'path': str(gate), 'key': 'gate.go', 'stop_exit_codes': [0, 2]}
            stage['gate'] = {'path': str(gate), 'key': 'gate.go', 'passes': stop != sid,
                             'sha256': report.file_hash(gate)}
        stage['output_hashes'] = {str(output): report.output_hash(output)}
        stages[sid] = stage
        recipes.append(recipe)
    if stop == 'physical_probe_gate':
        stages.pop('prospective_acquisition')
    config = {'version': 1, 'workflow_id': 'synthetic', 'working_directory': str(tmp_path),
              'state_dir': 'state', 'stages': recipes}
    cfg = tmp_path / 'workflow.yaml'
    cfg.write_text(yaml.safe_dump(config))
    _, identity = report.load_workflow(cfg, report.REPO)
    state = write(tmp_path / 'workflow.json', {'workflow_id': 'synthetic', **identity,
                  'status': 'gate_stopped' if stop else 'completed', 'stages': stages})
    return state, cfg


def test_synthetic_multipage_render(tmp_path):
    source = composition(tmp_path)
    before = report.file_hash(source)
    result = report.render(source, tmp_path / 'pdf', tmp_path / 'previews')
    assert result['page_count'] >= 7
    assert report.file_hash(source) == before
    assert result['inputs_unchanged'] and result['draft']
    pdf = (tmp_path / 'pdf/report.pdf').read_bytes()
    assert pdf.startswith(b'%PDF-') and b'/FontFile2' in pdf
    layout = json.loads((tmp_path / 'pdf/layout.json').read_text())
    assert all(t['font_pt'] >= 7.5 for t in layout['text'])
    assert sum(t['text'] == 'Case' for t in layout['text']) >= 3
    assert all(f['caption_bottom_pt'] <= report.BOTTOM for f in layout['figures'])
    assert len(list((tmp_path / 'previews').glob('*.png'))) == result['page_count']
    with pytest.raises(ValueError, match='already exists'):
        report.render(source, tmp_path / 'pdf')


@pytest.mark.parametrize('stop', [None, 'physical_probe_gate', 'prospective_acquisition'])
def test_final_terminal_paths(tmp_path, stop):
    source = composition(tmp_path)
    state, cfg = workflow(tmp_path, stop)
    content = json.loads(source.read_text())
    content.update(draft=False, walker_workflow_state=str(state), walker_workflow_config=str(cfg), sections=[])
    write(source, content)
    result = report.render(source, tmp_path / 'final')
    assert not result['draft'] and result['walker_workflow_output_identities']


def test_pending_and_tampered_workflow_rejected(tmp_path):
    state, cfg = workflow(tmp_path)
    data = json.loads(state.read_text())
    data['stages']['prospective_acquisition']['status'] = 'running'
    write(state, data)
    with pytest.raises(ValueError, match='not terminal'):
        report.terminal_workflow(state, cfg)
    data['stages']['prospective_acquisition']['status'] = 'completed'
    write(state, data)
    (tmp_path / 'lewm_a_full/record.json').write_text('{}')
    with pytest.raises(ValueError, match='Changed Walker output'):
        report.terminal_workflow(state, cfg)


def test_overlap_and_nonfinite_rejected(tmp_path):
    source = composition(tmp_path)
    data = json.loads(source.read_text())
    data['evidence'] = ['.']
    write(source, data)
    with pytest.raises(ValueError, match='overlaps'):
        report.render(source, tmp_path / 'new')
    source.write_text('{"value": NaN}')
    with pytest.raises(ValueError, match='Nonfinite'):
        report.render(source, tmp_path / 'new')


def test_em_dash_and_missing_final_guard_rejected(tmp_path):
    source = composition(tmp_path)
    data = json.loads(source.read_text())
    data.update(draft=False)
    write(source, data)
    with pytest.raises(ValueError, match='requires terminal'):
        report.render(source, tmp_path / 'missing')
    data.update(draft=True, title='Synthetic \u2014 forbidden punctuation')
    write(source, data)
    with pytest.raises(ValueError, match='em dashes'):
        report.render(source, tmp_path / 'dash')


def test_rehash_barrier_no_complete_manifest(tmp_path, monkeypatch):
    source = composition(tmp_path)
    original = report.Pages.finish
    mutated = False
    def finish(self):
        nonlocal mutated
        original(self)
        if self.page and not mutated:
            (tmp_path / 'evidence.json').write_text('{}')
            mutated = True
    monkeypatch.setattr(report.Pages, 'finish', finish)
    with pytest.raises(ValueError, match='Evidence changed'):
        report.render(source, tmp_path / 'partial')
    assert not (tmp_path / 'partial/manifest.json').exists()


def test_saved_gate_nonfinite_unrelated_metric_and_config_identity(tmp_path):
    state, cfg = workflow(tmp_path)
    data = json.loads(state.read_text())
    gate = tmp_path / 'physical_probe_gate/gate.json'
    gate.write_text('{"gate": {"go": true}, "unrelated_metric": NaN}')
    stage = data['stages']['physical_probe_gate']
    stage['gate']['sha256'] = report.file_hash(gate)
    stage['output_hashes'][str(gate.parent)] = report.output_hash(gate.parent)
    write(state, data)
    assert report.terminal_workflow(state, cfg)
    cfg.write_text(cfg.read_text() + '\n# changed configuration\n')
    with pytest.raises(ValueError, match='config identity'):
        report.terminal_workflow(state, cfg)


def test_stage_after_gate_stop_refused(tmp_path):
    state, cfg = workflow(tmp_path, 'physical_probe_gate')
    data = json.loads(state.read_text())
    data['stages']['prospective_acquisition'] = {'status': 'pending', 'attempts': [{'returncode': 1}]}
    write(state, data)
    with pytest.raises(ValueError, match='S4 ran after'):
        report.terminal_workflow(state, cfg)


def test_near_bottom_figure_moves_without_shrinking(tmp_path):
    source = composition(tmp_path)
    content = json.loads(source.read_text())
    content['sections'] = [{'title': 'Readable figure placement', 'blocks': [
        {'type': 'paragraph', 'text': '\n'.join(['Synthetic line'] * 40)},
        {'type': 'figure', 'path': 'synthetic.png', 'caption': 'A detailed figure remains full width.'}]}]
    write(source, content)
    report.render(source, tmp_path / 'pdf', tmp_path / 'previews')
    layout = json.loads((tmp_path / 'pdf/layout.json').read_text())
    figure = layout['figures'][0]
    paragraph_page = max(t['page'] for t in layout['text'] if t['text'] == 'Synthetic line')
    assert figure['page'] == paragraph_page + 1
    assert figure['top_pt'] == report.TOP
    assert figure['figure_height_pt'] == pytest.approx((report.PAGE_W - 2 * report.MARGIN) * 540 / 1200)
    assert figure['caption_bottom_pt'] <= report.BOTTOM


def test_json_numeric_overflow_refused(tmp_path):
    source = tmp_path / 'overflow.json'
    source.write_text('{"value": 1e309}')
    with pytest.raises(ValueError, match='Nonfinite JSON number'):
        report.strict_json(source)


def test_spec_change_after_read_before_hash_rejected(tmp_path, monkeypatch):
    source = composition(tmp_path)
    original = report.strict_json
    def altered(path):
        value = original(path)
        if path == source:
            path.write_text(path.read_text() + '\n')
        return value
    monkeypatch.setattr(report, 'strict_json', altered)
    with pytest.raises(ValueError, match='Input changed between binding'):
        report.render(source, tmp_path / 'changed')
    assert not (tmp_path / 'changed').exists()


@pytest.mark.parametrize('target', ['workflow_state', 'workflow_output'])
def test_change_after_gate_before_input_capture_rejected(tmp_path, monkeypatch, target):
    source = composition(tmp_path)
    state, cfg = workflow(tmp_path)
    content = json.loads(source.read_text())
    content.update(draft=False, walker_workflow_state=str(state), walker_workflow_config=str(cfg), sections=[])
    write(source, content)
    original = report.terminal_workflow
    def altered(*args):
        identities = original(*args)
        path = state if target == 'workflow_state' else tmp_path / 'lewm_a_full/record.json'
        path.write_text(path.read_text() + '\n')
        return identities
    monkeypatch.setattr(report, 'terminal_workflow', altered)
    with pytest.raises(ValueError, match='changed between binding|changed after qualification'):
        report.render(source, tmp_path / 'changed')
    assert not (tmp_path / 'changed').exists()
