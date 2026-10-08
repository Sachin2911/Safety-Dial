#!/usr/bin/env python3
"""Render structured, saved-evidence report content to A4 PDF; no scientific inference.

Spec: version=1, draft=bool, scope="through_s4", title, optional subtitle/date,
sections=[{title, blocks:[{type: paragraph|source|table|figure|page_break, ...}]}].
Paragraph/source: text, optional url. Table: headers, rows, optional widths/title.
Figure: path, caption, optional source. Paths resolve relative to the spec file.
Evidence lists files/directories; final specs also require walker_workflow_state and walker_workflow_config.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments'))
from helpers.managedWorkflow import file_hash, load_workflow, output_hash

PAGE_W, PAGE_H, MARGIN = 595.276, 841.890, 48.0
TOP, BOTTOM = 72.0, PAGE_H - 58.0


def require(ok, message):
    if not ok:
        raise ValueError(message)


def strict_json(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), f'Nonfinite JSON number: {value}')
        return number
    return json.loads(path.read_text(), parse_constant=invalid, parse_float=finite_float)


def terminal_workflow(path, config_path):
    """Verify recorded termination and identities, without recomputing scientific gates."""
    state = strict_json(path)
    require(state.get('status') in ('completed', 'gate_stopped'), 'Walker workflow is not terminal')
    config, identity = load_workflow(config_path, REPO)
    require(all(state.get(k) == v for k, v in identity.items()) and
            state.get('workflow_id') == config['workflow_id'], 'Workflow config identity differs')
    recipes = {stage['id']: stage for stage in config['stages']}
    require(set(recipes) == {'lewm_a_full', 'physical_probe_gate', 'prospective_acquisition'},
            'Final workflow must end at S4 with the declared S2/S3/S4 stages')
    stages, identities = state['stages'], {}
    for sid in ('lewm_a_full', 'physical_probe_gate', 'prospective_acquisition'):
        stage = stages.get(sid, {})
        require(stage.get('status') in ('completed', 'gate_stopped'), f'{sid} is not terminal')
        attempts = stage.get('attempts', [])
        require(attempts and attempts[-1].get('finished_unix') is not None and
                attempts[-1].get('returncode') in (0, 2), f'{sid} has no finalized subprocess')
        require(stage.get('output_hashes'), f'{sid} lacks immutable output identities')
        require(all(str((Path(identity['cwd']) / p).resolve()) in stage['output_hashes']
                    for p in recipes[sid]['outputs']), f'{sid} omitted declared output identities')
        for name, expected in stage['output_hashes'].items():
            require(output_hash(Path(name)) == expected, f'Changed Walker output: {name}')
            identities[name] = expected
        if sid == 'lewm_a_full':
            require(stage['status'] == 'completed' and attempts[-1]['returncode'] == 0,
                    'S2 must complete before a final report')
            continue
        gate = stage.get('gate', {})
        gate_path = Path(gate.get('path', ''))
        declared_gate = recipes[sid]['gate']
        require(gate_path == (Path(identity['cwd']) / declared_gate['path']).resolve() and
                gate.get('key') == declared_gate['key'], 'Recorded gate differs from recipe')
        require(gate_path.is_file() and file_hash(gate_path) == gate.get('sha256'), 'Gate identity changed')
        value = json.loads(gate_path.read_text())
        for key in gate['key'].split('.'):
            value = value[key]
        require(type(value) is bool and value == gate.get('passes'), 'Saved gate flag differs')
        identities[str(gate_path)] = {'kind': 'file', 'sha256': gate['sha256']}
        require((stage['status'] == 'completed') == value, 'Stage termination disagrees with its gate')
        require(attempts[-1]['returncode'] == 0 if value else
                attempts[-1]['returncode'] in declared_gate.get('stop_exit_codes', [0, 2]), 'Invalid terminal return code')
        if not value:
            require(state['status'] == 'gate_stopped', 'Workflow must preserve scientific gate stop')
            if sid == 'physical_probe_gate':
                later = stages.get('prospective_acquisition', {})
                require(later.get('status', 'pending') in ('pending', 'waiting') and not later.get('attempts'),
                        'S4 ran after an S3 scientific stop')
            return identities
    require(state['status'] == 'completed', 'All passed stages require completed workflow')
    return identities


class Pages:
    """All positions and measured text widths are in typographic points."""
    def __init__(self, pdf, preview, draft):
        self.pdf, self.preview, self.draft = pdf, preview, draft
        self.fig, self.page, self.audit, self.figures = None, 0, [], []
        self.new()

    def new(self):
        self.finish()
        self.page += 1
        self.fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72), dpi=144, facecolor='white')
        self.renderer = self.fig.canvas.get_renderer()
        self.y = TOP
        self.text(MARGIN, 32, 'SAFETYDIAL  /  RESEARCH REPORT', 9, color='#526176')
        self.text(MARGIN, PAGE_H - 34, 'Scope: through S4; S5 excluded', 8, color='#526176')
        self.text(PAGE_W - MARGIN - 22, PAGE_H - 34, str(self.page), 8, color='#526176')
        if self.draft:
            self.fig.text(.5, .48, 'DRAFT', ha='center', va='center', rotation=35,
                          fontsize=68, color='#eeeeee', zorder=-1, parse_math=False)

    def width(self, text, size, weight='normal'):
        return self.renderer.get_text_width_height_descent(
            text, FontProperties(family='DejaVu Sans', size=size, weight=weight), False)[0] * 72 / self.fig.dpi

    def wrap(self, text, width, size=10, weight='normal'):
        require(isinstance(text, str) and '\u2014' not in text, 'Text must be plain text without em dashes')
        require(math.isfinite(width) and width > 2 and size >= 7.5, 'Invalid text bounds/font')
        lines = []
        for paragraph in text.split('\n'):
            line = ''
            for word in paragraph.split():
                if line and self.width(line + ' ' + word, size, weight) > width:
                    lines.append(line)
                    line = ''
                while self.width(word, size, weight) > width:
                    n = 1
                    while n < len(word) and self.width(word[:n + 1], size, weight) <= width:
                        n += 1
                    require(self.width(word[:n], size, weight) <= width, 'Column narrower than one character')
                    lines.append(word[:n])
                    word = word[n:]
                line = (line + ' ' + word).strip()
            lines.append(line)
        return lines

    def text(self, x, y, text, size, weight='normal', color='#172536', url=None):
        require(all(math.isfinite(v) for v in (x, y, size)), 'Nonfinite layout')
        require(x >= 0 and y >= 0 and x + self.width(text, size, weight) <= PAGE_W - 1,
                f'Text outside page: {text[:50]}')
        artist = self.fig.text(x / PAGE_W, 1 - y / PAGE_H, text, va='top', fontsize=size,
                               weight=weight, color=color, family='DejaVu Sans', parse_math=False, url=url)
        box = artist.get_window_extent(self.renderer)
        require(box.y0 >= 0 and box.x0 >= 0 and box.y1 <= self.fig.bbox.height and box.x1 <= self.fig.bbox.width,
                'Measured text exceeds page')
        self.audit.append({'page': self.page, 'text': text, 'font_pt': size,
                           'bounds_pt': [v * 72 / self.fig.dpi for v in box.bounds]})

    def room(self, height):
        require(math.isfinite(height) and 0 <= height <= BOTTOM - TOP, 'Block exceeds one page')
        if self.y + height > BOTTOM:
            self.new()

    def paragraph(self, text, size=10, weight='normal', url=None, after=9):
        for line in self.wrap(text, PAGE_W - 2 * MARGIN, size, weight):
            self.room(size * 1.4)
            self.text(MARGIN, self.y, line, size, weight, url=url)
            self.y += size * 1.4
        self.y += after

    def heading(self, text, size=18):
        height = len(self.wrap(text, PAGE_W - 2 * MARGIN, size, 'bold')) * size * 1.4 + 28
        self.room(height)
        self.paragraph(text, size, 'bold', after=12)

    def table(self, block):
        headers, rows = block['headers'], block['rows']
        require(headers and all(len(row) == len(headers) for row in rows), 'Ragged table')
        weights = block.get('widths', [1] * len(headers))
        require(len(weights) == len(headers) and all(math.isfinite(v) and v > 0 for v in weights), 'Invalid columns')
        widths = [(PAGE_W - 2 * MARGIN) * v / sum(weights) for v in weights]
        def cells(row):
            return [self.wrap('undefined' if value is None else str(value), width - 12, 8.5)
                    for value, width in zip(row, widths)]
        header = cells(headers)
        header_h = max(map(len, header)) * 12 + 10
        require(header_h < (BOTTOM - TOP) / 2, 'Table header too tall')
        def draw(lines, count, shaded=False):
            height, x = count * 12 + 10, MARGIN
            for column, width in zip(lines, widths):
                self.fig.add_artist(Rectangle((x / PAGE_W, 1 - (self.y + height) / PAGE_H),
                    width / PAGE_W, height / PAGE_H, facecolor='#eaf0f5' if shaded else '#ffffff',
                    edgecolor='#ccd5df', linewidth=.45, transform=self.fig.transFigure, zorder=-.5))
                for i, line in enumerate(column):
                    self.text(x + 6, self.y + 5 + 12 * i, line, 8.5)
                x += width
            self.y += height
        if block.get('title'):
            title_h = len(self.wrap(block['title'], PAGE_W - 2 * MARGIN, 13, 'bold')) * 18.2 + 12
            self.room(title_h + header_h + 22)
            self.heading(block['title'], 13)
        self.room(header_h + 22)
        draw(header, max(map(len, header)), True)
        for row in rows:
            lines = cells(row)
            while any(lines):
                capacity = int((BOTTOM - self.y - 10) // 12)
                if capacity < 1:
                    self.new()
                    draw(header, max(map(len, header)), True)
                    capacity = int((BOTTOM - self.y - 10) // 12)
                count = min(max(map(len, lines)), capacity)
                draw([column[:count] for column in lines], count)
                lines = [column[count:] for column in lines]
        self.y += 12

    def figure(self, block, base):
        path = (base / block['path']).resolve()
        with Image.open(path) as source:
            image = source.convert('RGB')
        caption = block['caption'] + (f"\nSource: {block['source']}" if block.get('source') else '')
        lines = self.wrap(caption, PAGE_W - 2 * MARGIN, 9)
        caption_h = len(lines) * 12.6 + 16
        require(caption_h + 48 <= BOTTOM - TOP, 'Caption cannot stay with a legible figure on one page')
        width = PAGE_W - 2 * MARGIN
        height = min(width * image.height / image.width, BOTTOM - TOP - caption_h)
        self.room(height + caption_h)
        self.figures.append({'page': self.page, 'path': str(path), 'top_pt': self.y,
                             'figure_height_pt': height, 'caption_bottom_pt': self.y + height + caption_h})
        actual_w = height * image.width / image.height
        axes = self.fig.add_axes([(PAGE_W - actual_w) / 2 / PAGE_W,
                                  1 - (self.y + height) / PAGE_H, actual_w / PAGE_W, height / PAGE_H])
        axes.imshow(image)
        axes.axis('off')
        self.y += height + 8
        for line in lines:
            self.text(MARGIN, self.y, line, 9)
            self.y += 12.6
        self.y += 8

    def finish(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            if self.preview:
                self.fig.savefig(self.preview / f'page-{self.page:03}.png', dpi=144)
            plt.close(self.fig)
            self.fig = None


def render(spec_path, output, preview=None):
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    preview = Path(preview).resolve() if preview else None
    initial = {str(spec_path): output_hash(spec_path)}
    sources = {str(p): file_hash(p) for p in (Path(__file__).resolve(), REPO / 'experiments/helpers/managedWorkflow.py')}
    spec, base = strict_json(spec_path), spec_path.parent
    require(spec.get('version') == 1 and type(spec.get('draft')) is bool and spec.get('scope') == 'through_s4',
            'Spec requires version 1, explicit draft boolean and through_s4 scope')
    require(spec.get('author', 'Sachin Mohan') == 'Sachin Mohan', 'Unexpected report author')
    inputs = {spec_path, *( (base / p).resolve() for p in spec.get('evidence', []))}
    for section in spec['sections']:
        for block in section['blocks']:
            require(block['type'] in ('paragraph', 'source', 'table', 'figure', 'page_break'), 'Unknown content block')
            if block['type'] == 'figure':
                inputs.add((base / block['path']).resolve())
    workflow_identities = {}
    if not spec['draft']:
        require(spec.get('walker_workflow_state') and spec.get('walker_workflow_config'),
                'Final report requires terminal Walker workflow state and frozen config')
        state_path = (base / spec['walker_workflow_state']).resolve()
        config_path = (base / spec['walker_workflow_config']).resolve()
        inputs.update((state_path, config_path))
        initial.update({str(p): output_hash(p) for p in (state_path, config_path)})
        workflow_identities = terminal_workflow(state_path, config_path)
        inputs.update(Path(p) for p in workflow_identities)
    destinations = [p for p in (output, preview) if p]
    for destination in destinations:
        require(not destination.exists(), f'Output already exists: {destination}')
        require(all(destination != p and destination not in p.parents and p not in destination.parents for p in inputs),
                'Output overlaps an evidence input')
    if preview:
        require(output != preview and output not in preview.parents and preview not in output.parents, 'Outputs overlap')
    before = {str(p): output_hash(p) for p in sorted(inputs)}
    require(all(before[p] == value for p, value in initial.items()), 'Input changed between binding and interpretation')
    require(all(before[p] == value for p, value in workflow_identities.items()), 'Workflow output changed after qualification')
    for destination in destinations:
        destination.mkdir(parents=True)
    with matplotlib.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42, 'text.usetex': False}), PdfPages(
            output / 'report.pdf', metadata={'Title': spec['title'], 'Author': 'Sachin Mohan', 'Subject': 'SafetyDial study through S4'}) as pdf:
        pages = Pages(pdf, preview, spec['draft'])
        pages.y = 150
        pages.heading(spec['title'], 27)
        pages.paragraph(spec.get('subtitle', ''), 13)
        pages.paragraph('Sachin Mohan', 12)
        pages.paragraph(spec.get('date', datetime.now(timezone.utc).date().isoformat()), 10)
        pages.paragraph('DRAFT: results may be pending.' if spec['draft'] else 'Final saved-evidence report.', 10)
        pages.paragraph('Scope ends at S4. S5 is excluded.', 10)
        for section in spec['sections']:
            pages.new()
            pages.heading(section['title'])
            for block in section['blocks']:
                kind = block['type']
                if kind in ('paragraph', 'source'):
                    pages.paragraph(block['text'], url=block.get('url'))
                elif kind == 'page_break':
                    pages.new()
                else:
                    pages.table(block) if kind == 'table' else pages.figure(block, base)
        pages.finish()
    require(all(output_hash(Path(p)) == h for p, h in before.items()), 'Evidence changed during rendering')
    require(all(file_hash(Path(p)) == h for p, h in sources.items()), 'Renderer source changed')
    (output / 'layout.json').write_text(json.dumps({'page_count': pages.page, 'text': pages.audit, 'figures': pages.figures}, indent=2) + '\n')
    manifest = {'status': 'complete', 'draft': spec['draft'], 'scope': 'through_s4', 'author': 'Sachin Mohan',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'page_count': pages.page, 'inputs': before,
        'source_sha256': sources, 'walker_workflow_output_identities': workflow_identities,
        'inputs_unchanged': True, 'model_calls': 0, 'simulator_steps': 0, 'network_calls': 0,
        'layout': {'page_points': [PAGE_W, PAGE_H], 'body_font_pt': 10, 'caption_font_pt': 9,
                   'table_font_pt': 8.5, 'font': 'DejaVu Sans', 'pdf_fonttype': 42},
        'matplotlib_version': matplotlib.__version__,
        'files_sha256': {p.name: file_hash(p) for p in sorted(output.iterdir())},
        'previews_sha256': {str(p): file_hash(p) for p in sorted(preview.iterdir())} if preview else {}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    result = render(args.spec, args.output_dir, args.preview_dir)
    print(json.dumps({'status': result['status'], 'pages': result['page_count'], 'output_dir': str(args.output_dir)}))


if __name__ == '__main__':
    main()
