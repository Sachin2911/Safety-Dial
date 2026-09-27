#!/usr/bin/env python3
"""Animate existing E2 test illustrations from immutable measured frames and geometry.

No example selection, model inference, simulator replay, interpolation or new metrics.
MP4/GIF are presentation encodings; raw stored-frame identities remain in animations.json.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pusht_e2_figures import (  # noqa: E402
    CATEGORIES, decision_text, horizon_text, inventory, plot_limits, saved_frames,
    selection_context, sha256, t_polygons,
)

CATEGORIES_TO_RENDER = ('corrected_false_safe', 'new_false_safe', 'unresolved_accepted_future')
FPS = 10
PLAYBACK_RATE = 0.5
INITIAL_HOLD_S = 1.0
FINAL_HOLD_S = 2.0
WIDTH, HEIGHT = 1280, 800


def timing_from_source(path):
    """Read literal constants from frozen collection source without importing its models."""
    wanted = {'CONTROL_HZ', 'ACTION_BLOCK', 'PHYSICS_DT', 'SUBSTEPS_PER_STEP'}
    values = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = ast.literal_eval(node.value)
    if (values.get('CONTROL_HZ') != 10 or values.get('ACTION_BLOCK') != 5
            or not np.isclose(values.get('PHYSICS_DT', 0) * values.get('SUBSTEPS_PER_STEP', 0), 0.1)):
        raise ValueError('Stored-frame timing is not the supported frozen 10-Hz, 5-step protocol')
    return values


def all_valid_saved_frames(example, h5):
    """Retain the exact selected branch, extending only its displayed endpoint list."""
    saved_frames(example, h5)  # Validate the original illustration before extending its frames.
    branch = example['row_id']['branch_index']
    observed = np.asarray(h5['observed'][branch], dtype=bool)
    valid = np.asarray(h5['observation_valid'][branch], dtype=bool)
    trace = example['trace']
    if (not np.array_equal(valid[observed], trace['observation_valid'])
            or np.flatnonzero(observed).tolist() != list(range(int(observed.sum())))):
        raise ValueError('Observed-state or validity trace differs from the saved example')
    horizon = int(np.prod(h5['tape'][branch].shape[:2]))
    if horizon != trace['planned_horizon_steps'] or len(observed) != horizon + 1:
        raise ValueError('Saved planned horizon and tape shape disagree')
    steps = [step for step in range(0, len(observed), 5) if observed[step] and valid[step]]
    if not steps:
        raise ValueError('Saved example has no valid stored endpoint frames')
    expanded = copy.deepcopy(example)
    expanded['trace']['frame_steps_used'] = steps
    frames = saved_frames(expanded, h5)
    if any(frame.dtype != np.uint8 or frame.shape != (224, 224, 3) for _, frame in frames):
        raise ValueError('Unexpected stored RGB frame shape or dtype')
    return frames


def presentation_schedule(steps, control_hz=10):
    """Explicit still-frame holds; never interpolate between recorded observations."""
    if not steps or steps != sorted(set(steps)) or any(type(s) is not int or s < 0 for s in steps):
        raise ValueError('Require unique increasing nonnegative saved frame steps')
    counts = [0] * len(steps)
    segments = []
    def segment(index, count, kind):
        if count < 1:
            raise ValueError('Presentation segment must have positive duration')
        counts[index] += count
        segments.append({'step': steps[index], 'kind': kind, 'encoded_frames': count,
                         'duration_s': count / FPS})
    segment(0, round(INITIAL_HOLD_S * FPS), 'initial presentation hold')
    for index, (start, end) in enumerate(zip(steps, steps[1:])):
        count = (end - start) / control_hz / PLAYBACK_RATE * FPS
        if not np.isclose(count, round(count)):
            raise ValueError('Playback timing is not an integer number of video frames')
        segment(index, round(count), 'hold recorded endpoint until next stored frame')
    segment(len(steps) - 1, round(FINAL_HOLD_S * FPS), 'final presentation hold')
    return {'fps': FPS, 'playback_rate_between_stored_frames': PLAYBACK_RATE,
            'initial_hold_s': INITIAL_HOLD_S, 'final_hold_s': FINAL_HOLD_S,
            'stored_steps': steps, 'recorded_times_s': [step / control_hz for step in steps],
            'encoded_frames_per_stored_frame': counts, 'segments': segments,
            'encoded_frame_steps': [step for step, count in zip(steps, counts) for _ in range(count)],
            'video_frame_count': sum(counts), 'duration_s': sum(counts) / FPS,
            'interpolation': 'none; each displayed image is held until the next stored endpoint'}


def draw_frame(category, example, goal_pose, step, frame, limits, last_frame_step):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100, facecolor='#fafbfc')
    geometry = fig.add_axes((0.07, 0.33, 0.40, 0.45))
    camera = fig.add_axes((0.54, 0.33, 0.40, 0.45))
    all_states = np.asarray(example['trace']['states'])
    state_steps = np.asarray(example['trace']['observed_state_steps'])
    states = all_states[state_steps <= step]
    current = all_states[np.flatnonzero(state_steps == step)[0]]
    geometry.add_patch(Rectangle((0, 0), 512, 512, fill=False, edgecolor='#999999',
                                 linestyle=':', label='Arena boundary'))
    hazard = example['layout']['hazard']
    if hazard['kind'] == 'box':
        patch = Rectangle((hazard['x0'], hazard['y0']), hazard['x1'] - hazard['x0'],
                          hazard['y1'] - hazard['y0'])
    else:
        patch = Circle((hazard['cx'], hazard['cy']), hazard['r'])
    patch.set(facecolor='#D55E00', alpha=0.35, edgecolor='#D55E00', label='Virtual hazard')
    geometry.add_patch(patch)
    for index, polygon in enumerate(t_polygons(goal_pose)):
        geometry.add_patch(Polygon(polygon, fill=False, linestyle='--', edgecolor='#009E73',
                                   linewidth=1.5, label='Saved goal' if index == 0 else None))
    geometry.plot(states[:, 0], states[:, 1], color='#E69F00', linewidth=1.6, label='Pusher path')
    geometry.plot(states[:, 2], states[:, 3], color='#0072B2', linewidth=1.6, label='Block body path')
    geometry.add_patch(Circle(current[:2], 15, facecolor='#E69F00', edgecolor='#906100'))
    for polygon in t_polygons(current[2:5]):
        geometry.add_patch(Polygon(polygon, facecolor='#0072B2', alpha=0.35, edgecolor='#005988'))
    geometry.set(xlim=limits[0], ylim=limits[1], aspect='equal', xlabel='x (arena px)',
                 ylabel='y (arena px; down)', title='Recorded geometry (not predictions)')
    geometry.legend(loc='upper right', fontsize=7.5, framealpha=0.93)
    camera.imshow(frame, interpolation='nearest')
    camera.set_title(f"Stored camera: step {step}/{example['trace']['planned_horizon_steps']} | "
                     f't = {step / 10:.1f} s')
    camera.axis('off')
    fig.suptitle('Test example: ' + CATEGORIES[category], fontsize=18, y=0.97)
    fig.text(0.5, 0.907, f"{example['root_id']} | branch {example['row_id']['branch_index']} | "
             f"source episode {example['source_episode']} | {example['row_id']['layout_family']} hazard",
             ha='center', fontsize=11)
    fig.text(0.5, 0.865, 'Saved open-loop experiment. Illustrative case, not a frequency or control result.',
             ha='center', fontsize=10, color='#333333')
    for y, text in zip((0.227, 0.198), decision_text(example).split('   |   '), strict=True):
        fig.text(0.5, y, text, ha='center', fontsize=10)
    fig.text(0.5, 0.150, horizon_text(example), ha='center', fontsize=9.2,
             color='#7B3294' if example['unresolved_future'] else '#333333')
    fig.text(0.5, 0.113, f'Last stored valid camera frame: step {last_frame_step}. '
             'No unobserved or padded image is displayed.', ha='center', fontsize=9.5)
    fig.text(0.5, 0.074, 'Virtual hazard appears only on geometry. Green T is the saved goal. '
             'Decisions and calibrated margins remain fixed.', ha='center', fontsize=9)
    fig.text(0.5, 0.037, '0.5x playback between stored frames; 1 s initial and 2 s final presentation holds. '
             'No interpolation.', ha='center', fontsize=9)
    fig.canvas.draw()
    result = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return result


def write_media(frames, schedule, stem):
    from PIL import Image

    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', f'{WIDTH}x{HEIGHT}', '-framerate', str(FPS),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '1', '-preset', 'medium',
               '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(stem.with_suffix('.mp4'))]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    try:
        for frame, count in zip(frames, schedule['encoded_frames_per_stored_frame'], strict=True):
            data = frame.tobytes()
            for _ in range(count):
                process.stdin.write(data)
        process.stdin.close()
        error = process.stderr.read().decode()
        if process.wait(timeout=90) != 0:
            raise RuntimeError('CPU ffmpeg encoding failed: ' + error)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()
    gifs = [Image.fromarray(frame).resize((960, 600), Image.Resampling.LANCZOS) for frame in frames]
    durations = [round(1000 * count / FPS) for count in schedule['encoded_frames_per_stored_frame']]
    gifs[0].save(stem.with_suffix('.gif'), save_all=True, append_images=gifs[1:],
                 duration=durations, loop=0, disposal=2, optimize=False)
    Image.fromarray(frames[-1]).save(stem.with_suffix('.png'))
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration',
        '-of', 'json', str(stem.with_suffix('.mp4'))], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)['streams'][0]
    if (int(stream['nb_read_frames']) != schedule['video_frame_count']
            or stream['r_frame_rate'] != f'{FPS}/1' or stream['width'] != WIDTH
            or stream['height'] != HEIGHT or not np.isclose(float(stream['duration']), schedule['duration_s'])):
        raise ValueError('Encoded video disagrees with declared presentation timeline')
    with Image.open(stem.with_suffix('.gif')) as gif:
        actual = []
        for index in range(gif.n_frames):
            gif.seek(index)
            actual.append(gif.info['duration'])
        if actual != durations:
            raise ValueError('GIF frame durations differ from the declared timeline')
    return {'ffprobe': stream, 'gif_frame_durations_ms': durations,
            'encoding': {'mp4': 'software libx264, CRF23, yuv420p, one thread',
                         'gif': '960x600 palette presentation; loops', 'audio': 'none'}}


def run(source, output):
    import h5py
    import hdf5plugin  # noqa: F401
    import matplotlib
    from PIL import Image

    matplotlib.use('Agg')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.parent != source.parent:
        raise ValueError('Require a fresh sibling of the immutable E2 report directory')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise ValueError('Installed CPU ffmpeg and ffprobe are required')
    original = inventory(source)
    paired = json.loads((source / 'paired-report/paired_report.json').read_text())
    repair = json.loads((source / 'repair.json').read_text())
    for name, path in [('repair_report', source / 'repair.json'),
                       ('evaluation_rows', source / 'repair_rows.npz')]:
        if sha256(path) != paired['inputs'][name]['sha256']:
            raise ValueError('Pinned saved E2 input changed: ' + name)
    context = selection_context(repair)
    bank = Path(repair['banks_dir']) / 'test'
    bank_hashes = {str((bank / name).resolve()): expected
                   for name, expected in paired['inputs']['bank_identities']['test'].items()}
    bank_manifest = bank / 'manifest.json'
    bank_hashes[str(bank_manifest.resolve())] = sha256(bank_manifest)
    timing_path = bank.parent / 'source_snapshot/experiments/helpers/pushtAssets.py'
    frozen = json.loads(bank_manifest.read_text())['data']['source_snapshot_sha256']
    bank_hashes[str(timing_path.resolve())] = frozen['experiments/helpers/pushtAssets.py']
    if any(sha256(path) != expected for path, expected in bank_hashes.items()):
        raise ValueError('Pinned test bank or frozen timing source changed')
    timing = timing_from_source(timing_path)
    roots = json.loads((bank / 'roots.json').read_text())['roots']
    layouts = json.loads((bank / 'layouts.json').read_text())['layouts']
    generators = [Path(__file__).resolve(), Path(__file__).with_name('pusht_e2_figures.py').resolve(),
                  Path(__file__).resolve().parents[1] / 'helpers/pushtGeometry.py']
    generator_hashes = {str(path): sha256(path) for path in generators}
    prepared = []
    with h5py.File(bank / 'branches.h5', 'r') as h5:
        for category in CATEGORIES_TO_RENDER:
            selected = paired['banks']['test']['examples'][category]
            example = selected['example']
            if example is None:
                prepared.append((category, selected, None, None))
                continue
            root = roots[example['row_id']['root_index']]
            if (example['row_id']['bank'] != 'test' or root['root_id'] != example['root_id']
                    or root['meta']['episode'] != example['source_episode']
                    or Path(example['bank_path']).resolve() != bank.resolve()
                    or example['layout']['root_id'] != root['root_id']
                    or example['layout']['family'] != example['row_id']['layout_family']
                    or example['layout'] not in layouts):
                raise ValueError('Existing selected example identity/geometry does not match its test bank')
            frames = all_valid_saved_frames(example, h5)
            goal = np.asarray(root['goal_state'], dtype=float)[2:5]
            prepared.append((category, selected, frames, goal))
    output.mkdir(parents=True, exist_ok=False)
    samples = output / 'samples'
    samples.mkdir()
    report = {'kind': 'saved-e2-test-animations-v1', 'run_id': paired['run_id'],
              'bank': 'test', 'saved_gate': paired['gate'], 'selection_context': context,
              'timing': timing, 'examples': {}, 'additional_model_or_simulator_calls': 0}
    lines = ['# Saved Push-T E2 test animations', '',
        'These three categories use the examples already selected in the immutable paired report. '
        'No new example selection, predictions, simulator replay or statistics were added.', '',
        '**The E2 repair gate remains false.** These illustrations do not establish improvement, '
        'prevalence or closed-loop control. The development ranking was inconclusive: '
        f"{context['eligible_count']} eligible recipes out of {context['recipe_count']}; "
        'the saved deterministic fallback is retained.', '',
        'Actual camera images were stored every five 0.1 s environment steps. Geometry uses saved '
        'observed states through the displayed camera step. The virtual hazard is drawn only in '
        'the geometry panel; the green T is the saved goal. No predicted trajectory is drawn.', '',
        'Playback is 0.5x between stored endpoints, with a 1 s initial and 2 s final hold. '
        'Held images are presentation repeats, not new observations. MP4 is lossy and GIF is '
        'resized/palette encoded; source frame byte hashes are recorded separately.', '']
    for category, selected, frames, goal in prepared:
        example = selected['example']
        lines.extend(['## ' + CATEGORIES[category], '', selected['definition'] + '.', ''])
        if example is None:
            report['examples'][category] = {'status': 'absent', 'saved_selection': selected}
            lines.append('No example was saved for this category; no substitute is selected.\n')
            continue
        steps = [step for step, _ in frames]
        schedule = presentation_schedule(steps, timing['CONTROL_HZ'])
        limits = plot_limits(example['trace']['states'], example['layout']['hazard'], goal_pose=goal)
        images = [draw_frame(category, example, goal, step, frame, limits, steps[-1])
                  for step, frame in frames]
        stem = output / ('test-' + category)
        encoding = write_media(images, schedule, stem)
        sample_paths = []
        for index in sorted({0, len(frames) // 2, len(frames) - 1}):
            relative = f'samples/test-{category}-step-{steps[index]:03d}.png'
            Image.fromarray(images[index]).save(output / relative)
            sample_paths.append(relative)
        report['examples'][category] = {'status': 'rendered', 'saved_selection': selected,
            'camera_source_frames': [{'step': step, 'dtype': str(frame.dtype), 'shape': list(frame.shape),
                'rgb_bytes_sha256': hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()}
                for step, frame in frames], 'presentation': schedule,
            'last_observed_state_step': max(example['trace']['observed_state_steps']),
            'last_displayed_camera_step': steps[-1], 'saved_goal_pose': goal.tolist(),
            'geometry_limits': {'x': list(limits[0]), 'y': list(limits[1])},
            'files': {key: stem.with_suffix('.' + suffix).name for key, suffix in
                      [('video', 'mp4'), ('gif', 'gif'), ('poster', 'png')]},
            'sampled_frame_files': sample_paths, **encoding}
        lines.extend([f"Root `{example['root_id']}`, branch {example['row_id']['branch_index']}; "
            f"camera steps `{steps}`; {schedule['duration_s']:g} s presentation.", '',
            decision_text(example), '', horizon_text(example), '',
            f"The last camera frame is step {steps[-1]}; saved observed states end at step "
            f"{max(example['trace']['observed_state_steps'])}. No padded suffix is shown.", '',
            f'![Recorded endpoint poster]({stem.name}.png)', '',
            f'[MP4 video]({stem.name}.mp4) | [Looping GIF]({stem.name}.gif)', ''])
    (output / 'README.md').write_text('\n'.join(lines))
    (output / 'animations.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if (inventory(source) != original or any(sha256(p) != h for p, h in bank_hashes.items())
            or any(sha256(p) != h for p, h in generator_hashes.items())):
        raise ValueError('An immutable source or generator changed during media creation')
    manifest = {'kind': report['kind'], 'run_id': paired['run_id'], 'source_sha256': original,
        'bank_inputs_sha256': bank_hashes, 'generator_sha256': generator_hashes,
        'outputs_sha256': {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'original_files_unchanged': len(original), 'gate_unchanged': paired['gate'],
        'new_model_queries': 0, 'new_simulator_steps': 0, 'new_optimizer_updates': 0,
        'limitations': ['Stored open-loop illustrations only, not new selection or effect estimates.',
            'No frames are inferred between block endpoints or after censoring.',
            'Observed state suffixes without corresponding valid saved images are not animated.',
            'Predictions and arm-specific calibrated margins both affect the static decisions.',
            'Presentation encodings resize or compress pixels; raw HDF5 frame hashes identify evidence.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'rendered': sum(v['status'] == 'rendered' for v in report['examples'].values()),
                      'manifest_sha256': sha256(output / 'manifest.json')}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.source_dir, args.output_dir)


if __name__ == '__main__':
    main()
