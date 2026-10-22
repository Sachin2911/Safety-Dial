"""Render a matched recorded comparison with explicit terminal-frame holds."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unisafe_adaptation_evaluate import matched_prefix


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--candidate-label', default='Selected task adapter')
    parser.add_argument('--selection-description', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    prefix = matched_prefix(args.baseline, args.candidate)
    assert prefix['pass'], 'Only matched observed prefixes can be labelled as such'
    recordings = []
    artifacts = {}
    for directory in (args.baseline, args.candidate):
        rows = [json.loads(s) for s in (directory / 'steps.jsonl').read_text().splitlines()]
        cameras = np.load(directory / 'observations.npz')['cameras']
        summary = json.loads((directory / 'summary.json').read_text())
        assert len(cameras) == len(rows) + 1 == summary['steps'] + 1
        recordings.append((cameras, rows, summary))
        for filename in ('summary.json', 'steps.jsonl', 'observations.npz'):
            artifacts[str(directory / filename)] = digest(directory / filename)
    dt = recordings[0][1][0]['simulation_time_s']
    assert dt == recordings[1][1][0]['simulation_time_s']
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title = ImageFont.truetype(font_path, 27)
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 18)
    width, height = 1120, 820
    frame_count = max(len(item[0]) for item in recordings)
    labels = ('Frozen task policy', args.candidate_label)

    def frame(index):
        canvas = Image.new('RGB', (width, height), '#101722')
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 10), 'Recorded comparison | same frozen safety filter', font=title, fill='white')
        draw.text((20, 48), f'Playback time {index * dt:.2f} s | fixed front cameras', font=font, fill='#cbd5e1')
        for side, (cameras, rows, summary) in enumerate(recordings):
            left = 20 + side * 560
            current = min(index, len(cameras) - 1)
            ended = index >= len(cameras) - 1
            color = '#8fe3ad' if summary['safe_success'] else '#ffbc85'
            draw.text((left, 88), labels[side], font=font, fill='white')
            text = (f"{summary['outcome'].upper()} at {current * dt:.2f}s; frame held"
                    if ended else f'Simulation: {current * dt:.2f} s')
            draw.text((left, 121), text, font=small, fill=color if ended else '#cbd5e1')
            canvas.paste(Image.fromarray(cameras[current, 0]).resize((520, 520), Image.Resampling.NEAREST), (left, 155))
            if current:
                row = rows[current - 1]
                raw = 'OPEN' if row['task_proposed_action'][-1] >= 0 else 'CLOSE'
                executed = 'OPEN' if row['executed_normalized_action'][-1] >= 0 else 'CLOSE'
                draw.text((left, 688), f'Gripper: task {raw}, executed {executed}', font=small, fill='white')
                draw.text((left, 716), f"Filter replacement: {int(row['intervened'])} | delay override: {int(row['adapter_gripper_override'])}", font=small, fill='white')
                draw.text((left, 744), f"Block separation {100 * row['after']['middle_base_xy_distance_m']:.2f} cm | height {100 * row['after']['cube_positions'][1][2]:.2f} cm", font=small, fill='white')
        draw.text((20, 785), 'Terminal frames are held after termination. No later stability was simulated.', font=small, fill='#cbd5e1')
        return canvas

    movie = args.output / 'comparison.mp4'
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(1 / dt), '-i', '-',
        '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'medium', '-crf', '19',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(movie),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for i in range(frame_count):
            encoder.stdin.write(frame(i).tobytes())
        encoder.stdin.close()
        assert encoder.wait() == 0, 'Video encoding failed'
    first = prefix['first_executed_action_difference_step']
    preview_index = min((first + 1) if first is not None else frame_count - 1, frame_count - 1)
    frame(preview_index).save(args.output / 'first_change.png')
    frame(frame_count - 1).save(args.output / 'terminal_comparison.png')
    result = {
        'reference': str(args.baseline), 'candidate': str(args.candidate),
        'candidate_label': args.candidate_label, 'selection': args.selection_description,
        'prefix_audit': prefix, 'frames': frame_count, 'fps': 1 / dt,
        'movie_duration_seconds_including_reset_frame': frame_count * dt,
        'environment_steps': 0, 'model_queries': 0, 'training_updates': 0,
        'elapsed_seconds': time.perf_counter() - started,
        'source_artifact_sha256': artifacts, 'movie_sha256': digest(movie),
        'source': 'Recorded front RGB frames, nearest-neighbor enlargement and synchronized labels; no new simulation.',
        'limits': 'Individual illustration selected by the stated rule; frequencies come from the full outcome table. Terminal frames are held and labelled. No arbitrary snapshot restoration or post-termination stability claim.',
    }
    (args.output / 'provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(movie)


if __name__ == '__main__':
    main()
