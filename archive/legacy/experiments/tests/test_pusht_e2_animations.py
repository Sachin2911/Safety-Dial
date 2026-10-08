import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/pusht_e2_animations.py'
spec = importlib.util.spec_from_file_location('e2_animations_test', path)
animation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(animation)


def fixture(last=19):
    observed = np.arange(26) <= last
    states = np.arange(26 * 7, dtype=float).reshape(26, 7)
    frames = np.stack([np.full((224, 224, 3), step, dtype=np.uint8)
                       for step in range(0, 26, 5)])
    h5 = {'root_index': np.array([2]), 'observed': observed[None],
          'observation_valid': observed[None].copy(), 'states': states[None],
          'frames': frames[None], 'tape': np.zeros((1, 5, 5, 2), dtype=np.float32)}
    example = {'row_id': {'branch_index': 0, 'root_index': 2},
               'trace': {'observed_state_steps': list(range(last + 1)),
                         'states': states[observed].tolist(),
                         'observation_valid': observed[observed].tolist(),
                         'frame_steps_used': [0, last // 5 * 5], 'planned_horizon_steps': 25}}
    return example, h5


def test_censored_example_uses_only_observed_valid_endpoint_images_without_mutation():
    example, h5 = fixture()
    original = copy.deepcopy(example)
    frames = animation.all_valid_saved_frames(example, h5)
    assert [step for step, _ in frames] == [0, 5, 10, 15]
    assert all(np.all(image == step) for step, image in frames)
    assert example == original
    assert all(step <= 19 for step, _ in frames)


def test_full_example_includes_all_six_real_stored_endpoints():
    example, h5 = fixture(25)
    assert [step for step, _ in animation.all_valid_saved_frames(example, h5)] == list(range(0, 26, 5))


def test_invalid_observed_endpoint_is_not_displayed():
    example, h5 = fixture()
    h5['observation_valid'][0, 10] = False
    example['trace']['observation_valid'][10] = False
    assert [step for step, _ in animation.all_valid_saved_frames(example, h5)] == [0, 5, 15]


@pytest.mark.parametrize('change', ['state', 'root', 'validity', 'horizon', 'gap'])
def test_changed_source_identity_or_masks_fail_closed(change):
    example, h5 = fixture()
    if change == 'state':
        h5['states'][0, 2, 0] += 1
    elif change == 'root':
        h5['root_index'][0] = 3
    elif change == 'validity':
        h5['observation_valid'][0, 2] = False
    elif change == 'horizon':
        example['trace']['planned_horizon_steps'] = 30
    else:
        h5['observed'][0, 2] = False
        example['trace']['observed_state_steps'].remove(2)
        example['trace']['states'].pop(2)
        example['trace']['observation_valid'].pop(2)
    with pytest.raises(ValueError):
        animation.all_valid_saved_frames(example, h5)


@pytest.mark.parametrize('steps,total', [([0, 5, 10, 15, 20, 25], 80), ([0, 5, 10, 15], 60)])
def test_playback_schedule_explicitly_accounts_for_slowdown_and_presentation_holds(steps, total):
    schedule = animation.presentation_schedule(steps)
    assert schedule['video_frame_count'] == total
    assert len(schedule['encoded_frame_steps']) == total
    assert schedule['duration_s'] == total / 10
    assert schedule['encoded_frames_per_stored_frame'][0] == 20
    assert schedule['encoded_frames_per_stored_frame'][-1] == 20
    assert set(schedule['encoded_frame_steps']) == set(steps)
    assert schedule['recorded_times_s'][-1] == steps[-1] / 10
    assert schedule['segments'][0]['kind'] == 'initial presentation hold'
    assert schedule['segments'][-1]['kind'] == 'final presentation hold'


@pytest.mark.parametrize('steps', [[], [5, 0], [0, 5, 5], [-5, 0], [0, 1.5]])
def test_schedule_refuses_ambiguous_frame_sequences(steps):
    with pytest.raises(ValueError):
        animation.presentation_schedule(steps)


def test_frozen_timing_constants_are_parsed_without_executing_source(tmp_path):
    source = tmp_path / 'timing.py'
    source.write_text('raise RuntimeError("must not execute")\nCONTROL_HZ=10\nACTION_BLOCK=5\n'
                      'PHYSICS_DT=0.01\nSUBSTEPS_PER_STEP=10\n')
    assert animation.timing_from_source(source)['CONTROL_HZ'] == 10
    source.write_text(source.read_text().replace('CONTROL_HZ=10', 'CONTROL_HZ=20'))
    with pytest.raises(ValueError, match='10-Hz'):
        animation.timing_from_source(source)
