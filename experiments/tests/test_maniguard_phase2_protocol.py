"""Verify actual schedule arithmetic, exposure isolation and readiness enforcement."""

import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_phase2_model import genome_sha256
from helpers.maniguard_phase2_optim import WeightSearch, record_hash
from helpers.maniguard_phase2_protocol import (
    FAMILIES,
    METHODS,
    SEARCH_SEEDS,
    build_pilot,
    compile_schedule,
    pilot_report,
    resource_envelope,
    validate_protocol,
)


def resign(protocol):
    protocol['sha256'] = record_hash({k: v for k, v in protocol.items() if k != 'sha256'})
    return protocol


def test_compiled_schedule_has_exact_counts_and_cost_envelope():
    protocol = build_pilot()
    jobs = compile_schedule(protocol)
    assert len(jobs) == 522
    assert Counter(job['stage'] for job in jobs) == {
        'training': 432, 'finalist_validation': 72, 'reference_validation': 18}
    assert len({job['job_id'] for job in jobs}) == 522
    assert {job['family'] for job in jobs} == set(FAMILIES)
    assert {job['split'] for job in jobs} == {'train', 'validation'}
    resources = resource_envelope(jobs)
    assert resources['scheduled_policy_actions_max'] == 1270200
    assert resources['preparation_policy_actions_max'] == 100800
    assert resources['short_policy_actions_max'] == 384
    assert resources['policy_actions_max'] == 1371384
    assert resources['control_physics_ticks_max'] == 8228304
    assert resources['normal_interval_successful_queries_max'] == 342846
    assert resources['all_full_episodes_max'] == 546
    assert resources['gpu_hours_status'] == 'provisional_until_preparation_costing'


def test_every_method_gets_identical_scene_seed_exposure_on_each_panel():
    protocol = build_pilot()
    jobs = compile_schedule(protocol)
    for seed in SEARCH_SEEDS:
        for panel_index, panel in enumerate(protocol['training_panels']):
            assert all(case['policy_seed'] == panel_index for case in panel['cases'])
            exposures = []
            for method in METHODS:
                rows = [job for job in jobs if job['stage'] == 'training' and
                        job['optimizer_seed'] == seed and job['method'] == method and
                        job['round'] == panel_index]
                assert len(rows) == 24
                exposures.append(Counter((j['family'], j['scene'], j['seed']) for j in rows))
            assert exposures[0] == exposures[1] == exposures[2]
            assert list(exposures[0].values()) == [8, 8, 8]
    for family in FAMILIES:
        scene_ids = [next(c['scene'] for c in p['cases'] if c['family'] == family)
                     for p in protocol['training_panels']]
        assert len(set(scene_ids)) == 3


def test_ppo_order_balances_each_update_and_nomination_is_48_72():
    jobs = compile_schedule(build_pilot())
    for seed in SEARCH_SEEDS:
        rows = [job for job in jobs if job['stage'] == 'training' and
                job['method'] == 'ppo' and job['optimizer_seed'] == seed]
        assert [job['ppo_episode'] for job in rows] == list(range(1, 73))
        for start in range(0, 72, 3):
            batch = rows[start:start + 3]
            assert [job['family'] for job in batch] == list(FAMILIES)
            assert len({job['ppo_batch'] for job in batch}) == 1
            assert len({job['controller_ref'] for job in batch}) == 1
        references = {job['controller_ref'] for job in jobs if
                      job['stage'] == 'finalist_validation' and job['method'] == 'ppo' and
                      job['optimizer_seed'] == seed}
        assert references == {f'ppo/s{seed}/episodes48', f'ppo/s{seed}/episodes72'}


def test_initial_candidate_hashes_match_both_actual_search_methods():
    protocol = build_pilot()
    for seed in SEARCH_SEEDS:
        wanted = protocol['initialization']['shared_first_panel_sha256'][str(seed)]
        for method in ('evolution', 'random'):
            candidates = WeightSearch(method, seed).ask()
            assert [row['sha256'] for row in candidates] == wanted
            assert [genome_sha256(row['genome']) for row in candidates] == wanted


def test_validation_cases_identical_for_every_finalist_and_reference():
    jobs = compile_schedule(build_pilot())
    groups = {}
    for job in jobs:
        if job['split'] == 'validation':
            groups.setdefault(job['controller_ref'], []).append(
                (job['family'], job['scene'], job['seed']))
    assert len(groups) == 15  # twelve finalists plus three references
    first = next(iter(groups.values()))
    assert len(first) == 6 and all(seed == 10 for _, _, seed in first)
    assert all(cases == first for cases in groups.values())


def test_read_only_report_explicitly_refuses_unfinished_freeze():
    protocol = build_pilot()
    report = pilot_report(protocol)
    assert report['ready_to_run'] is False
    assert report['simulator_executed'] is False
    assert report['held_out_outcomes_loaded'] is False
    assert 'measured_costing' in report['unresolved_freeze_checks']
    with pytest.raises(ValueError, match='freeze incomplete'):
        validate_protocol(protocol, require_frozen=True)
    protocol['status'] = 'frozen'
    resign(protocol)
    with pytest.raises(ValueError, match='unresolved requirements'):
        validate_protocol(protocol)


def test_transfer_exposure_reserved_scenes_and_shortened_horizons_rejected():
    original = build_pilot()
    changed = deepcopy(original)
    changed['training_panels'][0]['cases'][0]['family'] = 'cabinet'
    with pytest.raises(ValueError, match='training_panels'):
        validate_protocol(resign(changed))
    changed = deepcopy(original)
    changed['training_panels'][0]['cases'][0]['scene'] = 'task_0023/base'
    with pytest.raises(ValueError, match='training_panels'):
        validate_protocol(resign(changed))
    changed = deepcopy(original)
    changed['training_panels'][0]['cases'][0]['max_steps'] = 32
    with pytest.raises(ValueError, match='training_panels'):
        validate_protocol(resign(changed))


def test_cost_override_and_unsubstantiated_readiness_are_rejected():
    protocol = build_pilot()
    changed = deepcopy(protocol)
    changed['resource_envelope']['gpu_hours_ceiling'] = 720
    with pytest.raises(ValueError, match='Resource bound changed'):
        validate_protocol(resign(changed))
    changed = deepcopy(protocol)
    changed['freeze_checks']['feature_contract']['passed'] = True
    with pytest.raises(ValueError, match='hashed evidence'):
        validate_protocol(resign(changed))


def test_changed_inputs_or_source_pins_fail_before_execution():
    protocol = build_pilot()
    path = next(iter(protocol['input_files']))
    protocol['input_files'][path] = '0' * 64
    with pytest.raises(ValueError, match='input/source changed'):
        validate_protocol(resign(protocol))
    protocol = build_pilot()
    protocol['optimizer_seeds'] = [17, 30]
    with pytest.raises(ValueError, match='optimizer_seeds'):
        validate_protocol(resign(protocol))
