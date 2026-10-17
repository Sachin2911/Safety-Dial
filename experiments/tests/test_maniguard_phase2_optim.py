"""Check panel isolation, exact resumption, and recurrent episodic gradients."""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_phase2_model import (
    decode,
    forward,
    genome_sha256,
    load_model,
    log_prob,
    random_genome,
    save_model,
    torch_log_prob,
)
from helpers.maniguard_phase2_optim import (
    PPOConfig,
    RecurrentPPO,
    WeightSearch,
    episode_family_mean,
    genome_hash,
)


def round_trip(state):
    return json.loads(json.dumps(state))


def test_search_common_initial_panel_and_resumption_before_and_after_ask():
    evolution = WeightSearch('evolution', 31)
    random = WeightSearch('random', 31)
    assert [c['sha256'] for c in evolution.ask()] == [c['sha256'] for c in random.ask()]
    resumed = WeightSearch.from_state_dict(round_trip(evolution.state_dict()))
    assert resumed.ask() == evolution.ask()
    values = [-2, -1, 0, 1, 0, -1, -2, 0]
    for search in (evolution, resumed):
        search.tell(values)
    assert evolution.state_dict() == resumed.state_dict()
    resumed = WeightSearch.from_state_dict(round_trip(evolution.state_dict()))
    assert evolution.ask() == resumed.ask()
    assert evolution.ask()[0]['sha256'] == evolution.archive[0]['candidates'][3]['sha256']
    assert all(c['origin'] == 'mutation' for c in evolution.ask()[1:])


@pytest.mark.parametrize('method', ['evolution', 'random'])
def test_finalists_use_only_common_last_panel_and_reevaluate_both_earlier_winners(method):
    search = WeightSearch(method, 32)
    first = search.ask()
    search.tell([1, 0, 0, 0, 0, 0, 0, 0])
    second = search.ask()
    search.tell([0, 0.7, 0, 0, 0, 0, 0, 0])
    third = search.ask()
    assert third[0]['sha256'] == first[0]['sha256']
    assert third[1]['sha256'] == second[1]['sha256']
    assert [c['source_round'] for c in third[:2]] == [0, 1]
    search.tell([-2, -2, 0, 0, 0.5, 0, 1, 0])
    finalists = search.finalists()
    assert [c['sha256'] for c in finalists] == [third[6]['sha256'], third[4]['sha256']]
    assert all(c['panel'] == 'panel-2' for c in finalists)
    with pytest.raises(ValueError, match='already completed'):
        search.ask()


def test_duplicate_winners_use_paid_slots_and_are_ranked_by_same_panel_mean():
    search = WeightSearch('random', 33)
    for _ in range(2):
        search.ask()
        search.tell([1, -2, -2, -2, -2, -2, -2, -2])
    third = search.ask()
    assert len(third) == 8 and third[0]['sha256'] == third[1]['sha256']
    search.tell([0, 1, 0.75, -2, -2, -2, -2, -2])
    first, second = search.finalists()
    assert first['sha256'] == third[2]['sha256']
    assert second['panel_mean_fitness'] == 0.5
    assert second['panel_evaluations'] == 2


def test_search_rejects_unaudited_or_tampered_state_and_early_finalists():
    search = WeightSearch('random', 34)
    with pytest.raises(ValueError, match='Call ask'):
        search.tell([0] * 8)
    with pytest.raises(ValueError, match='final common'):
        search.finalists()
    search.ask()
    with pytest.raises(ValueError, match='finite'):
        search.tell([np.nan] * 8)
    state = search.state_dict()
    state['pending'][0]['genome'][0] += 1
    with pytest.raises(ValueError, match='integrity'):
        WeightSearch.from_state_dict(state)


def fixture_batch(learner, prefix='batch', lengths=(2, 3, 4), utilities=(1, 0, -2)):
    rows = []
    rng = np.random.default_rng(701)
    for family, length, utility in zip(learner.config.families, lengths, utilities):
        features = rng.normal(size=(length, 64))
        hidden = None
        choices, logs = [], []
        for feature in features:
            result = learner.sample(feature, hidden)
            hidden = result['hidden']
            choices.append(result['actions'].tolist())
            logs.append(result['old_log_prob'])
        rows.append({'family': family, 'episode_id': f'{prefix}-{family}',
                     'features': features.tolist(), 'actions': choices, 'old_log_probs': logs,
                     'terminal_utility': utility,
                     'actor_sha256': genome_hash(learner.actor.export_genome())})
    return rows


def test_episode_family_weighting_preserves_full_trajectory_score_not_length_average():
    torch = pytest.importorskip('torch')
    parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    # Jar has two episodes with different lengths; each episode has equal weight within
    # Jar, and Jar has equal weight to the other two families. Each score remains a sum.
    episode_scores = [parameter * length for length in (1, 5, 3, 2)]
    result = episode_family_mean(episode_scores, ['jar', 'jar', 'stack', 'clutter'],
                                 ('jar', 'stack', 'clutter'))
    result.backward()
    assert result.item() == pytest.approx((3 + 3 + 2) / 3)
    assert parameter.grad.item() == pytest.approx(8 / 3)
    with pytest.raises(ValueError, match='every declared family'):
        episode_family_mean([1], ['jar'], ('jar', 'stack', 'clutter'))


def test_ppo_undiscounted_actor_loss_uses_episode_sums_with_unequal_lengths():
    pytest.importorskip('torch')
    learner = RecurrentPPO(seed=9, config=PPOConfig(epochs=1))
    rows = fixture_batch(learner, lengths=(1, 3, 5), utilities=(1, 1, -2))
    report = learner.update(rows)
    # Zero critic, ratio one: minus mean of the complete episode return-score sums.
    assert report['epochs'][0]['actor_loss'] == pytest.approx(-(1 + 3 - 10) / 3)
    assert report['max_collection_log_prob_error'] < 1e-12
    assert report['decisions'] == 9
    assert report['episodes_seen'] == 3
    assert report['family_episodes'] == {'jar': 1, 'stack': 1, 'clutter': 1}


def test_ppo_backpropagates_through_recurrent_history_and_resets_each_episode():
    torch = pytest.importorskip('torch')
    learner = RecurrentPPO(random_genome(np.random.default_rng(35)), config=PPOConfig(epochs=1))
    row = learner._trajectory(fixture_batch(learner)[0])
    row['features'].requires_grad_()
    logits, _ = learner._recompute(row)
    last_log = torch_log_prob(logits[-1], row['actions'][-1])
    last_log.backward()
    assert row['features'].grad[0].abs().sum() > 0
    assert learner.actor.W_rec.grad.abs().sum() > 0
    logits_again, _ = learner._recompute(row)
    assert torch.equal(logits, logits_again)


def test_ppo_state_restores_rng_adam_moments_and_next_update_exactly():
    pytest.importorskip('torch')
    learner = RecurrentPPO(seed=36, config=PPOConfig(epochs=2))
    learner.update(fixture_batch(learner, 'first'))
    restored = RecurrentPPO.from_state_dict(round_trip(learner.state_dict()))
    assert learner.state_dict() == restored.state_dict()
    first_batch = fixture_batch(learner, 'second')
    second_batch = fixture_batch(restored, 'second')
    assert first_batch == second_batch
    assert learner.update(first_batch) == restored.update(second_batch)
    assert learner.state_dict() == restored.state_dict()


def test_ppo_rejects_wrong_actor_log_probability_and_reused_episode():
    pytest.importorskip('torch')
    learner = RecurrentPPO(seed=37, config=PPOConfig(epochs=1))
    rows = fixture_batch(learner)
    changed = deepcopy(rows)
    changed[0]['old_log_probs'][0] += 0.1
    before = learner.state_dict()
    with pytest.raises(ValueError, match='log-probability audit'):
        learner.update(changed)
    assert learner.state_dict() == before
    changed = deepcopy(rows)
    changed[0]['actor_sha256'] = 'incorrect'
    with pytest.raises(ValueError, match='another collection actor'):
        learner.update(changed)
    learner.update(rows)
    repeat = fixture_batch(learner)
    with pytest.raises(ValueError, match='submitted twice'):
        learner.update(repeat)


def test_ppo_fixed_48_72_checkpoints_and_no_budget_overrun():
    pytest.importorskip('torch')
    learner = RecurrentPPO(seed=38, config=PPOConfig(epochs=1))
    for index in range(24):
        learner.update(fixture_batch(learner, str(index), lengths=(1, 1, 1)))
    assert set(learner.checkpoints) == {'48', '72'}
    assert learner.checkpoints['72']['sha256'] == genome_hash(learner.actor.export_genome())
    with pytest.raises(ValueError, match='budget'):
        learner.update(fixture_batch(learner, 'too-many', lengths=(1, 1, 1)))
    restored = RecurrentPPO.from_state_dict(round_trip(learner.state_dict()))
    assert restored.checkpoints == learner.checkpoints


@pytest.mark.parametrize('change', [{'gamma': 0.99}, {'epochs': 9},
                                  {'checkpoint_episodes': (24, 72)}])
def test_changed_objective_or_unbounded_learner_requires_new_protocol(change):
    with pytest.raises(ValueError):
        PPOConfig(**change).validate()


def test_saved_model_numpy_collection_hash_and_log_probs_are_accepted_by_ppo(tmp_path):
    """Exercise the real cross-module path, not the PPO sampler's self-consistency."""
    pytest.importorskip('torch')
    genome = random_genome(np.random.default_rng(130))
    checkpoint = tmp_path / 'runtime-actor.npz'
    artifact_hash = save_model(checkpoint, genome, metadata={'collection': 'runtime-style'})
    loaded, spec, _ = load_model(checkpoint)
    canonical = genome_sha256(loaded, spec)
    assert genome_hash(loaded) == canonical
    assert artifact_hash != canonical  # Whole artifact also authenticates run metadata.
    raw_hash = hashlib.sha256(np.asarray(loaded, dtype='<f8').tobytes()).hexdigest()
    assert raw_hash != canonical
    learner = RecurrentPPO(loaded, seed=130, config=PPOConfig(epochs=1))
    assert learner.state_dict()['genome_sha256'] == canonical
    rng = np.random.default_rng(131)
    rows = []
    for family, utility in zip(learner.config.families, (1, 0, -2)):
        features = rng.normal(size=(3, 64))
        hidden = None
        actions, logs = [], []
        for feature in features:
            logits, hidden = forward(loaded, feature, hidden)
            choices = decode(logits, rng)
            actions.append(choices.tolist())
            logs.append(log_prob(logits, choices))
        rows.append({'family': family, 'episode_id': f'runtime-{family}',
                     'features': features.tolist(), 'actions': actions, 'old_log_probs': logs,
                     'terminal_utility': utility, 'actor_sha256': canonical})
    legacy = deepcopy(rows)
    legacy[0]['actor_sha256'] = raw_hash
    with pytest.raises(ValueError, match='another collection actor'):
        learner.update(legacy)
    report = learner.update(rows)
    assert report['max_collection_log_prob_error'] < 1e-12
    assert report['episodes_seen'] == 3
