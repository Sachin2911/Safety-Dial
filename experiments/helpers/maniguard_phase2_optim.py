"""Finite, resumable shared-controller search and recurrent on-policy learning.

This module does not run simulator episodes or inspect validation/test outcomes. Every
search slot and every PPO trajectory must be backed by a separately audited full episode.
"""

import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from helpers.maniguard_phase2_model import (
    DEFAULT_SPEC,
    SCHEMA,
    genome_sha256,
    identity_genome,
    make_torch_actor,
    random_genome,
    torch_entropy,
    torch_log_prob,
)

VERSION = 'maniguard-phase2-optim-v2'


def genome_hash(genome):
    """Canonical actor identity: model schema/layout plus weights, excluding run metadata.

    Version two replaces the old raw-weight-only digest so collection, search, model
    checkpoints and learner validation refer to the same actor. Old optimizer states
    require an explicit migration rather than silently changing their provenance.
    """
    return genome_sha256(genome, DEFAULT_SPEC)


def record_hash(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SearchConfig:
    version: str = VERSION
    initializer_version: str = SCHEMA
    mutation_version: str = 'fanin-gaussian-v1'
    population: int = 8
    rounds: int = 3
    identity_seed: int = 20261017
    identity_bias: float = 0.25
    mutation_sigma: float = 0.10
    parent_count: int = 2

    def validate(self):
        if self.version != VERSION or self.population != 8 or self.rounds != 3:
            raise ValueError('The frozen pilot uses eight slots and three rounds')
        if self.initializer_version != SCHEMA or self.mutation_version != 'fanin-gaussian-v1':
            raise ValueError('Unknown frozen initializer or mutation version')
        if self.parent_count != 2 or not 0 < self.mutation_sigma < 1:
            raise ValueError('Invalid frozen mutation configuration')
        if self.identity_bias != 0.25 or type(self.identity_seed) is not int:
            raise ValueError('Identity uses the declared seed and shared 0.25 output bias')


class WeightSearch:
    """Search on three common panels without comparing cross-panel fitness.

    Duplicate retained winners consume two separate evaluation slots in the final round.
    Within one panel, repeated genomes are ranked by their mean observed utility; finalists
    are distinct genomes. All fresh children, including exact duplicate mutations, count.
    """

    def __init__(self, method, seed, panels=('panel-0', 'panel-1', 'panel-2'), config=None):
        self.config = config or SearchConfig()
        self.config.validate()
        if method not in ('evolution', 'random'):
            raise ValueError('Weight search supports evolution or random')
        if len(panels) != 3 or len(set(panels)) != 3:
            raise ValueError('Declare three distinct rotating training panel identifiers')
        self.method = method
        self.panels = tuple(panels)
        self.rng = np.random.default_rng(seed)
        self.round_index = 0
        self.pending = None
        self.archive = []
        self.config_sha256 = record_hash({'config': asdict(self.config), 'panels': self.panels,
                                         'spec': asdict(DEFAULT_SPEC)})

    @staticmethod
    def _rank(candidates):
        grouped = {}
        for candidate in candidates:
            entry = grouped.setdefault(candidate['sha256'],
                                       {'candidate': candidate, 'utilities': []})
            entry['utilities'].append(candidate['fitness'])
        ranked = sorted(grouped.values(), key=lambda row: -np.mean(row['utilities']))
        return [deepcopy(row['candidate']) | {
            'panel_mean_fitness': float(np.mean(row['utilities'])),
            'panel_evaluations': len(row['utilities']),
        } for row in ranked]

    def _mutation(self, genome):
        spec = DEFAULT_SPEC
        sizes = (spec.hidden_dim * spec.input_dim, spec.hidden_dim * spec.hidden_dim,
                 spec.hidden_dim, spec.num_heads * spec.num_choices * spec.hidden_dim,
                 spec.num_heads * spec.num_choices)
        scales = (1 / np.sqrt(spec.input_dim), 1 / np.sqrt(spec.hidden_dim), 1.0,
                  1 / np.sqrt(spec.hidden_dim), 1.0)
        deviations = np.concatenate([np.full(size, scale) for size, scale in zip(sizes, scales)])
        return np.asarray(genome) + self.rng.normal(size=sum(sizes)) * deviations * (
            self.config.mutation_sigma)

    def ask(self):
        if self.pending is not None:
            return deepcopy(self.pending)
        if self.round_index >= self.config.rounds:
            raise ValueError('All three search rounds have already completed')
        candidates = []

        def add(genome, origin, parent=None, source_round=None):
            values = np.asarray(genome, dtype=np.float64)
            candidates.append({
                'candidate_id': f'{self.method}-round{self.round_index}-slot{len(candidates)}',
                'panel': self.panels[self.round_index], 'round': self.round_index,
                'genome': values.tolist(), 'sha256': genome_hash(values), 'origin': origin,
                'parent_sha256': parent, 'source_round': source_round,
            })

        if self.round_index == 0:
            add(identity_genome(seed=self.config.identity_seed, bias=self.config.identity_bias),
                'identity')
        elif self.round_index == 1:
            winner = self._rank(self.archive[0]['candidates'])[0]
            add(winner['genome'], 'winner_reevaluation', winner['sha256'], 0)
        else:
            for source_round in (0, 1):
                winner = self._rank(self.archive[source_round]['candidates'])[0]
                add(winner['genome'], 'winner_reevaluation', winner['sha256'], source_round)
        parents = (self._rank(self.archive[-1]['candidates'])[:self.config.parent_count]
                   if self.archive else [])
        while len(candidates) < self.config.population:
            if self.method == 'evolution' and parents:
                parent = parents[int(self.rng.integers(len(parents)))]
                add(self._mutation(parent['genome']), 'mutation', parent['sha256'],
                    self.round_index - 1)
            else:
                add(random_genome(self.rng), 'random_initialization')
        self.pending = candidates
        return deepcopy(candidates)

    def tell(self, fitness, evidence=None):
        if self.pending is None:
            raise ValueError('Call ask before recording measured fitness')
        values = np.asarray(fitness, dtype=float)
        if values.shape != (self.config.population,) or not np.isfinite(values).all():
            raise ValueError('Every paid candidate slot needs finite measured fitness')
        if ((values < -2) | (values > 1)).any():
            raise ValueError('Shared utility must be between -2 and 1')
        candidates = [candidate | {'fitness': float(value)}
                      for candidate, value in zip(self.pending, values)]
        row = {'round': self.round_index, 'panel': self.panels[self.round_index],
               'candidates': candidates, 'evidence': deepcopy(evidence)}
        row['sha256'] = record_hash(row)
        self.archive.append(row)
        self.pending = None
        self.round_index += 1

    def finalists(self, count=2):
        if self.round_index != self.config.rounds or self.pending is not None:
            raise ValueError('Finalists require the final common-panel evaluation')
        ranked = self._rank(self.archive[-1]['candidates'])
        if count != 2 or len(ranked) < count:
            raise ValueError('Nominate exactly two distinct measured genomes')
        return ranked[:count]

    def state_dict(self):
        state = {'version': VERSION, 'kind': 'weight_search', 'method': self.method,
                 'config': asdict(self.config), 'config_sha256': self.config_sha256,
                 'panels': list(self.panels), 'round_index': self.round_index,
                 'rng_state': deepcopy(self.rng.bit_generator.state),
                 'pending': deepcopy(self.pending), 'archive': deepcopy(self.archive)}
        return state | {'sha256': record_hash(state)}

    @classmethod
    def from_state_dict(cls, state):
        data = deepcopy(state)
        digest = data.pop('sha256')
        if digest != record_hash(data) or data['version'] != VERSION:
            raise ValueError('Search state integrity/version mismatch')
        search = cls(data['method'], 0, data['panels'], SearchConfig(**data['config']))
        if search.config_sha256 != data['config_sha256']:
            raise ValueError('Search configuration hash mismatch')
        for round_record in data['archive']:
            raw = {key: value for key, value in round_record.items() if key != 'sha256'}
            if round_record['sha256'] != record_hash(raw):
                raise ValueError('Search archive hash mismatch')
        for candidate in (data['pending'] or []) + [
                candidate for row in data['archive'] for candidate in row['candidates']]:
            if candidate['sha256'] != genome_hash(candidate['genome']):
                raise ValueError('Archived genome hash mismatch')
        search.round_index = data['round_index']
        search.pending = data['pending']
        search.archive = data['archive']
        if search.round_index != len(search.archive) or not 0 <= search.round_index <= 3:
            raise ValueError('Search round ledger is inconsistent')
        search.rng.bit_generator.state = data['rng_state']
        return search


@dataclass(frozen=True)
class PPOConfig:
    version: str = VERSION
    families: tuple = ('jar', 'stack', 'clutter')
    epochs: int = 4
    learning_rate: float = 0.0003
    critic_learning_rate: float = 0.001
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 0.5
    log_prob_tolerance: float = 1e-6
    gamma: float = 1.0
    checkpoint_episodes: tuple = (48, 72)

    def validate(self):
        if self.version != VERSION or self.gamma != 1:
            raise ValueError('PPO uses the versioned undiscounted episodic objective')
        if len(set(self.families)) != 3 or len(self.families) != 3:
            raise ValueError('PPO requires the three declared development families')
        if type(self.epochs) is not int or not 1 <= self.epochs <= 8 or not 0 < self.clip_ratio < 1:
            raise ValueError('Invalid bounded PPO update settings')
        if not 0 < self.learning_rate <= 0.01 or not 0 < self.critic_learning_rate <= 0.01:
            raise ValueError('Invalid PPO learning rates')
        if self.entropy_coefficient != 0:
            raise ValueError('The initial protocol adds no entropy reward to measured utility')
        if tuple(self.checkpoint_episodes) != (48, 72):
            raise ValueError('PPO finalists are fixed at 48 and 72 training episodes')
        if self.max_grad_norm <= 0 or self.log_prob_tolerance <= 0:
            raise ValueError('Gradient and audit bounds must be positive')


def episode_family_mean(values, families, expected_families):
    """Average scalar episode contributions within families, then over families.

    The actor must supply a SUM of decision-level score contributions for each episode.
    Dividing that sum by trajectory length would change the terminal-utility objective.
    """
    if len(values) != len(families) or set(families) != set(expected_families):
        raise ValueError('A PPO batch must represent every declared family')
    groups = defaultdict(list)
    for value, family in zip(values, families):
        groups[family].append(value)
    return sum(sum(groups[family]) / len(groups[family]) for family in expected_families) / len(
        expected_families)


def _encode_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {'tensor': value.detach().cpu().tolist(), 'dtype': str(value.dtype).split('.')[-1]}
    if isinstance(value, dict):
        return {'mapping': [[_encode_tree(k), _encode_tree(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {'sequence': [_encode_tree(v) for v in value], 'tuple': isinstance(value, tuple)}
    return value


def _decode_tree(value):
    import torch
    if not isinstance(value, dict):
        return value
    if 'tensor' in value:
        return torch.tensor(value['tensor'], dtype=getattr(torch, value['dtype']))
    if 'mapping' in value:
        return {_decode_tree(k): _decode_tree(v) for k, v in value['mapping']}
    sequence = [_decode_tree(v) for v in value['sequence']]
    return tuple(sequence) if value['tuple'] else sequence


class RecurrentPPO:
    """Full-trajectory PPO with a nonprivileged recurrent-feature value baseline.

    Collection must freeze this actor for the entire submitted batch. Each trajectory
    records features, chosen head actions, old joint log probabilities and terminal
    utility. The critic sees current features and the actor's pre-action recurrent
    representation, detached from actor gradients. No monitor labels enter either actor
    or critic. Advantages are undiscounted Monte Carlo return minus the collection-state
    critic prediction, fixed across update epochs; there is no pooled normalization.
    """

    def __init__(self, initial_genome=None, seed=0, config=None):
        import torch
        self.config = config or PPOConfig()
        self.config.validate()
        genome = identity_genome() if initial_genome is None else initial_genome
        self.actor = make_torch_actor(genome, device='cpu', dtype=torch.float64)
        self.critic = torch.nn.Linear(DEFAULT_SPEC.input_dim + DEFAULT_SPEC.hidden_dim, 1,
                                      dtype=torch.float64)
        with torch.no_grad():
            self.critic.weight.zero_()
            self.critic.bias.zero_()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.config.critic_learning_rate)
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self.episodes_seen = 0
        self.updates = 0
        self.episode_ids = set()
        self.checkpoints = {}
        self.config_sha256 = record_hash(asdict(self.config))

    def sample(self, features, hidden=None):
        import torch
        with torch.no_grad():
            feature = torch.as_tensor(features, dtype=torch.float64)
            logits, next_hidden = self.actor(feature, hidden)
            choices = torch.multinomial(logits.softmax(-1), 1, generator=self.rng).squeeze(-1)
            return {'actions': choices.cpu().numpy(),
                    'old_log_prob': float(torch_log_prob(logits, choices)),
                    'hidden': next_hidden.detach(),
                    'actor_sha256': genome_hash(self.actor.export_genome())}

    def _trajectory(self, row):
        import torch
        features = torch.as_tensor(row['features'], dtype=torch.float64)
        raw_actions = np.asarray(row['actions'])
        actions = torch.as_tensor(raw_actions, dtype=torch.long)
        old = torch.as_tensor(row['old_log_probs'], dtype=torch.float64)
        length = len(features)
        if (length == 0 or features.shape != (length, DEFAULT_SPEC.input_dim)
                or actions.shape != (length, DEFAULT_SPEC.num_heads) or old.shape != (length,)):
            raise ValueError('Malformed full recurrent trajectory')
        if (not torch.isfinite(features).all() or not torch.isfinite(old).all()
                or not np.isfinite(raw_actions).all()
                or not np.equal(raw_actions, actions.numpy()).all()
                or (actions < 0).any() or (actions >= DEFAULT_SPEC.num_choices).any()):
            raise ValueError('Nonfinite feature/log probability or invalid categorical action')
        if row['family'] not in self.config.families or row['terminal_utility'] not in (-2, 0, 1):
            raise ValueError('Unexpected family or measured terminal utility')
        if row.get('actor_sha256', genome_hash(self.actor.export_genome())) != genome_hash(
                self.actor.export_genome()):
            raise ValueError('Trajectory came from another collection actor')
        return {'features': features, 'actions': actions, 'old_log_probs': old,
                'family': row['family'], 'terminal_utility': float(row['terminal_utility']),
                'episode_id': str(row['episode_id'])}

    def _recompute(self, trajectory):
        import torch
        hidden = self.actor.initial_state()
        logits, values = [], []
        for features in trajectory['features']:
            step_logits, hidden = self.actor(features, hidden)
            logits.append(step_logits)
            values.append(self.critic(torch.cat((features, hidden.detach()))).squeeze(-1))
        return torch.stack(logits), torch.stack(values)

    def update(self, trajectories):
        import torch
        rows = [self._trajectory(row) for row in trajectories]
        if not rows:
            raise ValueError('An update requires complete episodes')
        ids = [row['episode_id'] for row in rows]
        if len(set(ids)) != len(ids) or self.episode_ids.intersection(ids):
            raise ValueError('An on-policy episode cannot be submitted twice')
        families = [row['family'] for row in rows]
        episode_family_mean([0.0] * len(rows), families, self.config.families)
        stop = self.episodes_seen + len(rows)
        if stop > 72 or any(self.episodes_seen < mark < stop
                            for mark in self.config.checkpoint_episodes):
            raise ValueError('Updates must preserve exact 48/72 episode checkpoints and budget')
        max_log_prob_error = 0.0
        advantages = []
        with torch.no_grad():
            for row in rows:
                logits, values = self._recompute(row)
                error = float((torch_log_prob(logits, row['actions']) -
                               row['old_log_probs']).abs().max())
                max_log_prob_error = max(max_log_prob_error, error)
                if error > self.config.log_prob_tolerance:
                    raise ValueError('Collection log-probability audit failed; actor or memory differs')
                advantages.append(row['terminal_utility'] - values)
        epoch_records = []
        for _ in range(self.config.epochs):
            policy_terms, value_terms, entropies, kls, clip_fractions = [], [], [], [], []
            for row, advantage in zip(rows, advantages):
                logits, values = self._recompute(row)
                log_probs = torch_log_prob(logits, row['actions'])
                log_ratio = log_probs - row['old_log_probs']
                ratio = log_ratio.exp()
                unclipped = ratio * advantage
                clipped = ratio.clamp(1 - self.config.clip_ratio,
                                      1 + self.config.clip_ratio) * advantage
                policy_terms.append(-torch.minimum(unclipped, clipped).sum())
                # Value fitting gives each episode equal influence; unlike policy score
                # contributions this is a prediction regression, not the utility objective.
                value_terms.append(((values - row['terminal_utility']) ** 2).mean())
                entropies.append(torch_entropy(logits).mean())
                kls.append(((ratio - 1) - log_ratio).mean())
                clip_fractions.append(((ratio - 1).abs() > self.config.clip_ratio).double().mean())
            actor_loss = episode_family_mean(policy_terms, families, self.config.families)
            critic_loss = episode_family_mean(value_terms, families, self.config.families)
            if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
                raise ValueError('Nonfinite PPO objective')
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            critic_loss.backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        self.config.max_grad_norm,
                                                        error_if_nonfinite=True)
            critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                                         self.config.max_grad_norm,
                                                         error_if_nonfinite=True)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            epoch_records.append({
                'actor_loss': float(actor_loss.detach()), 'critic_loss': float(critic_loss.detach()),
                'actor_gradient_norm': float(actor_norm), 'critic_gradient_norm': float(critic_norm),
                'entropy': float(episode_family_mean(entropies, families, self.config.families)),
                'approx_kl': float(episode_family_mean(kls, families, self.config.families)),
                'clip_fraction': float(episode_family_mean(clip_fractions, families,
                                                           self.config.families)),
            })
        self.episodes_seen = stop
        self.episode_ids.update(ids)
        self.updates += 1
        if stop in self.config.checkpoint_episodes:
            genome = self.actor.export_genome()
            self.checkpoints[str(stop)] = {'episodes': stop, 'genome': genome.tolist(),
                                           'sha256': genome_hash(genome)}
        return {'episodes': len(rows), 'episodes_seen': stop, 'updates': self.updates,
                'decisions': sum(len(row['features']) for row in rows),
                'family_episodes': {f: families.count(f) for f in self.config.families},
                'max_collection_log_prob_error': max_log_prob_error, 'epochs': epoch_records,
                'actor_sha256': genome_hash(self.actor.export_genome())}

    def state_dict(self):
        genome = self.actor.export_genome()
        state = {'version': VERSION, 'kind': 'recurrent_ppo', 'config': asdict(self.config),
                 'config_sha256': self.config_sha256, 'genome': genome.tolist(),
                 'genome_sha256': genome_hash(genome),
                 'critic': _encode_tree(self.critic.state_dict()),
                 'actor_optimizer': _encode_tree(self.actor_optimizer.state_dict()),
                 'critic_optimizer': _encode_tree(self.critic_optimizer.state_dict()),
                 'rng_state': self.rng.get_state().tolist(), 'episodes_seen': self.episodes_seen,
                 'updates': self.updates, 'episode_ids': sorted(self.episode_ids),
                 'checkpoints': deepcopy(self.checkpoints)}
        return state | {'sha256': record_hash(state)}

    @classmethod
    def from_state_dict(cls, state):
        import torch
        data = deepcopy(state)
        digest = data.pop('sha256')
        if digest != record_hash(data) or data['version'] != VERSION:
            raise ValueError('PPO state integrity/version mismatch')
        config = data['config'] | {'families': tuple(data['config']['families']),
                                  'checkpoint_episodes': tuple(data['config']['checkpoint_episodes'])}
        learner = cls(data['genome'], config=PPOConfig(**config))
        if learner.config_sha256 != data['config_sha256'] or genome_hash(
                data['genome']) != data['genome_sha256']:
            raise ValueError('PPO configuration/genome hash mismatch')
        learner.critic.load_state_dict(_decode_tree(data['critic']))
        learner.actor_optimizer.load_state_dict(_decode_tree(data['actor_optimizer']))
        learner.critic_optimizer.load_state_dict(_decode_tree(data['critic_optimizer']))
        learner.rng.set_state(torch.tensor(data['rng_state'], dtype=torch.uint8))
        learner.episodes_seen = data['episodes_seen']
        learner.updates = data['updates']
        learner.episode_ids = set(data['episode_ids'])
        learner.checkpoints = data['checkpoints']
        if len(learner.episode_ids) != learner.episodes_seen or not 0 <= learner.episodes_seen <= 72:
            raise ValueError('PPO episode ledger is inconsistent')
        for checkpoint in learner.checkpoints.values():
            if checkpoint['sha256'] != genome_hash(checkpoint['genome']):
                raise ValueError('PPO checkpoint genome hash mismatch')
        return learner
