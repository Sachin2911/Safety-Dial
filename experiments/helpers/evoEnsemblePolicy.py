"""Uniform functional ensemble sharing visual inputs and the executed past actions."""
import numpy as np
import torch

from helpers.evoHistoryPolicy import HistoryBlockPolicy


class EnsembleHistoryPolicy:
    def __init__(self, member, n_members):
        if int(n_members) != n_members or n_members < 1:
            raise ValueError('positive integer member count required')
        self.member, self.n_members = member, int(n_members)
        self.n_params = member.n_params * self.n_members
        self.device = member.device
        self.action_mean, self.action_std = member.action_mean, member.action_std

    def features(self, z_t, z_prev, previous_actions):
        return self.member.features(z_t, z_prev, previous_actions)

    def act(self, theta, feats):
        theta = torch.as_tensor(theta, dtype=torch.float32, device=feats.device)
        if theta.ndim != 2 or theta.shape[-1] != self.n_params:
            raise ValueError('expected candidate-major concatenated member parameters')
        members = theta.reshape(theta.shape[0], self.n_members, self.member.n_params)
        # One shared past action block is supplied by the executor, not each member's proposal.
        return torch.stack([self.member.act(members[:, j], feats) for j in range(self.n_members)]).mean(0)

    def model_block(self, action):
        return self.member.model_block(action)

    def state(self):
        return {**self.member.state(), 'policy_kind': 'ensemble_history_sequential_block_v1',
                'n_members': np.asarray(self.n_members),
                'aggregation': 'arithmetic_mean_of_clipped_member_actions'}

    @classmethod
    def from_state(cls, state, *, device='cpu'):
        if str(state.get('policy_kind', '')) != 'ensemble_history_sequential_block_v1':
            raise ValueError('not a history ensemble policy')
        member_state = dict(state, policy_kind='history_mlp_sequential_block_v1')
        return cls(HistoryBlockPolicy.from_state(member_state, device=device), int(state['n_members']))
