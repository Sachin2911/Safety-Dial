"""Visual block controller with the previous executed block as observable memory."""
import numpy as np
import torch

from helpers.evoMlpPolicy import MLPBlockPolicy

HISTORY_FEATURE_DIM = 444


class HistoryBlockPolicy(MLPBlockPolicy):
    def __init__(self, *args, use_history=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_history = bool(use_history)
        widths = (HISTORY_FEATURE_DIM, *self.hidden, 60)
        self.layer_shapes = tuple(zip(widths[1:], widths[:-1]))
        self.n_params = sum(o*i+o for o, i in self.layer_shapes)

    def features(self, z_t, z_prev, previous_actions):
        visual = super().features(z_t, z_prev)
        past = torch.as_tensor(previous_actions, dtype=torch.float32, device=visual.device)
        if past.shape != (*visual.shape[:-1], 60):
            raise ValueError('one preceding 60-action block per visual feature required')
        past = ((past.reshape(*past.shape[:-1], 10, 6) - self.action_mean) /
                self.action_std).reshape(*past.shape[:-1], 60)
        return torch.cat([visual, past if self.use_history else torch.zeros_like(past)], -1)

    def act(self, theta, feats):
        theta = torch.as_tensor(theta, dtype=torch.float32, device=feats.device)
        if theta.ndim != 2 or feats.shape[0] != theta.shape[0] or feats.shape[-1] != HISTORY_FEATURE_DIM:
            raise ValueError('expected candidate-major 444-dimensional features')
        x = feats.reshape(feats.shape[0], -1, HISTORY_FEATURE_DIM)
        for j, (w, b) in enumerate(self.unpack(theta)):
            x = torch.einsum('poi,pmi->pmo', w, x) + b[:, None, :]
            if j < len(self.layer_shapes)-1:
                x = x.relu()
        return x.reshape(*feats.shape[:-1], 60).clamp(-1, 1)

    def state(self):
        return {**super().state(), 'policy_kind': 'history_mlp_sequential_block_v1',
                'use_history': np.asarray(self.use_history)}

    @classmethod
    def from_state(cls, state, *, device='cpu'):
        if str(state.get('policy_kind','')) != 'history_mlp_sequential_block_v1':
            raise ValueError('not an action-history controller')
        return cls((state['action_mean'], state['action_std']), state['feature_mean'],
                   state['feature_std'], hidden=state['hidden'],
                   use_history=bool(state['use_history']), device=device)
