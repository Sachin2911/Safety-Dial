"""Use the three visual frames and two action blocks already supplied to the world model."""
import numpy as np
import torch
from helpers.evoHistoryPolicy import HistoryBlockPolicy

CONTEXT_FEATURE_DIM=696


class ContextBlockPolicy(HistoryBlockPolicy):
    def __init__(self,*args,use_context=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.use_context=bool(use_context)
        widths=(CONTEXT_FEATURE_DIM,*self.hidden,60)
        self.layer_shapes=tuple(zip(widths[1:],widths[:-1]))
        self.n_params=sum(o*i+o for o,i in self.layer_shapes)

    def features(self,z_t,z_prev,z_older,previous_actions):
        current=(torch.cat([z_t,z_t-z_prev],-1).float()-self.feature_mean)/self.feature_std
        older=((z_prev-z_older).float()-self.feature_mean[192:])/self.feature_std[192:]
        past=torch.as_tensor(previous_actions,dtype=torch.float32,device=current.device)
        if past.shape!=(*current.shape[:-1],120):
            raise ValueError('two preceding action blocks required')
        past=((past.reshape(*past.shape[:-1],20,6)-self.action_mean)/self.action_std).reshape(*past.shape[:-1],120)
        if not self.use_context:
            older=torch.zeros_like(older)
            past=torch.cat([torch.zeros_like(past[...,:60]),past[...,60:]],-1)
        return torch.cat([current,older,past],-1)

    def act(self,theta,feats):
        theta=torch.as_tensor(theta,dtype=torch.float32,device=feats.device)
        if theta.ndim!=2 or feats.shape[0]!=theta.shape[0] or feats.shape[-1]!=CONTEXT_FEATURE_DIM:
            raise ValueError('expected candidate-major 696-dimensional features')
        x=feats.reshape(feats.shape[0],-1,CONTEXT_FEATURE_DIM)
        for j,(w,b) in enumerate(self.unpack(theta)):
            x=torch.einsum('poi,pmi->pmo',w,x)+b[:,None,:]
            if j<len(self.layer_shapes)-1:
                x=x.relu()
        return x.reshape(*feats.shape[:-1],60).clamp(-1,1)

    def state(self):
        return {**super().state(),'policy_kind':'context_mlp_sequential_block_v1',
                'use_context':np.asarray(self.use_context)}

    @classmethod
    def from_state(cls,state,*,device='cpu'):
        if str(state.get('policy_kind',''))!='context_mlp_sequential_block_v1':
            raise ValueError('not a full-context block policy')
        return cls((state['action_mean'],state['action_std']),state['feature_mean'],state['feature_std'],
                   hidden=state['hidden'],use_context=bool(state['use_context']),device=device)


def context_training_features(visual,past,episode,*,use_context):
    """Cached rows are spaced five steps; row i-2 supplies the previous block's context."""
    visual,past,episode=np.asarray(visual),np.asarray(past),np.asarray(episode)
    if visual.shape!=(len(episode),384) or past.shape!=(len(episode),60):
        raise ValueError('misaligned visual and past-action caches')
    valid=np.zeros(len(episode),bool)
    valid[2:]=episode[2:]==episode[:-2]
    older_visual=np.zeros((len(episode),192),dtype=np.float32)
    older_actions=np.zeros_like(past)
    if use_context:
        rows=np.flatnonzero(valid)
        older_visual[rows]=visual[rows-2,192:]
        older_actions[rows]=past[rows-2]
    return np.concatenate([visual,older_visual,older_actions,past],axis=1),valid
