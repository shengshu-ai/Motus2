"""Compact timestep embeddings with index_select/index_add autograd."""
from typing import NamedTuple,Tuple
import torch

class CompactE0(NamedTuple):
    planes: Tuple[torch.Tensor, ...]   # six [B,3,D] views
    flat_group: torch.Tensor          # [B*L] long, includes 3*b row offsets
    batch: int
    tokens: int

def make_compact_e0(time_embed, action_t, video_t, group):
    if action_t.requires_grad or video_t.requires_grad:
        return None  # caller retains original full differentiable timestep path
    if group.ndim!=2 or group.dtype!=torch.long:
        raise ValueError('Explicit [B,L] int64 producer groups required')
    b,l=group.shape
    if action_t.shape!=(b,) or video_t.shape!=(b,):raise ValueError('One producer t per sample')
    table_t=torch.stack([torch.zeros_like(action_t),action_t,video_t],dim=1)
    _,table=time_embed(table_t)
    if table.ndim!=4 or table.shape[:3]!=(b,3,6):raise ValueError('Expected original [B,3,6,D] projection')
    offsets=torch.arange(b,device=group.device,dtype=torch.long).view(b,1)*3
    ids=(group+offsets).reshape(-1)
    return CompactE0(table.unbind(dim=2),ids,b,l)

def modulation_part(block,e0,index):
    if isinstance(e0,CompactE0):
        plane=e0.planes[index]
        if plane.dtype == torch.float32 and 0 < e0.tokens <= 8192:
            from .training_compact_blocked_vjp import blocked_gather
            part = blocked_gather(plane, e0.flat_group, e0.batch, e0.tokens)
        else:
            part=plane.reshape(e0.batch*3,plane.shape[-1]).index_select(0,e0.flat_group)
            part=part.reshape(e0.batch,e0.tokens,plane.shape[-1])
    else:
        part=e0[index] if isinstance(e0,tuple) else e0[:,:,index,:]
    return block.modulation[:,index,:].unsqueeze(0)+part

def _pointwise_mlp_reason(embedding, projection):
    """Only the original stateless pointwise Linear/SiLU architecture is eligible."""
    from torch import nn
    expected=((embedding,(nn.Linear,nn.SiLU,nn.Linear)),
              (projection,(nn.SiLU,nn.Linear)))
    for sequence,types in expected:
        if type(sequence) is not nn.Sequential or len(sequence)!=len(types):
            return 'non_pointwise_time_mlp'
        if 'forward' in sequence.__dict__:
            return 'custom_time_mlp'
        for layer,kind in zip(sequence,types):
            if type(layer) is not kind or 'forward' in layer.__dict__:
                return 'custom_time_mlp'
            if isinstance(layer,nn.SiLU) and layer.inplace:
                return 'inplace_time_mlp'
        # Hooks may inspect/alter sequence dimensions, so do not assume the
        # original architecture alone makes a shape-changing call safe.
        for module in sequence.modules():
            if any(getattr(module,name,{}) for name in (
                    '_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks')):
                return 'time_mlp_hooks'
    return None

def compact_e0_rejection(model,*,pure_policy,action_first,action_t,video_t,rtc_action_token_t):
    if not model.training:return 'not_training'
    if not torch.is_grad_enabled():return 'no_grad'
    if not pure_policy:return 'mixed_or_non_policy'
    cfg=model.config;wan=model.video_model.wan_model
    if cfg.mode!='joint':return 'non_joint'
    if not action_first:return 'layout'
    if cfg.use_mot:return 'mot'
    if cfg.value_enabled or cfg.value_tokens_per_frame!=0:return 'value'
    if cfg.rtc_training_enabled or rtc_action_token_t is not None:return 'rtc'
    if getattr(cfg,'gradient_checkpointing',False) or wan.gradient_checkpointing:return 'checkpoint'
    if action_t.requires_grad or video_t.requires_grad:return 'learnable_timestep'
    if action_t.ndim!=1 or video_t.shape!=action_t.shape or action_t.numel()==0:return 'timestep_shape'
    return _pointwise_mlp_reason(wan.time_embedding,wan.time_projection)

def record_compact_e0(model,e0,*,reason,batch,tokens):
    """Host-only receipt once per training forward. No activation/value cache."""
    counts=getattr(model,'_training_compact_e0_counts',None)
    if counts is None:
        counts={'compact_forwards':0,'original_forwards':0,'fallback_reasons':{}}
        model._training_compact_e0_counts=counts
    if isinstance(e0,CompactE0):
        counts['compact_forwards']+=1
        counts['last_layout']={'batch':batch,'tokens':tokens,'table_rows':3*batch,
            'table_plane_shape':tuple(e0.planes[0].shape),'table_dtype':str(e0.planes[0].dtype),
            'group_dtype':str(e0.flat_group.dtype),'table_requires_grad':e0.planes[0].requires_grad}
    else:
        counts['original_forwards']+=1
        reason=reason or 'unsupported_geometry'
        counts['fallback_reasons'][reason]=counts['fallback_reasons'].get(reason,0)+1

def compact_e0_summary(model):
    c=getattr(model,'_training_compact_e0_counts',{})
    return {'compact_forwards':c.get('compact_forwards',0),
            'original_forwards':c.get('original_forwards',0),
            'fallback_reasons':dict(c.get('fallback_reasons',{})),
            'last_layout':dict(c.get('last_layout',{}))}
