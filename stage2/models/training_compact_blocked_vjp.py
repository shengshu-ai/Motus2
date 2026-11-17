"""Gather forward and two-stage Triton gradient reduction.
"""
import os
import torch
from torch.library import triton_op,wrap_triton
from .training_compact_blocked_kernels import partial_reduce,merge_partial

@triton_op('motus_compact_blocked::reduce',mutates_args={})
def blocked_reduce(grad:torch.Tensor,ids:torch.Tensor,batch:int,tokens:int)->torch.Tensor:
    d=grad.shape[-1];nt=(tokens+127)//128
    partial=torch.empty((batch,nt,3,d),device=grad.device,dtype=torch.float32)
    out=torch.empty((batch,3,d),device=grad.device,dtype=torch.float32)
    wrap_triton(partial_reduce)[(batch*nt,(d+31)//32)](
        grad,ids,partial,tokens,d,nt,*grad.stride(),128,32,
        num_warps=4,enable_fp_fusion=False)
    wrap_triton(merge_partial)[(batch,(d+31)//32)](
        partial,out,d,nt,32,64,num_warps=4,enable_fp_fusion=False)
    return out

@blocked_reduce.register_kernel('cpu')
def _cpu_blocked_reduce(grad,ids,batch,tokens):
    # CPU diagnostic implementation of the same tile-then-merge reduction.
    # It is not used by CUDA or presented as actual CUDA numerics/codegen.
    nt=(tokens+127)//128;d=grad.shape[-1]
    g=torch.nn.functional.pad(grad,(0,0,0,nt*128-tokens)).reshape(batch,nt,128,d)
    groups=torch.nn.functional.pad(ids.reshape(batch,tokens),(0,nt*128-tokens),value=-1)
    groups=groups.reshape(batch,nt,128)-torch.arange(batch)[:,None,None]*3
    partial=torch.stack([torch.where(groups[:,:,:,None]==r,g,0.).sum(2) for r in range(3)],2)
    return partial.sum(1)

class BlockedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx,plane,ids,batch,tokens):
        ctx.save_for_backward(ids);ctx.batch=batch;ctx.tokens=tokens
        return plane.reshape(batch*3,plane.shape[-1]).index_select(0,ids).reshape(batch,tokens,plane.shape[-1])
    @staticmethod
    def backward(ctx,grad):
        ids,=ctx.saved_tensors
        return blocked_reduce(grad,ids,ctx.batch,ctx.tokens),None,None,None

def modulation_part(block,e0,index):
    plane=e0.planes[index]
    if plane.dtype!=torch.float32 or e0.tokens>8192:
        # Conservative original differentiable gather for unsupported dtype/L.
        part=plane.reshape(e0.batch*3,-1).index_select(0,e0.flat_group).reshape(e0.batch,e0.tokens,-1)
    else:
        part=BlockedGather.apply(plane,e0.flat_group,e0.batch,e0.tokens)
    return block.modulation[:,index,:].unsqueeze(0)+part

def interpreter_reduce(grad,ids):
    if os.environ.get('TRITON_INTERPRET')!='1' or grad.is_cuda:
        raise RuntimeError('Explicit CPU Triton interpreter only')
    b,l,d=grad.shape;nt=(l+127)//128
    partial=torch.empty((b,nt,3,d),dtype=torch.float32);out=torch.empty((b,3,d),dtype=torch.float32)
    partial_reduce[(b*nt,(d+31)//32)](grad,ids,partial,l,d,nt,*grad.stride(),128,32,num_warps=4,enable_fp_fusion=False)
    merge_partial[(b,(d+31)//32)](partial,out,d,nt,32,64,num_warps=4,enable_fp_fusion=False)
    return out,partial


def _gather_with_blocked_vjp(plane,ids,batch,tokens):
    return BlockedGather.apply(plane,ids,batch,tokens)

def _gather_checkpoint_policy(ctx,op,*args,**kwargs):
    from torch.utils.checkpoint import CheckpointPolicy
    if op==torch.ops.aten.index_select.default:
        return CheckpointPolicy.MUST_RECOMPUTE
    return CheckpointPolicy.MUST_SAVE

def blocked_gather(plane,ids,batch,tokens):
    # Recompute ONLY the parameterless gather, never MLP/norm/linear/GEMM.
    from functools import partial
    from torch.utils.checkpoint import checkpoint,create_selective_checkpoint_contexts
    context_fn=partial(create_selective_checkpoint_contexts,_gather_checkpoint_policy)
    return checkpoint(_gather_with_blocked_vjp,plane,ids,batch,tokens,
                      use_reentrant=False,preserve_rng_state=False,context_fn=context_fn)
