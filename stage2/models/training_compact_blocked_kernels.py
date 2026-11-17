"""Two-stage FP32 VJP, no token atomics and no six-plane token activation."""
import triton
import triton.language as tl

@triton.jit
def partial_reduce(G,IDS,P,L:tl.constexpr,D:tl.constexpr,NT:tl.constexpr,
                   SB:tl.constexpr,SL:tl.constexpr,SD:tl.constexpr,
                   TL:tl.constexpr,TD:tl.constexpr):
    tile=tl.program_id(0);b=tile//NT;t=tile%NT
    li=t*TL+tl.arange(0,TL)
    di=tl.program_id(1)*TD+tl.arange(0,TD)
    group=tl.load(IDS+b*L+li,li<L,other=-1)-3*b
    grad=tl.load(G+b*SB+li[:,None]*SL+di[None,:]*SD,
                 (li[:,None]<L)&(di[None,:]<D),other=0).to(tl.float32)
    for r in tl.static_range(3):
        value=tl.sum(tl.where(group[:,None]==r,grad,0.),axis=0)
        tl.store(P+((b*NT+t)*3+r)*D+di,value,di<D)

@triton.jit
def merge_partial(P,E,D:tl.constexpr,NT:tl.constexpr,TD:tl.constexpr,MT:tl.constexpr):
    b=tl.program_id(0)
    ti=tl.arange(0,MT)
    di=tl.program_id(1)*TD+tl.arange(0,TD)
    for r in tl.static_range(3):
        values=tl.load(P+((b*NT+ti[:,None])*3+r)*D+di[None,:],
                       (ti[:,None]<NT)&(di[None,:]<D),other=0)
        out=tl.sum(values,axis=0)
        tl.store(E+(b*3+r)*D+di,out,di<D)
