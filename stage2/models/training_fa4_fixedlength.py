"""FA4 b14 fixed-length joint-mask training backend with guarded Flex fallback.

No CUDA/FA4 imports or contexts are created when this module is imported.
Only the joint/value0, action-first, equal-length 4D path is eligible.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
from functools import lru_cache
from pathlib import Path

import torch


INTERFACE_SHA = '27aa8985be0464bc872cf773d39c1b3266ce6f1b3dd2ed63e0f9829861b59c9b'
QTILE, KVTILE = 256, 128


class UnsupportedFA4(ValueError):
    pass


def cpu_metadata(plan):
    """Prepare both directional index sets without allocating an L² mask."""
    if plan.length < 256 or plan.length % 128:
        raise UnsupportedFA4('Supported padded lengths must be >=256 and divisible by128')
    if any(s.kind not in ('clean', 'pred', 'pad') or s.sample != 0 for s in plan.spans):
        raise UnsupportedFA4('Only one joint/value0 logical sample geometry; no packing')
    if not plan.spans or plan.spans[0].start != 0 or plan.spans[-1].end != plan.length:
        raise UnsupportedFA4('Logical spans must partition the complete padded sequence')
    for left, right in zip(plan.spans, plan.spans[1:]):
        if left.end != right.start:
            raise UnsupportedFA4('Logical spans must be contiguous and nonoverlapping')
    tables = {d:plan.sparse_tables(QTILE,KVTILE,d) for d in ('forward','backward')}
    result = {}
    for direction, table in tables.items():
        rows = table['rows']
        cols = table['kv_blocks'] if direction == 'forward' else table['q_blocks']
        arrays = {}
        for kind in ('partial','full'):
            counts = torch.tensor([len(r[kind]) for r in rows], dtype=torch.int32).view(1,1,-1)
            indices = torch.zeros((1,1,len(rows),cols),dtype=torch.int32)
            for i,row in enumerate(rows):
                if row[kind]:indices[0,0,i,:len(row[kind])] = torch.tensor(row[kind],dtype=torch.int32)
            arrays[kind+'_cnt'],arrays[kind+'_idx'] = counts,indices
        result[direction] = arrays
    # Ascending KV contributor order for noncausal deterministic dQ accumulation.
    contributors = [sorted(r['partial']+r['full']) for r in tables['forward']['rows']]
    for kind in ('partial','full'):
        order = torch.zeros_like(result['backward'][kind+'_idx'])
        for kv,row in enumerate(tables['backward']['rows']):
            # Match the official gather even in unused index slots (index=0).
            order[0,0,kv,:] = sum(n < kv for n in contributors[0])
            for slot,qblock in enumerate(row[kind]):
                order[0,0,kv,slot] = contributors[qblock].index(kv)
        result['backward'][kind+'_write_order'] = order
    # Kernel tiles can overrun the logical L at their last Q row; sentinel aux
    # entries prevent any out-of-range load even before the kernel's length mask.
    aux_length = math.ceil(plan.length/QTILE)*QTILE
    kinds = torch.zeros(aux_length,dtype=torch.int32)
    chunks = torch.full((aux_length,),-1,dtype=torch.int32)
    for span in plan.spans:
        kinds[span.start:span.end] = {'pad':0,'clean':1,'pred':2}[span.kind]
        chunks[span.start:span.end] = span.chunk
    result['aux'] = [kinds,chunks,torch.tensor([plan.window],dtype=torch.int32)]
    result['tables'] = tables
    return result


@lru_cache(None)
def _cute_joint_mask():
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    @cute.jit
    def joint_mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
        kinds, chunks, windows = aux_tensors
        qt = utils.scalar_to_ssa(kinds[q_idx[0]], cutlass.Int32)
        kt = utils.scalar_to_ssa(kinds[kv_idx[0]], cutlass.Int32)
        qc = utils.scalar_to_ssa(chunks[q_idx[0]], cutlass.Int32)
        kc = utils.scalar_to_ssa(chunks[kv_idx[0]], cutlass.Int32)
        ws = utils.scalar_to_ssa(windows[0], cutlass.Int32)
        clean = (qt == 1) & (kt == 1) & (kc <= qc)
        pred_clean = (qt == 2) & (kt == 1) & (kc < qc)
        pred_pred = (qt == 2) & (kt == 2) & (kc == qc)
        delta = qc-kc
        combined = (clean | pred_clean | pred_pred) & (qt != 0) & (kt != 0)
        combined = combined & (delta <= ws) & (delta >= -ws)
        return (q_idx == kv_idx) | combined

    return joint_mask


class FixedLengthJoint:
    """Prepared metadata owns immutable tensors until all backwards complete."""
    def __init__(self, plan, device, *, deterministic=False, counters=None):
        # No caller should construct this on CPU merely to query eligibility.
        device = torch.device(device)
        if device.type != 'cuda':raise UnsupportedFA4('FA4 requires CUDA')
        if device.index is None:device = torch.device('cuda',torch.cuda.current_device())
        try:
            version = importlib.metadata.version('flash-attn-4')
        except importlib.metadata.PackageNotFoundError as exc:
            raise UnsupportedFA4('FA4 b14 is not installed') from exc
        if version != '4.0.0b14':
            raise UnsupportedFA4('Requires FA4 b14')
        try:
            from flash_attn.cute import interface
            from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
        except ImportError as exc:
            raise UnsupportedFA4('FA4 backend import failed: ' + type(exc).__name__) from exc
        path = Path(interface.__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest() != INTERFACE_SHA:
            raise UnsupportedFA4('Incompatible FA4 interface')
        if torch.cuda.get_device_capability(device) != (10,3):
            raise UnsupportedFA4('Requires an SM103 GPU')
        self.length = plan.length
        self.counters = counters if counters is not None else [0, 0]
        self.deterministic = bool(deterministic)
        self.plan = plan
        meta = cpu_metadata(plan)
        self.aux = [x.to(device) for x in meta['aux']]
        fwd,bwd = meta['forward'],meta['backward']
        def base(row):
            return dict(mask_block_cnt=row['partial_cnt'].to(device),
                        mask_block_idx=row['partial_idx'].to(device),
                        full_block_cnt=row['full_cnt'].to(device),
                        full_block_idx=row['full_idx'].to(device),block_size=(QTILE,KVTILE))
        self.forward = BlockSparseTensorsTorch(**base(fwd))
        self.backward = BlockSparseTensorsTorch(**base(bwd),
            dq_write_order=bwd['partial_write_order'].to(device),
            dq_write_order_full=bwd['full_write_order'].to(device),spt=False)
        self.call = interface.flash_attn_func
        self.mask_mod = _cute_joint_mask()
        self.device = device

    def rejection(self, q,k,v):
        if any(x.ndim != 4 for x in (q,k,v)):return 'requires BLHD 4D'
        if any(x.shape != q.shape for x in (k,v)):return 'requires equal Q/K/V shapes'
        if q.shape[1:] != (self.length,24,128):return 'requires matching L,H24,D128'
        if q.shape[0] not in (1,2,4):return 'requires B1/B2/B4'
        if any(x.device != self.device for x in (q,k,v)):return 'device mismatch'
        if any(x.dtype != torch.bfloat16 for x in (q,k,v)):return 'requires BF16'
        if any(not x.is_contiguous() for x in (q,k,v)):return 'requires contiguous BLHD'
        return None

    def __call__(self,q,k,v,*,flex_fallback=None):
        reason = self.rejection(q,k,v)
        if reason:
            if flex_fallback is not None:
                self.counters[1] += 1
                return flex_fallback(q,k,v)
            raise UnsupportedFA4(reason)
        # Never catch a CUDA failure and silently pretend the FA4 arm succeeded.
        out,_lse = self.call(q,k,v,causal=False,softmax_scale=None,pack_gqa=False,
            deterministic=self.deterministic,mask_mod=self.mask_mod,aux_tensors=self.aux,
            block_sparse_tensors=self.forward,block_sparse_tensors_bwd=self.backward)
        self.counters[0] += 1
        return out
