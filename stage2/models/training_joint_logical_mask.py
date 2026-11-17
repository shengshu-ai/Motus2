"""CPU-only logical block plan for the existing Motus joint training mask.

Not an attention backend. Production would lower this plan to its exact tile
geometry plus a token-level CuTe mask_mod; do not pass these Python objects to FA4.
"""
from dataclasses import dataclass
import math
import torch

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str
    chunk: int
    sample: int

@dataclass(frozen=True)
class Plan:
    spans: tuple
    length: int
    window: int

    def allows(self, q, k):
        if q.sample != k.sample or q.kind == 'pad' or k.kind == 'pad':
            return False
        if abs(q.chunk - k.chunk) > self.window:
            return False
        if q.kind == 'clean':
            return k.kind == 'clean' and k.chunk <= q.chunk
        if q.kind == 'pred':
            return ((k.kind == 'clean' and k.chunk < q.chunk)
                    or (k.kind == 'pred' and k.chunk == q.chunk))
        raise ValueError('unsupported logical token kind')

    def dense(self):
        mask = torch.zeros(self.length, self.length, dtype=torch.bool)
        for q in self.spans:
            for k in self.spans:
                if self.allows(q, k):
                    mask[q.start:q.end, k.start:k.end] = True
        # EXACT reference behavior, including each sequence-padding token.
        mask.diagonal().fill_(True)
        return mask

    def allowed_area(self, qlo, qhi, klo, khi):
        """Exact allowed element count, without allocating a dense L² mask."""
        area = 0
        for q in self.spans:
            qn = max(0, min(q.end, qhi) - max(q.start, qlo))
            if not qn:
                continue
            for k in self.spans:
                if self.allows(q, k):
                    kn = max(0, min(k.end, khi) - max(k.start, klo))
                    area += qn * kn
            # Rectangles form a partition. Add self-diagonal only where its own
            # rectangle does not already provide it (padding or window<0).
            if not self.allows(q, q):
                area += max(0, min(q.end, qhi, khi) - max(q.start, qlo, klo))
        return area

    def sparse_tables(self, q_tile, kv_tile, direction='forward'):
        """Sorted full/partial lists; backward is KV-major at ITS OWN tile size."""
        if direction not in ('forward', 'backward') or min(q_tile, kv_tile) <= 0:
            raise ValueError('unsupported tile protocol')
        nq = math.ceil(self.length / q_tile);nk = math.ceil(self.length / kv_tile)
        rows = [{'full': [], 'partial': []} for _ in range(nq if direction == 'forward' else nk)]
        for qi in range(nq):
            qlo=qi*q_tile;qhi=min(qlo+q_tile,self.length)
            for ki in range(nk):
                klo=ki*kv_tile;khi=min(klo+kv_tile,self.length)
                area=self.allowed_area(qlo,qhi,klo,khi)
                if not area:
                    continue
                full=(area==(qhi-qlo)*(khi-klo) and qhi-qlo==q_tile and khi-klo==kv_tile)
                row,col=(qi,ki) if direction=='forward' else (ki,qi)
                rows[row]['full' if full else 'partial'].append(col)
        return {'direction':direction,'q_tile':q_tile,'kv_tile':kv_tile,
                'q_blocks':nq,'kv_blocks':nk,'rows':rows}


def make_plan(W, video_tokens, action_tokens, chunk_size, window,
              state_tokens=0, value_tokens=0, attention_mode='joint', sample=0):
    if attention_mode != 'joint' or value_tokens != 0:
        raise ValueError('Flex fallback: only joint/value0 is supported')
    if W < chunk_size or chunk_size < 1 or video_tokens < 1 or min(action_tokens,state_tokens)<0:
        raise ValueError('unsupported geometry')
    N=W-chunk_size;spans=[];pos=0
    for start in range(0,N,chunk_size):
        count=min(chunk_size,N-start);size=count*video_tokens
        spans.append(Span(pos,pos+size,'clean',start//chunk_size,sample));pos+=size
    for start in range(0,N,chunk_size):
        count=min(chunk_size,N-start)
        # State prefix and all actions are TOKEN_ACTION in the real mask.
        # In joint/value0, action+noisy-video can be one contiguous dense block.
        size=state_tokens+count*(action_tokens+video_tokens)
        spans.append(Span(pos,pos+size,'pred',(chunk_size+start)//chunk_size,sample));pos+=size
    padded=math.ceil(pos/128)*128
    if padded>pos:spans.append(Span(pos,padded,'pad',-1,sample))
    return Plan(tuple(spans),padded,window)


def pack_plans(plans):
    if len({p.window for p in plans})>1:
        raise ValueError('different window contracts require explicit per-sample lowering')
    spans=[];offset=0
    for sample,p in enumerate(plans):
        spans += [Span(s.start+offset,s.end+offset,s.kind,s.chunk,sample) for s in p.spans]
        offset += p.length
    return Plan(tuple(spans),offset,plans[0].window if plans else 0)


def reconstruct(plan,tables,token_mask):
    """CPU reference: partial tiles apply the exact token predicate."""
    result=torch.zeros(plan.length,plan.length,dtype=torch.bool)
    qm,kn=tables['q_tile'],tables['kv_tile']
    for row,record in enumerate(tables['rows']):
        for tag in ('full','partial'):
            for col in record[tag]:
                qi,ki=(row,col) if tables['direction']=='forward' else (col,row)
                qs=slice(qi*qm,min((qi+1)*qm,plan.length))
                ks=slice(ki*kn,min((ki+1)*kn,plan.length))
                result[qs,ks]=True if tag=='full' else token_mask[qs,ks]
    return result


def cpu_fa4_fields(tables):
    """CPU metadata layout only; no FA4 import, compilation or device transfer.

    FWD rows are query blocks, BWD rows are key blocks. The field names are the
    same; their direction-dependent contents are NOT interchangeable objects.
    """
    fields={}
    for tag,prefix in [('partial','mask'),('full','full')]:
        rows=tables['rows'];width=max([len(r[tag]) for r in rows]+[1])
        counts=torch.tensor([len(r[tag]) for r in rows],dtype=torch.int32).view(1,1,len(rows))
        indices=torch.zeros((1,1,len(rows),width),dtype=torch.int32)
        for i,r in enumerate(rows):
            if r[tag]:indices[0,0,i,:len(r[tag])]=torch.tensor(r[tag],dtype=torch.int32)
        fields[prefix+'_block_cnt']=counts;fields[prefix+'_block_idx']=indices
    fields['block_size']=(tables['q_tile'],tables['kv_tile'])
    return fields
