"""Shared signal-free physical query validity for collation and tokenization."""
import torch


def supported_query_mask(offsets, valid, durations, *, grid_step_seconds,
                         grid_count, max_source_gap_seconds):
    """Return [N,G] supported queries, without manufacturing a fallback sample.

    Valid source timestamps must be increasing within each row. Invalid padding
    may occupy arbitrary positions; source compaction preserves the valid order.
    A query is supported within the observed extent and the nominal duration
    when it is an exact observation or its interpolation gap is sufficiently small.
    """
    if offsets.ndim != 2 or valid.shape != offsets.shape or durations.shape != offsets.shape[:1]:
        raise ValueError('Physical support expects offsets/valid [N,Q] and durations [N]')
    if grid_step_seconds <= 0 or grid_count < 1 or max_source_gap_seconds <= 0:
        raise ValueError('Physical query step, count and maximum gap must be positive')
    valid=valid.bool()
    rows,width=offsets.shape
    query=torch.arange(grid_count,device=offsets.device,dtype=torch.float32)*float(grid_step_seconds)
    if width == 0:
        return torch.zeros((rows,grid_count),dtype=torch.bool,device=offsets.device)
    counts=valid.sum(1)
    packed=torch.full_like(offsets,float('inf'),dtype=torch.float32)
    row,col=valid.nonzero(as_tuple=True)
    packed[row,valid.long().cumsum(1)[row,col]-1]=offsets[row,col].float()
    query_rows=query[None].expand(rows,-1).contiguous()
    high=torch.searchsorted(packed.contiguous(),query_rows,right=False)
    last=(counts-1).clamp_min(0)
    high=torch.minimum(high,last[:,None]);low=(high-1).clamp_min(0)
    lo=packed.gather(1,low);hi=packed.gather(1,high)
    exact=torch.isclose(query_rows,lo,atol=1e-7,rtol=0)|torch.isclose(query_rows,hi,atol=1e-7,rtol=0)
    support=counts[:,None].gt(0)&query_rows.lt(durations.float()[:,None]+1e-7)
    support &= query_rows.ge(packed[:,:1]-1e-7)
    support &= query_rows.le(packed.gather(1,last[:,None])+1e-7)
    return support & ((hi-lo).le(float(max_source_gap_seconds))|exact)
