"""Parameter-free ROI and network readouts for pretraining."""
import torch


def network_mean_readouts(tokens, valid, network_ids, num_networks):
    """Pool each network separately; missing networks are excluded by `valid`."""
    if tokens.ndim!=3 or valid.shape!=tokens.shape[:2] or network_ids.shape!=valid.shape:
        raise ValueError('Network readout expects tokens [B,L,D] and valid/network IDs [B,L]')
    if num_networks<1 or torch.any(valid & ((network_ids<0)|(network_ids>=num_networks))):
        raise ValueError('Valid token network IDs fall outside the atlas')
    membership=valid.bool()[...,None] & network_ids[...,None].eq(
        torch.arange(num_networks,device=tokens.device))
    counts=membership.sum(1)
    raw=torch.einsum('bln,bld->bnd',membership.float(),tokens.float())/counts.clamp_min(1)[...,None]
    return {'raw':raw,'valid':counts>0}




def roi_mean_readouts(encoded):
    """Time mean per ROI, then equal ROI mean (not a flattened token mean)."""
    tokens, valid, ids = encoded['roi_tokens'], encoded['roi_valid'], encoded['roi_ids']
    b, _, d = tokens.shape
    n = int(encoded['num_rois'])
    scan = torch.arange(b, device=tokens.device)[:, None].expand_as(ids)
    flat = scan[valid] * n + ids[valid]
    sums = tokens.new_zeros((b*n,d),dtype=torch.float32).index_add(0,flat,tokens[valid].float())
    counts = tokens.new_zeros(b*n,dtype=torch.float32).index_add(0,flat,torch.ones_like(flat,dtype=torch.float32))
    per_roi = (sums/counts.clamp_min(1)[:,None]).reshape(b,n,d)
    present = counts.reshape(b,n)>0
    if torch.any(present.sum(1)==0):
        raise ValueError('ROI readout requires an observed ROI per scan')
    raw = per_roi.sum(1)/present.sum(1)[:,None]
    return dict(raw=raw,per_roi=per_roi,valid=present)




def encoded_readouts(encoded):
    from model.brain.roi_identity import identity_readout, diagnostic_vectors, slot_readouts
    slots=slot_readouts(encoded)
    branches=diagnostic_vectors(encoded,slots)
    return dict(raw=identity_readout(encoded,slots=slots),layernorm=branches['joint'],slots=slots)
