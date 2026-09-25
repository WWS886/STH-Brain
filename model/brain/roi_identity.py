"""Identity-preserving readout, contextual ROI prediction and per-slot moments."""
import math
import torch
from torch import nn
import torch.nn.functional as F
def _pack_flat(values, groups, group_count):
    """Pack ROI groups, including identities without context or target rows."""
    counts=torch.bincount(groups,minlength=group_count)
    width=max(1,int(counts.max()))
    packed=values.new_zeros((group_count,width,values.shape[-1]))
    valid=torch.zeros((group_count,width),device=values.device,dtype=torch.bool)
    order=torch.argsort(groups,stable=True)
    starts=counts.cumsum(0)-counts
    positions=torch.empty_like(groups)
    positions[order]=torch.arange(len(groups),device=groups.device)-starts[groups[order]]
    packed[groups,positions]=values
    valid[groups,positions]=True
    return packed,valid,positions


ARCHITECTURE = 'roi_temporal_hierarchy'


def slot_readouts(encoded):
    from model.brain.readout import roi_mean_readouts, network_mean_readouts
    roi = roi_mean_readouts(encoded)
    net = network_mean_readouts(encoded['tokens'], encoded['token_valid'],
                               encoded['token_network_ids'], encoded['num_networks'])
    return [(roi['per_roi'], roi['valid']), (net['raw'], net['valid'])]


def identity_readout(encoded, include_masks=True, slots=None):
    branches, masks = [], []
    for raw, valid in (slot_readouts(encoded) if slots is None else slots):
        z = F.layer_norm(raw.float(), (raw.shape[-1],)).masked_fill(~valid[..., None], 0)
        branches.append(z.flatten(1) / math.sqrt(raw.shape[1] * raw.shape[2]))
        masks.append(valid.float())
    return torch.cat(branches + (masks if include_masks else []), -1)


class ROIPredictorLayer(nn.Module):
    """One routed cross-attention update; only later layers retain query state."""
    def __init__(self, dim, heads, ffn_dim, query_residual=False):
        super().__init__()
        self.query_residual = query_residual
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))

    def forward(self, query, local, network, local_valid, network_valid,
                active, nroi, has_local, routed, query_valid):
        # Network K/V are shared across ROI groups: project once per layer,
        # then bound attention memory with chunks of 128 groups.
        wq, wk, wv = self.attention.in_proj_weight.chunk(3)
        bq, bk, bv = self.attention.in_proj_bias.chunk(3)
        nq = F.linear(query, wq, bq)
        nk, nv = F.linear(network, wk, bk), F.linear(network, wv, bv)
        lk, lval = F.linear(local, wk, bk), F.linear(local, wv, bv)
        heads = self.attention.num_heads
        def split_heads(x):
            return x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(1, 2)
        out = query.new_zeros(query.shape)
        for selected in active.split(128):
            scans = selected // nroi
            keys = torch.cat((lk[selected], nk[scans]), 1)
            values = torch.cat((lval[selected], nv[scans]), 1)
            network_support = network_valid[scans]
            if routed:
                network_support = network_support & ~has_local[selected, None]
            valid = torch.cat((local_valid[selected], network_support), 1)
            if not bool(valid.any(1).all()):
                raise ValueError('Routed target has no visible memory')
            update = F.scaled_dot_product_attention(
                split_heads(nq[selected]), split_heads(keys), split_heads(values),
                attn_mask=valid[:, None, None, :], dropout_p=0.)
            update = update.transpose(1, 2).reshape(len(selected), query.shape[1], -1)
            update = self.attention.out_proj(update)
            # First-layer output is anchored in context, with no query-only
            # residual. Subsequent query states already contain context.
            hidden = query[selected] + update if self.query_residual else update
            out[selected] = (hidden + self.ffn(self.norm(hidden))).to(out.dtype)
        return out.masked_fill(~query_valid[..., None], 0.)


class ROIPredictor(nn.Module):
    """Target queries use own-ROI memory when present, otherwise network memory.

    Grouped by scan and ROI so no other scan/ROI observation enters local memory.
    The teacher supplies only target positions here, never target features.
    """
    def __init__(self, cfg, geometry):
        super().__init__()
        cfg.validate_predictors()
        d = cfg.roi_predictor_dim
        self.ids = nn.Embedding(len(geometry.roi_to_network), d)
        self.position = nn.Linear(19, d)
        self.roi = nn.Linear(cfg.local_dim, d)
        self.network = nn.Linear(cfg.model_dim, d)
        self.types = nn.Parameter(torch.zeros(2, d))
        self.layers = nn.ModuleList([
            ROIPredictorLayer(d, cfg.roi_predictor_heads, cfg.roi_predictor_ffn_dim,
                              query_residual=index > 0)
            for index in range(cfg.roi_predictor_layers)
        ])
        self.output = nn.Linear(d, cfg.local_dim)

    def forward(self, context, teacher, select, geometry, memory_mode='routed', stage_outputs=None):
        if memory_mode not in {'routed','joint','roi_only','network_only'}:
            raise ValueError('Unknown ROI memory intervention')
        batch, nroi = context['tokens'].shape[0], int(context['num_rois'])
        scan, pos = select.nonzero(as_tuple=True)
        ids = teacher['roi_ids'][scan, pos]
        times = teacher['roi_time_seconds'][scan, pos]
        phase = times[:, None] * geometry.bounded_time_frequencies()[None]
        coordinates = geometry.atlas_xyz[ids] / geometry.atlas_xyz.abs().amax(0).clamp_min(1)
        q = self.ids(ids) + self.position(torch.cat((coordinates, phase.sin(), phase.cos()), -1))
        group = scan * nroi + ids
        qp, qvalid, qpos = _pack_flat(q, group, batch*nroi)
        cs, cp = context['roi_valid'].nonzero(as_tuple=True)
        cg = cs*nroi + context['roi_ids'][cs, cp]
        local, lv, _ = _pack_flat(self.roi(context['roi_tokens'][cs, cp]) + self.types[0], cg, batch*nroi)
        # Positions are already present in encoded memory. Add explicit memory
        # seconds to distinguish otherwise similar visible temporal states.
        ct = context['roi_time_seconds'][cs, cp]
        cphase = ct[:, None] * geometry.bounded_time_frequencies()[None]
        cxyz = geometry.atlas_xyz[context['roi_ids'][cs, cp]] / geometry.atlas_xyz.abs().amax(0).clamp_min(1)
        lp, _, _ = _pack_flat(self.position(torch.cat((cxyz,cphase.sin(),cphase.cos()),-1)), cg,batch*nroi)
        local = local + lp
        network = self.network(context['tokens']) + self.types[1]
        active = qvalid.any(1).nonzero().flatten()
        has_local=lv.any(1)
        network_valid=context['token_valid'].clone()
        if memory_mode=='roi_only':network_valid.zero_()
        if memory_mode=='network_only':lv=torch.zeros_like(lv)
        if not (lv[active].any(1) | network_valid[active//nroi].any(1)).all():
            raise ValueError('ROI predictor requires visible context')
        def capture(name,value):
            if stage_outputs is not None:
                from model.brain.content_statistics import pool_slots
                stage_outputs[name]=pool_slots(value.detach(),scan,ids,batch,nroi)
        capture('target_query',qp[group,qpos])
        for number,layer in enumerate(self.layers,1):
            qp = layer(qp, local, network, lv, network_valid, active, nroi,
                       has_local, memory_mode == 'routed', qvalid)
            capture(f'layer{number:02d}',qp[group,qpos])
        output=self.output(qp[group,qpos]);capture('output',output)
        return output, (scan,pos)


def target_local_support(context,metadata,select):
    """Visibility-only route in the same ordering as select.nonzero()."""
    batch,nroi=context['tokens'].shape[0],int(context['num_rois'])
    support=torch.zeros((batch,nroi),device=select.device,dtype=torch.bool)
    scan,pos=context['roi_valid'].nonzero(as_tuple=True)
    support[scan,context['roi_ids'][scan,pos]]=True
    scan,pos=select.nonzero(as_tuple=True)
    return support[scan,metadata['roi_ids'][scan,pos]]






def identity_moments(encoded, subjects, objective, distributed, slots=None, reduction=None, return_branches=False, return_subject_slots=False):
    from model.brain.pretrain import global_rows
    from model.brain.subject_objectives import full_branch_moments
    import torch.distributed as dist
    if reduction is not None:
        subjects=reduction.subjects
    elif distributed and dist.is_initialized():
        groups=[None]*dist.get_world_size();dist.all_gather_object(groups,list(subjects))
        subjects=[s for group in groups for s in group]
    unique=sorted(set(subjects)); subject_slots=[]
    for raw,valid in (slot_readouts(encoded) if slots is None else slots):
        z=F.layer_norm(raw.float(),(raw.shape[-1],))
        shape=z.shape[1:]
        z=global_rows(z.flatten(1),distributed,sizes=reduction.sizes if reduction is not None else None).reshape(-1,*shape)
        valid=global_rows(valid.float(),distributed,sizes=reduction.sizes if reduction is not None else None).bool()
        ids=reduction.index if reduction is not None else torch.tensor([unique.index(s) for s in subjects],device=z.device)
        sums=z.new_zeros((len(unique),*shape)).index_add(0,ids,z*valid[...,None])
        count=z.new_zeros((len(unique),shape[0])).index_add(0,ids,valid.float())
        means=sums/count.clamp_min(1)[...,None]
        present=count>0
        subject_slots.append((means,present))
    average, evidence, branches = full_branch_moments(subject_slots, objective)
    if return_subject_slots:
        return average, evidence, branches, subject_slots
    return (average, evidence, branches) if return_branches else (average, evidence)


def diagnostic_vectors(encoded, slots=None):
    """Zero-filled vectors for online pretraining diagnostics."""
    vectors=[]
    for raw,valid in (slot_readouts(encoded) if slots is None else slots):
        vectors.append(F.layer_norm(raw.float(),(raw.shape[-1],)).masked_fill(~valid[...,None],0).flatten(1))
    return dict(roi=vectors[0],network=vectors[1],joint=torch.cat(vectors,-1))
