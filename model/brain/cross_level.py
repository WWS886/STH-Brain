"""Per-ROI temporal encoding before same-block Network aggregation.

Flat indices are (scan, block, ROI). Neither packing nor attention substitutes
packed offsets for physical timestamps. Only caller-selected view rows enter.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ROITemporalLayer(nn.Module):
    def __init__(self, dim, heads, ffn_dim):
        super().__init__()
        if dim % heads or dim // heads < 16:
            raise ValueError('ROI temporal heads require at least eight rotary channel pairs')
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))

    def forward(self, x, times, valid, frequencies):
        b, length, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, length, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        angle = times.float()[:, None, :, None] * frequencies.float()[None, None, None, :]
        def rotate(t):
            pairs = t[..., :16].reshape(b, self.heads, length, 8, 2)
            a, c = pairs.unbind(-1)
            rotated = torch.stack((a * angle.cos() - c * angle.sin(),
                                   a * angle.sin() + c * angle.cos()), -1).flatten(-2).to(t.dtype)
            return torch.cat((rotated, t[..., 16:]), -1)
        mask = valid[:, None, None, :]
        h = F.scaled_dot_product_attention(rotate(q), rotate(k), v, attn_mask=mask, dropout_p=0.)
        x = x + self.proj(h.transpose(1, 2).reshape(b, length, d))
        x = x + self.ffn(self.norm2(x))
        return x.masked_fill(~valid[..., None], 0.)


class ROITemporalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.validate_roi_temporal()
        dim = cfg.roi_temporal_dim
        self.activation_checkpointing = cfg.roi_temporal_checkpointing
        self.input_projection = (nn.Linear(cfg.local_dim, dim)
                                 if dim != cfg.local_dim else nn.Identity())
        self.layers = nn.ModuleList([ROITemporalLayer(dim, cfg.roi_temporal_heads,
                                    cfg.roi_temporal_ffn_dim) for _ in range(cfg.roi_temporal_layers)])
        self.norm = nn.LayerNorm(dim)
        self.output_projection = (nn.Linear(dim, cfg.local_dim)
                                  if dim != cfg.local_dim else nn.Identity())
        self.output_norm = (nn.LayerNorm(cfg.local_dim)
                            if dim != cfg.local_dim else nn.Identity())

    def forward(self, features, indices, centers, frequencies, roi_count, stage_outputs=None):
        groups = indices[:, 0] * roi_count + indices[:, 2]
        _, inverse, counts = torch.unique(groups, sorted=True, return_inverse=True, return_counts=True)
        # Explicit sorting preserves chronological order even if caller rows are permuted.
        order = torch.argsort(groups * centers.shape[1] + indices[:, 1], stable=True)
        sorted_groups = inverse[order]
        starts = counts.cumsum(0) - counts
        positions = torch.arange(len(order), device=order.device) - starts[sorted_groups]
        x = features.new_zeros((len(counts), int(counts.max()), features.shape[-1]))
        valid = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        times = centers.new_zeros(x.shape[:2])
        x[sorted_groups, positions] = features[order]
        valid[sorted_groups, positions] = True
        times[sorted_groups, positions] = centers[indices[order, 0], indices[order, 1]]
        def capture(name, value):
            if stage_outputs is None:return
            from model.brain.content_statistics import pool_slots
            with torch.no_grad():
                mean=(value.detach().float()*valid[...,None]).sum(1)/valid.sum(1)[:,None]
                group_ids=torch.unique(groups,sorted=True)
                stage_outputs['roi_temporal/'+name]=pool_slots(mean,group_ids//roi_count,group_ids%roi_count,len(centers),roi_count)
                residual=(value.detach().float()-mean[:,None]).masked_fill(~valid[...,None],0.)
                temporal_std=(residual.square().sum(1)/valid.sum(1)[:,None]).sqrt()
                stage_outputs['roi_temporal/temporal_std/'+name]=pool_slots(temporal_std,group_ids//roi_count,group_ids%roi_count,len(centers),roi_count)
        x = self.input_projection(x).masked_fill(~valid[..., None], 0.)
        capture('input_projection',x)
        for number,layer in enumerate(self.layers,1):
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                # Non-reentrant checkpointing also supports autograd.grad diagnostics.
                x = checkpoint(layer, x, times, valid, frequencies, use_reentrant=False)
            else:
                x = layer(x, times, valid, frequencies)
            capture(f'layer{number:02d}',x)
        x=self.norm(x);capture('pre_output_norm',x)
        x=self.output_projection(x);capture('output_projection',x)
        x=self.output_norm(x);capture('output_norm',x)
        x = x.masked_fill(~valid[..., None], 0.)
        result = x.new_empty(features.shape)
        result[order] = x[sorted_groups, positions]
        return result
