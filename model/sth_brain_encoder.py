"""支持变长/padding 的 STH-Brain Transformer encoder，并施加物理时间/空间 SoPE。

输入：``tokens[B,L,D]``、``geometry_basis[B,L,51]`` 与有效位掩码 ``[B,L]``。
输出：相同形状的 contextualized token；padding 行严格置零。student 输入只有 context，
teacher 与完整视图 student 可输入完整 token。该模块不负责生成 C/M/E 掩码。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _geometry_embedding(
    basis: torch.Tensor,
    width: int,
    scale: float = 0.05,
) -> torch.Tensor:
    """把 51 维 basis 截断/补入 D 维，形成小幅无参数加性几何通道。"""
    out = basis.new_zeros((*basis.shape[:-1], width))
    used = min(width, basis.shape[-1])
    out[..., :used] = basis[..., :used]
    return out * float(scale)


def _apply_physical_sope(
    x: torch.Tensor,
    basis: torch.Tensor,
    spatial_head_fraction: float,
) -> torch.Tensor:
    """使用物理时间和球面 atlas 相位旋转 Q/K 通道对。

    ``x`` is ``[B,H,L,Dh]`` and ``basis`` is ``[B,L,51]``.  Eight temporal
    phase pairs rotate all heads.  Theta, phi, radius, left/right x, and z
    rotate the next five pairs in the configured spatial-head fraction.
    """
    batch, heads, length, head_dim = x.shape
    pair_count = head_dim // 2
    if pair_count == 0:
        return x
    paired_width = pair_count * 2
    pairs = x[..., :paired_width].reshape(batch, heads, length, pair_count, 2)
    sin_phase = x.new_zeros((batch, heads, length, pair_count))
    cos_phase = x.new_ones((batch, heads, length, pair_count))

    time_pairs = min(8, pair_count)
    if time_pairs:
        time = basis[..., :16].reshape(batch, length, 8, 2)
        sin_phase[..., :time_pairs] = time[:, None, :, :time_pairs, 0]
        cos_phase[..., :time_pairs] = time[:, None, :, :time_pairs, 1]

    spatial_available = max(0, min(5, pair_count - time_pairs))
    selected_heads = max(0, min(heads, int(round(heads * spatial_head_fraction))))
    if spatial_available and selected_heads:
        spatial = basis[..., 40:49]
        rho_angle = math.pi * spatial[..., 0]
        x_angle = math.pi * spatial[..., 5]
        z_angle = math.pi * spatial[..., 7]
        spatial_sin = torch.stack(
            (spatial[..., 1], spatial[..., 3], torch.sin(rho_angle),
             torch.sin(x_angle), torch.sin(z_angle)),
            dim=-1,
        )[..., :spatial_available]
        spatial_cos = torch.stack(
            (spatial[..., 2], spatial[..., 4], torch.cos(rho_angle),
             torch.cos(x_angle), torch.cos(z_angle)),
            dim=-1,
        )[..., :spatial_available]
        start = time_pairs
        stop = start + spatial_available
        sin_phase[:, :selected_heads, :, start:stop] = spatial_sin[:, None]
        cos_phase[:, :selected_heads, :, start:stop] = spatial_cos[:, None]

    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack(
        (even * cos_phase - odd * sin_phase, even * sin_phase + odd * cos_phase),
        dim=-1,
    ).reshape(batch, heads, length, paired_width)
    if paired_width == head_dim:
        return rotated
    return torch.cat((rotated, x[..., paired_width:]), dim=-1)


class STHBrainTransformerLayer(nn.Module):
    """带显式 Q/K 几何旋转的 pre-norm dense Transformer 层。"""

    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        spatial_head_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.dropout_p = float(dropout)
        self.spatial_head_fraction = float(spatial_head_fraction)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_projection = nn.Linear(dim, dim)
        self.ffn_in = nn.Linear(dim, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        geometry_basis: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = _apply_physical_sope(q, geometry_basis, self.spatial_head_fraction)
        k = _apply_physical_sope(k, geometry_basis, self.spatial_head_fraction)
        additive_mask = torch.zeros(
            (batch, 1, 1, length), dtype=q.dtype, device=q.device
        )
        additive_mask = additive_mask.masked_fill(~valid_mask[:, None, None, :], float("-inf"))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.dim)
        x = x + self.dropout(self.out_projection(attended))
        h = self.norm2(x)
        x = x + self.dropout(self.ffn_out(self.dropout(F.gelu(self.ffn_in(h)))))
        return x.masked_fill(~valid_mask[..., None], 0.0)


class STHBrainEncoder(nn.Module):
    """堆叠 Transformer 层；输入/输出均为 padded token batch。"""
    def __init__(
        self,
        dim: int = 256,
        layers: int = 8,
        heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        spatial_head_fraction: float = 0.5,
        geometry_additive_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                STHBrainTransformerLayer(
                    dim=dim,
                    heads=heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    spatial_head_fraction=spatial_head_fraction,
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.dim = int(dim)
        self.geometry_additive_scale = float(geometry_additive_scale)

    def forward(
        self,
        tokens: torch.Tensor,
        geometry_basis: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        stage_outputs: dict | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or geometry_basis.ndim != 3 or valid_mask.ndim != 2:
            raise ValueError("STHBrainEncoder expects [B,L,D], [B,L,51], [B,L]")
        if tokens.shape[:2] != valid_mask.shape or geometry_basis.shape[:2] != valid_mask.shape:
            raise ValueError("STHBrainEncoder padded shapes do not agree")
        if bool((valid_mask.sum(dim=1) == 0).any()):
            raise ValueError("Every STH-Brain sample must contain at least one token")
        x = tokens + _geometry_embedding(
            geometry_basis,
            self.dim,
            self.geometry_additive_scale,
        )
        x = x.masked_fill(~valid_mask[..., None], 0.0)
        if stage_outputs is not None:
            stage_outputs['network_input'] = tokens.detach()
            stage_outputs['network_input_after_geometry'] = x.detach()
        for number,layer in enumerate(self.layers,1):
            x = layer(x, geometry_basis, valid_mask)
            if stage_outputs is not None:
                stage_outputs[f'network_layer{number:02d}'] = x.detach()
        return self.final_norm(x).masked_fill(~valid_mask[..., None], 0.0)


__all__ = ["STHBrainEncoder", "STHBrainTransformerLayer"]
