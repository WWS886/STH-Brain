"""STH-Brain signal-free target queries attend only to visible student context.

Queries contain position information; target BOLD never enters the predictor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from model.sth_brain_encoder import (
    _apply_physical_sope,
)
from model.sth_brain_geometry import QUERY_DIM


class STHBrainTargetCrossAttentionLayer(nn.Module):
    """STH-Brain target-to-context cross-attention。

    target hidden 只能形成 Q，冻结在本层输入侧的 student context 只能形成 K/V；
    因而 target token 之间不存在 attention 边。Q/K 仍使用各自的物理时间与空间
    SoPE，padding context 通过显式 additive mask 排除。
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        spatial_head_fraction: float = 0.5,
        query_residual: bool = True,
        output_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.dropout_p = float(dropout)
        self.spatial_head_fraction = float(spatial_head_fraction)
        self.query_residual = bool(query_residual)
        self.q_projection = nn.Linear(dim, dim)
        self.kv_projection = nn.Linear(dim, 2 * dim)
        self.out_projection = nn.Linear(dim, dim, bias=bool(output_bias))
        self.ffn_in = nn.Linear(dim, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        target_basis: torch.Tensor,
        context_basis: torch.Tensor,
        target_valid: torch.Tensor,
        context_valid: torch.Tensor,
    ) -> torch.Tensor:
        if bool((context_valid.sum(dim=1) == 0).any()):
            raise ValueError("Every STH-Brain predictor sample needs at least one context token")
        batch, target_length, _ = target.shape
        context_length = int(context.shape[1])
        target_norm = self.norm1(target)
        context_norm = self.norm1(context)
        q = self.q_projection(target_norm).reshape(
            batch, target_length, self.heads, self.head_dim
        ).transpose(1, 2)
        kv = self.kv_projection(context_norm).reshape(
            batch, context_length, 2, self.heads, self.head_dim
        )
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = _apply_physical_sope(q, target_basis, self.spatial_head_fraction)
        k = _apply_physical_sope(k, context_basis, self.spatial_head_fraction)
        additive_mask = torch.zeros(
            (batch, 1, 1, context_length), dtype=q.dtype, device=q.device
        ).masked_fill(
            ~context_valid[:, None, None, :],
            float("-inf"),
        )
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch, target_length, self.dim
        )
        attended_update = self.dropout(self.out_projection(attended))
        # STH-Brain anchor 只在第一层关闭 direct query-content residual；后续层的
        # residual 连接保留，因为其输入已经经过 context V。
        # 输出仍可能退化为常数，训练时需监测表征变化。
        target = target + attended_update if self.query_residual else attended_update
        hidden = self.norm2(target)
        target = target + self.dropout(
            self.ffn_out(self.dropout(F.gelu(self.ffn_in(hidden))))
        )
        return target.masked_fill(~target_valid[..., None], 0.0)


class STHBrainPredictor(nn.Module):
    """从可见 context latent 交叉注意力预测目标潜表示。"""
    def __init__(
        self,
        model_dim: int = 256,
        predictor_dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        spatial_head_fraction: float = 0.5,
        mode: str = "context_anchored_cross_attention",
        first_layer_query_residual: bool = False,
        cross_attention_output_bias: bool = False,
    ) -> None:
        super().__init__()
        self.context_projection = nn.Linear(model_dim, predictor_dim)
        # The bias is the learned target seed.
        self.query_projection = nn.Linear(QUERY_DIM, predictor_dim)
        self.mode = str(mode)
        if self.mode != "context_anchored_cross_attention":
            raise ValueError('STH-Brain requires context-anchored cross attention')
        self.layers = nn.ModuleList(
            [
                STHBrainTargetCrossAttentionLayer(
                    dim=predictor_dim,
                    heads=heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    spatial_head_fraction=spatial_head_fraction,
                    query_residual=(
                        bool(first_layer_query_residual) if index == 0 else True
                    ),
                    output_bias=bool(cross_attention_output_bias),
                )
                for index in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(predictor_dim)
        self.output_projection = nn.Linear(predictor_dim, model_dim)

    def forward(
        self,
        context_latent: torch.Tensor,
        context_basis: torch.Tensor,
        context_valid: torch.Tensor,
        target_basis: torch.Tensor,
        target_valid: torch.Tensor,
        stage_outputs: dict | None = None,
    ) -> torch.Tensor:
        """返回 ``[B, Lt, model_dim]`` target prediction，padding target 严格置零。"""
        if context_latent.shape[:2] != context_valid.shape:
            raise ValueError("Predictor context shapes do not agree")
        if target_basis.shape[:2] != target_valid.shape:
            raise ValueError("Predictor target shapes do not agree")
        context = self.context_projection(context_latent)
        target_queries = self.query_projection(target_basis)
        if stage_outputs is not None:stage_outputs['target_query']=target_queries.detach()
        # Context 已在 student encoder 内完成 context-context 交互；predictor
        # 每层只更新 target queries，禁止 target-target self-attention。
        target_hidden = target_queries.masked_fill(
            ~target_valid[..., None], 0.0
        )
        for number,layer in enumerate(self.layers,1):
            target_hidden = layer(
                target_hidden,
                context,
                target_basis,
                context_basis,
                target_valid,
                context_valid,
            )
            if stage_outputs is not None:stage_outputs[f'layer{number:02d}']=target_hidden.detach()
        target_hidden = self.final_norm(target_hidden)
        if stage_outputs is not None:stage_outputs['final_norm']=target_hidden.detach()
        output = self.output_projection(target_hidden).masked_fill(
            ~target_valid[..., None], 0.0
        )
        if stage_outputs is not None:stage_outputs['output']=output.detach()
        return output


__all__ = ["STHBrainPredictor", "STHBrainTargetCrossAttentionLayer"]
