"""STH-Brain 的 ROI-level spherical attention 与 network circular moments。

SoPE 在 network pooling 之前直接旋转每个真实 ROI 的 Q/K。输入是共享时间编码器
产生的 ``[N,R_n,D]`` ROI block summaries，空间坐标只来自固定 atlas，不读取 BOLD、
疾病、site 或 subject 标签。该实现是面向 ROI atlas 的 spherical rotary adaptation，
不是对原 SoPE 论文全部模块的复现。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from model.sth_brain_geometry import QUERY_DIM, STHBrainGeometry


def _rotate_pairs(values: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs; leave an odd tail channel unchanged."""
    pair_count = values.shape[-1] // 2
    if pair_count == 0:
        return values
    paired_width = pair_count * 2
    pairs = values[..., :paired_width].reshape(*values.shape[:-1], pair_count, 2)
    sine = torch.sin(phase)[..., :pair_count]
    cosine = torch.cos(phase)[..., :pair_count]
    even, odd = pairs.unbind(dim=-1)
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)
    return (
        rotated
        if paired_width == values.shape[-1]
        else torch.cat((rotated, values[..., paired_width:]), dim=-1)
    )


def roi_spherical_coordinates(xyz: torch.Tensor) -> torch.Tensor:
    """Return normalized radius, polar theta and azimuth phi for each ROI."""
    xyz = xyz.float()
    radius = torch.linalg.vector_norm(xyz, dim=-1).clamp_min(1.0e-6)
    radius_normalized = radius / radius.max().clamp_min(1.0e-6)
    theta = torch.acos((xyz[:, 2] / radius).clamp(-1.0, 1.0))
    phi = torch.atan2(xyz[:, 1], xyz[:, 0])
    return torch.stack((radius_normalized, theta, phi), dim=-1)


def circular_network_moments(
    xyz: torch.Tensor, roi_to_network: torch.Tensor, num_networks: int
) -> torch.Tensor:
    """Nine signal-free statistics retaining mean direction and concentration."""
    spherical = roi_spherical_coordinates(xyz)
    radius, theta, phi = spherical.unbind(dim=-1)
    xyz_scale = xyz.abs().amax(dim=0).clamp_min(1.0e-6)
    xyz_normalized = xyz / xyz_scale
    rows: list[torch.Tensor] = []
    for network_id in range(int(num_networks)):
        selected = roi_to_network.eq(network_id)
        if not bool(selected.any()):
            raise ValueError(f"STH-Brain atlas network {network_id} contains no ROI")
        theta_sine = torch.sin(theta[selected]).mean()
        theta_cosine = torch.cos(theta[selected]).mean()
        phi_sine = torch.sin(phi[selected]).mean()
        phi_cosine = torch.cos(phi[selected]).mean()
        rows.append(
            torch.stack(
                (
                    radius[selected].mean(),
                    torch.where(torch.hypot(theta_sine, theta_cosine) > 1e-6, torch.sin(torch.atan2(theta_sine, theta_cosine)), theta_sine.new_zeros(())),
                    torch.where(torch.hypot(theta_sine, theta_cosine) > 1e-6, torch.cos(torch.atan2(theta_sine, theta_cosine)), theta_cosine.new_ones(())),
                    torch.where(torch.hypot(phi_sine, phi_cosine) > 1e-6, torch.sin(torch.atan2(phi_sine, phi_cosine)), phi_sine.new_zeros(())),
                    torch.where(torch.hypot(phi_sine, phi_cosine) > 1e-6, torch.cos(torch.atan2(phi_sine, phi_cosine)), phi_cosine.new_ones(())),
                    xyz_normalized[selected, 0].mean(),
                    torch.hypot(theta_sine, theta_cosine),
                    xyz_normalized[selected, 2].mean(),
                    torch.hypot(phi_sine, phi_cosine),
                )
            )
        )
    return torch.stack(rows)


class RoiGeometry(STHBrainGeometry):
    """51-D queries with circular network moments and bounded time frequencies."""

    def __init__(
        self,
        atlas_path: str | Path,
        time_periods: Sequence[float],
        *,
        maximum_time_frequency_octaves: float = 1.0,
    ) -> None:
        super().__init__(atlas_path, time_periods)
        if float(maximum_time_frequency_octaves) <= 0.0:
            raise ValueError("maximum_time_frequency_octaves must be positive")
        self.maximum_time_frequency_octaves = float(maximum_time_frequency_octaves)
        self.time_frequency_logits = nn.Parameter(torch.zeros(8))
        spherical = roi_spherical_coordinates(self.atlas_xyz)
        self.register_buffer("roi_spherical", spherical, persistent=True)
        # Circular network moments occupy the fixed [8,9] spatial basis.
        self.network_spatial_basis.copy_(
            circular_network_moments(
                self.atlas_xyz, self.roi_to_network, self.num_networks
            )
        )

    def bounded_time_frequencies(self) -> torch.Tensor:
        base = 2.0 * math.pi / self.time_periods.float()
        multiplier = torch.exp(
            math.log(2.0)
            * self.maximum_time_frequency_octaves
            * torch.tanh(self.time_frequency_logits.float())
        )
        return base * multiplier

    def query_basis(
        self,
        center_seconds: torch.Tensor,
        duration_seconds: torch.Tensor,
        tr_seconds: torch.Tensor,
        network_ids: torch.Tensor,
        mask_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        center_seconds = center_seconds.float()
        duration_seconds = duration_seconds.float()
        tr_seconds = tr_seconds.float()
        network_ids = network_ids.long()
        if mask_types is None:
            mask_types = torch.zeros_like(network_ids)
        mask_types = mask_types.long()
        if not (
            center_seconds.shape
            == duration_seconds.shape
            == tr_seconds.shape
            == network_ids.shape
            == mask_types.shape
        ):
            raise ValueError("All STH-Brain query fields must have identical shapes")
        if network_ids.numel() and (
            int(network_ids.min()) < 0 or int(network_ids.max()) >= self.num_networks
        ):
            raise ValueError("STH-Brain network_ids violate the atlas contract")
        angles = center_seconds[..., None] * self.bounded_time_frequencies()
        time_features = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)
        duration_features = self._log_fourier_8(duration_seconds)
        tr_features = self._log_fourier_8(tr_seconds)
        network_features = F.one_hot(
            network_ids, num_classes=self.num_networks
        ).to(center_seconds.dtype)
        spatial_features = self.network_spatial_basis[network_ids]
        mask_features = torch.stack(
            (
                (mask_types == 1) | (mask_types == 3),
                (mask_types == 2) | (mask_types == 3),
            ),
            dim=-1,
        ).to(center_seconds.dtype)
        result = torch.cat(
            (
                time_features,
                duration_features,
                tr_features,
                network_features,
                spatial_features,
                mask_features,
            ),
            dim=-1,
        )
        if result.shape[-1] != QUERY_DIM:
            raise RuntimeError(f"STH-Brain query width drifted to {result.shape[-1]}")
        return result

    def roi_phase(self, roi_ids: torch.Tensor, pair_count: int) -> torch.Tensor:
        """Component-specific rho/theta/phi phases, shape ``[R,pair_count]``."""
        if pair_count < 1:
            return self.roi_spherical.new_empty((roi_ids.numel(), 0))
        coords = self.roi_spherical[roi_ids.long()]
        phases: list[torch.Tensor] = []
        component_counts = [0, 0, 0]
        for pair_id in range(int(pair_count)):
            component = pair_id % 3
            component_counts[component] += 1
            frequency = float(2 ** (component_counts[component] - 1))
            coordinate = coords[:, component]
            if component == 0:
                coordinate = math.pi * coordinate
            phases.append(frequency * coordinate)
        return torch.stack(phases, dim=-1)



class RoiSphericalSelfAttention(nn.Module):
    """Within-network ROI attention whose Q/K are rotated by atlas coordinates."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        spatial_head_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("ROI-SoPE dim must be divisible by heads")
        if not 0.0 < float(spatial_head_fraction) <= 1.0:
            raise ValueError("ROI-SoPE spatial_head_fraction must lie in (0,1]")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.spatial_heads = max(1, int(round(self.heads * spatial_head_fraction)))
        self.dropout = float(dropout)
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(
        self,
        values: torch.Tensor,
        roi_ids: torch.Tensor,
        geometry: RoiGeometry,
        roi_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("ROI-SoPE expects [N,R,D]")
        batch, roi_count, _ = values.shape
        if int(roi_ids.numel()) != roi_count:
            raise ValueError("ROI-SoPE coordinate count does not match ROI tokens")
        if roi_valid_mask is not None:
            roi_valid_mask = roi_valid_mask.to(values.device).bool()
            if roi_valid_mask.shape != (batch, roi_count):
                raise ValueError("ROI-SoPE validity mask must have shape [N,R]")
            if bool((~roi_valid_mask.any(dim=1)).any()):
                raise ValueError("ROI-SoPE requires at least one valid ROI per row")
        hidden = self.norm(values)
        qkv = self.qkv(hidden).reshape(
            batch, roi_count, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        phase = geometry.roi_phase(roi_ids, self.head_dim // 2)
        phase = phase.to(device=query.device, dtype=query.dtype)[None, None]
        rotated_query = _rotate_pairs(query[:, : self.spatial_heads], phase)
        rotated_key = _rotate_pairs(key[:, : self.spatial_heads], phase)
        query = torch.cat((rotated_query, query[:, self.spatial_heads :]), dim=1)
        key = torch.cat((rotated_key, key[:, self.spatial_heads :]), dim=1)
        attention_mask = (
            roi_valid_mask[:, None, None, :]
            if roi_valid_mask is not None else None
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, roi_count, self.dim)
        output = values + self.output(attended)
        return (
            output.masked_fill(~roi_valid_mask[..., None], 0.0)
            if roi_valid_mask is not None else output
        )


__all__ = [
    "RoiGeometry",
    "RoiSphericalSelfAttention",
    "circular_network_moments",
    "roi_spherical_coordinates",
]
