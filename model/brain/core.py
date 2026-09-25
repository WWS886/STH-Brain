"""STH-Brain sequence-only physical tokenizer primitives.

Extracted from the audited ROI/time implementation; no FC imports or pretrainer
inheritance. STH-Brain adds per-query gap rejection to both native and alternate grids.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import nn
from model.config import STHBrainConfig
from model.roi_spatial.roi_sope import RoiGeometry as PhysicalGeometry, RoiSphericalSelfAttention
from model.sth_brain_tokenizer import _local_time_embedding
from utils.physical_support import supported_query_mask



def _kernel_points(seconds: float, grid_step_seconds: float) -> int:
    points = max(1, int(round(float(seconds) / float(grid_step_seconds))))
    return points if points % 2 == 1 else points + 1

def _batched_smooth_source(
    values: torch.Tensor, valid: torch.Tensor, kernel: tuple[float, ...]
) -> torch.Tensor:
    if len(kernel) == 1:
        return values
    source = values.permute(0, 2, 1)
    support_source = valid[:, None].to(values.dtype)
    weight = values.new_tensor(kernel)[None, None].expand(values.shape[-1], 1, -1)
    support_weight = values.new_tensor(kernel)[None, None]
    radius = len(kernel) // 2
    numerator = F.conv1d(source, weight, padding=radius, groups=values.shape[-1])
    support = F.conv1d(support_source, support_weight, padding=radius).clamp_min(1.0e-12)
    return (numerator / support).permute(0, 2, 1)

def vectorized_resample_shared_physical_grid(
    values: torch.Tensor,
    valid: torch.Tensor,
    offsets: torch.Tensor,
    durations: torch.Tensor,
    *,
    grid_step_seconds: float,
    source_stride: int,
    min_source_samples: int,
    max_source_gap_seconds: float,
    antialias_kernel: tuple[float, ...],
    grid_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU-only batched interpolation with the STH-Brain support semantics.

    No row is copied to the host and no Python loop scales with token count.
    Unsupported alternate rows use the original source for a defined forward
    value while remaining excluded from the physical-consistency denominator.
    """
    if values.ndim != 3 or valid.shape != values.shape[:2] or offsets.shape != valid.shape:
        raise ValueError("STH-Brain physical-grid tensor shapes disagree")
    if grid_step_seconds <= 0 or source_stride < 1:
        raise ValueError("STH-Brain physical-grid step/stride is invalid")
    valid = valid.bool()
    row_count, source_width, roi_count = values.shape
    counts = valid.sum(dim=1)
    torch._assert_async((counts > 0).all(), "STH-Brain physical view has an empty source row")

    # Compact arbitrary valid positions first.  Current STH-Brain blocks are
    # normally left-packed, but the registered kernel must also agree with the
    # reviewed row-wise reference when a missing frame creates a hole.
    slots = torch.arange(source_width, device=values.device)[None].expand(
        row_count, -1
    )
    compact_position = valid.long().cumsum(dim=1) - 1
    base_valid = slots < counts[:, None]
    base_time = offsets.new_zeros(offsets.shape).float()
    base_value = torch.zeros_like(values)
    rows, cols = valid.nonzero(as_tuple=True)
    destinations = compact_position[rows, cols]
    base_time[rows, destinations] = offsets[rows, cols].float()
    base_value[rows, destinations] = values[rows, cols]

    source_valid = base_valid
    source_time = base_time.masked_fill(~base_valid, float("inf"))
    source_value = base_value
    eligible = torch.ones(row_count, dtype=torch.bool, device=values.device)
    if source_stride > 1:
        adjacent = base_valid[:, 1:] & base_valid[:, :-1]
        gaps = base_time[:, 1:] - base_time[:, :-1]
        max_gap = gaps.masked_fill(~adjacent, float("-inf")).amax(dim=1)
        eligible = counts.ge(int(min_source_samples)) & adjacent.any(dim=1)
        eligible &= max_gap.le(float(max_source_gap_seconds))

        smoothed = _batched_smooth_source(
            base_value, base_valid, antialias_kernel
        )
        chosen = slots.remainder(int(source_stride)).eq(0) & (slots < counts[:, None])
        last = (counts - 1).clamp_min(0)
        chosen.scatter_(1, last[:, None], True)
        compact_position = chosen.long().cumsum(dim=1) - 1
        compact_valid = torch.zeros_like(valid)
        compact_time = offsets.new_zeros(offsets.shape).float()
        compact_value = torch.zeros_like(values)
        rows, cols = chosen.nonzero(as_tuple=True)
        destinations = compact_position[rows, cols]
        compact_valid[rows, destinations] = True
        # ``cols`` indexes the compact, left-packed source rather than the
        # original padded row.  Reading ``offsets`` here would select the
        # wrong timestamps whenever an acquisition contains an interior
        # invalid frame.  Both time and value must therefore be gathered from
        # the compact compacted source.
        compact_time[rows, destinations] = base_time[rows, cols]
        compact_value[rows, destinations] = smoothed[rows, cols].to(
            compact_value.dtype
        )

        compact_adjacent = compact_valid[:, 1:] & compact_valid[:, :-1]
        compact_gaps = compact_time[:, 1:] - compact_time[:, :-1]
        compact_max_gap = compact_gaps.masked_fill(
            ~compact_adjacent, float("-inf")
        ).amax(dim=1)
        decimated_ok = compact_valid.sum(dim=1).ge(2) & compact_adjacent.any(dim=1)
        decimated_ok &= compact_max_gap.le(float(max_source_gap_seconds))
        eligible &= decimated_ok

        use_compact = eligible[:, None]
        source_valid = torch.where(use_compact, compact_valid, base_valid)
        source_time = torch.where(use_compact, compact_time, base_time)
        source_value = torch.where(
            use_compact[:, :, None], compact_value, base_value
        )
        source_time = source_time.masked_fill(~source_valid, float("inf"))

    # STH-Brain passes the statically registered maximum block-grid width.  It must
    # not be inferred from source_width: a 2 s TR block has four observations
    # but still needs the shared 0..7 s one-second query grid.
    grid_count = max(1, int(grid_count if grid_count is not None else source_width))
    query = torch.arange(grid_count, device=values.device, dtype=torch.float32)
    query = query * float(grid_step_seconds)
    first_time = source_time[:, 0]
    destination_valid = supported_query_mask(
        source_time,source_valid,durations,grid_step_seconds=grid_step_seconds,
        grid_count=grid_count,max_source_gap_seconds=max_source_gap_seconds)

    query_rows = query[None].expand(row_count, -1).contiguous()
    high = torch.searchsorted(source_time.contiguous(), query_rows, right=False)
    max_high = (source_valid.sum(dim=1) - 1).clamp_min(0)[:, None]
    high = torch.minimum(high, max_high)
    low = (high - 1).clamp_min(0)
    gather_index = low[:, :, None].expand(-1, -1, roi_count)
    low_value = source_value.gather(1, gather_index)
    high_value = source_value.gather(
        1, high[:, :, None].expand(-1, -1, roi_count)
    )
    low_time = source_time.gather(1, low)
    high_time = source_time.gather(1, high)
    no_destination = ~destination_valid.any(dim=1)
    eligible = eligible & ~no_destination
    alpha = (
        (query_rows - low_time) / (high_time - low_time).clamp_min(1.0e-6)
    ).to(values.dtype)
    interpolated = low_value + alpha[:, :, None] * (high_value - low_value)
    single = source_valid.sum(dim=1).eq(1)
    interpolated = torch.where(single[:, None, None], source_value[:, :1], interpolated)
    exact_first = query_rows.eq(first_time[:, None])
    interpolated = torch.where(exact_first[:, :, None], source_value[:, :1], interpolated)
    interpolated = interpolated.masked_fill(~destination_valid[:, :, None], 0.0)
    output_offsets = query_rows.to(offsets.dtype).masked_fill(~destination_valid, 0.0)
    return interpolated, destination_valid, output_offsets, eligible

def _pack_flat(
    values: torch.Tensor, batch_ids: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = torch.bincount(batch_ids.long(), minlength=batch_size)
    if bool((counts == 0).any()):
        raise ValueError("Every STH-Brain scan must contribute at least one row")
    max_count = int(counts.max())
    padded = values.new_zeros((batch_size, max_count, values.shape[-1]))
    valid = torch.zeros((batch_size, max_count), dtype=torch.bool, device=values.device)
    positions = torch.empty_like(batch_ids, dtype=torch.long)
    for batch_id in range(batch_size):
        rows = batch_ids.eq(batch_id).nonzero(as_tuple=False).flatten()
        size = int(rows.numel())
        padded[batch_id, :size] = values[rows]
        valid[batch_id, :size] = True
        positions[rows] = torch.arange(size, device=values.device)
    return padded, valid, positions

class PhysicalTokenizer(nn.Module):
    def __init__(self, geometry: PhysicalGeometry, cfg: STHBrainConfig) -> None:
        super().__init__()
        self.geometry = geometry
        self.block_seconds = cfg.block_seconds
        self.max_gap_seconds = cfg.max_interpolation_gap_seconds
        self.local_dim = int(cfg.local_dim)
        self.model_dim = int(cfg.model_dim)
        self.network_embedding_scale = float(cfg.network_embedding_scale)
        self.physical_grid_step_seconds = float(cfg.physical_grid_step_seconds)
        self.local_time_mode=getattr(cfg,'local_time_mode','relative_block_cycles')
        self.local_time_frequency_min_hz=float(getattr(cfg,'local_time_frequency_min_hz',.03125))
        self.local_time_frequency_max_hz=float(getattr(cfg,'local_time_frequency_max_hz',.5))
        if self.local_time_mode not in {'relative_block_cycles','physical_seconds'}:
            raise ValueError('Unknown STH-Brain local time encoding mode')
        if self.local_time_mode=='physical_seconds' and not (
                0<self.local_time_frequency_min_hz<=self.local_time_frequency_max_hz<=.5/self.physical_grid_step_seconds):
            raise ValueError('Local physical time frequencies must fit the registered query grid')
        self.physical_kernel_seconds = tuple(float(x) for x in cfg.physical_kernel_seconds)
        self.physical_kernel_points = tuple(
            _kernel_points(value, self.physical_grid_step_seconds)
            for value in self.physical_kernel_seconds
        )
        if len(set(self.physical_kernel_points)) != len(self.physical_kernel_points):
            raise ValueError("STH-Brain physical kernels collapse on the configured grid")
        self.value_embedding = nn.Linear(1, self.local_dim)
        self.temporal_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    self.local_dim,
                    self.local_dim,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=self.local_dim,
                    bias=False,
                )
                for kernel in self.physical_kernel_points
            ]
        )
        self.temporal_branch_logits = nn.Parameter(
            torch.zeros(len(self.physical_kernel_points))
        )
        self.local_encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.local_dim,
                    nhead=cfg.local_heads,
                    dim_feedforward=cfg.local_ffn_dim,
                    dropout=cfg.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(cfg.local_layers)
            ]
        )
        self.roi_sope = RoiSphericalSelfAttention(
            dim=self.local_dim,
            heads=cfg.roi_sope_heads,
            dropout=cfg.dropout,
            spatial_head_fraction=cfg.roi_sope_head_fraction,
        )
        self.local_pool_query = nn.Parameter(torch.empty(self.local_dim))
        cfg.validate_network_summary()
        self.network_summary_mode = cfg.network_summary_mode
        self.roi_pool_query = (nn.Parameter(torch.empty(self.local_dim))
                               if self.network_summary_mode == 'mean_std_attention' else None)
        self.local_block_type = nn.Parameter(torch.empty(self.local_dim))
        summary_count = 3 if self.network_summary_mode == 'mean_std_attention' else 2
        self.local_to_model = nn.Linear(summary_count * self.local_dim, self.model_dim)
        self.network_embedding = nn.Embedding(geometry.num_networks, self.model_dim)
        nn.init.normal_(self.local_pool_query, std=0.02)
        if self.roi_pool_query is not None:
            nn.init.normal_(self.roi_pool_query, std=0.02)
        nn.init.normal_(self.local_block_type, std=0.02)
        if cfg.network_embedding_init_std is not None:
            nn.init.normal_(
                self.network_embedding.weight,
                mean=0.0,
                std=float(cfg.network_embedding_init_std),
            )

    def _resample_physical_grid(self, *args, **kwargs):
        kwargs['grid_count'] = max(1, int(math.ceil(self.block_seconds/self.physical_grid_step_seconds)))
        kwargs['max_source_gap_seconds'] = self.max_gap_seconds
        return vectorized_resample_shared_physical_grid(*args, **kwargs)

    def forward(
        self,
        block_values: torch.Tensor,
        block_sample_valid: torch.Tensor,
        block_sample_offsets: torch.Tensor,
        block_duration_seconds: torch.Tensor,
        token_indices: torch.Tensor,
        *,
        roi_valid_mask: torch.Tensor | None = None,
        stage_outputs: dict | None = None,
        physical_source_stride: int = 1,
        physical_min_source_samples: int = 4,
        physical_max_source_gap_seconds: float = 3.0,
        physical_antialias_kernel: tuple[float, ...] = (0.25, 0.5, 0.25),
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if roi_valid_mask is None:
            raise ValueError("STH-Brain tokenizer requires roi_valid_mask")
        roi_valid_mask = roi_valid_mask.bool()
        if roi_valid_mask.shape != (block_values.shape[0], block_values.shape[-1]):
            raise ValueError("STH-Brain roi_valid_mask must have shape [B,R]")
        token_indices = token_indices.long()
        if token_indices.ndim != 2 or token_indices.shape[1] != 3:
            raise ValueError('Physical tokenizer indices must have shape [M,3]')
        bounds=token_indices.new_tensor((block_values.shape[0],block_values.shape[1],self.geometry.num_networks))
        if bool(((token_indices<0)|(token_indices>=bounds)).any()):
            raise ValueError('Physical tokenizer indices exceed scan/block/network bounds')
        count = int(token_indices.shape[0])
        if count == 0:
            raise ValueError('ROI tokenizer requires visible tokens')
        roi_features, roi_indices = [], []
        traced={}
        branch_weights = torch.softmax(self.temporal_branch_logits.float(), dim=0)
        for network_id in range(self.geometry.num_networks):
            rows = token_indices[:, 2].eq(network_id).nonzero(as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            selected = token_indices[rows]
            batch_ids, block_ids = selected[:, 0], selected[:, 1]
            roi_ids = self.geometry.roi_to_network.eq(network_id).nonzero(
                as_tuple=False
            ).flatten()
            selected_roi_valid = roi_valid_mask[batch_ids][:, roi_ids]
            values = block_values[batch_ids, block_ids].index_select(-1, roi_ids)
            sample_valid = block_sample_valid[batch_ids, block_ids].bool()
            offsets = block_sample_offsets[batch_ids, block_ids]
            durations = block_duration_seconds[batch_ids, block_ids]
            values, sample_valid, offsets, eligible = self._resample_physical_grid(
                values,
                sample_valid,
                offsets,
                durations,
                grid_step_seconds=self.physical_grid_step_seconds,
                source_stride=physical_source_stride,
                min_source_samples=physical_min_source_samples,
                max_source_gap_seconds=physical_max_source_gap_seconds,
                antialias_kernel=physical_antialias_kernel,
            )
            if bool((~sample_valid.any(dim=1)).any()):
                raise ValueError('Selected block has no supported physical query; filter blocks before constructing C/M/E masks')
            values = values.permute(0, 2, 1).reshape(-1, values.shape[1], 1)
            sample_valid = sample_valid[:, None, :].expand(-1, roi_ids.numel(), -1)
            sample_valid = sample_valid.reshape(-1, sample_valid.shape[-1])
            flat_roi_valid = selected_roi_valid.reshape(-1)
            safe_valid = sample_valid.clone()
            safe_valid[~flat_roi_valid] = False
            if bool((safe_valid.sum(dim=1) == 0).any()):
                safe_valid[safe_valid.sum(dim=1) == 0, 0] = True
            offsets = offsets[:, None, :].expand(-1, roi_ids.numel(), -1)
            offsets = offsets.reshape(-1, offsets.shape[-1])
            durations = durations[:, None].expand(-1, roi_ids.numel()).reshape(-1)
            hidden = self.value_embedding(values)
            hidden = hidden + self.local_block_type
            hidden = hidden + _local_time_embedding(offsets, durations, self.local_dim,
                mode=self.local_time_mode,frequency_min_hz=self.local_time_frequency_min_hz,
                frequency_max_hz=self.local_time_frequency_max_hz)
            hidden = hidden.masked_fill(~sample_valid[..., None], 0.0)
            def capture(name, value, temporal=True):
                if stage_outputs is None:return
                with torch.no_grad():
                    value=value.detach().float()
                    if temporal:value=(value*sample_valid[...,None]).sum(1)/sample_valid.sum(1).clamp_min(1)[:,None]
                    value=value.reshape(rows.numel(),roi_ids.numel(),self.local_dim)
                    traced.setdefault(name,[]).append(value[selected_roi_valid])
            capture('value_time_embedding',hidden)
            temporal_input = hidden.transpose(1, 2)
            hidden = sum(
                branch_weights[index].to(hidden.dtype)
                * F.gelu(branch(temporal_input) + temporal_input)
                for index, branch in enumerate(self.temporal_branches)
            ).transpose(1, 2)
            hidden = hidden.masked_fill(~sample_valid[..., None], 0.0)
            capture('multiscale_convolution',hidden)
            for number,layer in enumerate(self.local_encoder,1):
                hidden = layer(hidden, src_key_padding_mask=~safe_valid)
                hidden = hidden.masked_fill(~sample_valid[..., None], 0.0)
                capture(f'local_layer{number:02d}',hidden)
            time_scores = torch.einsum("sqd,d->sq", hidden, self.local_pool_query)
            time_weights = torch.softmax(
                (time_scores / math.sqrt(float(self.local_dim))).masked_fill(
                    ~safe_valid, float("-inf")
                ),
                dim=-1,
            )
            pooled_roi = torch.einsum("sq,sqd->sd", time_weights, hidden)
            pooled_roi = pooled_roi.masked_fill(~flat_roi_valid[:, None], 0.0)
            capture('time_attention_pool',pooled_roi,temporal=False)
            roi_grid = pooled_roi.reshape(rows.numel(), roi_ids.numel(), self.local_dim)
            missing = ~selected_roi_valid.any(dim=1)
            active = ~missing
            spatial_grid = torch.zeros_like(roi_grid)
            if bool(active.any()):
                spatial_grid[active] = self.roi_sope(
                    roi_grid[active],
                    roi_ids,
                    self.geometry,
                    roi_valid_mask=selected_roi_valid[active],
                )
            roi_grid = spatial_grid
            roi_grid = roi_grid.masked_fill(~selected_roi_valid[..., None], 0.0)
            capture('roi_spatial_attention',roi_grid,temporal=False)
            local_rows, local_cols = selected_roi_valid.nonzero(as_tuple=True)
            roi_features.append(roi_grid[local_rows, local_cols])
            roi_indices.append(torch.stack((batch_ids[local_rows], block_ids[local_rows],
                                            roi_ids[local_cols]), dim=1))
        components = dict(roi_features=torch.cat(roi_features), roi_indices=torch.cat(roi_indices))
        if stage_outputs is not None:
            from model.brain.content_statistics import pool_slots
            ri=components['roi_indices']
            for name,values in traced.items():
                stage_outputs['tokenizer/'+name]=pool_slots(torch.cat(values),ri[:,0],ri[:,2],len(block_values),len(self.geometry.roi_to_network))
        return components


class PhysicalBackboneMethods:
    def _basis_for_indices(
            self,
            indices: torch.Tensor,
            block_center_seconds: torch.Tensor,
            block_duration_seconds: torch.Tensor,
            tr_seconds: torch.Tensor,
            mask_types: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch_ids, block_ids, network_ids = indices.long().unbind(dim=1)
            return self.geometry.query_basis(
                block_center_seconds[batch_ids, block_ids],
                block_duration_seconds[batch_ids, block_ids],
                tr_seconds[batch_ids],
                network_ids,
                mask_types,
            )

    def _encode_rows(
            self,
            indices: torch.Tensor,
            block_values: torch.Tensor,
            block_sample_valid: torch.Tensor,
            block_sample_offsets: torch.Tensor,
            block_center_seconds: torch.Tensor,
            block_duration_seconds: torch.Tensor,
            tr_seconds: torch.Tensor,
            *,
            prepared_tokens: torch.Tensor | None = None,
            roi_valid_mask: torch.Tensor | None = None,
            stage_outputs: dict | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            tokens = prepared_tokens
            if tokens is None:
                raise ValueError('Network encoding requires temporal ROI summaries')
            basis = self._basis_for_indices(
                indices,
                block_center_seconds,
                block_duration_seconds,
                tr_seconds,
            )
            batch_size = int(block_values.shape[0])
            padded_tokens, valid, positions = _pack_flat(tokens, indices[:, 0], batch_size)
            padded_basis, basis_valid, _ = _pack_flat(basis, indices[:, 0], batch_size)
            if not torch.equal(valid, basis_valid):
                raise RuntimeError("STH-Brain token/basis packing drift")
            encoded = self.encoder(padded_tokens, padded_basis, valid, stage_outputs=stage_outputs)
            return encoded[indices[:, 0], positions], encoded, padded_basis, valid, positions

    def _network_valid_mask(self, roi_valid_mask: torch.Tensor) -> torch.Tensor:
            if roi_valid_mask.shape[-1] != self.geometry.roi_to_network.numel():
                raise ValueError("STH-Brain ROI mask and atlas width disagree")
            mapping = self.geometry.roi_to_network.to(roi_valid_mask.device)
            return torch.stack(
                [
                    roi_valid_mask[:, mapping.eq(network_id)].bool().any(dim=1)
                    for network_id in range(self.geometry.num_networks)
                ],
                dim=-1,
            )

