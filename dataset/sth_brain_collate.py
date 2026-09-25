"""STH-Brain  物理时间 block 与防泄漏 C/M/E 掩码构造。

输入：变长 ``T×200`` ROI 样本、verified TR、atlas 网络、block 秒数和 mask 超参数。
输出：``[B,K,Q,200]`` blocks、sample/block valid、秒级中心/持续时间，以及互斥且完备的
context/target/exclusion ``[B,K,8]`` mask 与 target type。相同 seed/epoch/scan/view
得到确定性掩码，并与 DDP rank/worker 无关。
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from utils.physical_support import supported_query_mask


def stable_sth_brain_mask_seed(
    global_seed: int,
    epoch: int,
    scan_uid: str,
    view_id: int = 0,
) -> int:
    """生成与 DDP rank/worker 无关的稳定 mask seed，保证重跑/多卡一致。"""
    payload = f"{int(global_seed)}|{int(epoch)}|{scan_uid}|{int(view_id)}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _atlas_network_mapping(path: str | Path) -> torch.Tensor:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            if len(fields) < 4:
                raise ValueError(f"Malformed atlas line: {line!r}")
            rows.append(int(float(fields[3])))
    if len(rows) != 200:
        raise ValueError(f"STH-Brain atlas must have 200 ROI rows, got {len(rows)}")
    labels = sorted(set(rows))
    count = len(labels)
    if count != 8:
        raise ValueError(f"STH-Brain atlas must define 8 networks, got {count}")
    remap = {label: index for index, label in enumerate(labels)}
    return torch.tensor([remap[label] for label in rows], dtype=torch.long)




class STHBrainCollate:
    """把 ROI 序列切成非重叠物理时间 block，并构造有效 token 的 C/M/E 分区。"""
    """Create non-overlapping blocks and a context/target/exclusion partition."""

    def __init__(
        self,
        *,
        roi_network_path: str,
        block_seconds: float = 8.0,
        min_block_coverage: float = 0.75,
        target_coverage: Sequence[float] = (0.30, 0.45),
        target_coverage_start: Sequence[float] | None = (0.30, 0.40),
        coverage_curriculum_epochs: int = 5,
        exclusion_seconds: Sequence[float] = (6.0, 12.0),
        min_exclusion_blocks: int = 0,
        min_context_fraction: float = 0.25,
        max_mask_resamples: int = 32,
        seed: int = 42,
        view_id: int = 0,
        pad_value: float = 0.0,
        support_aware_masks: bool = False,
        minimum_valid_networks: int = 1,
        minimum_supported_mask_tokens: int = 1,
        physical_grid_step_seconds: float = 1.0,
        max_interpolation_gap_seconds: float = 6.5,
    ) -> None:
        self.pad_value = pad_value
        self.block_seconds = float(block_seconds)
        self.min_block_coverage = float(min_block_coverage)
        self.target_coverage = tuple(float(x) for x in target_coverage)
        self.target_coverage_start = (
            None if target_coverage_start is None
            else tuple(float(x) for x in target_coverage_start)
        )
        self.coverage_curriculum_epochs = int(coverage_curriculum_epochs)
        self.exclusion_seconds = tuple(float(x) for x in exclusion_seconds)
        self.min_exclusion_blocks = int(min_exclusion_blocks)
        self.min_context_fraction = float(min_context_fraction)
        self.max_mask_resamples = int(max_mask_resamples)
        self.seed = int(seed)
        self.view_id = int(view_id)
        self.mask_strategy = "union"
        self.epoch = 0
        self.roi_to_network = _atlas_network_mapping(roi_network_path)
        self.num_networks = int(self.roi_to_network.max()) + 1
        self.support_aware_masks = bool(support_aware_masks)
        self.minimum_valid_networks = int(minimum_valid_networks)
        self.minimum_supported_mask_tokens = int(minimum_supported_mask_tokens)
        self.physical_grid_step_seconds = float(physical_grid_step_seconds)
        self.max_interpolation_gap_seconds = float(max_interpolation_gap_seconds)
        if self.block_seconds <= 0:
            raise ValueError("sth_brain_block_seconds must be positive")
        if not 0 < self.min_block_coverage <= 1:
            raise ValueError("sth_brain_min_block_coverage must lie in (0,1]")
        for name, bounds in (
            ("target_coverage", self.target_coverage),
            ("exclusion_seconds", self.exclusion_seconds),
        ):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must be ordered [low, high]")
        if not 0 < self.target_coverage[0] <= self.target_coverage[1] < 1:
            raise ValueError("target coverage must lie strictly between 0 and 1")
        if self.exclusion_seconds[0] < 0:
            raise ValueError("exclusion seconds must be non-negative")
        if self.min_exclusion_blocks < 0:
            raise ValueError("min_exclusion_blocks must be non-negative")
        if self.minimum_valid_networks < 1:
            raise ValueError("minimum_valid_networks must be positive")
        if self.minimum_supported_mask_tokens < 1:
            raise ValueError("minimum_supported_mask_tokens must be positive")
        if self.physical_grid_step_seconds <= 0:
            raise ValueError("physical_grid_step_seconds must be positive")
        if self.max_interpolation_gap_seconds <= 0:
            raise ValueError("max_interpolation_gap_seconds must be positive")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _coverage_for_epoch(self) -> tuple[float, float]:
        if self.target_coverage_start is None or self.coverage_curriculum_epochs <= 0:
            return self.target_coverage
        progress = min(1.0, max(0.0, self.epoch / self.coverage_curriculum_epochs))
        return tuple(
            start + progress * (final - start)
            for start, final in zip(self.target_coverage_start, self.target_coverage)
        )

    def _make_blocks(self, item: Dict) -> list[tuple[torch.Tensor, torch.Tensor, float, float]]:
        values = item["x"].float()
        times = item.get("timestamps_seconds", item.get("time_values"))
        if times is None:
            times = torch.arange(values.shape[0], dtype=torch.float32) * float(item["tr"])
        times = times[: values.shape[0]].float()
        if values.ndim != 2 or values.shape[1] != 200:
            raise ValueError(f"STH-Brain input must be [T,200], got {tuple(values.shape)}")
        if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(times).all()):
            raise ValueError("STH-Brain blocks reject non-finite input or time values")
        if times.numel() > 1 and not bool((times[1:] > times[:-1]).all()):
            raise ValueError("STH-Brain physical times must be strictly increasing")
        tr = float(item["tr"])
        if tr <= 0:
            raise ValueError("STH-Brain requires a positive TR")
        start_time = float(times[0])
        block_ids = torch.floor((times - start_time) / self.block_seconds + 1e-7).long()
        blocks: list[tuple[torch.Tensor, torch.Tensor, float, float]] = []
        for block_id in range(int(block_ids.max()) + 1):
            selected = block_ids.eq(block_id).nonzero(as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            block_start = start_time + block_id * self.block_seconds
            block_values = values.index_select(0, selected)
            selected_times = times.index_select(0, selected)
            offsets = selected_times - block_start
            if selected_times.numel() > 1:
                estimated_step = torch.diff(selected_times).median()
            else:
                estimated_step = selected_times.new_tensor(tr)
            observed_span = (
                selected_times[-1] - selected_times[0] + estimated_step
            ).clamp(min=0.0, max=self.block_seconds)
            coverage = float(observed_span / self.block_seconds)
            if coverage + 1.0e-8 < self.min_block_coverage:
                continue
            query_supported = supported_query_mask(
                offsets[None],
                torch.ones_like(offsets[None], dtype=torch.bool),
                offsets.new_tensor([self.block_seconds]),
                grid_step_seconds=self.physical_grid_step_seconds,
                grid_count=max(
                    1,
                    int(math.ceil(self.block_seconds / self.physical_grid_step_seconds)),
                ),
                max_source_gap_seconds=self.max_interpolation_gap_seconds,
            )
            if not bool(query_supported.any()):
                continue
            blocks.append(
                (
                    block_values,
                    offsets,
                    block_start + 0.5 * self.block_seconds,
                    self.block_seconds,
                )
            )
        if not blocks:
            raise ValueError(
                f"Scan {item.get('scan_uid', '<unknown>')} has no block meeting "
                f"coverage={self.min_block_coverage} with supported physical queries "
                f"for block_seconds={self.block_seconds}"
            )
        return blocks

    def _structured_target(
        self, block_count: int, rng: random.Random, network_count: int | None = None
    ) -> tuple[torch.Tensor, float, float]:
        """Sample a genuine temporal/network cross-stripe without fallback.

        A target is the union of one contiguous temporal stripe and a random
        subset of complete functional-network stripes. Type 1 is temporal-only,
        type 2 network-only and type 3 their intersection. Every returned mask
        therefore contains all three semantic regions.
        """
        low, high = self._coverage_for_epoch()
        network_count = self.num_networks if network_count is None else int(network_count)
        if network_count < 1:
            raise ValueError("STH-Brain support-aware masks need a valid network")
        desired = rng.uniform(low, high)
        if self.mask_strategy == 'mixture':
            kind = rng.choice((1,2))
            size = block_count if kind == 1 else network_count
            options = [i for i in range(1,size) if low <= i/size <= high]
            if not options:
                raise RuntimeError('Selected slice masking strategy infeasible at this support')
            count = min(options, key=lambda i:abs(i/size-desired))
            types = torch.zeros((block_count,network_count),dtype=torch.long)
            if kind == 1:
                start = rng.randrange(block_count-count+1);types[start:start+count,:] = 1
            else:
                types[:,rng.sample(range(network_count),count)] = 2
            return types, float(desired), count/size
        candidates: list[tuple[float, int, int]] = []
        for time_count in range(1, block_count):
            qt = time_count / block_count
            for selected_network_count in range(1, network_count):
                qn = selected_network_count / network_count
                realized = qt + qn - qt * qn
                if low <= realized <= high:
                    candidates.append(
                        (abs(realized - desired), time_count, selected_network_count)
                    )
        target_type = torch.zeros((block_count, network_count), dtype=torch.long)
        if candidates:
            best = min(x[0] for x in candidates)
            _, time_count, selected_network_count = rng.choice(
                [x for x in candidates if abs(x[0] - best) < 1e-12]
            )
            start = rng.randrange(block_count - time_count + 1)
            chosen_networks = rng.sample(range(network_count), selected_network_count)
            time_selected = torch.zeros(block_count, dtype=torch.bool)
            network_selected = torch.zeros(network_count, dtype=torch.bool)
            time_selected[start:start + time_count] = True
            network_selected[chosen_networks] = True
            time_grid = time_selected[:, None].expand_as(target_type)
            network_grid = network_selected[None, :].expand_as(target_type)
            target_type[time_grid & ~network_grid] = 1
            target_type[~time_grid & network_grid] = 2
            target_type[time_grid & network_grid] = 3
            realized = (
                time_count / block_count
                + selected_network_count / network_count
                - (time_count * selected_network_count)
                / (block_count * network_count)
            )
            return target_type, float(desired), float(realized)

        minimum = 1.0 / block_count + 1.0 / network_count - 1.0 / (
            block_count * network_count
        )
        raise RuntimeError(
            "STH-Brain structured cross-stripe is discretely infeasible; random-token "
            "fallback is forbidden: "
            f"blocks={block_count} valid_networks={network_count} "
            f"requested_bounds=({low:.8f},{high:.8f}) "
            f"minimum_cross_stripe_coverage={minimum:.8f}"
        )

    def _partition(
        self,
        block_centers: torch.Tensor,
        rng: random.Random,
        network_count: int | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        float, int, bool, float, float, bool,
    ]:
        block_count = int(block_centers.numel())
        best = None
        for _ in range(max(1, self.max_mask_resamples)):
            target_type, requested_coverage, realized_coverage = self._structured_target(
                block_count, rng, network_count
            )
            target = target_type.gt(0)
            exclusion = torch.zeros_like(target)
            guards: list[float] = []
            adjacent_guard_count = 0
            block_ids = torch.arange(block_count)
            for block_id, network_id in target.nonzero(as_tuple=False).tolist():
                guard = rng.uniform(*self.exclusion_seconds)
                guards.append(guard)
                distance = (block_centers - block_centers[block_id]).abs()
                physical_guard = distance.le(guard)
                block_guard = (block_ids - block_id).abs().le(
                    self.min_exclusion_blocks
                )
                newly_adjacent = (
                    block_guard & ~target[:, network_id] & ~exclusion[:, network_id]
                )
                adjacent_guard_count += int(newly_adjacent.sum())
                exclusion[:, network_id] |= (
                    physical_guard | block_guard
                ) & ~target[:, network_id]
            context = ~(target | exclusion)
            context_fraction = float(context.float().mean())
            candidate = (
                context,
                target,
                exclusion,
                target_type,
                sum(guards) / max(1, len(guards)),
                adjacent_guard_count,
                False,
                requested_coverage,
                realized_coverage,
            )
            if best is None or int(context.sum()) > int(best[0].sum()):
                best = candidate
            if (
                context_fraction >= self.min_context_fraction
                and bool(context.any())
                and bool(target.any())
            ):
                return (*candidate, False)
        assert best is not None
        raise RuntimeError(
            "STH-Brain could not construct a structured cross-stripe C/M/E partition; "
            "resample fallback is forbidden: "
            f"min_context_fraction={self.min_context_fraction} "
            f"best_context_fraction={float(best[0].float().mean()):.8f} "
            f"attempts={self.max_mask_resamples}"
        )

    def _collate_blocks(
        self, batch: List[Dict]
    ) -> tuple[Dict, list[Dict], list[list[tuple[torch.Tensor, torch.Tensor, float, float]]]]:
        """构建确定性的非重叠物理时间 block。

        输出预训练元数据、排序后样本与物理时间 block，随后构造 C/M/E 掩码。
        """
        ordered = sorted(batch, key=lambda item: item["length"], reverse=True)
        base = {
            'subject_uid': [item['subject_uid'] for item in ordered],
            'dataset': [item['dataset'] for item in ordered],
            'tr': torch.tensor([item['tr'] for item in ordered], dtype=torch.float32),
            'roi_valid_mask': torch.stack([item['roi_valid_mask'].bool() for item in ordered]),
        }
        per_scan_blocks = [self._make_blocks(item) for item in ordered]
        batch_size = len(ordered)
        max_blocks = max(len(x) for x in per_scan_blocks)
        max_samples = max(
            block[0].shape[0] for scan in per_scan_blocks for block in scan
        )
        block_values = torch.full(
            (batch_size, max_blocks, max_samples, 200),
            self.pad_value,
            dtype=torch.float32,
        )
        sample_valid = torch.zeros(
            (batch_size, max_blocks, max_samples), dtype=torch.bool
        )
        sample_offsets = torch.zeros_like(sample_valid, dtype=torch.float32)
        block_valid = torch.zeros((batch_size, max_blocks), dtype=torch.bool)
        block_centers = torch.zeros((batch_size, max_blocks), dtype=torch.float32)
        block_durations = torch.zeros((batch_size, max_blocks), dtype=torch.float32)
        for batch_id, blocks in enumerate(per_scan_blocks):
            for block_id, (values, offsets, center, duration) in enumerate(blocks):
                sample_count = int(values.shape[0])
                block_values[batch_id, block_id, :sample_count] = values
                sample_valid[batch_id, block_id, :sample_count] = True
                sample_offsets[batch_id, block_id, :sample_count] = offsets
                block_valid[batch_id, block_id] = True
                block_centers[batch_id, block_id] = center
                block_durations[batch_id, block_id] = duration
        base.update(
            {
                "sth_brain_block_values": block_values,
                "sth_brain_block_sample_valid": sample_valid,
                "sth_brain_block_sample_offsets": sample_offsets,
                "sth_brain_block_valid": block_valid,
                "sth_brain_block_center_seconds": block_centers,
                "sth_brain_block_duration_seconds": block_durations,
            }
        )
        return base, ordered, per_scan_blocks

    def __call__(self, batch: List[Dict]) -> Dict:
        base, ordered, per_scan_blocks = self._collate_blocks(batch)
        batch_size = len(ordered)
        block_values = base["sth_brain_block_values"]
        sample_valid = base["sth_brain_block_sample_valid"]
        sample_offsets = base["sth_brain_block_sample_offsets"]
        block_valid = base["sth_brain_block_valid"]
        block_centers = base["sth_brain_block_center_seconds"]
        block_durations = base["sth_brain_block_duration_seconds"]
        max_blocks = int(block_valid.shape[1])
        context_mask = torch.zeros(
            (batch_size, max_blocks, self.num_networks), dtype=torch.bool
        )
        target_mask = torch.zeros_like(context_mask)
        exclusion_mask = torch.zeros_like(context_mask)
        target_type = torch.zeros_like(context_mask, dtype=torch.long)
        context_fractions = torch.zeros(batch_size)
        target_fractions = torch.zeros(batch_size)
        exclusion_fractions = torch.zeros(batch_size)
        guard_means = torch.zeros(batch_size)
        adjacent_guard_fractions = torch.zeros(batch_size)
        target_fallback_used = torch.zeros(batch_size, dtype=torch.bool)
        resample_fallback_used = torch.zeros(batch_size, dtype=torch.bool)
        requested_target_fractions = torch.zeros(batch_size)
        realized_target_fractions = torch.zeros(batch_size)
        target_type_token_counts = torch.zeros((batch_size, 3), dtype=torch.long)
        target_type_fractions = torch.zeros((batch_size, 3))
        network_valid_masks = torch.ones(
            (batch_size, self.num_networks), dtype=torch.bool
        )

        for batch_id, (item, blocks) in enumerate(zip(ordered, per_scan_blocks)):
            scan_uid = str(item.get("scan_uid", item.get("sample_id", batch_id)))
            rng = random.Random(
                stable_sth_brain_mask_seed(
                    self.seed,
                    self.epoch,
                    scan_uid,
                    self.view_id,
                )
            )
            centers = block_centers[batch_id, : len(blocks)]
            if self.support_aware_masks:
                roi_valid = item.get("roi_valid_mask")
                if not isinstance(roi_valid, torch.Tensor) or roi_valid.shape != (200,):
                    raise ValueError(
                        "Support-aware STH-Brain masks require roi_valid_mask[200]"
                    )
                network_valid = torch.stack(
                    [
                        roi_valid[self.roi_to_network.eq(network_id)].bool().any()
                        for network_id in range(self.num_networks)
                    ]
                )
                valid_network_ids = network_valid.nonzero(as_tuple=False).flatten()
                if valid_network_ids.numel() < self.minimum_valid_networks:
                    raise ValueError(
                        "Support-aware STH-Brain mask is infeasible: "
                        f"scan_uid={scan_uid!r} valid_networks="
                        f"{int(valid_network_ids.numel())} required="
                        f"{self.minimum_valid_networks}"
                    )
                supported_count = len(blocks) * int(valid_network_ids.numel())
                if supported_count < self.minimum_supported_mask_tokens:
                    raise ValueError(
                        "Support-aware STH-Brain mask is infeasible: "
                        f"scan_uid={scan_uid!r} supported_tokens={supported_count} "
                        f"required={self.minimum_supported_mask_tokens}"
                    )
                network_valid_masks[batch_id] = network_valid
            else:
                valid_network_ids = torch.arange(self.num_networks)
            (
                compact_context,
                compact_target,
                compact_excluded,
                compact_types,
                guard_mean,
                adjacent_guard_count,
                target_fallback,
                requested_coverage,
                realized_coverage,
                resample_fallback,
            ) = self._partition(
                centers, rng, network_count=int(valid_network_ids.numel())
            )
            context = torch.zeros(
                (len(blocks), self.num_networks), dtype=torch.bool
            )
            target = torch.zeros_like(context)
            excluded = torch.zeros_like(context)
            types = torch.zeros_like(context, dtype=torch.long)
            context[:, valid_network_ids] = compact_context
            target[:, valid_network_ids] = compact_target
            excluded[:, valid_network_ids] = compact_excluded
            types[:, valid_network_ids] = compact_types
            if not bool(context.any()) or not bool(target.any()):
                raise RuntimeError(
                    "Support-aware STH-Brain mask construction returned an "
                    "empty context or target: "
                    f"scan_uid={scan_uid!r} context_tokens={int(context.sum())} "
                    f"target_tokens={int(target.sum())} valid_networks="
                    f"{int(valid_network_ids.numel())} blocks={len(blocks)}"
                )
            context_mask[batch_id, : len(blocks)] = context
            target_mask[batch_id, : len(blocks)] = target
            exclusion_mask[batch_id, : len(blocks)] = excluded
            target_type[batch_id, : len(blocks)] = types
            total = float(len(blocks) * int(valid_network_ids.numel()))
            context_fractions[batch_id] = context.sum() / total
            target_fractions[batch_id] = target.sum() / total
            exclusion_fractions[batch_id] = excluded.sum() / total
            guard_means[batch_id] = guard_mean
            adjacent_guard_fractions[batch_id] = adjacent_guard_count / total
            target_fallback_used[batch_id] = target_fallback
            resample_fallback_used[batch_id] = resample_fallback
            requested_target_fractions[batch_id] = requested_coverage
            realized_target_fractions[batch_id] = realized_coverage
            for type_index in range(3):
                count = int(types.eq(type_index + 1).sum())
                target_type_token_counts[batch_id, type_index] = count
                target_type_fractions[batch_id, type_index] = count / total

        valid_tokens = (
            block_valid[..., None]
            & network_valid_masks[:, None, :]
        )
        partition = (
            context_mask.to(torch.int8)
            + target_mask.to(torch.int8)
            + exclusion_mask.to(torch.int8)
        )
        partition_error = torch.zeros(batch_size)
        for batch_id in range(batch_size):
            partition_error[batch_id] = float(
                (partition[batch_id][valid_tokens[batch_id]] != 1).float().mean()
            )
        base.update(
            {
                "sth_brain_context_mask": context_mask,
                "sth_brain_target_mask": target_mask,
                "sth_brain_exclusion_mask": exclusion_mask,
                "sth_brain_target_type": target_type,
                "sth_brain_network_valid_mask": network_valid_masks,
                "mask_context_fraction": context_fractions,
                "mask_target_fraction": target_fractions,
                "mask_excluded_fraction": exclusion_fractions,
                "mask_masked_fraction": target_fractions + exclusion_fractions,
                "mask_partition_error": partition_error,
                "mask_guard_seconds_mean": guard_means,
                "mask_min_exclusion_blocks": torch.full(
                    (batch_size,), self.min_exclusion_blocks, dtype=torch.long
                ),
                "mask_adjacent_guard_excluded_fraction": adjacent_guard_fractions,
                "mask_target_fallback_used": target_fallback_used,
                "mask_resample_fallback_used": resample_fallback_used,
                "mask_target_requested_fraction": requested_target_fractions,
                "mask_target_realized_fraction": realized_target_fractions,
                "mask_target_coverage_error_abs": (
                    requested_target_fractions - realized_target_fractions
                ).abs(),
                "mask_target_type1_tokens": target_type_token_counts[:, 0],
                "mask_target_type2_tokens": target_type_token_counts[:, 1],
                "mask_target_type3_tokens": target_type_token_counts[:, 2],
                "mask_target_type1_fraction": target_type_fractions[:, 0],
                "mask_target_type2_fraction": target_type_fractions[:, 1],
                "mask_target_type3_fraction": target_type_fractions[:, 2],
            }
        )
        return base


__all__ = ["STHBrainCollate", "stable_sth_brain_mask_seed"]
