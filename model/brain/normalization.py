"""Mask-safe signal normalization for STH-Brain ROI time-series tokenization."""
import torch


def selected_network_grid(
        token_indices: torch.Tensor,
        *,
        batch_size: int,
        block_count: int,
        network_count: int,
) -> torch.Tensor:
    """Return a deduplicated [B,K,N] selection grid after strict validation."""
    indices = token_indices.long()
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError('Physical tokenizer indices must have shape [M,3]')
    bounds = indices.new_tensor((batch_size, block_count, network_count))
    if bool(((indices < 0) | (indices >= bounds)).any()):
        raise ValueError('Physical tokenizer indices exceed scan/block/network bounds')
    selected = torch.zeros(
        (batch_size, block_count, network_count),
        dtype=torch.bool,
        device=indices.device,
    )
    if indices.numel():
        selected[tuple(indices.T)] = True
    return selected


def visible_scan_roi_zscore(
        block_values: torch.Tensor,
        block_sample_valid: torch.Tensor,
        roi_valid_mask: torch.Tensor,
        token_indices: torch.Tensor,
        roi_to_network: torch.Tensor,
        fallback_location: torch.Tensor,
        fallback_scale: torch.Tensor,
        *,
        network_count: int,
        minimum_samples: int,
        epsilon: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize each scan/ROI using only samples exposed by selected tokens."""
    if block_values.ndim != 4:
        raise ValueError('ROI block values must have shape [B,K,S,R]')
    batch_size, block_count, _, roi_count = block_values.shape
    if block_sample_valid.shape != block_values.shape[:3]:
        raise ValueError('Block sample validity must have shape [B,K,S]')
    if roi_valid_mask.shape != (batch_size, roi_count):
        raise ValueError('ROI validity must have shape [B,R]')
    if roi_to_network.shape != (roi_count,):
        raise ValueError('ROI-to-network mapping does not match ROI width')
    if fallback_location.shape != (roi_count,) or fallback_scale.shape != (roi_count,):
        raise ValueError('Fallback normalization statistics must be ROI aligned')
    if type(minimum_samples) is not int or minimum_samples < 1 or not epsilon > 0:
        raise ValueError('Visible normalization support and epsilon must be positive')
    if not torch.isfinite(fallback_location).all() or not torch.isfinite(fallback_scale).all() or not (fallback_scale > 0).all():
        raise ValueError('Fallback normalization statistics must be finite with positive scale')

    mapping = roi_to_network.to(token_indices.device).long()
    if network_count < 1 or bool((mapping < 0).any()) or bool((mapping >= network_count).any()):
        raise ValueError('ROI-to-network mapping exceeds declared networks')
    selected = selected_network_grid(
        token_indices,
        batch_size=batch_size,
        block_count=block_count,
        network_count=network_count,
    )
    selected_roi = selected[:, :, mapping]

    values = block_values.float()
    finite_values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    support = (
        block_sample_valid.bool()[..., None]
        & roi_valid_mask.bool()[:, None, None, :]
        & selected_roi[:, :, None, :]
        & torch.isfinite(values)
    )
    support_count = support.sum(dim=(1, 2))
    safe_values = torch.where(support, finite_values, torch.zeros_like(values))
    denominator = support_count.clamp_min(1).to(values.dtype)
    scan_location = safe_values.sum(dim=(1, 2)) / denominator

    centered = torch.where(
        support,
        finite_values - scan_location[:, None, None, :],
        torch.zeros_like(values),
    )
    scan_variance = centered.square().sum(dim=(1, 2)) / denominator
    scan_scale = scan_variance.clamp_min(0).sqrt()
    fallback = (
        (support_count < minimum_samples)
        | ~torch.isfinite(scan_location)
        | ~torch.isfinite(scan_scale)
        | (scan_scale <= epsilon)
    )
    location = torch.where(fallback, fallback_location.float()[None, :], scan_location)
    scale = torch.where(fallback, fallback_scale.float()[None, :], scan_scale)

    input_valid = (
        block_sample_valid.bool()[..., None]
        & roi_valid_mask.bool()[:, None, None, :]
        & torch.isfinite(values)
    )
    safe_input = torch.where(
        input_valid, finite_values, location[:, None, None, :]
    )
    normalized = (safe_input - location[:, None, None, :]) / scale[:, None, None, :]
    debug = {
        'support_count': support_count.detach(),
        'fallback_mask': fallback.detach(),
        'location': location.detach(),
        'scale': scale.detach(),
        'selected_network_grid': selected.detach(),
    }
    return normalized, debug
