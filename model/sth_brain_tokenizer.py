"""STH-Brain  的 block 内、atlas-network 级 BOLD tokenizer。

输入：``[B,K,Q,200]`` block BOLD、sample valid/offset/duration 与显式 token indices。
输出：按 indices 原顺序排列的 ``[M, model_dim]`` token。student 只传 context indices，
teacher 传全部有效 indices，因此 target/exclusion BOLD 不会进入 student tokenizer。
"""

from __future__ import annotations

import math

import torch



def _local_time_embedding(
    offsets: torch.Tensor,
    duration: torch.Tensor,
    width: int,
    *,
    mode: str = "relative_block_cycles",
    frequency_min_hz: float = 0.03125,
    frequency_max_hz: float = 0.5,
) -> torch.Tensor:
    """把 block 内秒偏移转换为无参数正余弦编码，输出最后一维为 ``width``。"""
    if mode == "relative_block_cycles":
        phase = 2.0 * math.pi * offsets / duration[:, None].clamp_min(1e-6)
        # Preserve the STH-Brain shared helper's relative-block convention.
        lower, upper = 0.25, 4.0
    elif mode == "physical_seconds":
        if not (math.isfinite(frequency_min_hz) and math.isfinite(frequency_max_hz)
                and 0 < frequency_min_hz <= frequency_max_hz):
            raise ValueError("Physical time frequencies must be finite, positive and ordered")
        phase = 2.0 * math.pi * offsets
        lower, upper = frequency_min_hz, frequency_max_hz
    else:
        raise ValueError("Unknown local time encoding mode")
    frequencies = torch.logspace(math.log10(lower), math.log10(upper), width // 2,
                                device=offsets.device, dtype=offsets.dtype)
    angles = phase[..., None] * frequencies
    encoded = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)
    if encoded.shape[-1] >= width:
        return encoded[..., :width]
    return torch.nn.functional.pad(encoded, (0, width - encoded.shape[-1]))

