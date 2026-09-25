"""STH-Brain  的确定性物理时间与 CC200 空间几何。

输入：``Node_CC200.node`` 的 200 个 MNI 坐标/8 网络标签，以及 block 中心时间、
持续时间、TR、network id 和 mask type。输出：固定 51 维 query/SoPE basis 与 atlas
哈希合同。本模块只编码时间和空间位置，不接收 BOLD、表型或 site。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import torch
from torch import nn


QUERY_DIM = 51


def _read_cc200_node(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, list[str], str]:
    """读取 atlas，返回坐标、紧凑网络编号、ROI 名称和文件 SHA-256。"""
    node_path = Path(path)
    if not node_path.is_file():
        raise FileNotFoundError(f"STH-Brain atlas file does not exist: {node_path}")
    raw = node_path.read_bytes()
    xyz: list[list[float]] = []
    raw_networks: list[int] = []
    names: list[str] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 6:
            raise ValueError(
                f"Malformed atlas row {line_number} in {node_path}: expected >=6 columns"
            )
        xyz.append([float(fields[0]), float(fields[1]), float(fields[2])])
        raw_networks.append(int(float(fields[3])))
        names.append(str(fields[5]))
    if len(xyz) != 200:
        raise ValueError(f"STH-Brain  requires exactly 200 ROI rows, got {len(xyz)}")
    unique = sorted(set(raw_networks))
    if len(unique) != 8:
        raise ValueError(f"STH-Brain  requires exactly 8 atlas networks, got {unique}")
    remap = {raw_id: compact_id for compact_id, raw_id in enumerate(unique)}
    roi_to_network = torch.tensor([remap[x] for x in raw_networks], dtype=torch.long)
    return (
        torch.tensor(xyz, dtype=torch.float32),
        roi_to_network,
        names,
        hashlib.sha256(raw).hexdigest(),
    )




class STHBrainGeometry(nn.Module):
    """保存不可训练 atlas 几何，并生成每个 (time, network, mask-type) 的 51 维 basis。"""
    """Atlas contract and the 51-D target-query whitelist."""

    def __init__(
        self,
        atlas_path: str | Path,
        time_periods: Sequence[float] = (8, 16, 32, 64, 120, 240, 480, 960),
    ) -> None:
        super().__init__()
        periods = tuple(float(x) for x in time_periods)
        if len(periods) != 8 or any(x <= 0 for x in periods):
            raise ValueError("sth_brain_time_periods must contain 8 positive seconds values")
        xyz, roi_to_network, roi_names, atlas_sha256 = _read_cc200_node(atlas_path)
        self.atlas_path = str(Path(atlas_path).resolve())
        self.atlas_sha256 = atlas_sha256
        self.roi_names = tuple(roi_names)
        self.register_buffer("atlas_xyz", xyz, persistent=True)
        self.register_buffer("roi_to_network", roi_to_network, persistent=True)
        self.register_buffer(
            "network_spatial_basis",
            torch.zeros((8, 9), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "time_periods",
            torch.tensor(periods, dtype=torch.float32),
            persistent=True,
        )

    @property
    def num_networks(self) -> int:
        return int(self.network_spatial_basis.shape[0])

    @staticmethod
    def _log_fourier_8(values: torch.Tensor) -> torch.Tensor:
        values = values.float().clamp_min(1e-6)
        phase = torch.log1p(values)[..., None] * values.new_tensor((1.0, 2.0, 4.0, 8.0))
        return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(-2)




__all__ = ["QUERY_DIM", "STHBrainGeometry"]
