"""
FFA-Net: Feature Fusion Attention Network (Qin et al., AAAI 2020)

Architecture:
  - Multiple Feature Attention (FA) blocks grouped into G groups of B blocks
  - Each FA block: Conv -> ReLU -> Conv + Channel Attention + Pixel Attention
  - Groups are concatenated then fused with 1x1 conv
  - Global residual learning: output = input + learned_residual

Default: G=3, B=3 (~0.5M params) for parameter parity with our lite model.
Full paper config: G=3, B=36 (~4.7M params) — set via yaml.

Reference:
  X. Qin, Z. Wang, Y. Bai, X. Xie, H. Jia,
  "FFA-Net: Feature Fusion Attention Network for Single Image Dehazing,"
  AAAI 2020.
"""

import torch
import torch.nn as nn

from .common import BaseDehazeModel, GatedFusion, PhysicsHead


class ChannelAttention(nn.Module):
    """Channel attention: global avg pool -> FC -> sigmoid gating."""

    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class PixelAttention(nn.Module):
    """Pixel attention: 1x1 conv -> sigmoid spatial gating."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sig(self.conv(x))


class FABlock(nn.Module):
    """Feature Attention block: Conv-ReLU-Conv + CA + PA + residual."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )
        self.ca = ChannelAttention(ch)
        self.pa = PixelAttention(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv(x)
        f = self.ca(f)
        f = self.pa(f)
        return x + f


class FAGroup(nn.Module):
    """A group of B FA blocks."""

    def __init__(self, ch: int, n_blocks: int):
        super().__init__()
        self.blocks = nn.Sequential(*[FABlock(ch) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class FFANet(BaseDehazeModel):
    """
    FFA-Net with optional LiDAR input and physics head.

    Args:
        mode:     'rgb' | 'lidar' | 'lidar_physics' | 'lidar_gated'
        ch:       base channel width (default 64)
        n_groups: number of FA groups G (default 3)
        n_blocks: number of FA blocks per group B (default 3)
    """

    def __init__(
        self,
        mode: str = "rgb",
        ch: int = 64,
        n_groups: int = 3,
        n_blocks: int = 3,
    ):
        super().__init__(mode=mode)
        in_ch = self.in_channels  # 3 for rgb/gated, 5 for lidar/lidar_physics

        # Input projection
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # G groups of B FA blocks
        self.groups = nn.ModuleList([
            FAGroup(ch, n_blocks) for _ in range(n_groups)
        ])

        # Fusion: concat G group outputs -> 1x1 conv
        self.fusion = nn.Conv2d(ch * n_groups, ch, 1, bias=False)

        # Gated mode: parallel depth encoder + gated fusion after FA groups
        if self.has_gated:
            self.dep_head = nn.Sequential(
                nn.Conv2d(2, ch, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
            self.dep_groups = nn.ModuleList([
                FAGroup(ch, n_blocks) for _ in range(n_groups)
            ])
            self.dep_fusion = nn.Conv2d(ch * n_groups, ch, 1, bias=False)
            self.gated_fusion = GatedFusion(ch)

        # Output projection (global residual: restored = hazy + residual)
        self.tail = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, 1),
        )

        if self.has_physics:
            self.physics_head = PhysicsHead(in_channels=ch)

    def _forward_features(self, x, depth_input=None):
        rgb = x[:, :3]  # for global residual

        f = self.head(x)

        # Run all groups and collect outputs
        group_outs = []
        for group in self.groups:
            f = group(f)
            group_outs.append(f)

        # Fuse group outputs
        fused = self.fusion(torch.cat(group_outs, dim=1))  # B x ch x H x W

        # Gated fusion with depth branch
        if self.has_gated and depth_input is not None:
            df = self.dep_head(depth_input)
            dep_outs = []
            for dg in self.dep_groups:
                df = dg(df)
                dep_outs.append(df)
            dep_fused = self.dep_fusion(torch.cat(dep_outs, dim=1))
            fused = self.gated_fusion(fused, dep_fused)

        # Global residual learning
        residual = self.tail(fused)
        restored = torch.clamp(rgb + residual, 0.0, 1.0)

        features = fused if self.has_physics else None
        return restored, features
