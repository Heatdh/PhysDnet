"""
Plain U-Net ablation baseline.

Same depthwise-separable U-Net architecture as LiDARDehazeNet but with a
SINGLE encoder (no dual-branch, no GatedFusion). This isolates the
contribution of:

  (a) LiDAR input  — rgb vs lidar mode
  (b) Physics head  — lidar vs lidar_physics mode
  (c) Dual-branch gated fusion — plain_unet vs LiDARDehazeNet

Architecture:
  Single-encoder U-Net with DepthwiseSeparableConv blocks, same channel
  widths and depth as LiDARDehazeNet-lite by default (base_ch=32, ~0.37M).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseDehazeModel, GatedFusion, PhysicsHead


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv (same as model.py)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class ConvBlock(nn.Module):
    """Two depthwise-separable convs with residual shortcut."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)
        self.skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + self.skip(x)


class UpBlock(nn.Module):
    """Upsample + concat skip + ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class PlainUNet(BaseDehazeModel):
    """
    Plain single-encoder U-Net ablation.

    Args:
        mode:     'rgb' | 'lidar' | 'lidar_physics' | 'lidar_gated'
        base_ch:  base channel width (default 32, same as LiDARDehazeNet-lite)
    """

    def __init__(self, mode: str = "rgb", base_ch: int = 32):
        super().__init__(mode=mode)
        c = base_ch
        in_ch = self.in_channels  # 3 for rgb/gated, 5 for lidar/lidar_physics

        # Single encoder (no dual branch — except in gated mode)
        self.enc1 = ConvBlock(in_ch, c)        # H
        self.enc2 = ConvBlock(c, c * 2)        # H/2
        self.enc3 = ConvBlock(c * 2, c * 4)    # H/4
        self.enc4 = ConvBlock(c * 4, c * 8)    # H/8

        self.pool = nn.MaxPool2d(2)

        # Gated mode: parallel depth encoder + per-scale gated fusion
        if self.has_gated:
            self.dep_enc1 = ConvBlock(2, c)
            self.dep_enc2 = ConvBlock(c, c * 2)
            self.dep_enc3 = ConvBlock(c * 2, c * 4)
            self.dep_enc4 = ConvBlock(c * 4, c * 8)
            self.fuse1 = GatedFusion(c)
            self.fuse2 = GatedFusion(c * 2)
            self.fuse3 = GatedFusion(c * 4)
            self.fuse4 = GatedFusion(c * 8)

        # Decoder
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)
        self.dec1 = UpBlock(c * 2, c, c)

        # Direct restoration head
        self.head_restore = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(c, 3, 1),
            nn.Sigmoid(),
        )

        if self.has_physics:
            self.physics_head = PhysicsHead(in_channels=c)

    def _forward_features(self, x, depth_input=None):
        # Encoder
        e1 = self.enc1(x)                      # c, H
        e2 = self.enc2(self.pool(e1))           # 2c, H/2
        e3 = self.enc3(self.pool(e2))           # 4c, H/4
        e4 = self.enc4(self.pool(e3))           # 8c, H/8

        # Gated fusion with depth branch at each scale
        if self.has_gated and depth_input is not None:
            d1 = self.dep_enc1(depth_input)
            d2 = self.dep_enc2(self.pool(d1))
            d3 = self.dep_enc3(self.pool(d2))
            d4 = self.dep_enc4(self.pool(d3))
            e1 = self.fuse1(e1, d1)
            e2 = self.fuse2(e2, d2)
            e3 = self.fuse3(e3, d3)
            e4 = self.fuse4(e4, d4)

        # Decoder
        d3 = self.dec3(e4, e3)                  # 4c, H/4
        d2 = self.dec2(d3, e2)                  # 2c, H/2
        d1 = self.dec1(d2, e1)                  # c, H

        restored = self.head_restore(d1)

        features = d1 if self.has_physics else None
        return restored, features
