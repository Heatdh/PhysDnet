"""
DEA-Net: Detail-Enhanced Attention Network (Chen et al., IEEE TIP 2024)

Architecture:
  - 3-level U-Net with Detail-Enhanced Convolution (DEConv) blocks
  - Content-Guided Attention (CGA) fusion at skip connections
  - DEBlocks at levels 1-2, DEABlocks (with CGA attention) at level 3

Default: base_dim=32 (~1.3M params).

NOTE: The original implementation uses `torch.cuda.FloatTensor` for
intermediate weight buffers in DEConv, which breaks CPU / ONNX export.
We fix this by using `torch.zeros(..., device=weight.device)` instead.

Reference:
  Z. Chen, Z. He, Z.-M. Lu,
  "DEA-Net: Single image dehazing based on detail-enhanced convolution
   and content-guided attention,"
  IEEE Transactions on Image Processing, vol. 33, pp. 1002-1015, 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .common import BaseDehazeModel, GatedFusion, PhysicsHead


# ── Detail-Enhanced Convolution components ────────────────────────────────

class Conv2d_cd(nn.Module):
    """Central difference convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=bias)

    def get_weight(self):
        w = self.conv.weight
        s = w.shape
        w_flat = Rearrange('co ci k1 k2 -> co ci (k1 k2)')(w)
        # Fixed: use device-agnostic zeros instead of torch.cuda.FloatTensor
        w_cd = torch.zeros(s[0], s[1], 9, device=w.device, dtype=w.dtype)
        w_cd[:, :, :] = w_flat[:, :, :]
        w_cd[:, :, 4] = w_flat[:, :, 4] - w_flat[:, :, :].sum(2)
        w_cd = w_cd.view(s[0], s[1], s[2], s[3])
        return w_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    """Angular difference convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=bias)

    def get_weight(self):
        w = self.conv.weight
        s = w.shape
        w_flat = Rearrange('co ci k1 k2 -> co ci (k1 k2)')(w)
        w_ad = w_flat - w_flat[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        w_ad = w_ad.view(s[0], s[1], s[2], s[3])
        return w_ad, self.conv.bias


class Conv2d_hd(nn.Module):
    """Horizontal difference convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=bias)

    def get_weight(self):
        w = self.conv.weight
        s = w.shape
        # Fixed: device-agnostic
        w_hd = torch.zeros(s[0], s[1], 9, device=w.device, dtype=w.dtype)
        w_hd[:, :, [0, 3, 6]] = w[:, :, :]
        w_hd[:, :, [2, 5, 8]] = -w[:, :, :]
        w_hd = w_hd.view(s[0], s[1], 3, 3)
        return w_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    """Vertical difference convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=bias)

    def get_weight(self):
        w = self.conv.weight
        s = w.shape
        # Fixed: device-agnostic
        w_vd = torch.zeros(s[0], s[1], 9, device=w.device, dtype=w.dtype)
        w_vd[:, :, [0, 1, 2]] = w[:, :, :]
        w_vd[:, :, [6, 7, 8]] = -w[:, :, :]
        w_vd = w_vd.view(s[0], s[1], 3, 3)
        return w_vd, self.conv.bias


class DEConv(nn.Module):
    """
    Detail-Enhanced Convolution: sums 5 gradient-aware convolution branches.

    At inference the 5 weight matrices are summed into a single 3x3 conv
    (re-parameterizable), but we keep the multi-branch form for training.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias
        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        return F.conv2d(x, w, b, stride=1, padding=1)


# ── Attention modules ─────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        return self.sa(torch.cat([x_avg, x_max], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=True),
        )

    def forward(self, x):
        return self.ca(self.gap(x))


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa = nn.Conv2d(2 * dim, dim, 7, padding=3,
                            padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x2 = torch.cat([x.unsqueeze(2), pattn1.unsqueeze(2)], dim=2)
        x2 = x2.view(B, C * 2, H, W)
        return self.sigmoid(self.pa(x2))


class CGAFusion(nn.Module):
    """Content-Guided Attention fusion for skip connections."""

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        return self.conv(result)


# ── DEBlock / DEABlock ────────────────────────────────────────────────────

def _default_conv(in_ch, out_ch, kernel_size, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=bias)


class DEBlock(nn.Module):
    """Basic residual block with standard convolutions."""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv1 = _default_conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = _default_conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.act1(self.conv1(x)) + x
        res = self.conv2(res) + x
        return res


class DEABlock(nn.Module):
    """DEBlock + Content-Guided Attention (channel + spatial + pixel)."""

    def __init__(self, dim, kernel_size=3, reduction=8):
        super().__init__()
        self.conv1 = _default_conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = _default_conv(dim, dim, kernel_size, bias=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x)) + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2 + x
        return res


# ── Main DEA-Net architecture ────────────────────────────────────────────

class DEANet(BaseDehazeModel):
    """
    DEA-Net with optional LiDAR input and physics head.

    Args:
        mode:     'rgb' | 'lidar' | 'lidar_physics' | 'lidar_gated'
        base_dim: base channel width (default 32)
        n_blocks: number of blocks per encoder/decoder level (default 4)
        n_bottleneck: number of DEABlocks at bottleneck (default 8)
    """

    def __init__(
        self,
        mode: str = "rgb",
        base_dim: int = 32,
        n_blocks: int = 4,
        n_bottleneck: int = 8,
    ):
        super().__init__(mode=mode)
        in_ch = self.in_channels
        d1, d2, d3 = base_dim, base_dim * 2, base_dim * 4

        # ── Encoder ──
        self.down1 = nn.Conv2d(in_ch, d1, 3, stride=1, padding=1)
        self.down2 = nn.Sequential(
            nn.Conv2d(d1, d2, 3, stride=2, padding=1), nn.ReLU(True))
        self.down3 = nn.Sequential(
            nn.Conv2d(d2, d3, 3, stride=2, padding=1), nn.ReLU(True))

        # Level 1 encoder blocks
        self.enc1 = nn.Sequential(*[DEBlock(d1) for _ in range(n_blocks)])
        # Level 2 encoder blocks
        self.fe_level2 = nn.Conv2d(d2, d2, 3, stride=1, padding=1)
        self.enc2 = nn.Sequential(*[DEBlock(d2) for _ in range(n_blocks)])
        # Level 3 bottleneck (DEABlocks with attention)
        self.fe_level3 = nn.Conv2d(d3, d3, 3, stride=1, padding=1)
        self.bottleneck = nn.Sequential(
            *[DEABlock(d3) for _ in range(n_bottleneck)])

        # ── Decoder ──
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(d3, d2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True))
        self.dec2 = nn.Sequential(*[DEBlock(d2) for _ in range(n_blocks)])

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(d2, d1, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True))
        self.dec1 = nn.Sequential(*[DEBlock(d1) for _ in range(n_blocks)])

        self.up3 = nn.Conv2d(d1, 3, 3, stride=1, padding=1)

        # CGA skip fusions
        self.mix1 = CGAFusion(d3, reduction=8)
        self.mix2 = CGAFusion(d2, reduction=4)

        # ── Gated mode: parallel depth branch ──
        if self.has_gated:
            self.dep_down1 = nn.Conv2d(2, d1, 3, stride=1, padding=1)
            self.dep_down2 = nn.Sequential(
                nn.Conv2d(d1, d2, 3, stride=2, padding=1), nn.ReLU(True))
            self.dep_enc1 = nn.Sequential(*[DEBlock(d1) for _ in range(n_blocks)])
            self.dep_enc2 = nn.Sequential(*[DEBlock(d2) for _ in range(n_blocks)])
            self.gated_fusion1 = GatedFusion(d1)
            self.gated_fusion2 = GatedFusion(d2)

        # ── Physics head ──
        if self.has_physics:
            self.physics_head = PhysicsHead(in_channels=d3)

    def _forward_features(self, x, depth_input=None):
        rgb = x[:, :3]

        # Encoder level 1
        x1 = self.enc1(self.down1(x))
        # Encoder level 2
        x2 = self.enc2(self.fe_level2(self.down2(x1)))
        # Bottleneck level 3
        x3 = self.bottleneck(self.fe_level3(self.down3(x2)))

        # Gated depth fusion at levels 1 & 2
        if self.has_gated and depth_input is not None:
            d1 = self.dep_enc1(self.dep_down1(depth_input))
            d2 = self.dep_enc2(self.dep_down2(d1))
            x1 = self.gated_fusion1(x1, d1)
            x2 = self.gated_fusion2(x2, d2)

        # Decoder with CGA skip fusions
        x3_mix = self.mix1(self.down3(x2), x3)  # skip + bottleneck
        u2 = self.dec2(self.up1(x3_mix))
        x2_mix = self.mix2(self.down2(x1), u2)  # skip + decoder
        u1 = self.dec1(self.up2(x2_mix))

        out = self.up3(u1)
        restored = torch.clamp(rgb + out, 0.0, 1.0)  # global residual

        features = x3 if self.has_physics else None
        return restored, features
