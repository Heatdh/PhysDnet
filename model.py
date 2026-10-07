"""
PhysDNet — physics-driven dehazing network.

Three variants share a common dual-encoder U-Net backbone with gated
multi-scale fusion, depthwise-separable convolutions, and a dual output
head (direct restoration + physics inversion):

  S (lite,   base_ch=32,  ~0.72M params)
  M (robust, base_ch=64,  ~2.95M params, +SE attention)
  L (robust, base_ch=96,  ~6.59M params, +SE attention)

All variants share:
  - Two output heads:
      1) Direct restored image  J_hat
      2) Physics-guided: transmission t_hat + airlight A_hat
  - Same forward signature: (rgb_hazy, sparse_depth, mask) -> dict
  - NPU-friendly operators (DWSep conv, BatchNorm, ReLU6, bilinear up)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Variant registry --------------------------------------------------------

MODEL_VARIANTS = {
    "lite":   {"base_ch": 32, "use_attention": False},
    "robust": {"base_ch": 64, "use_attention": True},
}


def get_model(variant: str = "lite", **overrides):
    """Factory: create a PhysDNet model by variant name."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}', choose from {list(MODEL_VARIANTS)}"
        )
    kwargs = {**MODEL_VARIANTS[variant], **overrides}
    return LiDARDehazeNet(**kwargs)


# Building blocks ---------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: depthwise 3x3 + pointwise 1x1."""

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


class GatedFusion(nn.Module):
    """Mask-aware gated fusion of RGB and depth features."""

    def __init__(self, ch: int):
        super().__init__()
        self.gate_conv = nn.Conv2d(ch * 2, ch, 1, bias=True)

    def forward(self, f_rgb: torch.Tensor, f_depth: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_conv(torch.cat([f_rgb, f_depth], dim=1)))
        return gate * f_rgb + (1.0 - gate) * f_depth


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


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention (SE block)."""

    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(ch // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        w = self.sig(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# Main model --------------------------------------------------------------

class LiDARDehazeNet(nn.Module):
    """
    Lightweight dual-encoder U-Net for LiDAR-guided image dehazing.

    Inputs:
        rgb_hazy    : B x 3 x H x W   (hazy RGB image, [0,1])
        sparse_depth: B x 1 x H x W   (sparse LiDAR depth, 0=missing)
        mask        : B x 1 x H x W   (validity mask, 1=valid depth)

    Outputs (dict):
        'restored'         : B x 3 x H x W   direct restored image
        'transmission'     : B x 1 x H x W   predicted transmission map
        'airlight'         : B x 3           predicted global airlight
        'physics_restored' : B x 3 x H x W   physics-reconstructed image
    """

    def __init__(self, base_ch: int = 32, use_attention: bool = False,
                 use_cbam: bool = False, use_residual: bool = False,
                 lidar_drop_rate: float = 0.0,
                 use_physics_head: bool = True):
        super().__init__()
        c = base_ch
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.lidar_drop_rate = lidar_drop_rate
        self.use_physics_head = use_physics_head

        # RGB encoder
        self.rgb_enc1 = ConvBlock(3, c)
        self.rgb_enc2 = ConvBlock(c, c * 2)
        self.rgb_enc3 = ConvBlock(c * 2, c * 4)
        self.rgb_enc4 = ConvBlock(c * 4, c * 8)

        # Depth encoder (sparse_depth + mask = 2ch)
        self.dep_enc1 = ConvBlock(2, c)
        self.dep_enc2 = ConvBlock(c, c * 2)
        self.dep_enc3 = ConvBlock(c * 2, c * 4)
        self.dep_enc4 = ConvBlock(c * 4, c * 8)

        # Gated fusion at each scale
        self.fuse1 = GatedFusion(c)
        self.fuse2 = GatedFusion(c * 2)
        self.fuse3 = GatedFusion(c * 4)
        self.fuse4 = GatedFusion(c * 8)

        # Bottleneck attention (robust variant)
        if use_cbam:
            self.bottleneck_attn = CBAM(c * 8)
        elif use_attention:
            self.bottleneck_attn = ChannelAttention(c * 8)
        else:
            self.bottleneck_attn = nn.Identity()

        # Shared decoder
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)
        self.dec1 = UpBlock(c * 2, c, c)

        # Head 1: direct restoration
        if use_residual:
            self.head_restore = nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, 3, 1),
            )
            nn.init.zeros_(self.head_restore[-1].bias)
        else:
            self.head_restore = nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.ReLU6(inplace=True),
                nn.Conv2d(c, 3, 1),
                nn.Sigmoid(),
            )

        # Head 2: physics (transmission + airlight)
        if use_physics_head:
            self.head_trans = nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.ReLU6(inplace=True),
                nn.Conv2d(c, 1, 1),
            )
            self.t_min = 0.05
            self.t_max = 0.95
            self.head_airlight = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c * 8, 3),
                nn.Sigmoid(),
            )

        self.pool = nn.MaxPool2d(2)

    def forward(
        self,
        rgb_hazy: torch.Tensor,
        sparse_depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        if self.training and self.lidar_drop_rate > 0:
            B = sparse_depth.size(0)
            drop_mask = torch.rand(B, 1, 1, 1, device=sparse_depth.device) < self.lidar_drop_rate
            sparse_depth = sparse_depth.masked_fill(drop_mask, 0.0)
            mask = mask.masked_fill(drop_mask, 0.0)

        depth_in = torch.cat([sparse_depth, mask], dim=1)

        r1 = self.rgb_enc1(rgb_hazy)
        d1 = self.dep_enc1(depth_in)
        f1 = self.fuse1(r1, d1)

        r2 = self.rgb_enc2(self.pool(r1))
        d2 = self.dep_enc2(self.pool(d1))
        f2 = self.fuse2(r2, d2)

        r3 = self.rgb_enc3(self.pool(r2))
        d3 = self.dep_enc3(self.pool(d2))
        f3 = self.fuse3(r3, d3)

        r4 = self.rgb_enc4(self.pool(r3))
        d4 = self.dep_enc4(self.pool(d3))
        f4 = self.fuse4(r4, d4)
        f4 = self.bottleneck_attn(f4)

        x = self.dec3(f4, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)

        if self.use_residual:
            restored = torch.clamp(rgb_hazy + self.head_restore(x), 0.0, 1.0)
        else:
            restored = self.head_restore(x)

        if self.use_physics_head:
            t_raw = self.head_trans(x)
            t_hat = self.t_min + (self.t_max - self.t_min) * torch.sigmoid(t_raw)
            A_hat = self.head_airlight(f4)

            # Physics inversion: J_tilde = (I - A*(1-t)) / (t + eps)
            eps = 1e-4
            A_spatial = A_hat[:, :, None, None]
            physics_restored = (rgb_hazy - A_spatial * (1.0 - t_hat)) / (t_hat + eps)
            physics_restored = torch.clamp(physics_restored, 0.0, 1.0)
        else:
            t_hat = torch.zeros(rgb_hazy.shape[0], 1, rgb_hazy.shape[2],
                                rgb_hazy.shape[3], device=rgb_hazy.device)
            A_hat = torch.zeros(rgb_hazy.shape[0], 3, device=rgb_hazy.device)
            physics_restored = torch.zeros_like(restored)

        return {
            "restored": restored,
            "transmission": t_hat,
            "airlight": A_hat,
            "physics_restored": physics_restored,
        }


if __name__ == "__main__":
    B = 2
    for variant in MODEL_VARIANTS:
        rgb = torch.rand(B, 3, 256, 256)
        depth = torch.rand(B, 1, 256, 256)
        mask = (torch.rand(B, 1, 256, 256) > 0.95).float()

        model = get_model(variant)
        total = sum(p.numel() for p in model.parameters())
        print(f"[{variant}] {total/1e6:.2f}M params")
        with torch.no_grad():
            out = model(rgb, depth, mask)
        for k, v in out.items():
            print(f"  {k}: {v.shape}")
