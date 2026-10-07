"""
Shared components for baseline dehazing models.

Provides:
  - BaseDehazeModel:  abstract base enforcing forward signature + output contract
  - LiDARInputAdapter: expands a 3-ch first conv to 5-ch (RGB + sparse_depth + mask)
  - PhysicsHead:      predicts transmission + airlight and reconstructs J via
                      the atmospheric scattering model (same formula as LiDARDehazeNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseDehazeModel(nn.Module, ABC):
    """
    Abstract base for all baseline dehazing models.

    Enforces the same interface as LiDARDehazeNet:
      forward(rgb_hazy, sparse_depth, mask) -> dict

    Subclasses must implement:
      _forward_features(x) -> (restored, features)
        where x is the (possibly expanded) input tensor and
        features is a tensor suitable for PhysicsHead (or None).
    """

    # Modes
    MODE_RGB = "rgb"
    MODE_LIDAR = "lidar"
    MODE_LIDAR_PHYSICS = "lidar_physics"
    MODE_LIDAR_GATED = "lidar_gated"
    VALID_MODES = {MODE_RGB, MODE_LIDAR, MODE_LIDAR_PHYSICS, MODE_LIDAR_GATED}

    def __init__(self, mode: str = "rgb"):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode = mode

    @property
    def has_lidar(self) -> bool:
        return self.mode in (self.MODE_LIDAR, self.MODE_LIDAR_PHYSICS,
                             self.MODE_LIDAR_GATED)

    @property
    def has_physics(self) -> bool:
        return self.mode in (self.MODE_LIDAR_PHYSICS, self.MODE_LIDAR_GATED)

    @property
    def has_gated(self) -> bool:
        return self.mode == self.MODE_LIDAR_GATED

    @property
    def in_channels(self) -> int:
        """Input channels for the RGB encoder (3 in gated mode, 5 in concat modes)."""
        if self.has_gated:
            return 3  # depth goes through separate branch
        return 5 if self.has_lidar else 3

    @abstractmethod
    def _forward_features(
        self, x: torch.Tensor, depth_input: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: B x C x H x W input (C=3 for rgb/gated, C=5 for lidar concat modes)
            depth_input: B x 2 x H x W (sparse_depth + mask) in gated mode, else None
        Returns:
            restored: B x 3 x H x W  direct restored image [0, 1]
            features: B x F x H' x W' intermediate features for PhysicsHead,
                      or None if physics mode is not active.
        """
        ...

    def forward(
        self,
        rgb_hazy: torch.Tensor,
        sparse_depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Build input tensor
        if self.has_gated:
            x = rgb_hazy  # B x 3 x H x W — depth handled by separate branch
            depth_input = torch.cat([sparse_depth, mask], dim=1)  # B x 2
        elif self.has_lidar:
            x = torch.cat([rgb_hazy, sparse_depth, mask], dim=1)  # B x 5 x H x W
            depth_input = None
        else:
            x = rgb_hazy  # B x 3 x H x W
            depth_input = None

        restored, features = self._forward_features(x, depth_input)

        out = {"restored": restored}

        if self.has_physics and features is not None:
            phys_out = self.physics_head(features, rgb_hazy)
            out.update(phys_out)

        return out


class GatedFusion(nn.Module):
    """
    Mask-aware gated fusion of RGB features and depth features.
    Same design as LiDARDehazeNet for fair comparison.

        gate = sigmoid(W * [f_rgb; f_depth])
        fused = gate * f_rgb + (1-gate) * f_depth
    """

    def __init__(self, ch: int):
        super().__init__()
        self.gate_conv = nn.Conv2d(ch * 2, ch, 1, bias=True)

    def forward(self, f_rgb: torch.Tensor, f_depth: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_conv(torch.cat([f_rgb, f_depth], dim=1)))
        return gate * f_rgb + (1.0 - gate) * f_depth


class PhysicsHead(nn.Module):
    """
    Predicts transmission map t(x) and global airlight A, then reconstructs
    the clear image via the atmospheric scattering model:

        J = (I - A * (1 - t)) / (t + eps)

    Same formula as LiDARDehazeNet for fair comparison.
    """

    def __init__(self, in_channels: int, spatial_size: str = "full"):
        """
        Args:
            in_channels: number of channels in the feature map
            spatial_size: "full" if features are at input resolution,
                         "bottleneck" if features are downsampled
        """
        super().__init__()
        self.spatial_size = spatial_size
        self.t_min = 0.05
        self.t_max = 0.95

        # Transmission head
        self.trans_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
        )

        # Airlight head — global pooling → 3 values
        self.airlight_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 3),
            nn.Sigmoid(),
        )

    def forward(
        self, features: torch.Tensor, rgb_hazy: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        H, W = rgb_hazy.shape[2:]

        # Transmission map
        t_raw = self.trans_head(features)
        if t_raw.shape[2:] != (H, W):
            t_raw = F.interpolate(t_raw, size=(H, W), mode="bilinear",
                                  align_corners=False)
        t_hat = self.t_min + (self.t_max - self.t_min) * torch.sigmoid(t_raw)

        # Airlight
        A_hat = self.airlight_head(features)  # B x 3

        # Physics reconstruction
        eps = 1e-4
        A_spatial = A_hat[:, :, None, None]
        physics_restored = (rgb_hazy - A_spatial * (1.0 - t_hat)) / (t_hat + eps)
        physics_restored = torch.clamp(physics_restored, 0.0, 1.0)

        return {
            "transmission": t_hat,
            "airlight": A_hat,
            "physics_restored": physics_restored,
        }


def expand_first_conv(conv: nn.Conv2d, new_in_channels: int = 5) -> nn.Conv2d:
    """
    Replace a Conv2d(3, out, ...) with Conv2d(new_in_channels, out, ...),
    copying existing RGB weights and zero-initializing the extra channels.
    """
    assert conv.in_channels == 3, f"Expected 3 input channels, got {conv.in_channels}"
    new_conv = nn.Conv2d(
        new_in_channels, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        # Copy RGB weights
        new_conv.weight[:, :3] = conv.weight
        # Zero-init extra channels
        new_conv.weight[:, 3:] = 0.0
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv
