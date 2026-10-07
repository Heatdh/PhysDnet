"""
AOD-Net: All-in-One Dehazing Network (Li et al., ICCV 2017)

Architecture:
  Five conv layers producing K(x), then:
    J(x) = K(x) * I(x) - K(x) + b
  where b=1 (bias term).

  Extremely lightweight (~1.8K params RGB-only).

Reference:
  B. Li, X. Peng, Z. Wang, J. Xu, D. Feng,
  "AOD-Net: All-in-One Dehazing Network," ICCV 2017.
"""

import torch
import torch.nn as nn

from .common import BaseDehazeModel, GatedFusion, PhysicsHead


class AODNet(BaseDehazeModel):
    """
    AOD-Net with optional LiDAR input and physics head.

    Modes:
      'rgb':            3-ch input, direct restoration only
      'lidar':          5-ch input (RGB + depth + mask), direct restoration
      'lidar_physics':  5-ch input + physics head (transmission + airlight)
      'lidar_gated':    dual-branch (3ch RGB + 2ch depth) with gated fusion + physics
    """

    def __init__(self, mode: str = "rgb"):
        super().__init__(mode=mode)
        in_ch = self.in_channels  # 3 for rgb/gated, 5 for lidar/lidar_physics

        # Five conv layers matching the original AOD-Net architecture
        self.conv1 = nn.Conv2d(in_ch, 3, 1, padding=0)
        self.conv2 = nn.Conv2d(in_ch, 3, 3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, 5, padding=2)   # cat(conv1, conv2) -> 6ch
        self.conv4 = nn.Conv2d(6, 3, 7, padding=3)   # cat(conv2, conv3) -> 6ch
        self.conv5 = nn.Conv2d(12, 3, 3, padding=1)  # cat(conv1..conv4) -> 12ch

        self.relu = nn.ReLU(inplace=True)

        # Gated mode: parallel depth encoder + fusion
        if self.has_gated:
            self.dep_conv1 = nn.Conv2d(2, 3, 1, padding=0)
            self.dep_conv2 = nn.Conv2d(2, 3, 3, padding=1)
            self.dep_conv3 = nn.Conv2d(6, 3, 5, padding=2)
            self.dep_conv4 = nn.Conv2d(6, 3, 7, padding=3)
            self.dep_conv5 = nn.Conv2d(12, 3, 3, padding=1)
            # Fuse at the 12-ch concat level
            self.gated_fusion = GatedFusion(12)

        # Intermediate feature channels for physics head
        if self.has_physics:
            self.physics_head = PhysicsHead(in_channels=12)

    def _forward_features(self, x, depth_input=None):
        # RGB path
        f1 = self.relu(self.conv1(x))
        f2 = self.relu(self.conv2(x))
        f3 = self.relu(self.conv3(torch.cat([f1, f2], dim=1)))
        f4 = self.relu(self.conv4(torch.cat([f2, f3], dim=1)))
        concat_feats = torch.cat([f1, f2, f3, f4], dim=1)  # B x 12

        if self.has_gated and depth_input is not None:
            # Depth path (same structure, 2ch input)
            d1 = self.relu(self.dep_conv1(depth_input))
            d2 = self.relu(self.dep_conv2(depth_input))
            d3 = self.relu(self.dep_conv3(torch.cat([d1, d2], dim=1)))
            d4 = self.relu(self.dep_conv4(torch.cat([d2, d3], dim=1)))
            depth_feats = torch.cat([d1, d2, d3, d4], dim=1)  # B x 12
            concat_feats = self.gated_fusion(concat_feats, depth_feats)

        K = self.relu(self.conv5(concat_feats))

        # AOD-Net formula: J = K * I - K + b (b=1)
        rgb = x[:, :3]
        restored = K * rgb - K + 1.0
        restored = torch.clamp(restored, 0.0, 1.0)

        features = concat_feats if self.has_physics else None
        return restored, features
