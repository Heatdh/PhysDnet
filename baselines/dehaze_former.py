"""
DehazeFormer: Vision Transformers for Single Image Dehazing (Song et al., IEEE TIP 2023)

Architecture:
  - Window-based transformer encoder-decoder (Swin-like)
  - Multi-scale stages with downsampling/upsampling via strided convolutions
  - SKFusion (Selective Kernel Fusion) for cross-scale feature merging
  - Soft reconstruction: output = input + learned_residual

Configs:
  -T (tiny):   embed_dim=24,  depths=[1,1,1,1],  num_heads=[2,4,6,8]  ~0.68M
  -S (small):  embed_dim=48,  depths=[2,2,2,2],  num_heads=[2,4,6,8]  ~5.5M
  -B (base):   embed_dim=48,  depths=[4,4,4,4],  num_heads=[2,4,6,8]  ~10.1M
  -L (large):  embed_dim=96,  depths=[4,4,4,4],  num_heads=[2,4,6,8]  ~25.4M

Default: -T for parameter parity with our lite model. Configurable via yaml.

Reference:
  Y. Song, Z. He, H. Qian, X. Du,
  "Vision Transformers for Single Image Dehazing," IEEE TIP 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .common import BaseDehazeModel, GatedFusion, PhysicsHead


# ---------------------------------------------------------------------------
# Window attention helpers
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x: B x H x W x C
        window_size: window size
    Returns:
        (B * nW) x window_size x window_size x C
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition.

    Args:
        windows: (B * nW) x window_size x window_size x C
        window_size: window size
        H, W: original spatial dimensions
    Returns:
        B x H x W x C
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ---------------------------------------------------------------------------
# Core transformer blocks
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = torch.flatten(coords, 1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        head_dim = C // self.num_heads

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Window attention + FFN with pre-norm (LayerNorm)."""

    def __init__(self, dim: int, num_heads: int, window_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        ws = self.attn.window_size

        # Pad to multiple of window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        # Window attention
        x_windows = window_partition(x, ws)        # (B*nW) x ws x ws x C
        x_windows = x_windows.view(-1, ws * ws, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, ws, ws, C)
        x = window_reverse(attn_windows, ws, Hp, Wp)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # Residual + FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoder / Decoder stages
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to patch embedding via overlapping convolution."""

    def __init__(self, in_ch: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B x (H*W) x C
        return x, H, W


class Downsample(nn.Module):
    """2x spatial downsampling via strided convolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        _, C2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H2, W2


class Upsample(nn.Module):
    """2x spatial upsampling via transposed convolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.deconv(x)
        _, C2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H2, W2


class SKFusion(nn.Module):
    """
    Selective Kernel Fusion: adaptively fuses two feature maps
    using channel-wise attention (simplified SK convolution).
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(dim, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim * 2, bias=False),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """x1 and x2 are both B x L x C."""
        u = x1 + x2
        # Global pool over spatial
        s = u.mean(dim=1)  # B x C
        z = self.fc(s)     # B x 2C
        a, b = z.chunk(2, dim=-1)  # each B x C
        a = a.unsqueeze(1)
        b = b.unsqueeze(1)
        weights = torch.softmax(torch.stack([a, b], dim=0), dim=0)
        return weights[0] * x1 + weights[1] * x2


class EncoderStage(nn.Module):
    """N transformer blocks at the same resolution."""

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, window_size)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


class DecoderStage(nn.Module):
    """N transformer blocks + SKFusion with skip connection."""

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int):
        super().__init__()
        self.sk_fusion = SKFusion(dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, window_size)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, skip: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.sk_fusion(x, skip)
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DehazeFormer(BaseDehazeModel):
    """
    DehazeFormer with optional LiDAR input and physics head.

    Args:
        mode:       'rgb' | 'lidar' | 'lidar_physics' | 'lidar_gated'
        embed_dim:  base embedding dimension (doubled at each downscale)
        depths:     list of transformer block counts per stage [enc1, enc2, enc3, enc4]
        num_heads:  list of attention head counts per stage
        window_size: window size for windowed attention (default 8)
    """

    def __init__(
        self,
        mode: str = "rgb",
        embed_dim: int = 24,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_size: int = 8,
    ):
        super().__init__(mode=mode)

        if depths is None:
            depths = [1, 1, 1, 1]
        if num_heads is None:
            num_heads = [2, 4, 6, 8]

        in_ch = self.in_channels  # 3 for rgb/gated, 5 for lidar/lidar_physics
        dims = [embed_dim * (2 ** i) for i in range(4)]  # e.g., [24, 48, 96, 192]

        # Patch embedding
        self.patch_embed = PatchEmbed(in_ch, dims[0])

        # Encoder stages
        self.enc1 = EncoderStage(dims[0], depths[0], num_heads[0], window_size)
        self.down1 = Downsample(dims[0])
        self.enc2 = EncoderStage(dims[1], depths[1], num_heads[1], window_size)
        self.down2 = Downsample(dims[1])
        self.enc3 = EncoderStage(dims[2], depths[2], num_heads[2], window_size)
        self.down3 = Downsample(dims[2])

        # Bottleneck
        self.bottleneck = EncoderStage(dims[3], depths[3], num_heads[3], window_size)

        # Gated mode: lightweight depth CNN encoder + gated fusion at bottleneck
        if self.has_gated:
            self.dep_embed = nn.Conv2d(2, dims[0], 3, padding=1)
            self.dep_down1 = nn.Sequential(
                nn.Conv2d(dims[0], dims[1], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[1]), nn.ReLU(inplace=True))
            self.dep_down2 = nn.Sequential(
                nn.Conv2d(dims[1], dims[2], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[2]), nn.ReLU(inplace=True))
            self.dep_down3 = nn.Sequential(
                nn.Conv2d(dims[2], dims[3], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[3]), nn.ReLU(inplace=True))
            # Fuse in token space at bottleneck level
            self.gated_fusion = GatedFusion(dims[3])

        # Decoder stages
        self.up3 = Upsample(dims[3])
        self.dec3 = DecoderStage(dims[2], depths[2], num_heads[2], window_size)
        self.up2 = Upsample(dims[2])
        self.dec2 = DecoderStage(dims[1], depths[1], num_heads[1], window_size)
        self.up1 = Upsample(dims[1])
        self.dec1 = DecoderStage(dims[0], depths[0], num_heads[0], window_size)

        # Output projection (soft reconstruction: output = input + residual)
        self.output_proj = nn.Conv2d(dims[0], 3, 3, padding=1)

        if self.has_physics:
            self.physics_head = PhysicsHead(in_channels=dims[0])

    def _forward_features(self, x, depth_input=None):
        rgb = x[:, :3]
        B, _, H_orig, W_orig = x.shape

        # Pad spatial dims to multiples of 8 (window_size) and 16 (3 downsamples)
        factor = 16  # 2^3 downsamples + window_size alignment
        pad_h = (factor - H_orig % factor) % factor
        pad_w = (factor - W_orig % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Patch embed
        tokens, H, W = self.patch_embed(x)

        # Encoder
        e1 = self.enc1(tokens, H, W)
        tokens, H2, W2 = self.down1(e1, H, W)

        e2 = self.enc2(tokens, H2, W2)
        tokens, H3, W3 = self.down2(e2, H2, W2)

        e3 = self.enc3(tokens, H3, W3)
        tokens, H4, W4 = self.down3(e3, H3, W3)

        # Bottleneck
        tokens = self.bottleneck(tokens, H4, W4)

        # Gated fusion at bottleneck
        if self.has_gated and depth_input is not None:
            if pad_h > 0 or pad_w > 0:
                depth_input = F.pad(depth_input, (0, pad_w, 0, pad_h), mode="reflect")
            df = self.dep_embed(depth_input)
            df = self.dep_down1(df)
            df = self.dep_down2(df)
            df = self.dep_down3(df)  # B x dims[3] x H4 x W4
            # Convert tokens to spatial, fuse, convert back
            t_spatial = tokens.view(B, H4, W4, -1).permute(0, 3, 1, 2)
            t_spatial = self.gated_fusion(t_spatial, df)
            tokens = t_spatial.flatten(2).transpose(1, 2)

        # Decoder
        tokens, Hu3, Wu3 = self.up3(tokens, H4, W4)
        tokens = self.dec3(tokens, e3, H3, W3)

        tokens, Hu2, Wu2 = self.up2(tokens, H3, W3)
        tokens = self.dec2(tokens, e2, H2, W2)

        tokens, Hu1, Wu1 = self.up1(tokens, H2, W2)
        tokens = self.dec1(tokens, e1, H, W)

        # Back to spatial
        feat = tokens.view(B, H, W, -1).permute(0, 3, 1, 2)  # B x C x H x W

        # Output projection + global residual
        residual = self.output_proj(feat)
        # Crop to original size
        residual = residual[:, :, :H_orig, :W_orig]
        feat = feat[:, :, :H_orig, :W_orig]

        restored = torch.clamp(rgb + residual, 0.0, 1.0)

        features = feat if self.has_physics else None
        return restored, features


# ---------------------------------------------------------------------------
# Preset configs (matching paper)
# ---------------------------------------------------------------------------

DEHAZE_FORMER_PRESETS = {
    "tiny": {
        "embed_dim": 24,
        "depths": [1, 1, 1, 1],
        "num_heads": [2, 4, 6, 8],
    },
    "small": {
        "embed_dim": 48,
        "depths": [2, 2, 2, 2],
        "num_heads": [2, 4, 6, 8],
    },
    "base": {
        "embed_dim": 48,
        "depths": [4, 4, 4, 4],
        "num_heads": [2, 4, 6, 8],
    },
    "large": {
        "embed_dim": 96,
        "depths": [4, 4, 4, 4],
        "num_heads": [2, 4, 6, 8],
    },
}
