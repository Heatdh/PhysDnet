"""
Faithful implementation of DehazeFormer from:
  'Vision Transformers for Single Image Dehazing'
  Song et al., IEEE TIP 2023
  Official: https://github.com/IDKiro/DehazeFormer

Key components reproduced:
  - Rescale Layer Normalization (RLN)
  - Window-based spatial self-attention with relative position bias
  - Shifted window mechanism (alternating blocks)
  - Depth-wise convolution parallel branch
  - Selective Kernel Fusion (SKFusion) for skip connections
  - Depth-aware weight initialization

Modified to accept arbitrary in_channels for multi-modal input (e.g., RGB + Depth).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import trunc_normal_


# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in/out requires >= 2 dimensions")
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    if dimensions > 2:
        receptive_field_size = 1
        for s in tensor.shape[2:]:
            receptive_field_size *= s
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    else:
        fan_in = num_input_fmaps
        fan_out = num_output_fmaps
    return fan_in, fan_out


def window_partition(x, window_size):
    """Partition into non-overlapping windows. (B,C,H,W) -> (B*nW, C, ws, ws)"""
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    return x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)


def window_reverse(windows, window_size, H, W):
    """Reverse window partition. (B*nW, C, ws, ws) -> (B, C, H, W)"""
    nH, nW = H // window_size, W // window_size
    B = windows.shape[0] // (nH * nW)
    x = windows.view(B, nH, nW, -1, window_size, window_size)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)


# ---------------------------------------------------------------------------
#  Rescale Layer Normalization
# ---------------------------------------------------------------------------

class RLN(nn.Module):
    """Rescale LayerNorm: normalizes input and produces rescale/rebias from
    input statistics via learnable meta-networks (1x1 convs on std/mean)."""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super().__init__()
        self.eps = eps
        self.detach_grad = detach_grad
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)
        trunc_normal_(self.meta1.weight, std=0.02)
        nn.init.constant_(self.meta1.bias, 1)
        trunc_normal_(self.meta2.weight, std=0.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((x - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        normalized = (x - mean) / std
        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)
        return normalized * self.weight + self.bias, rescale, rebias


# ---------------------------------------------------------------------------
#  MLP with depth-aware init and RLN rescale/rebias
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.network_depth = network_depth
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_features, out_features, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, rescale, rebias):
        return self.mlp(x) * rescale + rebias


# ---------------------------------------------------------------------------
#  Attention: window MHSA + optional DWConv parallel branch
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Window-based multi-head self-attention with relative position bias
    and an optional depth-wise convolution parallel branch."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 use_attn=True, conv_type=None, network_depth=1):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_attn = use_attn
        self.conv_type = conv_type

        # --- conv branch (created whenever conv_type is set) ---
        if conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1, padding_mode='reflect'),
            )
        if conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim,
                                  padding_mode='reflect')

        # --- V projection + output projection (shared by attn & conv) ---
        if conv_type in ('DWConv', 'Conv') or use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        # --- attention-only parameters ---
        if use_attn:
            self.scale = self.head_dim ** -0.5
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn_post = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
            )

            # relative position bias table  (2*ws-1)^2 x num_heads
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

            # compute pair-wise relative position index
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
            coords_flat = torch.flatten(coords, 1)                       # 2, ws*ws
            relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            self.register_buffer('relative_position_index',
                                 relative_coords.sum(-1))  # ws*ws x ws*ws

    def forward(self, x):
        B, C, H, W = x.shape

        # value for both branches
        if self.conv_type in ('DWConv', 'Conv') or self.use_attn:
            V = self.V(x)

        # ---------- window attention branch ----------
        attn_out = None
        if self.use_attn:
            QK = self.QK(x)
            QKV = torch.cat([QK, V], dim=1)           # B, 3C, H, W

            if self.shift_size > 0:
                QKV = torch.roll(QKV,
                                 shifts=(-self.shift_size, -self.shift_size),
                                 dims=(2, 3))

            QKV = window_partition(QKV, self.window_size)  # B_, 3C, ws, ws
            B_, _, Wh, Ww = QKV.shape
            N = Wh * Ww

            QK_, V_ = QKV.split([C * 2, C], dim=1)
            Q, K = QK_.reshape(B_, self.num_heads, 2, self.head_dim, N).unbind(2)
            V_ = V_.reshape(B_, self.num_heads, self.head_dim, N)

            # spatial attention within each window
            attn = (Q.transpose(-2, -1) @ K) * self.scale     # B_, heads, N, N

            # add relative position bias
            N_sq = self.window_size * self.window_size
            rpb = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(N_sq, N_sq, -1).permute(2, 0, 1).contiguous()
            attn = attn + rpb.unsqueeze(0)
            attn = attn.softmax(dim=-1)

            attn_out = (attn @ V_.transpose(-2, -1)).transpose(-2, -1)
            attn_out = attn_out.reshape(B_, C, Wh, Ww)

            attn_out = window_reverse(attn_out, self.window_size, H, W)

            if self.shift_size > 0:
                attn_out = torch.roll(attn_out,
                                      shifts=(self.shift_size, self.shift_size),
                                      dims=(2, 3))

            attn_out = self.attn_post(attn_out)

        # ---------- conv branch (operates on un-shifted V) ----------
        conv_out = None
        if self.conv_type in ('Conv', 'DWConv'):
            conv_out = self.conv(V)

        # ---------- combine ----------
        if attn_out is not None and conv_out is not None:
            return self.proj(attn_out + conv_out)
        elif attn_out is not None:
            return self.proj(attn_out)
        elif conv_out is not None:
            return self.proj(conv_out)
        return x


# ---------------------------------------------------------------------------
#  Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 window_size=8, shift_size=0, use_attn=True, conv_type='DWConv'):
        super().__init__()
        self.use_attn = use_attn

        self.norm1 = RLN(dim) if use_attn else nn.Identity()
        self.attn = Attention(dim, num_heads, window_size, shift_size,
                              use_attn, conv_type, network_depth)

        self.norm2 = RLN(dim)
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        identity = x
        if self.use_attn:
            x, rescale, rebias = self.norm1(x)
            x = self.attn(x)
            x = identity + x
            identity = x
        x, rescale, rebias = self.norm2(x)
        x = self.mlp(x, rescale, rebias)
        x = identity + x
        return x


# ---------------------------------------------------------------------------
#  Basic layer  (one encoder / decoder stage)
# ---------------------------------------------------------------------------

class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 window_size=8, attn_ratio=0., attn_loc='last', conv_type='DWConv'):
        super().__init__()

        # decide which blocks get attention
        if attn_loc == 'last':
            use_attns = [i >= depth - depth * attn_ratio for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < depth * attn_ratio for i in range(depth)]
        elif attn_loc == 'all':
            use_attns = [True] * depth
        else:
            raise ValueError(f"Unknown attn_loc: {attn_loc}")

        self.blocks = nn.ModuleList([
            TransformerBlock(
                network_depth=network_depth,
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                use_attn=use_attns[i],
                conv_type=conv_type,
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
#  Patch embed / unembed  (down / up sampling)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Strided convolution for spatial down-sampling (or identity when stride=1)."""
    def __init__(self, in_chans, embed_dim, patch_size=2, kernel_size=3):
        super().__init__()
        padding = (kernel_size - patch_size + 1) // 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=kernel_size, stride=patch_size,
                              padding=padding, padding_mode='reflect')

    def forward(self, x):
        return self.proj(x)


class PatchUnEmbed(nn.Module):
    """Conv + PixelShuffle for spatial up-sampling."""
    def __init__(self, embed_dim, out_chans, patch_size=2, kernel_size=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2,
                      kernel_size=kernel_size, padding=kernel_size // 2,
                      padding_mode='reflect'),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
#  Selective Kernel Fusion
# ---------------------------------------------------------------------------

class SKFusion(nn.Module):
    """Adaptive channel-attention fusion of two feature maps (skip + decoder)."""
    def __init__(self, dim, height=2, reduction=8):
        super().__init__()
        self.height = height
        d = max(int(dim / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, dim * height, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1).view(B, self.height, C, H, W)
        feats_sum = in_feats.sum(dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))
        return (in_feats * attn).sum(dim=1)


# ---------------------------------------------------------------------------
#  DehazeFormer
# ---------------------------------------------------------------------------

class DehazeFormer(nn.Module):
    """
    Faithful reproduction of DehazeFormer (Song et al., TIP 2023).

    5-stage U-shaped encoder-decoder:
      Stage 1 (full res) -> merge (2x down) -> Stage 2 -> merge (2x down)
      -> Stage 3 (bottleneck) -> split (2x up) + SKFusion -> Stage 4
      -> split (2x up) + SKFusion -> Stage 5 (full res) -> head

    Variants (select via `variant`):
      'T'  Tiny   dims=[24,48,96,48,24]  depths=[1,1,2,1,1]   no attn
      'S'  Small  dims=[24,48,96,48,24]  depths=[1,1,2,1,1]   partial attn
      'B'  Base   dims=[48,96,192,96,48] depths=[4,6,6,6,4]   partial attn
      'L'  Large  dims=[48,96,192,96,48] depths=[8,8,12,8,8]  partial attn
    """

    VARIANTS = {
        'T': dict(
            embed_dims=[24, 48, 96, 48, 24],
            mlp_ratios=[2., 4., 4., 2., 2.],
            depths=[1, 1, 2, 1, 1],
            num_heads=[2, 4, 6, 4, 2],
            attn_ratio=[0, 0, 0, 0, 0],
        ),
        'S': dict(
            embed_dims=[24, 48, 96, 48, 24],
            mlp_ratios=[2., 4., 4., 2., 2.],
            depths=[1, 1, 2, 1, 1],
            num_heads=[2, 4, 6, 4, 2],
            attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        ),
        'B': dict(
            embed_dims=[48, 96, 192, 96, 48],
            mlp_ratios=[2., 4., 4., 2., 2.],
            depths=[4, 6, 6, 6, 4],
            num_heads=[2, 4, 6, 4, 2],
            attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        ),
        'L': dict(
            embed_dims=[48, 96, 192, 96, 48],
            mlp_ratios=[2., 4., 4., 2., 2.],
            depths=[8, 8, 12, 8, 8],
            num_heads=[2, 4, 6, 4, 2],
            attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        ),
    }

    def __init__(self, in_channels=3, out_channels=3, variant='B', window_size=8,
                 embed_dims=None, mlp_ratios=None, depths=None, num_heads=None,
                 attn_ratio=None, conv_type=None, attn_loc='last'):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size

        # load variant defaults, allow per-parameter overrides
        cfg = self.VARIANTS.get(variant, self.VARIANTS['B']).copy()
        if embed_dims is not None:  cfg['embed_dims'] = embed_dims
        if mlp_ratios is not None:  cfg['mlp_ratios'] = mlp_ratios
        if depths is not None:      cfg['depths'] = depths
        if num_heads is not None:   cfg['num_heads'] = num_heads
        if attn_ratio is not None:  cfg['attn_ratio'] = attn_ratio

        e = cfg['embed_dims']
        m = cfg['mlp_ratios']
        d = cfg['depths']
        h = cfg['num_heads']
        a = cfg['attn_ratio']

        if conv_type is None:
            conv_type = ['DWConv'] * 5
        elif isinstance(conv_type, str):
            conv_type = [conv_type] * 5

        network_depth = sum(d)

        # --- input embedding (no spatial downsampling, just 3x3 projection) ---
        self.patch_embed = PatchEmbed(in_channels, e[0], patch_size=1, kernel_size=3)

        # --- encoder stage 1 (full resolution) ---
        self.layer1 = BasicLayer(network_depth, e[0], d[0], h[0], m[0],
                                 window_size, a[0], attn_loc, conv_type[0])
        self.patch_merge1 = PatchEmbed(e[0], e[1], patch_size=2, kernel_size=3)

        # --- encoder stage 2 (1/2 resolution) ---
        self.layer2 = BasicLayer(network_depth, e[1], d[1], h[1], m[1],
                                 window_size, a[1], attn_loc, conv_type[1])
        self.patch_merge2 = PatchEmbed(e[1], e[2], patch_size=2, kernel_size=3)

        # --- bottleneck stage 3 (1/4 resolution) ---
        self.layer3 = BasicLayer(network_depth, e[2], d[2], h[2], m[2],
                                 window_size, a[2], attn_loc, conv_type[2])

        # --- decoder stage 4 (1/2 resolution) ---
        self.patch_split1 = PatchUnEmbed(e[2], e[3], patch_size=2, kernel_size=3)
        self.fusion1 = SKFusion(e[3])
        self.layer4 = BasicLayer(network_depth, e[3], d[3], h[3], m[3],
                                 window_size, a[3], attn_loc, conv_type[3])

        # --- decoder stage 5 (full resolution) ---
        self.patch_split2 = PatchUnEmbed(e[3], e[4], patch_size=2, kernel_size=3)
        self.fusion2 = SKFusion(e[4])
        self.layer5 = BasicLayer(network_depth, e[4], d[4], h[4], m[4],
                                 window_size, a[4], attn_loc, conv_type[4])

        # --- output head (3x3 conv, no spatial change) ---
        self.patch_unembed = nn.Conv2d(e[4], out_channels, kernel_size=3,
                                        padding=1, padding_mode='reflect')

    def _pad_input(self, x):
        """Pad spatial dims to be divisible by window_size * 4."""
        _, _, h, w = x.shape
        factor = self.window_size * 4
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x

    def forward(self, x_in):
        B, C, H, W = x_in.shape
        base_img = x_in[:, :3] if C > 3 else x_in

        x_in = self._pad_input(x_in)

        # encoder
        feat = self.patch_embed(x_in)
        feat = self.layer1(feat)
        skip1 = feat

        feat = self.patch_merge1(feat)
        feat = self.layer2(feat)
        skip2 = feat

        # bottleneck
        feat = self.patch_merge2(feat)
        feat = self.layer3(feat)

        # decoder
        feat = self.patch_split1(feat)
        feat = self.fusion1([feat, skip2])
        feat = self.layer4(feat)

        feat = self.patch_split2(feat)
        feat = self.fusion2([feat, skip1])
        feat = self.layer5(feat)

        # output + residual to input RGB
        out = self.patch_unembed(feat)
        out = out[:, :, :H, :W] + base_img
        return {"preds": torch.clamp(out, 0, 1)}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for variant in ['T', 'S', 'B', 'L']:
        model = DehazeFormer(in_channels=4, variant=variant).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        dummy = torch.rand(1, 4, 256, 256, device=device)
        out = model(dummy)
        print(f"DehazeFormer-{variant}: {n_params:.2f}M params, "
              f"output={out['preds'].shape}")
